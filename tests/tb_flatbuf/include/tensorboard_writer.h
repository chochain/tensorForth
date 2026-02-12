/*
 * tensorboard_writer.h  —  TensorBoard/tbparse compatible event file writer
 *
 * ── Correct proto field numbers (cross-checked against TB compat protos) ─────
 *
 * tensorflow.Event  (event.proto):
 *   1: wall_time (double)
 *   2: step (int64)
 *   3: file_version (string)
 *   +4: graph_def (bytes)
 *   5: summary (Summary)
 *   +6: log_message (LogMessage)
 *   +7: session_log (SessionLog)
 *   +8: tagged_run_metadata (TaggedRunMetaData)
 *   9: meta_graph_def (bytes)  ← do NOT write here
 *
 * = Tensor ============================================
 * tensorflow.TensorProto:
 *   1: dtype (DT_FLOAT=1)
 *   2: tensor_shape (empty=scalar)
 *   +3: version_number (int32)
 *   +4: tensor_content (bytes)
 *   5: float_val (packed)
 *   +6: double_val (packed)
 *   +7: int_val (packed)
 *   +8: string_val (bytes repeated)
 *   +9: scomplex_val (float packed)
 *   +10: int64_val (packed)
 *   +11: bool_val (packed)
 *   +12: dcomplex_val (double packed)
 *   +13: half_val (int32 packed)
 *   +14: resource_handle_val (ResourceHandleProto)
 *   +15: variant_val (VariantTensorDataProto)
 *   +16: uint32_val (packed)
 *   +17: uint64_val (packed)
 *
 * +tensorflow.ResourceHandleProto
 *
 * +tensorflow.VariantTensorDataProto:
 *   1: type_name (string)
 *   2: metadata: (bytes)
 *   3: tensors (TensorProto repeated)
 *
 * = Summary ============================================
 * tensorflow.Summary.Value  (summary.proto):
 *   1: tag (string)
 *   oneof {
 *     +2: simple_value (float)
 *     +3: obsolete_old_style_histogram (bytes)
 *     4: image (Summary.Image)
 *     5: histo (HistogramProto)
 *     +6: audio (Summary.Audio)
 *     8: tensor (TensorProto)
 *   }
 *   9: metadata (SummaryMetadata)
 *
 * tensorflow.SummaryMetadata:
 *   1: plugin_data (PluginData)
 *   +2: display_name (string)
 *   +3: summary_description (string)
 *   4: data_class (DataClass: SCALAR=1, BLOB=2, TENSOR=3)
 *
 * tensorflow.SummaryMetadata.PluginData:
 *   1: plugin_name (string)    2: content (bytes)
 *
 * tensorflow.Summary.Image:
 *   1: height
 *   2: width
 *   3: colorspace (grascale=1, grascale+alpha=2, RGB=3, RGBA=4, DIGITAL_YUV=5, RGBA=6)
 *   4: encoded_image_string (bytes)
 *
 * +tensorflow.Summary.Audio
 *
 * tensorflow.HistogramProto:
 *   1:min 2:max 3:num 4:sum 5:sum_squares
 *   6: bucket_limit 7:bucket
 *   +6: repeated bucket_limit, +7: repeated bucket
 *
 * = Projector ============================================
 * +tensorflow.ProjectorConfig
 *   1: model_checkpoint_path (string)
 *   2: embeddings (EmbeddingInfo repeated)
 *   3: model_checkpoint_dir (string)
 *
 * +tensorflow.EmbeddingInfo
 *   1: tensor_name (string)
 *   2: metadata_path (string)
 *   3: bookmark_path (string)
 *   4: tensor_shape (uint32 repeated)
 *   5: sprite (SpriteMetadata)
 *
 * +tensorflow.SpriteMetadata
 *   1: image_path (string)
 *   2: single_image_dim (uint32 repeated)
 *
 * ── TFRecord framing ─────────────────────────────────────────────────────────
 * [length:uint64 LE][masked_crc32c(length):uint32 LE][data][masked_crc32c(data):uint32 LE]
 * masking: ((crc >> 15 | crc << 17) + 0xa282ead8) & 0xFFFFFFFF
 */
#pragma once

#include "crc32c.h"
#include "proto_encode.h"
#include "tb_flatbuffers_schema.h"

#include <cstdint>
#include <cstring>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <unistd.h>

namespace tensorboard {

// ─── Path helper ─────────────────────────────────────────────────────────────
inline std::string MakeEventFilePath(const std::string& dir, int seq = 0) {
    char hostname[256] = "localhost";
    gethostname(hostname, sizeof(hostname));
    hostname[sizeof(hostname)-1] = '\0';
    for (char* p = hostname; *p; ++p)
        if (*p == '/' || *p == '\\' || *p == ':') *p = '_';
    std::ostringstream ss;
    ss << dir << "/events.out.tfevents."
       << static_cast<long>(std::time(nullptr)) << "."
       << hostname << "."
       << static_cast<int>(getpid()) << "." << seq;
    return ss.str();
}

// ─── PNG Encoder (uncompressed deflate stored blocks) ────────────────────────
namespace png {
static uint32_t adler32(const uint8_t* d, size_t n) {
    uint32_t s1=1,s2=0; for(size_t i=0;i<n;i++){s1=(s1+d[i])%65521;s2=(s2+s1)%65521;}
    return (s2<<16)|s1;
}
static uint32_t crc32_png(const uint8_t* d, size_t n) {
    static uint32_t t[256]; static bool init=false;
    if(!init){for(uint32_t i=0;i<256;i++){uint32_t c=i;for(int k=0;k<8;k++)c=(c&1)?(0xEDB88320u^(c>>1)):(c>>1);t[i]=c;}init=true;}
    uint32_t c=0xFFFFFFFFu; for(size_t i=0;i<n;i++)c=t[(c^d[i])&0xFF]^(c>>8); return c^0xFFFFFFFFu;
}
static void push_be32(std::vector<uint8_t>& v,uint32_t n){
    v.push_back((n>>24)&0xFF);v.push_back((n>>16)&0xFF);v.push_back((n>>8)&0xFF);v.push_back(n&0xFF);}
static void write_chunk(std::vector<uint8_t>& out,const char type[4],const std::vector<uint8_t>& data){
    push_be32(out,static_cast<uint32_t>(data.size()));
    out.insert(out.end(),type,type+4); out.insert(out.end(),data.begin(),data.end());
    std::vector<uint8_t> ci; ci.insert(ci.end(),type,type+4); ci.insert(ci.end(),data.begin(),data.end());
    push_be32(out,crc32_png(ci.data(),ci.size()));
}
inline std::vector<uint8_t> EncodePNG(int w,int h,const std::vector<uint8_t>& px,int ch=3){
    std::vector<uint8_t> out;
    static const uint8_t sig[]={137,80,78,71,13,10,26,10}; out.insert(out.end(),sig,sig+8);
    {std::vector<uint8_t> ihdr; push_be32(ihdr,w); push_be32(ihdr,h);
     ihdr.push_back(8); ihdr.push_back(ch==1?0:(ch==4?6:2));
     ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0); write_chunk(out,"IHDR",ihdr);}
    int rb=w*ch; std::vector<uint8_t> sl;
    for(int y=0;y<h;y++){sl.push_back(0);sl.insert(sl.end(),px.begin()+y*rb,px.begin()+(y+1)*rb);}
    std::vector<uint8_t> zlib; zlib.push_back(0x78); zlib.push_back(0x01);
    size_t pos=0,tot=sl.size();
    while(pos<tot){size_t bsz=std::min(tot-pos,(size_t)65535);bool last=(pos+bsz>=tot);
        uint16_t bl=static_cast<uint16_t>(bsz),bn=static_cast<uint16_t>(~bl);
        zlib.push_back(last?0x01:0x00);
        zlib.push_back(bl&0xFF);zlib.push_back((bl>>8)&0xFF);
        zlib.push_back(bn&0xFF);zlib.push_back((bn>>8)&0xFF);
        zlib.insert(zlib.end(),sl.begin()+pos,sl.begin()+pos+bsz);pos+=bsz;}
    push_be32(zlib,adler32(sl.data(),sl.size()));
    write_chunk(out,"IDAT",zlib); write_chunk(out,"IEND",{}); return out;
}
} // namespace png

// ─── EventWriter ─────────────────────────────────────────────────────────────
class EventWriter {
public:
    explicit EventWriter(const std::string& path)
        : file_(path, std::ios::binary | std::ios::trunc) {
        if (!file_.is_open())
            throw std::runtime_error("Cannot open event file: " + path);
        WriteFileVersionEvent();
    }
    ~EventWriter() { if (file_.is_open()) file_.close(); }

    // ── Scalar ────────────────────────────────────────────────────────────────
    void WriteScalar(const std::string& tag, float value, int64_t step) {
        proto::Encoder val;
        val.write_str(1, tag);                                 // tag
        val.write_f32(2, value);                               // simple_value
//        val.write_msg_raw(9, BuildScalarMetadata());           // metadata → field 9
//        val.write_msg_raw(8, BuildFloatTensorProto(value));    // tensor   → field 8
        WriteRecord(BuildEvent(val.buf(), step));
    }

    // ── Image (RGB row-major, 3 bytes/pixel) ──────────────────────────────────
    void WriteImage(const std::string& tag, int width, int height,
                    const std::vector<uint8_t>& pixels_rgb, int64_t step) {
        auto png_bytes = png::EncodePNG(width, height, pixels_rgb, 3);
        proto::Encoder img;
        img.write_s32(1, height);
        img.write_s32(2, width);
        img.write_s32(3, 3);  // colorspace = RGB
        img.write_bytes(4, png_bytes.data(), png_bytes.size()); // encoded_image_string
        proto::Encoder val;
        val.write_str(1, tag);                                 // tag
        val.write_msg_raw(9, BuildImageMetadata());            // metadata → field 9
        val.write_msg_raw(4, img.buf());                       // image    → field 4
        WriteRecord(BuildEvent(val.buf(), step));
    }

    // ── Histogram ─────────────────────────────────────────────────────────────
    void WriteHistogram(const std::string& tag, const std::vector<double>& values,
                        int64_t step, int num_buckets = 30) {
        if (values.empty()) return;
        double vmin = *std::min_element(values.begin(), values.end());
        double vmax = *std::max_element(values.begin(), values.end());
        double vsum = 0, vsumsq = 0;
        for (double v : values) { vsum += v; vsumsq += v*v; }
        std::vector<double> limits, counts;
        BuildBuckets(vmin, vmax, values, num_buckets, limits, counts);
        proto::Encoder histo;
        
        histo.write_f64(1, vmin);
        histo.write_f64(2, vmax);
        histo.write_f64(3, static_cast<double>(values.size()));
        histo.write_f64(4, vsum);   histo.write_f64(5, vsumsq);
        histo.write_f64_packed(6, limits);
        histo.write_f64_packed(7, counts);
        
        proto::Encoder val;
        val.write_str(1, tag);                                 // tag
        val.write_msg_raw(9, BuildHistogramMetadata());        // metadata → field 9
        val.write_msg_raw(5, histo.buf());               // histo    → field 5
        WriteRecord(BuildEvent(val.buf(), step));
    }

    void WriteHistogram(const std::string& tag, const std::vector<float>& values,
                        int64_t step, int num_buckets = 30) {
        std::vector<double> dv(values.begin(), values.end());
        WriteHistogram(tag, dv, step, num_buckets);
    }

private:
    std::ofstream file_;

    void WriteRecord(const std::vector<uint8_t>& data) {
        uint64_t len = data.size();
        uint32_t lc = crc32c::Masked(crc32c::Value(reinterpret_cast<const uint8_t*>(&len), 8));
        uint32_t dc = crc32c::Masked(crc32c::Value(data.data(), data.size()));
        file_.write(reinterpret_cast<const char*>(&len),       8);
        file_.write(reinterpret_cast<const char*>(&lc),        4);
        file_.write(reinterpret_cast<const char*>(data.data()), data.size());
        file_.write(reinterpret_cast<const char*>(&dc),        4);
        file_.flush();
    }

    std::vector<uint8_t> BuildEvent(const std::vector<uint8_t>& value_bytes, int64_t step) {
        proto::Encoder summary;
        summary.write_msg_raw(1, value_bytes);   // repeated Value
        proto::Encoder event;
        event.write_f64(1, static_cast<double>(std::time(nullptr)));  // wall_time
        event.write_s64(2, step);                                     // step
        event.write_msg_raw(5, summary.buf());                        // summary
        return event.buf();
    }

    void WriteFileVersionEvent() {
        proto::Encoder event;
        event.write_f64(1, static_cast<double>(std::time(nullptr)));
        event.write_s64(2, 0);
        event.write_str(3, "brain.Event:2");   // field 3 = file_version
        WriteRecord(event.buf());
    }

    // TensorProto for scalar float. Canonical encoding matching protobuf serializer:
    //   dtype=DT_FLOAT(1), tensor_shape OMITTED (empty=proto3 default),
    //   float_val uses packed encoding (wire type 2), NOT non-packed (wire type 5).
    std::vector<uint8_t> BuildFloatTensorProto(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, 4);
        uint8_t fb[4] = {
            static_cast<uint8_t>( bits        & 0xFF),
            static_cast<uint8_t>((bits >>  8) & 0xFF),
            static_cast<uint8_t>((bits >> 16) & 0xFF),
            static_cast<uint8_t>((bits >> 24) & 0xFF),
        };
        proto::Encoder tp;
        tp.write_s32(1, 1);          // dtype = DT_FLOAT
        tp.write_msg_raw(2, {});    // tensor_shape = empty (scalar)
        tp.write_bytes(5, fb, 4);      // float_val at field 5 (packed)
        return tp.buf();
    }

    // Plugin metadata — note: metadata goes in Summary.Value field 9
    std::vector<uint8_t> BuildScalarMetadata() {
        proto::Encoder pd;
        pd.write_str(1, "scalars");
        // content: empty = ScalarPluginData{mode=DEFAULT} in proto3
        proto::Encoder meta;
        meta.write_msg_raw(1, pd.buf());
        meta.write_s32(4, 1);           // data_class = DATA_CLASS_SCALAR
        return meta.buf();
    }

    std::vector<uint8_t> BuildImageMetadata(int32_t max_images = 1) {
        proto::Encoder pc;
        pc.write_s32(1, max_images);    // max_images_per_step
        proto::Encoder pd;
        pd.write_str(1, "images");
        const auto& pcb = pc.buf();
        pd.write_bytes(2, pcb.data(), pcb.size());
        proto::Encoder meta;
        meta.write_msg_raw(1, pd.buf());
        meta.write_s32(4, 2);           // data_class = DATA_CLASS_BLOB_SEQUENCE
        return meta.buf();
    }

    std::vector<uint8_t> BuildHistogramMetadata() {
        proto::Encoder pd;
        pd.write_str(1, "histograms");
        proto::Encoder meta;
        meta.write_msg_raw(1, pd.buf());
        meta.write_s32(4, 1);           // data_class = DATA_CLASS_SCALAR
        return meta.buf();
    }

    void BuildBuckets(double vmin, double vmax, const std::vector<double>& values,
                      int nb, std::vector<double>& limits, std::vector<double>& counts) {
        if (vmin == vmax) { limits.push_back(vmin+1e-10); counts.push_back((double)values.size()); return; }
        double bw = (vmax-vmin)/nb;
        for(int i=0;i<nb;i++){limits.push_back(vmin+(i+1)*bw);counts.push_back(0.0);}
        limits.back()=vmax+1e-10;
        for(double v:values){int b=std::max(0,std::min(nb-1,(int)((v-vmin)/bw)));counts[b]+=1.0;}
    }
};

} // namespace tensorboard
