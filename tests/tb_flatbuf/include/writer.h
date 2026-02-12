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
#include "schema.h"
#include "png.h"
#include "encoder.h"

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
inline std::string logdir(const std::string& dir, int seq = 0) {
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

// ─── EventWriter ─────────────────────────────────────────────────────────────
class EventWriter {
public:
    explicit EventWriter(const std::string& path)
        : file_(path, std::ios::binary | std::ios::trunc) {
        if (!file_.is_open())
            throw std::runtime_error("Cannot open event file: " + path);
        add_version();
    }
    ~EventWriter() { if (file_.is_open()) file_.close(); }

    void add_version() {
        proto::Encoder event;
        event.f64(1, static_cast<double>(std::time(nullptr)));
        event.s64(2, 0);
        event.str(3, "brain.Event:2");   // field 3 = file_version
        
        write(event.buf());
    }

    // ── Scalar ────────────────────────────────────────────────────────────────
    void WriteScalar(const std::string& tag, float value, int64_t step) {
        proto::Encoder enc;
        enc.str(1, tag);                                 // tag
        enc.f32(2, value);                               // simple_value
//        enc.raw(9, scalar_meta());                     // metadata → field 9
//        enc.raw(8, scalar_tensor(value));              // tensor   → field 8
        write(build(enc.buf(), step));
    }

    // ── Image (RGB row-major, 3 bytes/pixel) ──────────────────────────────────
    void WriteImage(const std::string& tag, int width, int height,
                    const std::vector<uint8_t>& pixels_rgb, int64_t step) {
        auto png_bytes = png::raw2png(width, height, pixels_rgb, 3);
        
        proto::Encoder img;
        img.s32(1, height);
        img.s32(2, width);
        img.s32(3, 3);  // colorspace = RGB
        img.raw(4, png_bytes.data(), png_bytes.size()); // encoded_image_string
        
        proto::Encoder enc;
        enc.str(1, tag);                                 // tag
        enc.raw(9, image_meta());                    // metadata → field 9
        enc.raw(4, img.buf());                       // image    → field 4
        write(build(enc.buf(), step));
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
        histo.f64(1, vmin);
        histo.f64(2, vmax);
        histo.f64(3, static_cast<double>(values.size()));
        histo.f64(4, vsum);   histo.f64(5, vsumsq);
        histo.f64_packed(6, limits);
        histo.f64_packed(7, counts);
        
        proto::Encoder enc;
        enc.str(1, tag);                       // tag
        enc.raw(9, histo_meta());              // metadata → field 9
        enc.raw(5, histo.buf());               // histo    → field 5
        write(build(enc.buf(), step));
    }

    void WriteHistogram(const std::string& tag, const std::vector<float>& values,
                        int64_t step, int num_buckets = 30) {
        std::vector<double> dv(values.begin(), values.end());
        WriteHistogram(tag, dv, step, num_buckets);
    }

private:
    std::ofstream file_;

    void write(const std::vector<uint8_t>& data) {
        uint64_t len = data.size();
        uint32_t lc = crc32c::mask(crc32c::value(reinterpret_cast<const uint8_t*>(&len), 8));
        uint32_t dc = crc32c::mask(crc32c::value(data.data(), data.size()));
        file_.write(reinterpret_cast<const char*>(&len),       8);
        file_.write(reinterpret_cast<const char*>(&lc),        4);
        file_.write(reinterpret_cast<const char*>(data.data()), data.size());
        file_.write(reinterpret_cast<const char*>(&dc),        4);
        file_.flush();
    }

    std::vector<uint8_t> build(const std::vector<uint8_t>& value_bytes, int64_t step) {
        proto::Encoder summary;
        summary.raw(1, value_bytes);   // repeated Value
        
        proto::Encoder event;
        event.f64(1, static_cast<double>(std::time(nullptr)));  // wall_time
        event.s64(2, step);                                           // step
        event.raw(5, summary.buf());                            // summary
        
        return event.buf();
    }

    // TensorProto for scalar float. Canonical encoding matching protobuf serializer:
    //   dtype=DT_FLOAT(1), tensor_shape OMITTED (empty=proto3 default),
    //   float_val uses packed encoding (wire type 2), NOT non-packed (wire type 5).
#if 0    
    std::vector<uint8_t> scalar_tensor(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, 4);
        uint8_t fb[4] = {
            static_cast<uint8_t>( bits        & 0xFF),
            static_cast<uint8_t>((bits >>  8) & 0xFF),
            static_cast<uint8_t>((bits >> 16) & 0xFF),
            static_cast<uint8_t>((bits >> 24) & 0xFF),
        };
        
        proto::Encoder tp;
        tp.s32(1, 1);            // dtype = DT_FLOAT
        tp.raw(2, {});     // tensor_shape = empty (scalar)
        tp.raw(5, fb, 4);        // float_val at field 5 (packed)
        
        return tp.buf();
    }

    // Plugin metadata — note: metadata goes in Summary.Value field 9
    std::vector<uint8_t> scalar_meta() {
        proto::Encoder pd;
        pd.str(1, "scalars");
        
        // content: empty = ScalarPluginData{mode=DEFAULT} in proto3
        proto::Encoder meta;
        meta.raw(1, pd.buf());
        meta.s32(4, 1);           // data_class = DATA_CLASS_SCALAR
        
        return meta.buf();
    }
#endif
    std::vector<uint8_t> image_meta(int32_t max_images = 1) {
        proto::Encoder pc;
        pc.s32(1, max_images);    // max_images_per_step
        
        proto::Encoder pd;
        pd.str(1, "images");
        
        const auto& pcb = pc.buf();
        pd.raw(2, pcb.data(), pcb.size());
        
        proto::Encoder meta;
        meta.raw(1, pd.buf());
        meta.s32(4, 2);           // data_class = DATA_CLASS_BLOB_SEQUENCE
        
        return meta.buf();
    }

    std::vector<uint8_t> histo_meta() {
        proto::Encoder pd;
        pd.str(1, "histograms");
        
        proto::Encoder meta;
        meta.raw(1, pd.buf());
        meta.s32(4, 1);           // data_class = DATA_CLASS_SCALAR
        
        return meta.buf();
    }

    void BuildBuckets(
        double vmin,
        double vmax,
        const std::vector<double>& values,
        int nb, std::vector<double>& limits,
        std::vector<double>& counts) {
        if (vmin == vmax) {
            limits.push_back(vmin+1e-10);
            counts.push_back((double)values.size());
            return;
        }
        
        double bw = (vmax-vmin)/nb;
        for (int i=0;i<nb;i++) {
            limits.push_back(vmin+(i+1)*bw);
            counts.push_back(0.0);
        }
        limits.back()=vmax+1e-10;
        for (double v:values) {
            int b=std::max(0,std::min(nb-1,(int)((v-vmin)/bw)));
            counts[b]+=1.0;
        }
    }
};

} // namespace tensorboard
