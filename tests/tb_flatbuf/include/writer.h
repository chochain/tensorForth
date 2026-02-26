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
 * +tensorflow.ResourceHandleProto
 *
 * +tensorflow.DataType
 *   DT_FLOAT=1, DT_DOUBLE=2, ..., DT_STRING=7
 *
 * = Tensor =====================
 * tensorflow.TensorProto:
 *   1: dtype (DT_FLOAT=1)
 *   2: tensor_shape (empty=scalar)
 *   +3: version_number (int32)
 *   +4: tensor_content (bytes)
 *   5: float_val (packed)
 *   +6: double_val (packed)
 *   +7: int_val (packed)
 *   8: string_val (bytes repeated)
 *   +9: scomplex_val (float packed)
 *   +10: int64_val (packed)
 *   +11: bool_val (packed)
 *   +12: dcomplex_val (double packed)
 *   +13: half_val (int32 packed)
 *   +14: resource_handle_val (ResourceHandleProto)
 *   x15: variant_val (VariantTensorDataProto) 
 *   +16: uint32_val (packed)
 *   +17: uint64_val (packed)
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
 *   4: data_class (DataClass: SCALAR=1, TENSOR=2, BLOB=3)
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
 * [length:uint64 LE]
 * [masked_crc32c(length):uint32 LE]
 * [data][masked_crc32c(data):uint32 LE]
 * masking: ((crc >> 15 | crc << 17) + 0xa282ead8) & 0xFFFFFFFF
 */
#pragma once

#include "types.h"
#include "crc32c.h"
#include "schema.h"    // not needed for now
#include "png.h"
#include "encoder.h"

#include <ctime>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <map>

namespace tensorboard {

// ─── Path helper ────────────────────────────────────────────────────────────
inline STR logdir(const STR& dir, int seq = 0) {
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

// ─── EventWriter ────────────────────────────────────────────────────────────
class EventWriter {
public:
    explicit EventWriter(const STR& path)
        : _file(path, std::ios::binary | std::ios::trunc) {
        if (!_file.is_open())
            throw std::runtime_error("Cannot open event file: " + path);
//CC        add_version();
    }
    ~EventWriter() { if (_file.is_open()) _file.close(); }

    void add_version() {
        proto::Encoder event;
        event.f64(1, static_cast<F64>(std::time(nullptr))); // wall_time
        event.s64(2, 0);                                    // step
        event.str(3, "brain.Event:2");                      // file_version

        _write(event.buf());
    }

    // ── Scalar ──────────────────────────────────────────────────────────────
    void add_scalar(const STR& tag, F32 v, S64 step) {
        proto::Encoder enc;
        enc.str(1, tag);                                 // tag
        enc.f32(2, v);                                   // simple_value

        _write(_summary(enc.buf(), step));
    }
   
    void add_scalar_tensor(const STR& tag, F32 v, S64 step) {
        proto::Encoder enc;
        enc.str(1, tag);                                 // tag
        enc.raw(9, _scalar_meta());                      // metadata → field 9
        enc.raw(8, _scalar_tensor(v));                   // tensor   → field 8

        _write(_summary(enc.buf(), step));               // for new time-series
    }

    // ── Text (NEW) ──────────────────────────────────────────────────────────
    void add_text(const STR& tag, const STR& txt, S64 step) {
        proto::Encoder enc;
        enc.str(1, tag);                                 // tag
        enc.raw(9, _text_meta());                        // metadata → field 9
        enc.raw(8, _text_tensor(txt));                   // tensor   → field 8

        _write(_summary(enc.buf(), step));
    }

    // ── Image (RGB row-major, 3 bytes/pixel) ────────────────────────────────
    void add_image(
        const STR& tag,
        int width,
        int height,
        const U8V& pixels_rgb,
        S64 step) {
        auto png = png::raw2png(width, height, pixels_rgb, 3);
        
        proto::Encoder img;
        img.s32(1, height);
        img.s32(2, width);
        img.s32(3, 3);                                   // colorspace = RGB
        img.raw(4, png.data(), png.size());              // encoded_image_string
        
        proto::Encoder enc;
        enc.str(1, tag);                                 // tag
        enc.raw(9, _image_meta());                       // metadata → field 9
        enc.raw(4, img.buf());                           // image    → field 4
        
        _write(_summary(enc.buf(), step));
    }

    // ── Histogram ───────────────────────────────────────────────────────────
    void add_histo(
        const STR& tag,
        const F64V& values,
        S64 step,
        int num_buckets = 30) {
        if (values.empty()) return;
        
        F64 vmin = *std::min_element(values.begin(), values.end());
        F64 vmax = *std::max_element(values.begin(), values.end());
        F64 vsum = 0, vsumsq = 0;
        for (F64 v : values) { vsum += v; vsumsq += v*v; }
        
        F64V limits, counts;
        _buckets(vmin, vmax, values, num_buckets, limits, counts);
        
        proto::Encoder histo;
        histo.f64(1, vmin);
        histo.f64(2, vmax);
        histo.f64(3, static_cast<F64>(values.size()));
        histo.f64(4, vsum);
        histo.f64(5, vsumsq);
        histo.f64_packed(6, limits);
        histo.f64_packed(7, counts);
        
        proto::Encoder enc;
        enc.str(1, tag);                       // tag
        enc.raw(9, _histo_meta());             // metadata → field 9
        enc.raw(5, histo.buf());               // histo    → field 5
        
        _write(_summary(enc.buf(), step));
    }

    void add_histo(
        const STR& tag,
        const F32V& values,
        S64 step,
        int num_buckets = 30) {
        F64V dv(values.begin(), values.end());
        add_histo(tag, dv, step, num_buckets);
    }

protected:
    std::ofstream _file;

    void _dump(const U8V& buf, const char *hdr, const char *pfx="") {
        int sz = (int)buf.size();
        printf("%s%s len=%d(%x)\n", pfx, hdr, sz, sz);
        for (int i=0; i < sz; i+=16) {
            printf("%s%04x:", pfx, i);
            for (int j=0; j<16; j++) {
                U8 c = (i+j) < sz ? buf.data()[i+j] : 0;
                printf(" %02x", c);
            }
            printf("  ");
            for (int j=0; j < 16; j++) {   // print and advance to next byte
                U8 c = ((i+j) < sz ? buf.data()[i+j] : 0) & 0x7f;
                printf("%c", (char)((c==0x7f||c<0x20) ? '_' : c));
            }
            printf("\n");
        }
    }

    void _write(const U8V& buf) {
        U64 len = buf.size();
        U32 lc = crc32c::mask(crc32c::value(reinterpret_cast<const U8*>(&len), 8));
        U32 dc = crc32c::mask(crc32c::value(buf.data(), buf.size()));
        _file.write(reinterpret_cast<const char*>(&len),       8);
        _file.write(reinterpret_cast<const char*>(&lc),        4);
        _file.write(reinterpret_cast<const char*>(buf.data()), buf.size());
        _file.write(reinterpret_cast<const char*>(&dc),        4);
        
        _file.flush();
    }

    U8V _summary(const U8V& buf, S64 step) {
        proto::Encoder summary;
        summary.raw(1, buf);                                    // repeated Value
        _dump(summary.buf(), "summary", "");
        
        proto::Encoder event;
//CC        event.f64(1, static_cast<F64>(std::time(nullptr)));   // wall_time
//CC        event.s64(2, step);                                   // step
        event.raw(5, summary.buf());                            // summary
        _dump(event.buf(), "event", "");
        
        return event.buf();
    }

    // TensorProto for scalar F32. Canonical encoding matching protobuf serializer:
    //   dtype=DT_FLOAT(1), tensor_shape OMITTED (empty=proto3 default),
    //   float_val uses packed encoding (wire type 2), NOT non-packed (wire type 5).
    U8V _scalar_tensor(F32 v) {
        U32 bits;
        std::memcpy(&bits, &v, 4);
        U8 fb[4] = {
            static_cast<U8>( bits        & 0xFF),
            static_cast<U8>((bits >>  8) & 0xFF),
            static_cast<U8>((bits >> 16) & 0xFF),
            static_cast<U8>((bits >> 24) & 0xFF),
        };
        
        proto::Encoder tp;        // TensorProto
        tp.s32(1, 1);             // dtype = DT_FLOAT
//        tp.raw(2, {});            // tensor_shape = empty (scalar), optional
        tp.raw(5, fb, 4);         // float_val at field 5 (packed)
        
        return tp.buf();
    }

    // TensorProto for string (text)
    U8V _text_tensor(const STR& txt) {
        proto::Encoder tp;        // TensorProto
        tp.s32(1, 7);             // dtype = DT_STRING (7)
//        tp.raw(2, {});            // tensor_shape = empty (scalar), optional
        tp.str(8, txt);           // string_val
        
        return tp.buf();
    }
    
    // Plugin metadata – note: metadata goes in Summary.Value field 9
    U8V _scalar_meta() {
        proto::Encoder pd;        // SummaryMetadata.PluginData
        pd.str(1, "scalars");     // plugin name
        
        // content: empty = scalar_plugin{mode=DEFAULT} in proto3
        proto::Encoder meta;      // SummaryMetadata
        meta.raw(1, pd.buf());
        meta.s32(4, 1);           // data_class = DATA_CLASS_SCALAR
        
        return meta.buf();
    }

    U8V _text_meta() {
        proto::Encoder pd;        // SummaryMetadata.PluginData
        pd.str(1, "text");        // plugin_name
        // empty content for text
        
        proto::Encoder meta;      // SummaryMetadata
        meta.raw(1, pd.buf());    // PluginData
        meta.s32(4, 2);           // data_class = DATA_CLASS_TENSOR
        
        return meta.buf();
    }

    U8V _image_meta(S32 max_images = 1) {
        proto::Encoder pc;
        pc.s32(1, max_images);    // max_images_per_step
        
        proto::Encoder pd;
        pd.str(1, "images");      // plugin_name
        pd.raw(2, pc.buf());      // content
        
        proto::Encoder meta;
        meta.raw(1, pd.buf());
        meta.s32(4, 3);           // data_class = DATA_CLASS_BLOB_SEQUENCE
        
        return meta.buf();
    }

    U8V _histo_meta() {
        proto::Encoder pd;
        pd.str(1, "histograms");  // plugin_name
        // empty content for histogram
        
        proto::Encoder meta;
        meta.raw(1, pd.buf());
        meta.s32(4, 1);           // data_class = DATA_CLASS_SCALAR
        
        return meta.buf();
    }

    void _buckets(
        F64 vmin,
        F64 vmax,
        const F64V& values,
        int nb, F64V& limits,
        F64V& counts) {
        if (vmin == vmax) {
            limits.push_back(vmin+1e-10);
            counts.push_back((F64)values.size());
            return;
        }
        
        F64 bw = (vmax-vmin)/nb;
        for (int i=0;i<nb;i++) {
            limits.push_back(vmin+(i+1)*bw);
            counts.push_back(0.0);
        }
        
        limits.back()=vmax+1e-10;
        for (F64 v:values) {
            int b=std::max(0,std::min(nb-1,(int)((v-vmin)/bw)));
            counts[b]+=1.0;
        }
    }
};

} // namespace tensorboard
