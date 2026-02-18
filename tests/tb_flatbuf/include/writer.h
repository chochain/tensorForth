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
 * = Tensor ===================== (implemented in encoder.h)
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
 *   +15: variant_val (VariantTensorDataProto)
 *   +16: uint32_val (packed)
 *   +17: uint64_val (packed)
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

// ─── HParam value types ─────────────────────────────────────────────────────
struct HParamValue {
    enum Type { HP_FLOAT, HP_INT, HP_STR, HP_BOOL } type;
    union {
        F64 f;
        S64 i;
    };
    STR s;
    
    HParamValue(double v)        : type(HP_FLOAT), f(v) {}
    HParamValue(float v)         : type(HP_FLOAT), f(v) {}
    HParamValue(int64_t v)       : type(HP_INT), i(v) {}
    HParamValue(int32_t v)       : type(HP_INT), i(v) {}
    HParamValue(const char* v)   : type(HP_STR), s(v) {}
//    HParamValue(const string& v) : type(HP_STR), s(v.c_str()) {}
    HParamValue(bool v)          : type(HP_BOOL), i(v) {}
};

// ─── EventWriter ────────────────────────────────────────────────────────────
class EventWriter {
public:
    explicit EventWriter(const STR& path)
        : _file(path, std::ios::binary | std::ios::trunc) {
        if (!_file.is_open())
            throw std::runtime_error("Cannot open event file: " + path);
        add_version();
    }
    ~EventWriter() { if (_file.is_open()) _file.close(); }

    void add_version() {
        proto::Encoder event;
        event.f64(1, static_cast<F64>(std::time(nullptr)));
        event.s64(2, 0);
        event.str(3, "brain.Event:2");   // field 3 = file_version

        _dump(event.buf(), "event");
        _write(event.buf());
    }

    // ── Scalar ──────────────────────────────────────────────────────────────
    void add_scalar(const STR& tag, F32 value, S64 step) {
        proto::Encoder enc;
        enc.str(1, tag);                                 // tag
        enc.f32(2, value);                               // simple_value

        _dump(_summary(enc.buf(), step), "scalar.summary");
        _dump(enc.buf(), "scalar", "    ");
        _write(_summary(enc.buf(), step));
    }

    void add_scalar_tensor(const STR& tag, F32 value, S64 step) {
        proto::Encoder enc;
        enc.str(1, tag);                                 // tag
        enc.raw(8, _scalar_tensor(value));               // tensor   → field 8
        enc.raw(9, _scalar_meta());                      // metadata → field 9

//        _dump(_summary(enc.buf(), step), "scalarT.summary");
//        _dump(enc.buf(), "scalarT", "    ");
        _write(_summary(enc.buf(), step));
    }

    // ── Text (NEW) ──────────────────────────────────────────────────────────
    void add_text(const STR& tag, const STR& text, S64 step) {
        proto::Encoder enc;
        enc.str(1, tag);                                 // tag
        enc.raw(8, _text_tensor(text));                  // tensor   → field 8
        enc.raw(9, _text_meta());                        // metadata → field 9

//        _dump(_summary(enc.buf(), step), "textT.summary");
//        _dump(enc.buf(), "textT", "    ");
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

    // ── HParams (NEW) ───────────────────────────────────────────────────────
    // Initialize hparams experiment with parameter and metric definitions
    void add_hparams_config(
        const std::map<STR, HParamValue>& hparam_defaults,
        const std::vector<STR>& metric_tags) {
        
        // Build HParamsPluginData for session_start_info
        proto::Encoder hparams_proto;
        
        // Field 1: hparams (repeated HParamInfo)
        for (const auto& kv : hparam_defaults) {
            proto::Encoder hparam_info;
            hparam_info.str(1, kv.first);  // name
            
            // Type and domain based on value type
            switch (kv.second.type) {
                case HParamValue::HP_FLOAT:
                    hparam_info.s32(2, 1);  // type = DATA_TYPE_FLOAT64
                    break;
                case HParamValue::HP_INT:
                    hparam_info.s32(2, 3);  // type = DATA_TYPE_FLOAT64 (TB converts ints)
                    break;
                case HParamValue::HP_STR:
                    hparam_info.s32(2, 2);  // type = DATA_TYPE_STRING
                    break;
                case HParamValue::HP_BOOL:
                    hparam_info.s32(2, 4);  // type = DATA_TYPE_BOOL
                    break;
            }
            
            hparams_proto.raw(1, hparam_info.buf());
        }
        
        // Field 2: metric_infos (repeated MetricInfo)
        for (const auto& tag : metric_tags) {
            proto::Encoder metric_info;
            metric_info.str(1, tag);  // name.tag
            hparams_proto.raw(2, metric_info.buf());
        }
        
        // Create session_start_info
        proto::Encoder session_start;
        session_start.raw(1, hparams_proto.buf());  // hparams
        session_start.str(2, "");  // model_uri (empty)
        session_start.str(3, "default");  // monitor_url (group name)
        session_start.s64(4, 0);  // group_name as version
        
        // Wrap in plugin data
        proto::Encoder plugin_data;
        plugin_data.str(1, "hparams");  // plugin_name
        plugin_data.raw(2, session_start.buf());  // content
        
        proto::Encoder metadata;
        metadata.raw(1, plugin_data.buf());
        
        // Create summary value
        proto::Encoder summary_value;
        summary_value.str(1, "_hparams_/session_start_info");
        summary_value.raw(9, metadata.buf());
        
        _write(_summary(summary_value.buf(), 0));
    }
    
    // Log actual hyperparameter values and corresponding metrics
    void add_hparams(
        const std::map<STR, HParamValue>& hparams,
        const std::map<STR, F64>& metrics,
        S64 step = 0) {
        
        // 1. Write session start with hparam values
        proto::Encoder session_start;
        
        for (const auto& kv : hparams) {
            proto::Encoder hparam;
            hparam.str(1, kv.first);  // name
            
            // Field 2: value (oneof)
            proto::Encoder value;
            switch (kv.second.type) {
                case HParamValue::HP_FLOAT:
                    value.f64(1, kv.second.f);  // number_value
                    break;
                case HParamValue::HP_INT:
                    value.f64(1, static_cast<F64>(kv.second.i));
                    break;
                case HParamValue::HP_STR:
                    value.str(2, kv.second.s);  // string_value
                    break;
                case HParamValue::HP_BOOL:
                    value.write_bool(3, kv.second.i);  // bool_value
                    break;
            }
            hparam.raw(2, value.buf());
            
            session_start.raw(1, hparam.buf());
        }
        
        session_start.str(3, "default");  // group_name
        session_start.s64(4, step);  // start_time_secs
        
        proto::Encoder plugin_data;
        plugin_data.str(1, "hparams");
        plugin_data.raw(2, session_start.buf());
        
        proto::Encoder metadata;
        metadata.raw(1, plugin_data.buf());
        
        proto::Encoder summary_value;
        summary_value.str(1, "_hparams_/session_start_info");
        summary_value.raw(9, metadata.buf());
        
        _write(_summary(summary_value.buf(), step));
        
        // 2. Write metrics as regular scalars
        for (const auto& kv : metrics) {
            add_scalar(kv.first, static_cast<F32>(kv.second), step);
        }
        
        // 3. Write session end
        proto::Encoder session_end;
        session_end.s32(1, 2);  // status = STATUS_SUCCESS
        session_end.s64(2, step);  // end_time_secs
        
        proto::Encoder plugin_data_end;
        plugin_data_end.str(1, "hparams");
        plugin_data_end.raw(2, session_end.buf());
        
        proto::Encoder metadata_end;
        metadata_end.raw(1, plugin_data_end.buf());
        
        proto::Encoder summary_value_end;
        summary_value_end.str(1, "_hparams_/session_end_info");
        summary_value_end.raw(9, metadata_end.buf());
        
        _write(_summary(summary_value_end.buf(), step));
    }

private:
    std::ofstream _file;

    void _dump(const U8V& buf, const char *hdr, const char *pfx="") {
        int sz = (int)buf.size();
        printf("%s%s len=%d(%x)\n%s", pfx, hdr, sz, sz, pfx);
        for (int i=0; i < sz; i+=16) {
            for (int j=0; j<16; j++) {
                U8 c = (i+j) < sz ? buf.data()[i+j] : 0;
                printf("%02x ", c);
            }
            for (int j=0; j < 16; j++) {   // print and advance to next byte
                U8 c = ((i+j) < sz ? buf.data()[i+j] : 0) & 0x7f;
                printf("%c", (char)((c==0x7f||c<0x20) ? '_' : c));
            }
            printf("\n%s", pfx);
        }
        printf("\n");
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
        
        proto::Encoder event;
        event.f64(1, static_cast<F64>(std::time(nullptr)));
        event.s64(2, step);
        event.raw(5, summary.buf());                            // summary
        
        return event.buf();
    }

    // TensorProto for scalar F32. Canonical encoding matching protobuf serializer:
    //   dtype=DT_FLOAT(1), tensor_shape OMITTED (empty=proto3 default),
    //   float_val uses packed encoding (wire type 2), NOT non-packed (wire type 5).
    U8V _scalar_tensor(F32 value) {
        U32 bits;
        std::memcpy(&bits, &value, 4);
        U8 fb[4] = {
            static_cast<U8>( bits        & 0xFF),
            static_cast<U8>((bits >>  8) & 0xFF),
            static_cast<U8>((bits >> 16) & 0xFF),
            static_cast<U8>((bits >> 24) & 0xFF),
        };
        
        proto::Encoder tp;
        tp.s32(1, 1);             // dtype = DT_FLOAT
//        tp.raw(2, {});            // tensor_shape = empty (scalar), optional
        tp.raw(5, fb, 4);         // float_val at field 5 (packed)
        
        _dump(tp.buf(), "f32_tensor");
        return tp.buf();
    }

    // TensorProto for string (text)
    U8V _text_tensor(const STR& text) {
        proto::Encoder tp;
        tp.s32(1, 7);             // dtype = DT_STRING (7)
//        tp.raw(2, {});            // tensor_shape = empty (scalar), optional
        tp.str(8, text);          // string_val
        
        _dump(tp.buf(), "str_tensor");
        return tp.buf();
    }
    
    // TensorProto for string batch (text) - 1D vector
    U8V _text_tensor_batch(const std::vector<STR>& texts) {
        proto::Encoder tp;
        tp.s32(1, 7);             // dtype = DT_STRING (7)
        
        // tensor_shape: 1D vector with size = texts.size()
        proto::Encoder shape;
        proto::Encoder dim;
        dim.s64(1, static_cast<S64>(texts.size()));  // size field
        shape.raw(2, dim.buf());                      // dim field (repeated)
        
        tp.raw(2, shape.buf());   // tensor_shape
        
        // string_val field 8 (repeated bytes)
        for (const auto& text : texts) {
            tp.str(8, text);      // repeated string_val
        }
        
        return tp.buf();
    }
    
    // Plugin metadata – note: metadata goes in Summary.Value field 9
    U8V _scalar_meta() {
        proto::Encoder pd;        // payload
        pd.str(1, "scalars");
        
        // content: empty = scalar_plugin{mode=DEFAULT} in proto3
        proto::Encoder meta;      // SummaryMetadata
        meta.raw(1, pd.buf());
        meta.s32(4, 1);           // data_class = DATA_CLASS_SCALAR
        
        return meta.buf();
    }

    U8V _text_meta() {
        proto::Encoder pd;      // payload
        pd.str(1, "text");
        // Empty content for text plugin
        
        proto::Encoder meta;    // SummaryMetadata
        meta.raw(1, pd.buf());
        meta.s32(4, 2);         // data_class = DATA_CLASS_TENSOR
        
        return meta.buf();
    }

    U8V _image_meta(S32 max_images = 1) {
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

    U8V _histo_meta() {
        proto::Encoder pd;
        pd.str(1, "histograms");
        
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
