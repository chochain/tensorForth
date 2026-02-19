#pragma once

#include "types.h"
#include "writer.h"

namespace tensorboard {

// ─── HParam value types ─────────────────────────────────────────────────────
struct HParamValue {
    enum Type { HP_FLOAT, HP_INT, HP_STR, HP_BOOL } type;
    union {
        F64 f;
        S64 i;
    };
    STR s;

    HParamValue()              : type(HP_FLOAT), f(0) {}
    HParamValue(double v)      : type(HP_FLOAT), f(v) {}
    HParamValue(float v)       : type(HP_FLOAT), f(v) {}
    HParamValue(int64_t v)     : type(HP_INT),   i(v) {}
    HParamValue(int32_t v)     : type(HP_INT),   i(v) {}
    HParamValue(const char* s) : type(HP_STR),   s(s) {}
    HParamValue(const STR& s)  : type(HP_STR),   s(s.c_str()) {}
    HParamValue(bool v)        : type(HP_BOOL),  i(v) {}
};

class HParamWriter : public EventWriter {
public:
    explicit HParamWriter(const STR& path)
        : EventWriter(path) {
    }

    // ── HParams (NEW) ───────────────────────────────────────────────────────
    // Initialize hparams experiment with parameter and metric definitions
    void add_config(
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
};

} // namespace tensorboard
