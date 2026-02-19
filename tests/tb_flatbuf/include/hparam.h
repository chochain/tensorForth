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
    explicit HParamWriter(const STR& path) : EventWriter(path) {}

    // ── HParams (NEW) ───────────────────────────────────────────────────────
    // Initialize hparams experiment with parameter and metric definitions
    void add_config(
        const std::map<STR, HParamValue>& defaults,
        const std::vector<STR>& tags) {
        
        // Build HParamsPluginData for session_start_info
        proto::Encoder hparams;
        
        // Field 1: hparams (repeated HParamInfo)
        for (const auto& kv : defaults) {
            hparams.raw(1, _info(kv.first, kv.second));
        }

        // Field 2: metric_infos (repeated MetricInfo)
        for (const auto& tag : tags) {
            proto::Encoder i;
            i.str(1, tag);            // name.tag
            hparams.raw(2, i.buf());
        }
        
        // Create session_start_info
        proto::Encoder ses;
        ses.raw(1, hparams.buf());    // hparams
        ses.str(2, "");               // model_uri (empty)
        ses.str(3, "default");        // monitor_url (group name)
        ses.s64(4, 0);                // group_name as version
        // Create summary value
        _write(_summary(_ses_meta(ses), 0));
    }
    
    // Log actual hyperparameter values and corresponding metrics
    void add_hparams(
        const std::map<STR, HParamValue>& hparams,
        const std::map<STR, F64>& metrics,
        S64 step = 0) {
        
        // 1. Write session start with hparam values
        proto::Encoder ses1;
        for (const auto& kv : hparams) {
            ses1.raw(1, _hparam(kv.first, kv.second).buf());
        }
        ses1.str(3, "default");  // group_name
        ses1.s64(4, step);       // start_time_secs
        
        _write(_summary(_ses_start(ses1.buf(), ), step));
        
        // 2. Write metrics as regular scalars
        for (const auto& kv : metrics) {
            add_scalar(kv.first, static_cast<F32>(kv.second), step);
        }
        
        // 3. Write session end
        proto::Encoder ses0;
        ses0.s32(1, 2);          // status = STATUS_SUCCESS
        ses0.s64(2, step);       // end_time_secs
        
        _write(_summary(_ses_end(ses0.buf()), step));
    }

private:
    U8V _ses_meta(U8V ses) {
        proto::Encoder pd;       // Wrap in plugin data
        pd.str(1, "hparams");    // plugin_name
        pd.raw(2, ses);          // content
        
        proto::Encoder meta;
        meta.raw(1, pd.buf());

        return meta.buf();
    }
    
    U8V _ses_start(U8V ses) {
        proto::Encoder enc;      // SummaryValue
        enc.str(1, "_hparams_/session_start_info");
        enc.raw(9, _ses_meta(ses));
        
        return enc.buf();
    }
    
    U8V _ses_end(U8V ses) {
        proto::Encoder enc;      // SummaryValue
        enc.str(1, "_hparams_/session_start_info");
        enc.raw(9, _ses_meta(ses));
        
        return enc.buf();
    }
    
    U8V _info(STR& name, HParamValue& v) {
        proto::Encoder i;
        i.str(1, name);   // name
            
        // Type and domain based on value type
        switch (v.type) {
        case HParamValue::HP_FLOAT: // DATA_TYPE_FLOAT64
            i.s32(2, 1);  break;
        case HParamValue::HP_INT:   // DATA_TYPE_FLOAT64 (TB converts ints)
            i.s32(2, 3);  break;
        case HParamValue::HP_STR:   // DATA_TYPE_STRING
            i.s32(2, 2);  break;
        case HParamValue::HP_BOOL:  // DATA_TYPE_BOOL
            i.s32(2, 4);  break;
        }
        return i.buf();
    }

    U8V _params(STR& name, HParamValue& v) {
        proto::Encoder enc;
        enc.str(1, name);           // name
            
        // Field 2: value (oneof)
        proto::Encoder i;
        switch (v) {
        case HParamValue::HP_FLOAT: // number_value
            i.f64(1, v.f);                   break;
        case HParamValue::HP_INT:
            i.f64(1, static_cast<F64>(v.i)); break;
        case HParamValue::HP_STR:   // string_value
            i.str(2, v.s);                   break;
        case HParamValue::HP_BOOL:  // bool_value
            i.write_bool(3, v.i);            break;
        }
        enc.raw(2, i.buf());

        return enc.buf();
    }
};

} // namespace tensorboard
