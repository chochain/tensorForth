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
        PTR p;     // uintptr_t (string pointer)
    };

    HParamValue()              : type(HP_FLOAT)       {}
    HParamValue(F64 v)         : type(HP_FLOAT), f(v) {}
    HParamValue(F32 v)         : type(HP_FLOAT), f(v) {}
    HParamValue(S64 v)         : type(HP_INT),   i(v) {}
    HParamValue(S32 v)         : type(HP_INT),   i(v) {}
    HParamValue(BOOL v)        : type(HP_BOOL),  i(v) {}
    HParamValue(const char *s) : type(HP_STR),   p(reinterpret_cast<PTR>(s)) {}
    HParamValue(STR& s)        : HParamValue(s.c_str()) {}
};

class HParamWriter : public EventWriter {
public:
    explicit HParamWriter(const STR& path) : EventWriter(path) {}

    // ── HParams (NEW) ───────────────────────────────────────────────────────
    // Initialize hparams experiment with parameter and metric definitions
    void add_config(
        const std::map<STR, HParamValue>& info,
        const std::vector<STR>& metrics) {

        // Build Experiment for HParamsPluginData
        proto::Encoder ex;             // Experiment
        ex.str(1, "my_test");
        _dump(ex.buf(), "ex1", "");
        
        // Field 4: hparams (repeated HParamInfo)
        for (const auto& kv : info) {
            ex.raw(4, _info(kv.first, kv.second));
        }
        _dump(ex.buf(), "ex4", "");

        // Field 5: metric_infos (repeated MetricInfo)
        for (const auto& name : metrics) {
            proto::Encoder n;         // MetricName
            n.str(1, "scalars");      // group
            n.str(2, name);           // tag
            
            proto::Encoder i;         // MetricInfo
            i.raw(1, n.buf());
            
            ex.raw(5, i.buf());       // Experiment enclose MetricInfo
        }
        _dump(ex.buf(), "ex5", "");
        
        // Create summary value
        _write(_summary(_plugin(ex.buf(), 2), 0));
    }
    
    // Log actual hyperparameter values and corresponding metrics
    void add_hparams(
        const std::map<STR, HParamValue>& hparams,
        const std::map<STR, F64>& metrics,
        const char *group = "default",
        S64 step = 0) {
        
        // 1. Write session start with hparam values
        proto::Encoder pd;
        for (const auto& kv : hparams) {
            pd.enc(_param(kv.first, kv.second));
        }
        
        proto::Encoder ses1;        // SessionStartInfo
        ses1.raw(1, pd.buf());      // map<protobuf.Value> hparams
        ses1.str(4, group);         // group_name
        ses1.s64(5, step);          // start_time_secs
        
        _write(_summary(_plugin(ses1.buf(), 3), step));

        // 2. Write metrics as regular scalars
        for (const auto& kv : metrics) {
            add_scalar(kv.first, static_cast<F32>(kv.second), step);
        }
        
        // 3. Write session end
        proto::Encoder ses0;        // SessionEndInfo
        ses0.s32(1, 2);             // status = STATUS_SUCCESS
        ses0.s64(2, step);          // end_time_secs
        
        _write(_summary(_plugin(ses0.buf(), 4), step));
    }

private:
    U8V _plugin(const U8V& ses, U32 idx) {
        proto::Encoder ppd;         // HParamsPluginData
        ppd.s32(1, 0);              // version (always 0)
        ppd.raw(idx, ses);          // content 2:experiment,3:start_info,4:end_info

        proto::Encoder pd;          // SummaryMetadata.PluginData
        pd.str(1, "hparams");       // plugin_name
        pd.raw(2, ppd.buf());       // content
        
        proto::Encoder meta;        // SummaryMetadata
        meta.raw(1, pd.buf());      // plugin_data
        meta.s32(4, 3);             // DATA_CLASS_BLOB_SEQUENCE
        _dump(ses, "meta", "");

        const char *tag[] = {
            "_hparams_/experiment",
            "_hparams_/session_start_info",
            "_hparams_/session_end_info"
        };
        proto::Encoder enc;         // SummaryValue
        enc.str(1, tag[idx-2]);     // HParam tag
        enc.raw(9, meta.buf());
        enc.raw(8, _scalar_tensor(0.0)); // blank Tensor

        _dump(enc.buf(), "enc.buf", "    ");
        
        return enc.buf();
    }
    
    U8V _info(STR name, HParamValue hpv) {
        proto::Encoder i;           // MetricInfo
        i.str(1, name);             // name

        // Type and domain based on value type
        const int tag = 4;          // DATA_TYPE
        switch (hpv.type) {
        case HParamValue::HP_FLOAT: // DATA_TYPE_FLOAT64
            i.s32(tag, 3);  break;
        case HParamValue::HP_INT:   // DATA_TYPE_FLOAT64 (TB converts ints)
            i.s32(tag, 3);  break;
        case HParamValue::HP_STR:   // DATA_TYPE_STRING
            i.s32(tag, 1);  break;
        case HParamValue::HP_BOOL:  // DATA_TYPE_BOOL
            i.s32(tag, 2); break;
        }
        _dump(i.buf(), "_info", "");
        return i.buf();
    }

    U8V _param(STR name, HParamValue hpv) {
        // Field 1: name
        proto::Encoder k;
        k.str(1, name);             // name
            
        // Field 2: value (oneof)   // protobuf.Value
        proto::Encoder v;
        switch (hpv.type) {
        case HParamValue::HP_FLOAT: // number_value
            v.f64(2, hpv.f);                                break;
        case HParamValue::HP_INT:
            v.f64(2, static_cast<F64>(hpv.i));              break;
        case HParamValue::HP_STR:   // string_value
            v.str(3, reinterpret_cast<const char*>(hpv.p)); break;
        case HParamValue::HP_BOOL:  // bool_value
            v.write_bool(4, hpv.i);                         break;
        }
        
        proto::Encoder enc;
        enc.raw(1, k.buf());
        enc.raw(2, v.buf());
        _dump(enc.buf(), "_param", "");

        return enc.buf();
    }
};

} // namespace tensorboard
