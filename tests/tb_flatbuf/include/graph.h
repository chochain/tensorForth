/*
 @file
 @brief - GraphDef event writer
 @note - see schema.h for details
 */
#pragma once

#include "writer.h"

namespace tensorboard {

// ── GraphDef ───────────────────────────────────────────────────────────
struct AttrValue {
    enum Type {  AV_FLOAT, AV_INT, AV_STR, AV_BOOL } type;
    int dt_type;
    union {
        F64 f;
        S64 i;
        PTR p;     // uintptr_t (string pointer)
    };
    AttrValue()              : type(AV_FLOAT), dt_type(2)       {}
    AttrValue(F64 v)         : type(AV_FLOAT), dt_type(2), f(v) {}
    AttrValue(F32 v)         : type(AV_FLOAT), dt_type(1), f(v) {}
    AttrValue(S64 v)         : type(AV_INT),   dt_type(9), i(v) {}
    AttrValue(S32 v)         : type(AV_INT),   dt_type(3), i(v) {}
    AttrValue(BOOL v)        : type(AV_BOOL),  dt_type(10),i(v) {}
    AttrValue(const char *s) : type(AV_STR),   dt_type(7), p(reinterpret_cast<PTR>(s)) {}
    AttrValue(STR& s)        : AttrValue(s.c_str()) {}
};

class Node : public proto::Encoder {
public:    
    explicit Node(const STR& name, const STR& op) {
        str(1, name);
        str(2, op);
    }
    explicit Node(const STR& name, const STR& op, const STR& input) { // NodeDef
        str(1, name);
        str(2, op);
        add_input(input);
    }
    void add_input(const STR& input) {
        if (input.size() > 0) str(3, input);
    }
    void add_type(const STR& name, int dt_type=1) {  // DT_FLOAT
        proto::Encoder dt;
        dt.s32(6, dt_type);

        _attr(name, dt.buf());
    }
    void add_value(const STR& k, const AttrValue& av) {
        proto::Encoder v;
        switch (av.type) {
        case AttrValue::AV_FLOAT: v.f32(4, av.f);        break;
        case AttrValue::AV_INT:   v.s64(3, av.i);        break;
        case AttrValue::AV_STR:   v.str(2, reinterpret_cast<const char*>(av.p)); break;
        case AttrValue::AV_BOOL:  v.write_bool(5, av.i); break;
        }
        _attr(k, v.buf());
    }
    void add_stride(U16V stride) {
        proto::Encoder v;
        for (U64 s : stride) {          // raw numbers
            v.pack(s);
        }
        proto::Encoder t;               // ListValue
        t.raw(3, v.buf());              // 3 = repeated int64

        proto::Encoder av;              // AttrValue
        av.raw(1, t.buf());             // ListValue as length limited

        _attr("strides", av.buf());
    }
    void add_shape(U32V shape) {
        proto::Encoder ts;              // TensorShapeProto
        ts.raw(7, _dim(shape));

        _attr("shape", ts.buf());
    }
    void add_tensor(U32V shape, F32V value) {
        proto::Encoder v;               // collect tensor values
        for (F32 f : value) {
            v.pack(f);
        }
        proto::Encoder tv;              // TensorProto.Value
        tv.s32(1, 1);                   // DataType 1:DT_FLOAT
        tv.raw(2, _dim(shape));
        tv.raw(5, v.buf());             // repeated float
        
        proto::Encoder t;               // TensorProto
        t.raw(8, tv.buf());

        _attr("value", t.buf());
    }
    
private:
    U8V _dim(U32V shape) {
        proto::Encoder v;
        for (auto s : shape) {
            proto::Encoder sv;
            sv.s32(1, s);
//            sv.str(2, "size");        // by default
            v.raw(2, sv.buf());
        }
        return v.buf();
    }
    void _attr(const STR& k, U8V buf) { // add attribute
        proto::Encoder at;
        at.str(1, k);
        at.raw(2, buf);

        _dump(at.buf(), k.c_str(), "");
        
        raw(5, at.buf());               // NodeDef.attr
    }
};

class GraphWriter : public EventWriter {
    std::vector<Node> _net;
        
public:
    explicit GraphWriter(const STR& path) : EventWriter(path) {}

    void add_node(const Node& node) {
        _net.push_back(node);
    }

    void write(S64 step=0) {
        proto::Encoder graph;
        for (auto n : _net) {
            graph.raw(1, n.buf());
        }
        
        proto::Encoder event;
        event.s64(2, step);
        event.raw(4, graph.buf());
        
        _write(event.buf());
    }
};

} // class Node
