/*
 @file
 @brief - GraphDef event writer
 @note - see schema.h for details
 */
#pragma once
#include "encoder.h"

namespace t4::tb::graph {

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
};

class Node : public Encoder {
public:
    explicit Node() {}
    explicit Node(const char *name, const char *op)                    { init(name, op);        }
    explicit Node(const char *name, const char *op, const char *input) { init(name, op, input); }
    
    void init(const char *name, const char *op) {
        str(1, name);
        str(2, op);
    };
    void init(const char *name, const char *op, const char *input) {
        init(name, op);
        add_input(input);
    };
    void add_input(const char *input) {
        if (strlen(input) > 0) str(3, input);
    }
    void add_type(const char *name, int dt_type=1) {  // DT_FLOAT
        Encoder dt;
        dt.s32(6, dt_type);

        _attr(name, dt.buf());
    }
    void add_value(const char *k, const AttrValue& av) {
        Encoder v;
        switch (av.type) {
        case AttrValue::AV_FLOAT: v.f32(4, av.f);        break;
        case AttrValue::AV_INT:   v.s64(3, av.i);        break;
        case AttrValue::AV_STR:   v.str(2, reinterpret_cast<const char*>(av.p)); break;
        case AttrValue::AV_BOOL:  v.write_bool(5, av.i); break;
        }
        _attr(k, v.buf());
    }
    void add_stride(U16V stride) {
        Encoder v;
        for (U64 s : stride) {          /// * add raw numbers
            v.pack(s);
        }
        Encoder t;                      ///< ListValue
        t.raw(3, v.buf());              /// * 3 = repeated int64

        Encoder av;                     ///< AttrValue
        av.raw(1, t.buf());             /// * ListValue as length limited

        _attr("strides", av.buf());
    }
    void add_shape(U32V shape) {
        Encoder ts;                     ///< TensorShapeProto
        ts.raw(7, _dim(shape));

        _attr("shape", ts.buf());
    }
    void add_tensor(U32V shape, F32V value) {
        Encoder v;                      ///< collect tensor values
        for (F32 f : value) {
            v.pack(f);
        }
        Encoder tv;                     ///< TensorProto.Value
        tv.s32(1, 1);                   /// * DataType 1:DT_FLOAT
        tv.raw(2, _dim(shape));
        tv.raw(5, v.buf());             /// * repeated float
        
        Encoder t;                      ///< TensorProto
        t.raw(8, tv.buf());

        _attr("value", t.buf());
    }
    
private:
    U8V _dim(U32V shape) {             ///< add tensor dimensions
        Encoder v;
        for (auto s : shape) {
            Encoder sv;
            sv.s32(1, s);
//            sv.str(2, "size");       /// * by default
            v.raw(2, sv.buf());
        }
        return v.buf();
    }
    void _attr(const char *k, U8V buf) { ///< add attribute
        Encoder at;
        at.str(1, k);
        at.raw(2, buf);

//        _dump(at.buf(), k, "");       /// * for tracing
        
        raw(5, at.buf());               /// * NodeDef.attr
    }
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
            for (int j=0; j < 16; j++) { /// * print and advance to next byte
                U8 c = ((i+j) < sz ? buf.data()[i+j] : 0) & 0x7f;
                printf("%c", (char)((c==0x7f||c<0x20) ? '_' : c));
            }
            printf("\n");
        }
    }
};  // class Node

} // namespace t4::tb::graph
