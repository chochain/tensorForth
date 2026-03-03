/*
 * proto_encode.h - Minimal Protocol Buffers wire-format encoder
 *
 * Supports:
 *   - Varint (field types: int32, int64, uint32, uint64, bool, enum)
 *   - 64-bit (double)
 *   - 32-bit (float)
 *   - Length-delimited (string, bytes, embedded messages, repeated fields)
 *
 * Wire types:
 *   0 = Varint
 *   1 = 64-bit
 *   2 = Length-delimited
 *   5 = 32-bit
 *
 */
#pragma once

#include "types.h"

namespace proto {

class Encoder {
public:
    // ── Key/Tag encoding ─────────────────────────────────────────────────────
    // field_number << 3 | wire_type
    void tag(U32 field_number, U32 wire_type) {
        pack((U64)((field_number << 3) | wire_type));
    }

    // ── Packing ───────────────────────────────────────────────────────────────
    void pack(U64 value) {
        while (value > 0x7F) {
            _buf.push_back(static_cast<U8>((value & 0x7F) | 0x80));
            value >>= 7;
        }
        _buf.push_back(static_cast<U8>(value));
    }

    void pack(F64 value) {
        U64 bits;
        memcpy(&bits, &value, sizeof bits);
        for (int i = 0; i < 8; ++i) {
            _buf.push_back(static_cast<U8>(bits & 0xFF));
            bits >>= 8;
        }
    }

    void pack(F32 value) {
        U32 bits;
        memcpy(&bits, &value, sizeof bits);
        for (int i = 0; i < 4; ++i) {
            _buf.push_back(static_cast<U8>(bits & 0xFF));
            bits >>= 8;
        }
    }
    
    // ── Varint ───────────────────────────────────────────────────────────────
    void s32(U32 field, S32 value) {
        tag(field, 0);
        pack(static_cast<U64>(static_cast<U32>(value)));
    }

    void s64(U32 field, S64 value) {
        tag(field, 0);
        pack(static_cast<U64>(value));
    }

    void u32(U32 field, U32 value) {
        tag(field, 0);
        pack((U64)value);
    }

    void write_bool(U32 field, BOOL value) {
        tag(field, 0);
        pack((U64)(value ? 1 : 0));
    }

    void write_enum(U32 field, S32 value) {
        s32(field, value);
    }

    // ── 64-bit (F64) ──────────────────────────────────────────────────────
    void f64(U32 field, F64 value) {
        tag(field, 1);
        U64 bits;
        memcpy(&bits, &value, sizeof bits);
        for (int i = 0; i < 8; ++i) {
            _buf.push_back(static_cast<U8>(bits & 0xFF));
            bits >>= 8;
        }
    }

    // Packed repeated F64s
    void f64(U32 field, const F64V& values) {
        tag(field, 2);
        pack(values.size() * 8);
        for (F64 v : values) {
            pack(v);
        }
    }

    // ── 32-bit (F32) ───────────────────────────────────────────────────────
    void f32(U32 field, F32 value) {
        tag(field, 5);
        pack(value);
    }

    // Packed repeated F32s
    void f32(U32 field, const F32V& values) {
        tag(field, 2);
        pack(values.size() * 4);
        for (F32 v : values) {
            pack(v);
        }
    }
    // ── Length-delimited ─────────────────────────────────────────────────────
    void raw(U32 field, const U8 *data, USZ len) {
        tag(field, 2);
        pack(len);
        _buf.insert(_buf.end(), data, data + len);
    }

    // Write raw bytes of a message (no field tag) — used for nested messages
    void raw(U32 field, const U8V& data) {
        tag(field, 2);
        pack(data.size());
        _buf.insert(_buf.end(), data.begin(), data.end());
    }

    void str(U32 field, const STR& s) {
        raw(field, reinterpret_cast<const U8*>(s.c_str()), s.size());
    }
    
    void str(U32 field, const char *s) {
        raw(field, (const U8*)s, strlen(s));
    }

    void enc(const Encoder& sub) {
        const auto& sub_buf = sub.buf();
        _buf.insert(_buf.end(), sub_buf.begin(), sub_buf.end());
    }
    
    void enc(const U8V& buf) {
        _buf.insert(_buf.end(), buf.begin(), buf.end());
    }

    // ── Access ───────────────────────────────────────────────────────────────
    const U8V& buf() const { return _buf; }
    
    USZ  size() const { return _buf.size(); }
    void clear() { _buf.clear(); }

private:
    U8V _buf;
};

} // namespace proto
