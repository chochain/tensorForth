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
 */
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace proto {

class Encoder {
public:
    // ── Key/Tag encoding ─────────────────────────────────────────────────────
    // field_number << 3 | wire_type
    void tag(uint32_t field_number, uint32_t wire_type) {
        u64((field_number << 3) | wire_type);
    }

    // ── Varint ───────────────────────────────────────────────────────────────
    void u64(uint64_t value) {
        while (value > 0x7F) {
            _buf.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
            value >>= 7;
        }
        _buf.push_back(static_cast<uint8_t>(value));
    }

    void write_bool(uint32_t field, bool value) {
        tag(field, 0);
        u64(value ? 1 : 0);
    }

    void s32(uint32_t field, int32_t value) {
        tag(field, 0);
        u64(static_cast<uint64_t>(static_cast<uint32_t>(value)));
    }

    void s64(uint32_t field, int64_t value) {
        tag(field, 0);
        u64(static_cast<uint64_t>(value));
    }

    void write_u32(uint32_t field, uint32_t value) {
        tag(field, 0);
        u64(value);
    }

    void write_enum(uint32_t field, int32_t value) {
        s32(field, value);
    }

    // ── 64-bit (double) ──────────────────────────────────────────────────────
    void f64(uint32_t field, double value) {
        tag(field, 1);
        uint64_t bits;
        memcpy(&bits, &value, sizeof bits);
        for (int i = 0; i < 8; ++i) {
            _buf.push_back(static_cast<uint8_t>(bits & 0xFF));
            bits >>= 8;
        }
    }

    // ── 32-bit (float) ───────────────────────────────────────────────────────
    void f32(uint32_t field, float value) {
        tag(field, 5);
        uint32_t bits;
        memcpy(&bits, &value, sizeof bits);
        for (int i = 0; i < 4; ++i) {
            _buf.push_back(static_cast<uint8_t>(bits & 0xFF));
            bits >>= 8;
        }
    }

    // ── Length-delimited ─────────────────────────────────────────────────────
    void str(uint32_t field, const std::string& s) {
        raw(field, reinterpret_cast<const uint8_t*>(s.c_str()), s.size());
    }

    void raw(uint32_t field, const uint8_t* data, size_t len) {
        tag(field, 2);
        u64(len);
        _buf.insert(_buf.end(), data, data + len);
    }

    // Write raw bytes of a message (no field tag) — used for nested messages
    void raw(uint32_t field, const std::vector<uint8_t>& data) {
        tag(field, 2);
        u64(data.size());
        _buf.insert(_buf.end(), data.begin(), data.end());
    }

    void msg(uint32_t field, const Encoder& sub) {
        const auto& sub_buf = sub.buf();
        raw(field, sub_buf.data(), sub_buf.size());
    }

    // Packed repeated floats
    void f32_packed(uint32_t field, const std::vector<float>& values) {
        tag(field, 2);
        u64(values.size() * 4);
        for (float v : values) {
            uint32_t bits;
            memcpy(&bits, &v, sizeof bits);
            for (int i = 0; i < 4; ++i) {
                _buf.push_back(static_cast<uint8_t>(bits & 0xFF));
                bits >>= 8;
            }
        }
    }

    // Packed repeated doubles
    void f64_packed(uint32_t field, const std::vector<double>& values) {
        tag(field, 2);
        u64(values.size() * 8);
        for (double v : values) {
            uint64_t bits;
            memcpy(&bits, &v, sizeof bits);
            for (int i = 0; i < 8; ++i) {
                _buf.push_back(static_cast<uint8_t>(bits & 0xFF));
                bits >>= 8;
            }
        }
    }

    // ── Access ───────────────────────────────────────────────────────────────
    const std::vector<uint8_t>& buf() const { return _buf; }
    
    size_t size() const { return _buf.size(); }
    void   clear() { _buf.clear(); }

private:
    std::vector<uint8_t> _buf;
};

} // namespace proto
