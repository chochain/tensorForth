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
    void write_tag(uint32_t field_number, uint32_t wire_type) {
        write_var((field_number << 3) | wire_type);
    }

    // ── Varint ───────────────────────────────────────────────────────────────
    void write_var(uint64_t value) {
        while (value > 0x7F) {
            buf_.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
            value >>= 7;
        }
        buf_.push_back(static_cast<uint8_t>(value));
    }

    void write_bool(uint32_t field, bool value) {
        write_tag(field, 0);
        write_var(value ? 1 : 0);
    }

    void write_s32(uint32_t field, int32_t value) {
        write_tag(field, 0);
        write_var(static_cast<uint64_t>(static_cast<uint32_t>(value)));
    }

    void write_s64(uint32_t field, int64_t value) {
        write_tag(field, 0);
        write_var(static_cast<uint64_t>(value));
    }

    void write_u32(uint32_t field, uint32_t value) {
        write_tag(field, 0);
        write_var(value);
    }

    void write_enum(uint32_t field, int32_t value) {
        write_s32(field, value);
    }

    // ── 64-bit (double) ──────────────────────────────────────────────────────
    void write_f64(uint32_t field, double value) {
        write_tag(field, 1);
        uint64_t bits;
        memcpy(&bits, &value, sizeof bits);
        for (int i = 0; i < 8; ++i) {
            buf_.push_back(static_cast<uint8_t>(bits & 0xFF));
            bits >>= 8;
        }
    }

    // ── 32-bit (float) ───────────────────────────────────────────────────────
    void write_f32(uint32_t field, float value) {
        write_tag(field, 5);
        uint32_t bits;
        memcpy(&bits, &value, sizeof bits);
        for (int i = 0; i < 4; ++i) {
            buf_.push_back(static_cast<uint8_t>(bits & 0xFF));
            bits >>= 8;
        }
    }

    // ── Length-delimited ─────────────────────────────────────────────────────
    void write_str(uint32_t field, const std::string& s) {
        write_bytes(field, reinterpret_cast<const uint8_t*>(s.c_str()), s.size());
    }

    void write_bytes(uint32_t field, const uint8_t* data, size_t len) {
        write_tag(field, 2);
        write_var(len);
        buf_.insert(buf_.end(), data, data + len);
    }

    void write_msg(uint32_t field, const Encoder& sub) {
        const auto& sub_buf = sub.buf();
        write_bytes(field, sub_buf.data(), sub_buf.size());
    }

    // Write raw bytes of a message (no field tag) — used for nested messages
    void write_msg_raw(uint32_t field, const std::vector<uint8_t>& data) {
        write_tag(field, 2);
        write_var(data.size());
        buf_.insert(buf_.end(), data.begin(), data.end());
    }

    // Packed repeated floats
    void write_f32_packed(uint32_t field, const std::vector<float>& values) {
        write_tag(field, 2);
        write_var(values.size() * 4);
        for (float v : values) {
            uint32_t bits;
            memcpy(&bits, &v, sizeof bits);
            for (int i = 0; i < 4; ++i) {
                buf_.push_back(static_cast<uint8_t>(bits & 0xFF));
                bits >>= 8;
            }
        }
    }

    // Packed repeated doubles
    void write_f64_packed(uint32_t field, const std::vector<double>& values) {
        write_tag(field, 2);
        write_var(values.size() * 8);
        for (double v : values) {
            uint64_t bits;
            memcpy(&bits, &v, sizeof bits);
            for (int i = 0; i < 8; ++i) {
                buf_.push_back(static_cast<uint8_t>(bits & 0xFF));
                bits >>= 8;
            }
        }
    }

    // ── Access ───────────────────────────────────────────────────────────────
    const std::vector<uint8_t>& buf() const { return buf_; }
    size_t Size() const { return buf_.size(); }
    void Clear() { buf_.clear(); }

private:
    std::vector<uint8_t> buf_;
};

} // namespace proto
