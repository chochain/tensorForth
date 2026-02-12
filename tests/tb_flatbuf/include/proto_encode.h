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
    void WriteTag(uint32_t field_number, uint32_t wire_type) {
        WriteVarint((field_number << 3) | wire_type);
    }

    // ── Varint ───────────────────────────────────────────────────────────────
    void WriteVarint(uint64_t value) {
        while (value > 0x7F) {
            buf_.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
            value >>= 7;
        }
        buf_.push_back(static_cast<uint8_t>(value));
    }

    void WriteBool(uint32_t field, bool value) {
        WriteTag(field, 0);
        WriteVarint(value ? 1 : 0);
    }

    void WriteInt32(uint32_t field, int32_t value) {
        WriteTag(field, 0);
        WriteVarint(static_cast<uint64_t>(static_cast<uint32_t>(value)));
    }

    void WriteInt64(uint32_t field, int64_t value) {
        WriteTag(field, 0);
        WriteVarint(static_cast<uint64_t>(value));
    }

    void WriteUInt32(uint32_t field, uint32_t value) {
        WriteTag(field, 0);
        WriteVarint(value);
    }

    void WriteEnum(uint32_t field, int32_t value) {
        WriteInt32(field, value);
    }

    // ── 64-bit (double) ──────────────────────────────────────────────────────
    void WriteDouble(uint32_t field, double value) {
        WriteTag(field, 1);
        uint64_t bits;
        memcpy(&bits, &value, sizeof bits);
        for (int i = 0; i < 8; ++i) {
            buf_.push_back(static_cast<uint8_t>(bits & 0xFF));
            bits >>= 8;
        }
    }

    // ── 32-bit (float) ───────────────────────────────────────────────────────
    void WriteFloat(uint32_t field, float value) {
        WriteTag(field, 5);
        uint32_t bits;
        memcpy(&bits, &value, sizeof bits);
        for (int i = 0; i < 4; ++i) {
            buf_.push_back(static_cast<uint8_t>(bits & 0xFF));
            bits >>= 8;
        }
    }

    // ── Length-delimited ─────────────────────────────────────────────────────
    void WriteString(uint32_t field, const std::string& s) {
        WriteBytes(field, reinterpret_cast<const uint8_t*>(s.c_str()), s.size());
    }

    void WriteBytes(uint32_t field, const uint8_t* data, size_t len) {
        WriteTag(field, 2);
        WriteVarint(len);
        buf_.insert(buf_.end(), data, data + len);
    }

    void WriteMessage(uint32_t field, const Encoder& sub) {
        const auto& sub_buf = sub.GetBuffer();
        WriteBytes(field, sub_buf.data(), sub_buf.size());
    }

    // Write raw bytes of a message (no field tag) — used for nested messages
    void WriteRawMessage(uint32_t field, const std::vector<uint8_t>& data) {
        WriteTag(field, 2);
        WriteVarint(data.size());
        buf_.insert(buf_.end(), data.begin(), data.end());
    }

    // Packed repeated floats
    void WritePackedFloats(uint32_t field, const std::vector<float>& values) {
        WriteTag(field, 2);
        WriteVarint(values.size() * 4);
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
    void WritePackedDoubles(uint32_t field, const std::vector<double>& values) {
        WriteTag(field, 2);
        WriteVarint(values.size() * 8);
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
    const std::vector<uint8_t>& GetBuffer() const { return buf_; }
    size_t Size() const { return buf_.size(); }
    void Clear() { buf_.clear(); }

private:
    std::vector<uint8_t> buf_;
};

} // namespace proto
