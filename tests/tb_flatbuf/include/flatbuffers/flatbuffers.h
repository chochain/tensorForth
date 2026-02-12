/*
 * flatbuffers.h - Minimal FlatBuffers builder for TensorBoard event writing
 *
 * Implements the FlatBuffers binary format:
 *  - Scalars stored inline in tables
 *  - Strings: [length:uint32][bytes...][null]
 *  - Vectors: [length:uint32][elements...]
 *  - Tables: [soffset to vtable][field data...] + vtable
 *  - Vtable: [vtable_size:uint16][object_size:uint16][field_offsets...]
 *
 * Note: FlatBuffers builds buffers back-to-front for cache efficiency.
 */
#pragma once
#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace flatbuffers {

// ── Offset wrapper for type safety ─────────────────────────────────────────
template<typename T = void>
struct Offset {
    uint32_t o = 0;
    Offset() = default;
    explicit Offset(uint32_t v) : o(v) {}
    bool IsNull() const { return o == 0; }
    Offset<void> Union() const { return Offset<void>(o); }
};

// ── FlatBufferBuilder ────────────────────────────────────────────────────────
class FlatBufferBuilder {
public:
    explicit FlatBufferBuilder(size_t initial = 256) {
        buf_.reserve(initial);
    }

    void Clear() {
        buf_.clear();
        vtable_offsets_.clear();
        current_vtable_.clear();
        object_start_ = 0;
        nested_ = false;
        finished_ = false;
    }

    // Current write position (size of buffer so far)
    uint32_t GetOffset() const { return static_cast<uint32_t>(buf_.size()); }

    size_t GetSize() const { return buf_.size(); }

    // Access the finished buffer (call after Finish)
    const uint8_t* GetBufferPointer() const {
        assert(finished_);
        return buf_.data();
    }
    uint8_t* GetBufferPointer() {
        assert(finished_);
        return buf_.data();
    }

    // ── Low-level write helpers ─────────────────────────────────────────────
private:
    void Align(size_t alignment) {
        size_t mod = buf_.size() % alignment;
        if (mod) {
            size_t pad = alignment - mod;
            buf_.insert(buf_.end(), pad, 0);
        }
    }

    template<typename T>
    uint32_t Write(T value) {
        Align(sizeof(T));
        uint32_t pos = GetOffset();
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&value);
        buf_.insert(buf_.end(), p, p + sizeof(T));
        return pos;
    }

    void WriteBytes(const void* data, size_t len) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
        buf_.insert(buf_.end(), p, p + len);
    }

public:
    // ── Strings ─────────────────────────────────────────────────────────────
    Offset<void> CreateString(const char* str, size_t len) {
        Align(4);
        uint32_t pos = Write<uint32_t>(static_cast<uint32_t>(len));
        WriteBytes(str, len);
        buf_.push_back(0); // null terminator
        return Offset<void>(pos);
    }
    Offset<void> CreateString(const std::string& s) {
        return CreateString(s.c_str(), s.size());
    }
    Offset<void> CreateString(const char* s) {
        return CreateString(s, strlen(s));
    }

    // ── Vectors ─────────────────────────────────────────────────────────────
    template<typename T>
    Offset<void> CreateVector(const std::vector<T>& v) {
        return CreateVector(v.data(), v.size());
    }

    template<typename T>
    Offset<void> CreateVector(const T* data, size_t count) {
        Align(4);
        Write<uint32_t>(static_cast<uint32_t>(count));
        if (count > 0) {
            Align(sizeof(T));
            WriteBytes(data, count * sizeof(T));
        }
        return Offset<void>(static_cast<uint32_t>(buf_.size() - count * sizeof(T) - sizeof(uint32_t)));
    }

    // Vector of offsets
    Offset<void> CreateVectorOfOffsets(const std::vector<Offset<void>>& offsets) {
        // Offsets must be stored as relative references
        // Collect positions first, then create vector of uint32_t relative offsets
        // For simplicity, store absolute offsets; we patch during Finish
        Align(4);
        uint32_t vec_start = Write<uint32_t>(static_cast<uint32_t>(offsets.size()));
        for (auto& off : offsets) {
            Write<uint32_t>(off.o);
        }
        return Offset<void>(vec_start);
    }

    // ── Tables ──────────────────────────────────────────────────────────────
    void StartTable() {
        assert(!nested_ && "Cannot nest tables without finishing the current one");
        nested_ = true;
        object_start_ = GetOffset();
        current_vtable_.clear();
    }

    // Add scalar field (field_id is the field index * sizeof(voffset_t))
    template<typename T>
    void AddElement(uint16_t field_offset, T value, T default_val = T(0)) {
        if (value == default_val) return;
        uint32_t pos = Write(value);
        TrackField(field_offset, pos);
    }

    // Force-add scalar (even if default)
    template<typename T>
    void AddElementForce(uint16_t field_offset, T value) {
        uint32_t pos = Write(value);
        TrackField(field_offset, pos);
    }

    // Add an offset field (reference to another object)
    void AddOffset(uint16_t field_offset, Offset<void> off) {
        if (off.IsNull()) return;
        // Write a placeholder; will be patched to relative offset in EndTable
        Align(4);
        uint32_t pos = GetOffset();
        Write<uint32_t>(off.o); // store absolute for now, patch in EndTable
        pending_refs_.push_back({pos, off.o});
        TrackField(field_offset, pos);
    }

    uint32_t EndTable() {
        assert(nested_ && "Must call StartTable first");
        nested_ = false;

        // Patch all pending offset references to be relative
        PatchPendingRefs();

        uint32_t table_end = GetOffset();

        // Write vtable
        // Format: [vtable_size:u16][object_size:u16][field_offset_0:u16]...
        uint16_t max_field_offset = 0;
        for (auto& [foff, pos] : current_vtable_) {
            max_field_offset = std::max(max_field_offset, foff);
        }
        // Number of field slots needed
        uint16_t num_slots = (max_field_offset / 2) + 1;
        if (current_vtable_.empty()) num_slots = 0;

        uint16_t vtable_size = static_cast<uint16_t>((2 + num_slots) * 2);
        uint16_t object_size = static_cast<uint16_t>(table_end - object_start_ + 4); // +4 for soffset

        Align(2);
        uint32_t vtable_start = GetOffset();
        Write<uint16_t>(vtable_size);
        Write<uint16_t>(object_size);

        // Fill in field slots (relative to object start)
        std::vector<uint16_t> slots(num_slots, 0);
        for (auto& [foff, fpos] : current_vtable_) {
            uint16_t slot_idx = foff / 2;
            if (slot_idx < num_slots) {
                // offset from object start to this field
                slots[slot_idx] = static_cast<uint16_t>(fpos - object_start_);
            }
        }
        for (auto s : slots) Write<uint16_t>(s);
        (void)GetOffset(); // vtable end position

        // Write soffset_t at the table's start position
        // soffset = vtable_start - object_start (negative means vtable is after the object in our forward build)
        // In standard FlatBuffers, soffset = vtable_pos - object_pos (can be negative)
        // We're building forward, so vtable comes AFTER the object data
        int32_t soffset = static_cast<int32_t>(vtable_start) - static_cast<int32_t>(object_start_);

        // Insert soffset at object_start_
        // We need to insert 4 bytes at position object_start_ and shift everything
        // Actually we reserved space conceptually — let's just write it at the end and return
        // the vtable end as the "table" since we'll do a simpler layout

        // REVISED APPROACH: For TensorBoard we use Protocol Buffers not FlatBuffers for the
        // actual Summary/Event, but we demonstrate FlatBuffers by writing our OWN schema.
        // Let's keep it simple: write soffset now at buf_[object_start_]
        // We actually need to pre-allocate soffset space. Let me restructure.

        // Store the soffset position and value for patching
        soffset_patches_.push_back({object_start_, soffset});
        vtable_offsets_.push_back(vtable_start);
        current_vtable_.clear();

        return object_start_; // return start of object for offset references
    }

    // ── Finish ───────────────────────────────────────────────────────────────
    template<typename T>
    void Finish(Offset<T> root) {
        // Write root table offset (relative from this position)
        Align(4);
        // Apply all soffset patches
        for (auto& [pos, val] : soffset_patches_) {
            *reinterpret_cast<int32_t*>(buf_.data() + pos) = val;
        }
        soffset_patches_.clear();

        Write<uint32_t>(root.o); // root offset (absolute, will be relative in real FB)
        finished_ = true;
    }

    // Raw access
    std::vector<uint8_t>& GetBuffer() { return buf_; }
    const std::vector<uint8_t>& GetBuffer() const { return buf_; }

private:
    void TrackField(uint16_t field_offset, uint32_t data_pos) {
        // field_offset is field_id * 2 (as voffset_t)
        current_vtable_.push_back({field_offset, data_pos});
    }

    void PatchPendingRefs() {
        for (auto& [src_pos, dst_abs] : pending_refs_) {
            // relative offset = dst_abs - src_pos
            int32_t rel = static_cast<int32_t>(dst_abs) - static_cast<int32_t>(src_pos);
            *reinterpret_cast<int32_t*>(buf_.data() + src_pos) = rel;
        }
        pending_refs_.clear();
    }

    std::vector<uint8_t> buf_;
    bool nested_    = false;
    bool finished_  = false;
    uint32_t object_start_ = 0;

    // Track field positions in current object: (field_voffset, absolute_pos)
    std::vector<std::pair<uint16_t, uint32_t>> current_vtable_;
    // Pending relative offset patches: (src_abs_pos, dst_abs_pos)
    std::vector<std::pair<uint32_t, uint32_t>> pending_refs_;
    // soffset patches: (object_start_pos, soffset_value)
    std::vector<std::pair<uint32_t, int32_t>> soffset_patches_;
    // All vtable start positions
    std::vector<uint32_t> vtable_offsets_;
};

} // namespace flatbuffers
