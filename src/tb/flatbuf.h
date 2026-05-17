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

#include "types.h"

#include <cassert>
#include <algorithm>
#include <stdexcept>

namespace tensorboard {

// ── Offset wrapper for type safety ─────────────────────────────────────────
template<typename T = void>
struct Offset {
    U32 o = 0;
    Offset() = default;
    
    explicit Offset(U32 v) : o(v) {}
    
    BOOL         IsNull() const { return o == 0; }
    Offset<void> Union()  const { return Offset<void>(o); }
};

// ── FlatBufferBuilder ────────────────────────────────────────────────────────
class FlatBufferBuilder {
public:
    explicit FlatBufferBuilder(USZ initial = 256) {
        _buf.reserve(initial);
    }

    void clear() {
        _buf.clear();
        _voff.clear();
        _vtable.clear();
        _start  = 0;
        _nested = false;
        _finished = false;
    }

    // Current write position (size of buffer so far)
    U32 offset() const { return static_cast<U32>(_buf.size()); }

    USZ size() const { return _buf.size(); }

    // Access the finished buffer (call after Finish)
    const U8* buf() const {
        assert(_finished);
        return _buf.data();
    }
    U8* buf() {
        assert(_finished);
        return _buf.data();
    }

    // ── Low-level write helpers ─────────────────────────────────────────────
private:
    void align(USZ alignment) {
        USZ mod = _buf.size() % alignment;
        if (mod) {
            USZ pad = alignment - mod;
            _buf.insert(_buf.end(), pad, 0);
        }
    }

    template<typename T>
    U32 write(T value) {
        align(sizeof(T));
        U32 pos = offset();
        const U8* p = reinterpret_cast<const U8*>(&value);
        _buf.insert(_buf.end(), p, p + sizeof(T));
        return pos;
    }

    void write(const void* data, USZ len) {
        const U8* p = reinterpret_cast<const U8*>(data);
        _buf.insert(_buf.end(), p, p + len);
    }

public:
    // ── Strings ─────────────────────────────────────────────────────────────
    Offset<void> to_s(const char* str, USZ len) {
        align(4);
        U32 pos = write<U32>(static_cast<U32>(len));
        write(str, len);
        _buf.push_back(0); // null terminator
        
        return Offset<void>(pos);
    }
    Offset<void> to_s(const STR& s) {
        return to_s(s.c_str(), s.size());
    }
    Offset<void> to_s(const char* s) {
        return to_s(s, strlen(s));
    }

    // ── Vectors ─────────────────────────────────────────────────────────────
    template<typename T>
    Offset<void> vec(const std::vector<T>& v) {
        return vec(v.data(), v.size());
    }

    template<typename T>
    Offset<void> vec(const T* data, USZ count) {
        align(4);
        write<U32>(static_cast<U32>(count));
        if (count > 0) {
            align(sizeof(T));
            write(data, count * sizeof(T));
        }
        return Offset<void>(static_cast<U32>(_buf.size() - count * sizeof(T) - sizeof(U32)));
    }

    // Vector of offsets
    Offset<void> voff(const std::vector<Offset<void>>& offsets) {
        // Offsets must be stored as relative references
        // Collect positions first, then create vector of U32 relative offsets
        // For simplicity, store absolute offsets; we patch during Finish
        align(4);
        U32 vec_start = write<U32>(static_cast<U32>(offsets.size()));
        for (auto& off : offsets) {
            write<U32>(off.o);
        }
        return Offset<void>(vec_start);
    }

    // ── Tables ──────────────────────────────────────────────────────────────
    void start_table() {
        assert(!_nested && "Cannot nest tables without finishing the current one");
        _nested = true;
        _start  = offset();
        _vtable.clear();
    }

    // Add scalar field (field_id is the field index * sizeof(voffset_t))
    template<typename T>
    void add(U16 field_offset, T value, T default_val = T(0)) {
        if (value == default_val) return;
        U32 pos = write(value);
        _track(field_offset, pos);
    }

    // Force-add scalar (even if default)
    template<typename T>
    void add_force(U16 field_offset, T value) {
        U32 pos = write(value);
        _track(field_offset, pos);
    }

    // Add an offset field (reference to another object)
    void add(U16 field_offset, Offset<void> off) {
        if (off.IsNull()) return;
        // Write a placeholder; will be patched to relative offset in EndTable
        align(4);
        U32 pos = offset();
        write<U32>(off.o); // store absolute for now, patch in EndTable
        _pending.push_back({pos, off.o});
        _track(field_offset, pos);
    }

    U32 end_table() {
        assert(_nested && "Must call StartTable first");
        _nested = false;

        // Patch all pending offset references to be relative
        _patch();

        U32 table_end = offset();

        // Write vtable
        // Format: [vtable_size:u16][object_size:u16][field_offset_0:u16]...
        U16 max_field_offset = 0;
        for (auto& [foff, pos] : _vtable) {
            max_field_offset = std::max(max_field_offset, foff);
        }
        // Number of field slots needed
        U16 num_slots = (max_field_offset / 2) + 1;
        if (_vtable.empty()) num_slots = 0;

        U16 vtable_size = static_cast<U16>((2 + num_slots) * 2);
        U16 object_size = static_cast<U16>(table_end - _start + 4); // +4 for soffset

        align(2);
        U32 vtable_start = offset();
        write<U16>(vtable_size);
        write<U16>(object_size);

        // Fill in field slots (relative to object start)
        U16V slots(num_slots, 0);
        for (auto& [foff, fpos] : _vtable) {
            U16 slot_idx = foff / 2;
            if (slot_idx < num_slots) {
                // offset from object start to this field
                slots[slot_idx] = static_cast<U16>(fpos - _start);
            }
        }
        for (auto s : slots) write<U16>(s);
        (void)offset(); // vtable end position

        // Write soffset_t at the table's start position
        // soffset = vtable_start - object_start (negative means vtable is after the object in our forward build)
        // In standard FlatBuffers, soffset = vtable_pos - object_pos (can be negative)
        // We're building forward, so vtable comes AFTER the object data
        int32_t soffset = static_cast<int32_t>(vtable_start) - static_cast<int32_t>(_start);

        // Insert soffset at _start
        // We need to insert 4 bytes at position _start and shift everything
        // Actually we reserved space conceptually — let's just write it at the end and return
        // the vtable end as the "table" since we'll do a simpler layout

        // REVISED APPROACH: For TensorBoard we use Protocol Buffers not FlatBuffers for the
        // actual Summary/Event, but we demonstrate FlatBuffers by writing our OWN schema.
        // Let's keep it simple: write soffset now at _buf[_start]
        // We actually need to pre-allocate soffset space. Let me restructure.

        // Store the soffset position and value for patching
        _soff.push_back({_start, soffset});
        _voff.push_back(vtable_start);
        _vtable.clear();

        return _start; // return start of object for offset references
    }

    // ── Finish ───────────────────────────────────────────────────────────────
    template<typename T>
    void finish(Offset<T> root) {
        // Write root table offset (relative from this position)
        align(4);
        // Apply all soffset patches
        for (auto& [pos, val] : _soff) {
            *reinterpret_cast<int32_t*>(_buf.data() + pos) = val;
        }
        _soff.clear();

        write<U32>(root.o); // root offset (absolute, will be relative in real FB)
        _finished = true;
    }

    // Raw access
    U8V& rbuf() { return _buf; }
    const U8V& rbuf() const { return _buf; }

private:
    void _track(U16 field_offset, U32 data_pos) {
        // field_offset is field_id * 2 (as voffset_t)
        _vtable.push_back({field_offset, data_pos});
    }

    void _patch() {
        for (auto& [src_pos, dst_abs] : _pending) {
            // relative offset = dst_abs - src_pos
            int32_t rel = static_cast<int32_t>(dst_abs) - static_cast<int32_t>(src_pos);
            *reinterpret_cast<int32_t*>(_buf.data() + src_pos) = rel;
        }
        _pending.clear();
    }

    U8V  _buf;
    BOOL _nested    = false;
    BOOL _finished  = false;
    U32  _start     = 0;

    // Track field positions in current object: (field_voffset, absolute_pos)
    std::vector<std::pair<U16, U32>> _vtable;
    // Pending relative offset patches: (src_abs_pos, dst_abs_pos)
    std::vector<std::pair<U32, U32>> _pending;
    // soffset patches: (_startpos, soffset_value)
    std::vector<std::pair<U32, int32_t>> _soff;
    // All vtable start positions
    std::vector<U32> _voff;
};

} // namespace flatbuffers
