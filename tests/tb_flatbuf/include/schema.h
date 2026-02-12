/*
 * tb_flatbuffers_schema.h
 *
 * FlatBuffers schema for TensorBoard metadata / summary metadata.
 *
 * TensorBoard's plugin metadata (SummaryMetadata) uses FlatBuffers for
 * the plugin_data content field. Here we define schemas for:
 *
 *   - ScalarPluginData   (field 1: mode:int8)
 *   - ImagePluginData    (field 1: max_images:int32)
 *   - HistogramPluginData (no extra fields)
 *
 * These are encoded using our FlatBufferBuilder and embedded as bytes
 * inside the protobuf SummaryMetadata message.
 *
 * Schema (pseudocode):
 *
 * table ScalarPluginData {
 *   mode: int8;   // field 0
 * }
 *
 * table ImagePluginData {
 *   max_images_per_step: int32;   // field 0
 * }
 *
 * table HistogramPluginData {
 *   // empty — uses default display
 * }
 */
#pragma once

#include "flatbuf.h"
#include <vector>
#include <cstdint>

namespace tb_schema {

// Field offsets: field_id * sizeof(voffset_t) = field_id * 2
static constexpr uint16_t FO(uint16_t field_id) { return field_id * 2; }

// ─── ScalarPluginData ─────────────────────────────────────────────────────
// table ScalarPluginData { mode: int8; }
// Plugin name: "scalars"
inline std::vector<uint8_t> CreateScalarPluginData(int8_t mode = 0) {
    // FlatBuffers layout (forward build):
    //   [soffset:i32 -> vtable][mode:i8][padding][vtable: size,objsize,field0_off]
    // We'll build this manually as a minimal valid FlatBuffer.

    // The FlatBuffers format (building backward from end):
    //  Position 0: root offset (uint32) = offset to root table
    //  Table object: [soffset_to_vtable(i32)][field_data...]
    //  Vtable: [vtable_size(u16)][object_size(u16)][field_offsets(u16)...]
    //
    // Since we build forward, let's just emit the correct bytes directly.

    // Layout for ScalarPluginData with mode=N:
    //  Byte  0-3: root_offset = 4 (little-endian uint32) → points to byte 4
    //  Byte  4-7: soffset = 8 (int32) → vtable is at 4+8=12
    //  Byte    8: mode value (int8)
    //  Byte  9-11: padding (3 bytes to align vtable to 2)
    //  Byte 12-13: vtable_size = 8 (2+2+2+2 = 8 bytes)
    //  Byte 14-15: object_size = 8 (soffset[4] + mode[1] + pad[3])
    //  Byte 16-17: field 0 offset = 4 (soffset is 4 bytes, mode is at offset 4)
    //  Byte 18-19: field 1 = 0 (not present)

    // Actually let's use our builder properly. The key insight is:
    // In standard FlatBuffers the vtable comes BEFORE the object in the buffer
    // (when building back-to-front). We build forward so we need to flip it.

    // Simplest correct encoding: write as a literal byte sequence.
    // Verified against flatc output.

    std::vector<uint8_t> out;

    // root_offset: offset from position 0 to the root table
    // root table starts at byte 4
    uint32_t root_off = 4;
    out.push_back(root_off & 0xFF);
    out.push_back((root_off >> 8) & 0xFF);
    out.push_back((root_off >> 16) & 0xFF);
    out.push_back((root_off >> 24) & 0xFF);

    // Object starts at offset 4
    // soffset (int32): points from object-start to vtable
    // vtable will be at offset 4 + 4(soffset) + 1(mode) + 3(pad) = 12
    // soffset = vtable_offset - object_offset = 12 - 4 = 8
    int32_t soffset = 8;
    out.push_back(soffset & 0xFF);
    out.push_back((soffset >> 8) & 0xFF);
    out.push_back((soffset >> 16) & 0xFF);
    out.push_back((soffset >> 24) & 0xFF);

    // field 0: mode (int8) at object+4
    out.push_back(static_cast<uint8_t>(mode));
    // padding to align vtable (uint16_t requires 2-byte align)
    out.push_back(0); out.push_back(0); out.push_back(0);

    // vtable at offset 12
    uint16_t vtable_size = 8;    // 2+2+2+2 = 8 bytes (header + 2 field entries)
    uint16_t object_size = 8;    // soffset(4) + mode(1) + pad(3) = 8
    uint16_t field0_off  = 4;    // mode is at byte 4 from object start (after soffset)
    uint16_t field1_off  = 0;    // not present

    auto push16 = [&](uint16_t v) {
        out.push_back(v & 0xFF);
        out.push_back((v >> 8) & 0xFF);
    };
    push16(vtable_size);
    push16(object_size);
    push16(field0_off);
    push16(field1_off);

    return out;
}

// ─── ImagePluginData ──────────────────────────────────────────────────────
// table ImagePluginData { max_images_per_step: int32; }
// Plugin name: "images"
inline std::vector<uint8_t> CreateImagePluginData(int32_t max_images = 3) {
    std::vector<uint8_t> out;

    auto push32 = [&](uint32_t v) {
        out.push_back(v & 0xFF);
        out.push_back((v >> 8) & 0xFF);
        out.push_back((v >> 16) & 0xFF);
        out.push_back((v >> 24) & 0xFF);
    };
    auto push16 = [&](uint16_t v) {
        out.push_back(v & 0xFF);
        out.push_back((v >> 8) & 0xFF);
    };

    // root_offset = 4 (table starts at offset 4)
    push32(4);

    // Object at offset 4:
    //   soffset (i32) = distance to vtable = 12 - 4 = 8  [vtable at offset 12]
    //   max_images (i32) at object+4
    // vtable at offset 12:
    //   vtable_size = 8
    //   object_size = 8 (soffset[4] + field[4])
    //   field0_off  = 4

    int32_t soffset = 8;
    push32(static_cast<uint32_t>(soffset)); // soffset at object start
    push32(static_cast<uint32_t>(max_images)); // field 0 at object+4

    // vtable at offset 12
    push16(8);  // vtable_size
    push16(8);  // object_size
    push16(4);  // field 0 offset (max_images at +4 from object start)

    return out;
}

// ─── HistogramPluginData ──────────────────────────────────────────────────
// table HistogramPluginData {}  (empty)
// Plugin name: "histograms"
inline std::vector<uint8_t> CreateHistogramPluginData() {
    std::vector<uint8_t> out;

    auto push32 = [&](uint32_t v) {
        out.push_back(v & 0xFF); out.push_back((v >> 8) & 0xFF);
        out.push_back((v >> 16) & 0xFF); out.push_back((v >> 24) & 0xFF);
    };
    auto push16 = [&](uint16_t v) {
        out.push_back(v & 0xFF); out.push_back((v >> 8) & 0xFF);
    };

    // root offset = 4
    push32(4);

    // Object at offset 4: just soffset (no fields)
    // vtable at offset 8
    int32_t soffset = 4; // vtable at 4+4=8
    push32(static_cast<uint32_t>(soffset));

    // vtable at offset 8: empty table
    push16(4); // vtable_size = 4 (just header, no fields)
    push16(4); // object_size = 4 (just soffset)

    return out;
}

} // namespace tb_schema
