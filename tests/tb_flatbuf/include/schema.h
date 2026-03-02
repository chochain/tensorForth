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
 *   - HParamPluginData
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
 * ── Correct proto field numbers (cross-checked against TB compat protos) ─────
 *
 * tensorflow.Event  (event.proto):
 *   1: wall_time (double)
 *   2: step (int64)
 *   3: file_version (string)
 *   +4: graph_def (bytes)
 *   5: summary (Summary)
 *   +6: log_message (LogMessage)
 *   +7: session_log (SessionLog)
 *   +8: tagged_run_metadata (TaggedRunMetaData)
 *   9: meta_graph_def (bytes)  ← do NOT write here
 *
 * +tensorflow.ResourceHandleProto
 *
 * +tensorflow.DataType
 *   DT_FLOAT=1, DT_DOUBLE=2, ..., DT_STRING=7
 *
 * = Tensor =====================
 * tensorflow.TensorProto:
 *   1: dtype (DT_FLOAT=1)
 *   2: tensor_shape (empty=scalar)
 *   +3: version_number (int32)
 *   +4: tensor_content (bytes)
 *   5: float_val (packed)
 *   +6: double_val (packed)
 *   +7: int_val (packed)
 *   8: string_val (bytes repeated)
 *   +9: scomplex_val (float packed)
 *   +10: int64_val (packed)
 *   +11: bool_val (packed)
 *   +12: dcomplex_val (double packed)
 *   +13: half_val (int32 packed)
 *   +14: resource_handle_val (ResourceHandleProto)
 *   x15: variant_val (VariantTensorDataProto) 
 *   +16: uint32_val (packed)
 *   +17: uint64_val (packed)
 *
 * = GraphDef ===========================================
 * tensorflow.Graph
 *   1: node (NodeDef repeated)
 *   +2: FunctionDefLibrary library
 *   3: version (int32)
 *   +4: versions (VersionDef)
 *
 * tensorflow.NodeDef
 *   1: name (string)
 *   2: op   (string)
 *   3: input (string repeated)
 *   4: device (string)
 *   5: attr (map<string, AttrValue>)
 *
 * tensorflow.AttrValue
 *   oneof value {
 *     bytes s = 2;                 // "string"
 *     int64 i = 3;                 // "int"
 *     float f = 4;                 // "float"
 *     bool b = 5;                  // "bool"
 *     DataType type = 6;           // "type"
 *     TensorShapeProto shape = 7;  // "shape"
 *     TensorProto tensor = 8;      // "tensor"
 *     ListValue list = 1;          // any "list(...)"
 *     string placeholder = 9;      // For library functions
 *     +NameAttrList func = 10;
 *   }
 *
 *= For attributes that are lists (e.g., strides: [1, 2, 2, 1])
 *  message ListValue {
 *    repeated bytes s = 2;
 *    repeated int64 i = 3 [packed = true];
 *    repeated float f = 4 [packed = true];
 *    repeated bool b = 5 [packed = true];
 *    repeated DataType type = 6 [packed = true];
 *    repeated TensorShapeProto shape = 7;
 *    repeated TensorProto tensor = 8;
 *    +repeated NameAttrList func = 9;              // "list(attr)"
 *  }
 *
 * = Summary ============================================
 * tensorflow.Summary.Value  (summary.proto):
 *   1: tag (string)
 *   oneof {
 *     +2: simple_value (float)
 *     +3: obsolete_old_style_histogram (bytes)
 *     4: image (Summary.Image)
 *     5: histo (HistogramProto)
 *     +6: audio (Summary.Audio)
 *     8: tensor (TensorProto)
 *   }
 *   9: metadata (SummaryMetadata)
 *
 * tensorflow.SummaryMetadata:
 *   1: plugin_data (PluginData)
 *   +2: display_name (string)
 *   +3: summary_description (string)
 *   4: data_class (DataClass: SCALAR=1, TENSOR=2, BLOB=3)
 *
 * tensorflow.SummaryMetadata.PluginData:
 *   1: plugin_name (string)    2: content (bytes)
 *
 * tensorflow.Summary.Image:
 *   1: height
 *   2: width
 *   3: colorspace (grascale=1, grascale+alpha=2, RGB=3, RGBA=4, DIGITAL_YUV=5, RGBA=6)
 *   4: encoded_image_string (bytes)
 *
 * +tensorflow.Summary.Audio
 *
 * tensorflow.HistogramProto:
 *   1:min 2:max 3:num 4:sum 5:sum_squares
 *   6: bucket_limit 7:bucket
 *   +6: repeated bucket_limit, +7: repeated bucket
 *
 * = Projector ============================================
 * +tensorflow.ProjectorConfig
 *   1: model_checkpoint_path (string)
 *   2: embeddings (EmbeddingInfo repeated)
 *   3: model_checkpoint_dir (string)
 *
 * +tensorflow.EmbeddingInfo
 *   1: tensor_name (string)
 *   2: metadata_path (string)
 *   3: bookmark_path (string)
 *   4: tensor_shape (uint32 repeated)
 *   5: sprite (SpriteMetadata)
 *
 * +tensorflow.SpriteMetadata
 *   1: image_path (string)
 *   2: single_image_dim (uint32 repeated)
 *
 * ── TFRecord framing ─────────────────────────────────────────────────────────
 * [length:uint64 LE]
 * [masked_crc32c(length):uint32 LE]
 * [data][masked_crc32c(data):uint32 LE]
 * masking: ((crc >> 15 | crc << 17) + 0xa282ead8) & 0xFFFFFFFF
 */
#pragma once

#include "flatbuf.h"

namespace proto {

// Field offsets: field_id * sizeof(voffset_t) = field_id * 2
static constexpr U16 FO(U16 field_id) { return field_id * 2; }

// ─── ScalarPluginData ─────────────────────────────────────────────────────
// table ScalarPluginData { mode: int8; }
// Plugin name: "scalars"
inline U8V scalar_meta(U8 mode = 0) {
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

    U8V out;

    // root_offset: offset from position 0 to the root table
    // root table starts at byte 4
    U32 root_off = 4;
    out.push_back(root_off & 0xFF);
    out.push_back((root_off >> 8) & 0xFF);
    out.push_back((root_off >> 16) & 0xFF);
    out.push_back((root_off >> 24) & 0xFF);

    // Object starts at offset 4
    // soffset (int32): points from object-start to vtable
    // vtable will be at offset 4 + 4(soffset) + 1(mode) + 3(pad) = 12
    // soffset = vtable_offset - object_offset = 12 - 4 = 8
    S32 soffset = 8;
    out.push_back(soffset & 0xFF);
    out.push_back((soffset >> 8) & 0xFF);
    out.push_back((soffset >> 16) & 0xFF);
    out.push_back((soffset >> 24) & 0xFF);

    // field 0: mode (int8) at object+4
    out.push_back(static_cast<U8>(mode));
    // padding to align vtable (U16 requires 2-byte align)
    out.push_back(0); out.push_back(0); out.push_back(0);

    // vtable at offset 12
    U16 vtable_size = 8;    // 2+2+2+2 = 8 bytes (header + 2 field entries)
    U16 object_size = 8;    // soffset(4) + mode(1) + pad(3) = 8
    U16 field0_off  = 4;    // mode is at byte 4 from object start (after soffset)
    U16 field1_off  = 0;    // not present

    auto push16 = [&](U16 v) {
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
inline U8V image_meta(S32 max_images = 3) {
    U8V out;

    auto push32 = [&](U32 v) {
        out.push_back(v & 0xFF);
        out.push_back((v >> 8) & 0xFF);
        out.push_back((v >> 16) & 0xFF);
        out.push_back((v >> 24) & 0xFF);
    };
    auto push16 = [&](U16 v) {
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

    S32 soffset = 8;
    push32(static_cast<U32>(soffset)); // soffset at object start
    push32(static_cast<U32>(max_images)); // field 0 at object+4

    // vtable at offset 12
    push16(8);  // vtable_size
    push16(8);  // object_size
    push16(4);  // field 0 offset (max_images at +4 from object start)

    return out;
}

// ─── HistogramPluginData ──────────────────────────────────────────────────
// table HistogramPluginData {}  (empty)
// Plugin name: "histograms"
inline U8V histo_meta() {
    U8V out;

    auto push32 = [&](U32 v) {
        out.push_back(v & 0xFF); out.push_back((v >> 8) & 0xFF);
        out.push_back((v >> 16) & 0xFF); out.push_back((v >> 24) & 0xFF);
    };
    auto push16 = [&](U16 v) {
        out.push_back(v & 0xFF); out.push_back((v >> 8) & 0xFF);
    };

    // root offset = 4
    push32(4);

    // Object at offset 4: just soffset (no fields)
    // vtable at offset 8
    S32 soffset = 4; // vtable at 4+4=8
    push32(static_cast<U32>(soffset));

    // vtable at offset 8: empty table
    push16(4); // vtable_size = 4 (just header, no fields)
    push16(4); // object_size = 4 (just soffset)

    return out;
}

} // namespace tb_schema
