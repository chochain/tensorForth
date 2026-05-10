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
 
## Architecture Overview

tfevents file
└── TFRecord frames (length + masked CRC32C framing)
    └── tensorflow.Event (Protocol Buffers)
        └── tensorflow.Summary
            └── Summary.Value[]
                ├── tag: string
                ├── SummaryMetadata (protobuf)
                │   └── PluginData
                │       ├── plugin_name: "scalars" | "text" | "images" | "histograms" | "hparams"
                │       └── content: bytes  ◄── FlatBuffers encoded!
                │           ├── ScalarPluginData    { mode: int8 }
                │           ├── (empty for text)
                │           ├── HParamPluginData (one of)
                │           │   ├── Experiment
                │           │   ├── SessionStartInfo
                │           │   └── SessionEndInfo
                │           ├── ImagePluginData     { max_images: int32 }
                │           └── HistogramPluginData { (empty) }
                └── payload (one of):
                    ├── simple_value: float   (scalar)
                    ├── tensor: TensorProto   (text, hparams, or other tensor types)
                    ├── image: Summary.Image  (PNG bytes)
                    └── histo: HistogramProto (bucket edges + counts)

       [ EVENT FILE (.v2) ]

                |
                v
      +-------------------+
      |      Summary      |
      +---------+---------+

                |
     +----------+----------+
     |                     |
[  Value  ]           [  Value  ] ...

     |
     +--> 1) tag: "loss"
     |
     +--> 8) tensor: [0.42, 0.38, ...]

     |
     +--> 9) metadata: [ SummaryMetadata ]
                |
                +--> display_name: "Training Loss"

                |
                +--> 4) data_class: DATA_CLASS_SCALAR
                |
                +--> 1) plugin_data: [ PluginData ] <---+

                           |                         |
                           +--> plugin_name: "scalars"
                           |
                           +--> content: <serialized bytes>
                                (e.g., version info)

       PluginData (Message)
 _________________________________
|                                 |
|  plugin_name: "pr_curves"       |-----> Tells TB which 
|  (string)                       |       Dashboard to load.
|_________________________________|
|                                 |
|  content: \x08\x01\x12\x04...   |-----> Opaque byte string.
|  (bytes)                        |       Plugin-specific 
|_________________________________|       config/metadata.

### HParams Session Structure

Summary
└── value (Summary.Value)
    ├── tag: "_hparams_/experiment"
    ├── metadata (SummaryMetadata)
    │   └── plugin_data (FeatureNameConfig)
    │       └── plugin_name: "hparams"  <-- Required
    └── tensor (TensorProto)
        └── string_val: [Serialized Experiment Proto]
    
# 1. Session Start
Summary
└── value (Summary.Value)
    ├── tag: "_hparams_/session_start_info"
    ├── metadata (SummaryMetadata)
    │   └── plugin_data (FeatureNameConfig)
    │       │── plugin_name: "hparams"  <-- Required
    │       └── content: [Serialized SessionStartInfo Proto]
    └── tensor (TensorProto)  <--

Summary.Value {
  tag: "_hparams_/session_start_info"
  metadata: {
    plugin_data: {
      plugin_name: "hparams"
      content: {
        hparams: [
          { name: "learning_rate", value: {number_value: 0.001} }
          { name: "batch_size", value: {number_value: 32} }
          { name: "optimizer", value: {string_value: "adam"} }
        ]
        group_name: "default"
        start_time_secs: 0
      }
    }
  }
}

# 2. Metrics (regular scalars)
Summary.Value { tag: "accuracy", simple_value: 0.925 }
Summary.Value { tag: "loss", simple_value: 0.234 }

# 3. Session End
Summary.Value {
  tag: "_hparams_/session_end_info"
  metadata: {
    plugin_data: {
      plugin_name: "hparams"
      content: {
        status: STATUS_SUCCESS (2)
        end_time_secs: 100
      }
    }
  }
}
 */
#pragma once

#include "flatbuf.h"
#include "encoder.h"

namespace schema {
    // TensorProto for scalar F32. Canonical encoding matching protobuf serializer:
    //   dtype=DT_FLOAT(1), tensor_shape OMITTED (empty=proto3 default),
    //   float_val uses packed encoding (wire type 2), NOT non-packed (wire type 5).
    U8V scalar_tensor(F32 v) {
        U32 bits;
        std::memcpy(&bits, &v, 4);
        U8 fb[4] = {
            static_cast<U8>( bits        & 0xFF),
            static_cast<U8>((bits >>  8) & 0xFF),
            static_cast<U8>((bits >> 16) & 0xFF),
            static_cast<U8>((bits >> 24) & 0xFF),
        };
        
        proto::Encoder tp;        // TensorProto
        tp.s32(1, 1);             // dtype = DT_FLOAT
//        tp.raw(2, {});            // tensor_shape = empty (scalar), optional
        tp.raw(5, fb, 4);         // float_val at field 5 (packed)
        
        return tp.buf();
    }

    // TensorProto for string (text)
    U8V text_tensor(const STR& txt) {
        proto::Encoder tp;        // TensorProto
        tp.s32(1, 7);             // dtype = DT_STRING (7)
//        tp.raw(2, {});            // tensor_shape = empty (scalar), optional
        tp.str(8, txt);           // string_val
        
        return tp.buf();
    }
    
    // Plugin metadata – note: metadata goes in Summary.Value field 9
    U8V scalar_meta() {
        proto::Encoder pd;        // SummaryMetadata.PluginData
        pd.str(1, "scalars");     // plugin name
        
        // content: empty = scalar_plugin{mode=DEFAULT} in proto3
        proto::Encoder meta;      // SummaryMetadata
        meta.raw(1, pd.buf());
        meta.s32(4, 1);           // data_class = DATA_CLASS_SCALAR
        
        return meta.buf();
    }

    U8V text_meta() {
        proto::Encoder pd;        // SummaryMetadata.PluginData
        pd.str(1, "text");        // plugin_name
        // empty content for text
        
        proto::Encoder meta;      // SummaryMetadata
        meta.raw(1, pd.buf());    // PluginData
        meta.s32(4, 2);           // data_class = DATA_CLASS_TENSOR
        
        return meta.buf();
    }

    U8V image_meta(S32 max_images = 1) {
        proto::Encoder pc;
        pc.s32(1, max_images);    // max_images_per_step
        
        proto::Encoder pd;
        pd.str(1, "images");      // plugin_name
        pd.raw(2, pc.buf());      // content
        
        proto::Encoder meta;
        meta.raw(1, pd.buf());
        meta.s32(4, 3);           // data_class = DATA_CLASS_BLOB_SEQUENCE
        
        return meta.buf();
    }

    U8V histo_meta() {
        proto::Encoder pd;
        pd.str(1, "histograms");  // plugin_name
        // empty content for histogram
        
        proto::Encoder meta;
        meta.raw(1, pd.buf());
//        meta.s32(4, 1);           // data_class = DATA_CLASS_SCALAR
        
        return meta.buf();
    }
} // namespace tb_schema
