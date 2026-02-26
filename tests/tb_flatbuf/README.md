# TensorBoard FlatBuffers Event Writer (C++)

A self-contained C++ library that writes TensorBoard `.tfevents` binary files
with **scalar**, **image**, and **histogram** summaries, using **FlatBuffers** for
plugin metadata encoding.

---

## Architecture Overview

```
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
```
### HParams Session Structure
```
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
```
---

## File Structure

```
tb_flatbuffers/
├── include/
│   ├── flatbuf.h              # Minimal FlatBuffers builder
│   ├── crc32c.h               # CRC32C + masked CRC for TFRecord framing
│   ├── encode.h               # Minimal protobuf wire-format encoder
│   ├── schema.h               # FlatBuffers encoders for TB plugin data
│   ├── hparam.h               # HParams tab handler
│   └── writer.h               # Main EventWriter class
├── src/
│   └── main.cpp               # Demo application
├── schemas/
│   └── tb_plugin_data.fbs     # FlatBuffers schema (for documentation/flatc)
├── Makefile
└── README.md
```

---

## Fixes for TensorBoard 2.x Compatibility

Three issues were corrected from the initial version:

**Fix 1 — `step` field missing from events.**  
Every `tensorflow.Event` must include the `step` field (proto field 2), even at step 0. Without it TensorBoard 2.x cannot build the timeline and silently ignores the events.

**Fix 2 — Wrong plugin data encoding.**  
`SummaryMetadata.PluginData.content` must be **proto3-encoded**, not FlatBuffers bytes. FlatBuffers defines the *schema* for plugin metadata (see `schemas/tb_plugin_data.fbs` and `include/tb_flatbuffers_schema.h`), but the actual wire bytes stored in the `content` field are proto3:
- `"scalars"` → `ScalarPluginData { mode: int32 }` — empty bytes when `mode=DEFAULT=0`
- `"images"` → `ImagePluginData { max_images_per_step: int32 }` — e.g. `0x08 0x01`
- `"histograms"` → `HistogramPluginData {}` — empty bytes

**Fix 3 — Incorrect event filename.**  
TensorBoard 2.x requires the full filename format:
```
events.out.tfevents.{unix_timestamp}.{hostname}.{pid}.{sequence}
```
The previous code used `events.out.tfevents.{timestamp}.demo` which TensorBoard's file discovery rejects.

---

```bash
make         # builds ./tb_demo
make run     # builds and runs
make clean   # removes binary
```

**Requirements:** `g++` with C++17, `make`. No external dependencies.

---

## Running

```bash
./tb_demo [logdir]
# Default logdir: /tmp/tb_flatbuffers_demo
```

This writes three runs under `logdir/`:

| Run         | Tags                                                      | Steps |
|-------------|-----------------------------------------------------------|-------|
| `scalars`   | `train/loss`, `train/accuracy`, `train/lr`, `val/*`       | 100   |
| `images`    | `images/sine_wave`, `images/gradient_heatmap`, `images/checkerboard` | 10 |
| `histograms`| `weights/layer1`, `weights/layer2`, `activations/relu`, `gradients/layer1` | 50 |

---

## Visualizing

```bash
pip install tensorboard
tensorboard --logdir /tmp/tb_flatbuffers_demo
# Open http://localhost:6006
```

---

## FlatBuffers Role

TensorBoard uses FlatBuffers to encode the `PluginData.content` bytes inside
`SummaryMetadata`. Each plugin (scalars, images, histograms) has its own schema
defined in `tensorboard/plugins/*/metadata.py` in the TensorBoard source.

The FlatBuffer tells TensorBoard:
- **Scalars**: the display mode (linear / log scale)
- **Images**: the maximum number of images to show per step
- **Histograms**: (currently no extra config)

### FlatBuffers Binary Layout

For `ScalarPluginData { mode: 0 }`:

```
Offset  Bytes   Meaning
──────  ─────   ───────
0-3     04 00 00 00   root_offset = 4 (table is at byte 4)
4-7     08 00 00 00   soffset = 8 (vtable is 8 bytes forward from here)
8       00            mode = 0 (DEFAULT)
9-11    00 00 00      padding (align vtable to 2 bytes)
12-13   08 00         vtable_size = 8
14-15   08 00         object_size = 8
16-17   04 00         field[0] offset = 4 (mode is at object+4)
18-19   00 00         field[1] = 0 (not present)
```

---

## TFRecord Format

Each record in the `.tfevents` file:

```
┌──────────────────────────────┐
│ length      (uint64, LE)     │  ← byte count of data
│ masked_crc32_of_length (u32) │  ← CRC32C of length bytes, masked
├──────────────────────────────┤
│ data        (bytes)          │  ← serialized tensorflow.Event proto
│ masked_crc32_of_data   (u32) │  ← CRC32C of data, masked
└──────────────────────────────┘
```

Masking: `masked = ((crc >> 15 | crc << 17) + 0xa282ead8)`

---

## PNG Encoding

Image summaries encode pixels as uncompressed PNG using **deflate stored blocks**
(BTYPE=00), which produces valid PNG without requiring zlib compression.

---

## License

MIT — see the source files for details.
