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
                │       ├── plugin_name: "scalars" | "images" | "histograms"
                │       └── content: bytes  ◄── FlatBuffers encoded!
                │           ├── ScalarPluginData    { mode: int8 }
                │           ├── ImagePluginData     { max_images: int32 }
                │           └── HistogramPluginData { (empty) }
                └── payload (one of):
                    ├── simple_value: float   (scalar)
                    ├── image: Summary.Image  (PNG bytes)
                    └── histo: HistogramProto (bucket edges + counts)
```

---

## File Structure

```
tb_flatbuffers/
├── include/
│   ├── flatbuffers/
│   │   └── flatbuffers.h          # Minimal FlatBuffers builder
│   ├── crc32c.h                   # CRC32C + masked CRC for TFRecord framing
│   ├── proto_encode.h             # Minimal protobuf wire-format encoder
│   ├── tb_flatbuffers_schema.h    # FlatBuffers encoders for TB plugin data
│   └── tensorboard_writer.h      # Main EventWriter class
├── src/
│   └── main.cpp                  # Demo application
├── schemas/
│   └── tb_plugin_data.fbs        # FlatBuffers schema (for documentation/flatc)
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
