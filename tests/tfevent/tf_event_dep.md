# Understanding the TensorFlow Proto Dependencies Issue

## The Problem

When trying to use TensorFlow's official proto files directly, you may encounter errors like:

```
tensorflow/core/framework/summary.proto: Import "xla/tsl/protobuf/histogram.proto" was not found or had errors.
```

This happens because TensorFlow's proto files have deep dependency chains:

```
event.proto
  └─ summary.proto
       ├─ xla/tsl/protobuf/histogram.proto
       ├─ tensorflow/core/framework/tensor.proto
       │    ├─ tensorflow/core/framework/types.proto
       │    ├─ tensorflow/core/framework/resource_handle.proto
       │    └─ tensorflow/core/framework/tensor_shape.proto
       └─ And many more...
```

## Why This Happens

TensorFlow's codebase is massive and modular. The proto files are organized across multiple directories:
- `tensorflow/core/framework/`
- `xla/tsl/protobuf/`
- `tensorflow/core/protobuf/`
- And many others

Each import path expects a specific directory structure that mirrors TensorFlow's repository layout.

## Solutions Overview

### Solution 1: Use Simplified Protos (Recommended for Standalone Projects)

**Pros:**
- No TensorFlow dependency
- Self-contained and portable
- Easy to integrate
- Small codebase

**Cons:**
- Must maintain proto definitions
- Missing some advanced features

**Use when:**
- Building standalone applications
- Don't need full TensorFlow integration
- Want minimal dependencies
- Cross-platform portability is important

### Solution 2: Use Full TensorFlow Source

**Pros:**
- Official proto definitions
- All features available
- Always up-to-date

**Cons:**
- Large dependency (~3GB source)
- Complex build setup
- Many transitive dependencies

**Use when:**
- Already using TensorFlow
- Need advanced features
- Contributing to TensorFlow ecosystem

### Solution 3: Use Pre-compiled TensorFlow

**Pros:**
- Official binaries
- Don't need to compile TensorFlow

**Cons:**
- Large installation
- Version compatibility issues

**Use when:**
- Using TensorFlow C++ API
- Platform has official builds

## Our Solution: Simplified Protos

We provide simplified versions of the key proto files that:

1. **Remove unnecessary dependencies**: Only include what's needed for basic logging
2. **Flatten the structure**: All protos in one directory
3. **Maintain compatibility**: Generate events TensorBoard can read
4. **Keep it simple**: Easy to understand and modify

### What's Included

#### histogram.proto
```protobuf
message HistogramProto {
  double min = 1;
  double max = 2;
  int64 num = 3;
  double sum = 4;
  double sum_squares = 5;
  repeated double bucket_limit = 6;
  repeated double bucket = 7;
}
```

This is **identical** to TensorFlow's version, just without the XLA/TSL import path.

#### summary.proto
```protobuf
message Summary {
  message Image { ... }
  message Audio { ... }
  message Value {
    string tag = 1;
    oneof value {
      float simple_value = 2;
      Image image = 4;
      HistogramProto histo = 5;
      // ... other types
    }
  }
  repeated Value value = 1;
}
```

Simplified to include only commonly used summary types.

#### event.proto
```protobuf
message Event {
  double wall_time = 1;
  int64 step = 2;
  oneof what {
    Summary summary = 5;
    // ... other types
  }
}
```

Core event structure, compatible with TensorBoard.

### Compatibility

The simplified protos generate events that are 100% compatible with TensorBoard because:

1. **Same field numbers**: Protocol Buffers are backward compatible by field number
2. **Same types**: We use the same data types (double, int64, string, etc.)
3. **Same structure**: The hierarchy and nesting are preserved
4. **Same binary format**: Protobuf serialization is deterministic

## Detailed Comparison

### Official TensorFlow Approach

```bash
# Clone entire TensorFlow repo (3+ GB)
git clone https://github.com/tensorflow/tensorflow.git

# Set up complex build environment
cd tensorflow
./configure

# Compile specific protos (with all dependencies)
bazel build //tensorflow/core:protos_all

# Include in your project
# - Add 20+ include paths
# - Link against multiple libraries
# - Deal with version conflicts
```

### Simplified Approach

```bash
# Copy 3 proto files (< 1 KB total)
cp histogram.proto summary.proto event.proto ./

# Compile protos
protoc --cpp_out=. *.proto

# Build your app
g++ your_app.cpp *.pb.cc -lprotobuf
```

## Feature Comparison

| Feature | Simplified | Full TensorFlow |
|---------|-----------|-----------------|
| Scalar logging | ✅ | ✅ |
| Image logging | ✅ | ✅ |
| Histogram logging | ✅ | ✅ |
| Audio logging | ✅ | ✅ |
| Text logging | ✅ | ✅ |
| Graph visualization | ❌ | ✅ |
| Profiler data | ❌ | ✅ |
| XLA HLO graphs | ❌ | ✅ |
| Custom plugins | ❌ | ✅ |

## Migration Guide

### From Full TensorFlow to Simplified

If you're currently using full TensorFlow protos:

1. **Replace includes:**
```cpp
// Before
#include "tensorflow/core/framework/summary.pb.h"
#include "xla/tsl/protobuf/histogram.pb.h"

// After
#include "summary.pb.h"
#include "histogram.pb.h"
```

2. **Update namespace (if needed):**
```cpp
// Both use tensorflow namespace
tensorflow::Summary summary;
tensorflow::HistogramProto hist;
```

3. **Rebuild:**
```bash
# Remove old proto objects
rm *.pb.cc *.pb.h

# Compile simplified protos
protoc --cpp_out=. histogram.proto summary.proto event.proto

# Rebuild your app
g++ your_app.cpp *.pb.cc -lprotobuf
```

### From Simplified to Full TensorFlow

If you need advanced features:

1. **Clone TensorFlow:**
```bash
git clone --depth 1 https://github.com/tensorflow/tensorflow.git
```

2. **Update proto paths:**
```cpp
// Update includes
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/util/event.pb.h"
```

3. **Set up build with Bazel:**
```python
# BUILD file
cc_binary(
    name = "your_app",
    srcs = ["your_app.cpp"],
    deps = [
        "//tensorflow/core:protos_all",
    ],
)
```

## When to Use Which Approach

### Use Simplified Protos When:
- ✅ Building standalone C++ applications
- ✅ Logging basic metrics (scalars, images, histograms)
- ✅ Want minimal dependencies
- ✅ Need cross-platform portability
- ✅ Don't have TensorFlow installed
- ✅ Prototyping or learning

### Use Full TensorFlow Protos When:
- ✅ Already using TensorFlow C++ API
- ✅ Need graph visualization
- ✅ Using TensorFlow profiler
- ✅ Need XLA/HLO integration
- ✅ Contributing to TensorFlow
- ✅ Using custom TensorBoard plugins

## Frequently Asked Questions

### Q: Will TensorBoard read files created with simplified protos?
**A:** Yes! The binary format is identical for the fields we use.

### Q: Can I mix files created with both approaches?
**A:** Yes, TensorBoard can read both in the same log directory.

### Q: What if I need a feature not in the simplified protos?
**A:** You can add it! Just copy the relevant message definition from TensorFlow's protos.

### Q: Are there performance differences?
**A:** No, the generated C++ code is essentially identical.

### Q: How do I update the simplified protos?
**A:** Check TensorFlow's proto files for changes and update field numbers/types accordingly.

### Q: Is this approach officially supported?
**A:** This is a community approach. For official support, use TensorFlow's protos directly.

## Example: Adding a New Summary Type

If you need a summary type not in the simplified version:

1. **Find it in TensorFlow's source:**
```bash
# Search TensorFlow repo
grep -r "message.*Summary" tensorflow/core/framework/
```

2. **Copy the message definition:**
```protobuf
// Add to summary.proto
message Summary {
  // ... existing messages ...
  
  message Metadata {
    PluginData plugin_data = 1;
    string display_name = 2;
    string summary_description = 3;
  }
}
```

3. **Recompile:**
```bash
protoc --cpp_out=. summary.proto
```

4. **Use in your code:**
```cpp
tensorflow::Summary::Metadata* metadata = 
    summary_value->mutable_metadata();
// ... set fields ...
```

## Conclusion

The simplified proto approach provides a practical middle ground:
- **Simple enough** for standalone projects
- **Compatible enough** with TensorBoard
- **Extensible enough** for most use cases
- **Maintainable enough** for production use

For 90% of TensorBoard logging use cases, the simplified protos are sufficient and far easier to work with than the full TensorFlow dependency tree.
