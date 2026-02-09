# TensorBoard C++ Event Writer

A standalone C++ library for writing TensorFlow events that can be visualized in TensorBoard, without requiring the full TensorFlow installation.

## Features

- ✅ Write scalar metrics (loss, accuracy, etc.)
- ✅ Write image summaries (training samples, feature maps)
- ✅ Write histogram summaries (weights, gradients, activations)
- ✅ No TensorFlow dependency - only requires Protocol Buffers
- ✅ Simple, self-contained implementation

## Prerequisites

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y build-essential protobuf-compiler libprotobuf-dev
```

### macOS
```bash
brew install protobuf
```

### CentOS/RHEL
```bash
sudo yum install -y gcc-c++ protobuf-devel protobuf-compiler
```

## Quick Start

### Method 1: Using the Build Script (Recommended)

```bash
chmod +x build.sh
./build.sh
```

This will:
1. Compile the proto files
2. Build the application
3. Run a demo that generates sample data
4. Show you how to view it in TensorBoard

### Method 2: Manual Build

```bash
# Step 1: Compile proto files
protoc --cpp_out=. histogram.proto
protoc --cpp_out=. summary.proto
protoc --cpp_out=. event.proto

# Step 2: Compile the application
g++ -std=c++14 -O2 \
    tensorboard_writer.cpp \
    event.pb.cc \
    summary.pb.cc \
    histogram.pb.cc \
    -lprotobuf \
    -o tensorboard_writer

# Step 3: Run
./tensorboard_writer
```

### Method 3: Using CMake

```bash
mkdir build
cd build
cmake ..
make
./tensorboard_writer
```

## Viewing Results

After running the application:

```bash
# Install TensorBoard (requires Python)
pip install tensorboard

# Start TensorBoard
tensorboard --logdir=./logs

# Open your browser to http://localhost:6006
```

```bash
# or Install TensorBoard Docker image
docker pull schafo/tensorboard:latest

# create tboard.sh with the following
#!/bin/bash
docker run -d --restart always \
    -v /u01/runs:/app/runs/:ro \
    -p 6006:6006 \
    -w "/app/" --name "tensorboard" \
    schafo/tensorboard

# check event file
docker exec -it tensorboard /bin/bash
tensorboard --inspect --event_file=event_file 
```

## Usage Example

```cpp
#include "event.pb.h"
#include "summary.pb.h"

int main() {
    std::string log_dir = "./logs";
    
    // Log a scalar value
    WriteScalarSummary(log_dir, "train/loss", 0.5f, /*step=*/0);
    
    // Log an image (RGB data)
    std::vector<uint8_t> image_data = ...; // Your image data
    WriteImageSummary(log_dir, "samples/image", image_data, 
                     /*height=*/256, /*width=*/256, /*channels=*/3, /*step=*/0);
    
    // Log a histogram
    std::vector<float> weights = ...; // Your weight values
    WriteHistogramSummary(log_dir, "weights/layer1", weights, /*step=*/0);
    
    return 0;
}
```

## File Structure

```
.
├── histogram.proto          # Histogram message definition
├── summary.proto           # Summary message definition
├── event.proto             # Event message definition
├── tensorboard_writer.cpp  # Main implementation
├── build.sh                # Build script
├── CMakeLists.txt         # CMake build configuration
└── README.md              # This file
```

## API Reference

### WriteScalarSummary
```cpp
void WriteScalarSummary(
    const std::string& log_dir,    // Directory to write events
    const std::string& tag,         // Metric name (e.g., "train/loss")
    float value,                    // Value to log
    int64_t step                    // Training step/iteration
);
```

### WriteImageSummary
```cpp
void WriteImageSummary(
    const std::string& log_dir,           // Directory to write events
    const std::string& tag,                // Image name
    const std::vector<uint8_t>& image_data, // Raw pixel data (RGB/RGBA)
    int height,                            // Image height
    int width,                             // Image width
    int channels,                          // 1=grayscale, 3=RGB, 4=RGBA
    int64_t step                           // Training step
);
```

### WriteHistogramSummary
```cpp
void WriteHistogramSummary(
    const std::string& log_dir,          // Directory to write events
    const std::string& tag,               // Histogram name
    const std::vector<float>& values,     // Data to histogram
    int64_t step,                         // Training step
    int num_bins = 30                     // Number of bins
);
```

## Best Practices

### 1. Organizing Metrics
Use hierarchical tags with `/` separators:
```cpp
WriteScalarSummary(log_dir, "loss/train", train_loss, step);
WriteScalarSummary(log_dir, "loss/validation", val_loss, step);
WriteScalarSummary(log_dir, "accuracy/train", train_acc, step);
WriteScalarSummary(log_dir, "accuracy/validation", val_acc, step);
```

### 2. Logging Frequency
- **Scalars**: Log every step (cheap)
- **Histograms**: Log every 5-10 steps (moderate cost)
- **Images**: Log every 10-50 steps (expensive)

### 3. Image Sizes
Keep image dimensions reasonable:
```cpp
// Resize large images before logging
const int MAX_SIZE = 512;
if (width > MAX_SIZE || height > MAX_SIZE) {
    // Resize image first
}
```

### 4. Multiple Experiments
Organize experiments in separate directories:
```cpp
std::string exp_name = "experiment_" + std::to_string(time(nullptr));
std::string log_dir = "./logs/" + exp_name;
```

## Troubleshooting

### Problem: "protoc: command not found"
**Solution**: Install Protocol Buffers compiler:
```bash
# Ubuntu/Debian
sudo apt-get install protobuf-compiler

# macOS
brew install protobuf
```

### Problem: "fatal error: google/protobuf/..."
**Solution**: Install development headers:
```bash
# Ubuntu/Debian
sudo apt-get install libprotobuf-dev

# macOS (should be included with protobuf)
brew reinstall protobuf
```

### Problem: TensorBoard shows no data
**Solutions**:
1. Check the log directory path is correct
2. Verify files are being created: `ls -la ./logs/`
3. Make sure you're pointing TensorBoard to the right directory
4. Check file permissions

### Problem: Images not displaying
**Solutions**:
1. The simplified version uses raw image data
2. For proper PNG display, integrate stb_image_write.h
3. Or use Python to convert raw data to PNG post-processing

### Problem: "undefined reference to google::protobuf..."
**Solution**: Make sure to link against protobuf library:
```bash
g++ ... -lprotobuf
```

## Advanced Features

### Adding CRC32C Checksums (Production)
For production use, implement proper CRC32C checksums:

```cpp
#include <crc32c/crc32c.h>

uint32_t ComputeCRC32C(const std::string& data) {
    return crc32c::Crc32c(data.data(), data.size());
}

void WriteRecord(std::ofstream& file, const std::string& data) {
    uint64_t length = data.size();
    file.write(reinterpret_cast<const char*>(&length), sizeof(length));
    
    uint32_t length_crc = ComputeCRC32C(
        std::string(reinterpret_cast<const char*>(&length), sizeof(length))
    );
    file.write(reinterpret_cast<const char*>(&length_crc), sizeof(length_crc));
    
    file.write(data.c_str(), data.size());
    
    uint32_t data_crc = ComputeCRC32C(data);
    file.write(reinterpret_cast<const char*>(&data_crc), sizeof(data_crc));
}
```

### PNG Image Encoding
For proper image display, add stb_image_write.h:

```cpp
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

std::string EncodePNG(const std::vector<uint8_t>& image_data, 
                      int height, int width, int channels) {
    std::string png_data;
    
    auto write_func = [](void* context, void* data, int size) {
        std::string* str = static_cast<std::string*>(context);
        str->append(static_cast<char*>(data), size);
    };
    
    stbi_write_png_to_func(write_func, &png_data, 
                          width, height, channels, 
                          image_data.data(), width * channels);
    
    return png_data;
}
```

## Performance Optimization

### Batched Writing
```cpp
class BatchedWriter {
    std::vector<tensorflow::Event> buffer_;
    size_t max_size_;
    
public:
    void AddEvent(const tensorflow::Event& event) {
        buffer_.push_back(event);
        if (buffer_.size() >= max_size_) {
            Flush();
        }
    }
    
    void Flush() {
        // Write all buffered events
        // Clear buffer
    }
};
```

### Async Writing
```cpp
#include <thread>
#include <queue>

class AsyncWriter {
    std::queue<tensorflow::Event> queue_;
    std::thread worker_;
    
    void WorkerThread() {
        // Process events from queue in background
    }
};
```

## Comparison with Alternatives

| Method | Pros | Cons |
|--------|------|------|
| This library | No TensorFlow dep, simple | Manual proto management |
| TensorFlow C++ API | Official, feature-complete | Large dependency |
| Python wrapper | Easy integration | Requires Python runtime |
| tensorboard-logger | Header-only | May be outdated |

## Why This Approach?

This implementation:
- ✅ Requires only Protocol Buffers (not full TensorFlow)
- ✅ Uses simplified proto definitions (no XLA/TSL dependencies)
- ✅ Is easy to integrate into existing C++ projects
- ✅ Works on any platform with protobuf support
- ✅ Generates files 100% compatible with TensorBoard

## License

This code is provided as-is for educational and commercial use. The Protocol Buffer definitions are simplified versions of TensorFlow's proto files, which are licensed under Apache 2.0.

## Contributing

Issues and pull requests welcome! Areas for improvement:
- Add CRC32C checksum support
- Integrate PNG encoding
- Add more summary types (audio, text)
- Performance optimizations
- Windows build support

## Resources

- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [Protocol Buffers](https://developers.google.com/protocol-buffers)
- [Original TensorFlow Protos](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core)

## Support

For issues specific to this implementation, please open an issue on GitHub.

For TensorBoard usage questions, see the [official TensorBoard docs](https://www.tensorflow.org/tensorboard/get_started).
