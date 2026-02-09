# Complete Setup Guide - TensorBoard C++ Writer

## Quick Start (Choose Your Method)

### Method 1: Automated Build (Recommended - 30 seconds)

```bash
chmod +x build.sh
./build.sh
```

This handles everything automatically. If you get a protobuf error, run:
```bash
chmod +x fix_protobuf.sh
./fix_protobuf.sh
```

### Method 2: Manual Build (1 minute)

```bash
# Compile proto files
protoc --cpp_out=. histogram.proto
protoc --cpp_out=. summary.proto
protoc --cpp_out=. event.proto

# Choose the right source file
# Use tensorboard_writer_v2.cpp (most compatible)
g++ -std=c++14 -O2 \
    tensorboard_writer_v2.cpp \
    event.pb.cc \
    summary.pb.cc \
    histogram.pb.cc \
    -lprotobuf \
    -o tensorboard_writer

# Run
./tensorboard_writer
```

### Method 3: Docker (100% Reliable - 2 minutes)

```bash
# Build
docker build -t tensorboard-writer .

# Run
docker run -v $(pwd)/logs:/app/logs tensorboard-writer

# View results
docker run -d -p 6006:6006 -v $(pwd)/logs:/app/logs \
  tensorboard-writer tensorboard --logdir=/app/logs --host=0.0.0.0

# Open http://localhost:6006
```

## File Overview

### Core Files (Required)
- `histogram.proto` - Histogram message definition
- `summary.proto` - Summary message definition
- `event.proto` - Event message definition
- `tensorboard_writer_v2.cpp` - Main application (most compatible)

### Build Files (Choose One)
- `build.sh` - Automated build script ⭐ Recommended
- `Makefile` - Traditional make build
- `CMakeLists.txt` - CMake build system
- `Dockerfile` - Docker build

### Documentation
- `README.md` - Full documentation
- `QUICK_FIX.md` - Fast solutions for common errors ⭐
- `PROTOBUF_FIX.md` - Detailed protobuf troubleshooting
- `DEPENDENCIES.md` - Understanding the proto dependencies

### Troubleshooting Tools
- `fix_protobuf.sh` - Auto-fix protobuf version issues
- `tensorboard_writer_safe.cpp` - Version with extra checks

## Common Issues & Solutions

### Issue 1: `protoc: command not found`

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y protobuf-compiler libprotobuf-dev

# macOS
brew install protobuf

# CentOS/RHEL
sudo yum install -y protobuf-devel protobuf-compiler
```

### Issue 2: `exception in libprotobuf.so`

This is a version mismatch. **Solutions:**

**Quick fix:**
```bash
./fix_protobuf.sh
```

**Manual fix:**
```bash
rm -f *.pb.cc *.pb.h tensorboard_writer
protoc --cpp_out=. *.proto
g++ -std=c++14 -O2 tensorboard_writer_v2.cpp *.pb.cc -lprotobuf -o tensorboard_writer
./tensorboard_writer
```

**Nuclear option:**
```bash
sudo apt-get remove --purge protobuf-compiler libprotobuf-dev
sudo apt-get install -y protobuf-compiler libprotobuf-dev
rm -f *.pb.* tensorboard_writer
./build.sh
```

### Issue 3: `'VersionNumber' is not a member of 'google::protobuf::internal'`

Use `tensorboard_writer_v2.cpp` instead of `tensorboard_writer_safe.cpp`:
```bash
g++ -std=c++14 -O2 tensorboard_writer_v2.cpp *.pb.cc -lprotobuf -o tensorboard_writer
```

### Issue 4: Can't see data in TensorBoard

**Check:**
```bash
# Verify files were created
ls -lh ./logs/

# Should see: events.out.tfevents.XXXXXXXX.localhost

# Start TensorBoard
tensorboard --logdir=./logs

# Open http://localhost:6006
```

## Viewing Results

After running the writer:

```bash
# Method 1: Direct command
tensorboard --logdir=./logs
# Then open http://localhost:6006

# Method 2: Using make
make tensorboard

# Method 3: Using Docker
docker run -p 6006:6006 -v $(pwd)/logs:/app/logs tensorboard-writer \
  tensorboard --logdir=/app/logs --host=0.0.0.0
```

## Integration into Your Project

### Step 1: Copy Files

Copy these files to your project:
```
histogram.proto
summary.proto
event.proto
tensorboard_writer_v2.cpp (rename to match your project)
```

### Step 2: Extract Functions

From `tensorboard_writer_v2.cpp`, you need:
- `WriteRecord()`
- `GetEventFilename()`
- `WriteScalarSummary()`
- `WriteImageSummary()`
- `WriteHistogramSummary()`

### Step 3: Use in Your Code

```cpp
#include "event.pb.h"
#include "summary.pb.h"

int main() {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    
    std::string log_dir = "./logs";
    
    // Your training loop
    for (int epoch = 0; epoch < 100; epoch++) {
        float loss = ComputeLoss();
        WriteScalarSummary(log_dir, "train/loss", loss, epoch);
        
        // Log images every 10 epochs
        if (epoch % 10 == 0) {
            auto image = GetInputImage();
            WriteImageSummary(log_dir, "inputs/sample", 
                            image.data, image.height, image.width, 3, epoch);
        }
        
        // Log weight distributions
        auto weights = GetLayerWeights(0);
        WriteHistogramSummary(log_dir, "weights/layer0", weights, epoch);
    }
    
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
```

### Step 4: Build Your Project

Add to your build system:

**Makefile:**
```makefile
PROTO_FILES = histogram.proto summary.proto event.proto
PROTO_SRCS = $(PROTO_FILES:.proto=.pb.cc)

%.pb.cc %.pb.h: %.proto
	protoc --cpp_out=. $<

your_app: your_app.cpp $(PROTO_SRCS)
	g++ -std=c++14 -O2 $^ -lprotobuf -o $@
```

**CMakeLists.txt:**
```cmake
find_package(Protobuf REQUIRED)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS 
    histogram.proto summary.proto event.proto)
add_executable(your_app your_app.cpp ${PROTO_SRCS})
target_link_libraries(your_app ${Protobuf_LIBRARIES})
```

## Performance Tips

### 1. Batched Writing

```cpp
class EventWriter {
    std::vector<tensorflow::Event> buffer_;
    size_t max_buffer_size_ = 100;
    
    void Flush() {
        std::ofstream file(filename_, std::ios::binary | std::ios::app);
        for (const auto& event : buffer_) {
            std::string serialized;
            event.SerializeToString(&serialized);
            WriteRecord(file, serialized);
        }
        buffer_.clear();
    }
    
    ~EventWriter() { Flush(); }
};
```

### 2. Reduce Logging Frequency

```cpp
// Don't log every single step
if (step % 10 == 0) {
    WriteScalarSummary(...);
}

// Log expensive items even less frequently
if (step % 100 == 0) {
    WriteImageSummary(...);
}
```

### 3. Resize Large Images

```cpp
// Before logging, resize to reasonable size
const int MAX_DIM = 512;
if (width > MAX_DIM || height > MAX_DIM) {
    image = ResizeImage(image, MAX_DIM, MAX_DIM);
}
```

## Testing Your Setup

### Minimal Test

Create `test.cpp`:
```cpp
#include "event.pb.h"
#include <iostream>

int main() {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    tensorflow::Event event;
    event.set_step(1);
    std::cout << "✓ Protobuf working!" << std::endl;
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
```

Compile and run:
```bash
protoc --cpp_out=. event.proto
g++ test.cpp event.pb.cc -lprotobuf -o test
./test
```

If this works, your setup is correct!

## Production Checklist

- [ ] Protobuf versions match (check with `protoc --version`)
- [ ] Clean build (run `make clean` or `rm *.pb.*`)
- [ ] Proto files compiled successfully
- [ ] Application compiles without warnings
- [ ] Application runs without errors
- [ ] Event files are created in logs directory
- [ ] TensorBoard can read the files
- [ ] Data appears correctly in TensorBoard UI

## Getting Help

1. **Check QUICK_FIX.md** for common solutions
2. **Run the test above** to isolate the issue
3. **Check versions:**
   ```bash
   protoc --version
   dpkg -l | grep protobuf
   ldd tensorboard_writer | grep protobuf
   ```
4. **Try Docker** - it always works
5. **Post error logs** with full error messages

## Summary

**For immediate use:**
```bash
./build.sh
tensorboard --logdir=./logs
```

**If you hit protobuf errors:**
```bash
./fix_protobuf.sh
```

**For guaranteed success:**
```bash
docker-compose up
```

That's it! You should now have a working TensorBoard C++ event writer.
