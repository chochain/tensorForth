# Writing TensorFlow Events in C++ for TensorBoard Visualization

## Introduction

TensorBoard is TensorFlow's visualization toolkit that allows you to track and visualize metrics during machine learning experiments. While most developers use TensorBoard through Python's TensorFlow API, there are scenarios where writing events directly from C++ is necessary or preferred:

- High-performance training loops implemented in C++
- Custom C++ frameworks or game engines with ML components
- Embedded systems or real-time applications
- Integration with existing C++ codebases

This guide demonstrates how to write TensorFlow event files from C++ that TensorBoard can read and visualize, covering scalars, images, and histograms.

## Understanding TensorBoard's File Format

TensorBoard reads event files with a specific binary format:

### File Naming Convention
```
events.out.tfevents.[timestamp].[hostname]
```

### Record Format
Each record in the file follows this structure:
1. **Length** (8 bytes): uint64 length of the data
2. **Length CRC** (4 bytes): CRC32C checksum of the length
3. **Data** (variable): Serialized protobuf message
4. **Data CRC** (4 bytes): CRC32C checksum of the data

### Protocol Buffers
TensorBoard uses Google Protocol Buffers for data serialization. The main proto definitions you need are:
- `tensorflow/core/util/event.proto` - Defines the Event message
- `tensorflow/core/framework/summary.proto` - Defines Summary and histogram structures

## Prerequisites

### Required Libraries
- **Protocol Buffers**: For serialization
- **TensorFlow proto files**: Event and summary definitions
- **stb_image_write.h** (optional): For PNG encoding of images

### Setting Up Proto Files

First, obtain the TensorFlow proto files and compile them:

```bash
# Download TensorFlow source or just the proto files
git clone --depth 1 https://github.com/tensorflow/tensorflow.git

# Compile the necessary proto files
protoc --cpp_out=. tensorflow/core/util/event.proto
protoc --cpp_out=. tensorflow/core/framework/summary.proto
protoc --cpp_out=. tensorflow/core/framework/tensor.proto
protoc --cpp_out=. tensorflow/core/framework/types.proto
protoc --cpp_out=. tensorflow/core/framework/resource_handle.proto
protoc --cpp_out=. tensorflow/core/framework/tensor_shape.proto
```

This generates `.pb.h` and `.pb.cc` files that you'll include in your project.

## Core Components

### 1. Helper Functions

Every implementation needs these basic utilities:

```cpp
#include <fstream>
#include <string>
#include <ctime>
#include <cstdint>
#include "tensorflow/core/util/event.pb.h"
#include "tensorflow/core/framework/summary.pb.h"

// Get current timestamp in microseconds
int64_t GetCurrentTimeMicros() {
    return static_cast<int64_t>(time(nullptr)) * 1000000;
}

// Write a length-delimited record
void WriteRecord(std::ofstream& file, const std::string& data) {
    // Write length as uint64_t
    uint64_t length = data.size();
    file.write(reinterpret_cast<const char*>(&length), sizeof(length));
    
    // Write CRC of length (use proper CRC32C in production)
    uint32_t length_crc = 0;  // Placeholder
    file.write(reinterpret_cast<const char*>(&length_crc), sizeof(length_crc));
    
    // Write data
    file.write(data.c_str(), data.size());
    
    // Write CRC of data (use proper CRC32C in production)
    uint32_t data_crc = 0;  // Placeholder
    file.write(reinterpret_cast<const char*>(&data_crc), sizeof(data_crc));
}
```

**Note on CRC32C**: For production use, implement proper CRC32C checksums. TensorBoard can work without them, but they provide data integrity verification. Libraries like `crc32c` or hardware-accelerated implementations are recommended.

### 2. Event Writer Base Class (Optional)

For cleaner code organization, you can create a base event writer class:

```cpp
class EventWriter {
private:
    std::string log_dir_;
    std::string filename_;
    
public:
    EventWriter(const std::string& log_dir) : log_dir_(log_dir) {
        filename_ = log_dir + "/events.out.tfevents." + 
                   std::to_string(time(nullptr)) + ".hostname";
    }
    
    void WriteEvent(const tensorflow::Event& event) {
        std::ofstream file(filename_, std::ios::binary | std::ios::app);
        std::string serialized;
        event.SerializeToString(&serialized);
        WriteRecord(file, serialized);
        file.close();
    }
};
```

## Writing Scalar Summaries

Scalar summaries are the most common type of metric logged to TensorBoard, tracking values like loss, accuracy, learning rate, etc.

### Implementation

```cpp
void WriteScalarSummary(const std::string& log_dir, 
                        const std::string& tag,
                        float value,
                        int64_t step) {
    // Create event file name
    std::string filename = log_dir + "/events.out.tfevents." + 
                          std::to_string(time(nullptr)) + ".hostname";
    
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    
    // Create the event
    tensorflow::Event event;
    event.set_wall_time(GetCurrentTimeMicros() / 1e6);
    event.set_step(step);
    
    // Create summary
    tensorflow::Summary* summary = event.mutable_summary();
    tensorflow::Summary::Value* summary_value = summary->add_value();
    summary_value->set_tag(tag);
    summary_value->set_simple_value(value);
    
    // Serialize and write
    std::string serialized;
    event.SerializeToString(&serialized);
    WriteRecord(file, serialized);
    
    file.close();
}
```

### Usage Example

```cpp
int main() {
    std::string log_dir = "./logs/experiment_1";
    
    // Simulate training loop
    for (int epoch = 0; epoch < 100; epoch++) {
        // Simulated metrics
        float train_loss = 2.0f * exp(-0.05f * epoch) + 0.1f;
        float train_acc = 1.0f - train_loss / 2.0f;
        float val_loss = train_loss * 1.1f;
        float val_acc = train_acc * 0.95f;
        float learning_rate = 0.001f * pow(0.95f, epoch / 10);
        
        // Log metrics
        WriteScalarSummary(log_dir, "loss/train", train_loss, epoch);
        WriteScalarSummary(log_dir, "loss/validation", val_loss, epoch);
        WriteScalarSummary(log_dir, "accuracy/train", train_acc, epoch);
        WriteScalarSummary(log_dir, "accuracy/validation", val_acc, epoch);
        WriteScalarSummary(log_dir, "hyperparameters/learning_rate", learning_rate, epoch);
    }
    
    return 0;
}
```

### Best Practices for Scalar Logging

1. **Use hierarchical tags**: Organize metrics with `/` separators (e.g., `train/loss`, `val/loss`)
2. **Log regularly**: Write at consistent intervals for smooth curves
3. **Multiple metrics**: Log related metrics together for easy comparison
4. **Naming conventions**: Use clear, descriptive names

## Writing Image Summaries

Image summaries allow you to visualize inputs, outputs, feature maps, and other visual data during training.

### Basic Implementation

```cpp
void WriteImageSummary(const std::string& log_dir,
                       const std::string& tag,
                       const std::vector<uint8_t>& image_data,
                       int height,
                       int width,
                       int channels,
                       int64_t step) {
    
    std::string filename = log_dir + "/events.out.tfevents." + 
                          std::to_string(time(nullptr)) + ".hostname";
    
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    
    // Create the event
    tensorflow::Event event;
    event.set_wall_time(GetCurrentTimeMicros() / 1e6);
    event.set_step(step);
    
    // Create summary
    tensorflow::Summary* summary = event.mutable_summary();
    tensorflow::Summary::Value* summary_value = summary->add_value();
    summary_value->set_tag(tag);
    
    // Create image proto
    tensorflow::Summary::Image* image = summary_value->mutable_image();
    image->set_height(height);
    image->set_width(width);
    image->set_colorspace(channels);  // 1=grayscale, 3=RGB, 4=RGBA
    
    // For now, store raw data (better: encode as PNG)
    std::string raw_data(image_data.begin(), image_data.end());
    image->set_encoded_image_string(raw_data);
    
    // Serialize and write
    std::string serialized;
    event.SerializeToString(&serialized);
    WriteRecord(file, serialized);
    
    file.close();
}
```

### PNG Encoding with stb_image_write

For proper image display in TensorBoard, encode images as PNG:

```cpp
// Download stb_image_write.h from https://github.com/nothings/stb
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

std::string EncodePNG(const std::vector<uint8_t>& image_data, 
                      int height, int width, int channels) {
    std::string png_data;
    
    // Callback to write to string instead of file
    auto write_func = [](void* context, void* data, int size) {
        std::string* str = static_cast<std::string*>(context);
        str->append(static_cast<char*>(data), size);
    };
    
    stbi_write_png_to_func(write_func, &png_data, 
                          width, height, channels, 
                          image_data.data(), width * channels);
    
    return png_data;
}

// Updated image writer with PNG encoding
void WriteImageSummaryPNG(const std::string& log_dir,
                          const std::string& tag,
                          const std::vector<uint8_t>& image_data,
                          int height,
                          int width,
                          int channels,
                          int64_t step) {
    
    std::string filename = log_dir + "/events.out.tfevents." + 
                          std::to_string(time(nullptr)) + ".hostname";
    
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    
    tensorflow::Event event;
    event.set_wall_time(GetCurrentTimeMicros() / 1e6);
    event.set_step(step);
    
    tensorflow::Summary* summary = event.mutable_summary();
    tensorflow::Summary::Value* summary_value = summary->add_value();
    summary_value->set_tag(tag);
    
    tensorflow::Summary::Image* image = summary_value->mutable_image();
    image->set_height(height);
    image->set_width(width);
    image->set_colorspace(channels);
    
    // Encode as PNG
    std::string encoded_image = EncodePNG(image_data, height, width, channels);
    image->set_encoded_image_string(encoded_image);
    
    std::string serialized;
    event.SerializeToString(&serialized);
    WriteRecord(file, serialized);
    
    file.close();
}
```

### Usage Examples

```cpp
// Example 1: Log input images
void LogInputBatch(const std::string& log_dir, 
                   const std::vector<std::vector<uint8_t>>& batch,
                   int step) {
    // Log first few images from batch
    for (int i = 0; i < std::min(4, (int)batch.size()); i++) {
        std::string tag = "inputs/image_" + std::to_string(i);
        WriteImageSummaryPNG(log_dir, tag, batch[i], 224, 224, 3, step);
    }
}

// Example 2: Visualize feature maps
std::vector<uint8_t> FeatureMapToImage(const std::vector<float>& feature_map,
                                       int height, int width) {
    std::vector<uint8_t> image(height * width * 3);
    
    // Find min/max for normalization
    float min_val = *std::min_element(feature_map.begin(), feature_map.end());
    float max_val = *std::max_element(feature_map.begin(), feature_map.end());
    float range = max_val - min_val;
    
    for (int i = 0; i < height * width; i++) {
        // Normalize to 0-255
        uint8_t val = static_cast<uint8_t>(
            ((feature_map[i] - min_val) / range) * 255
        );
        image[i * 3 + 0] = val;  // R
        image[i * 3 + 1] = val;  // G
        image[i * 3 + 2] = val;  // B
    }
    
    return image;
}

// Example 3: Create visualization grid
std::vector<uint8_t> CreateGrid(const std::vector<std::vector<uint8_t>>& images,
                               int img_h, int img_w, int channels,
                               int grid_rows, int grid_cols) {
    int grid_h = img_h * grid_rows;
    int grid_w = img_w * grid_cols;
    std::vector<uint8_t> grid(grid_h * grid_w * channels, 255);
    
    for (int r = 0; r < grid_rows; r++) {
        for (int c = 0; c < grid_cols; c++) {
            int img_idx = r * grid_cols + c;
            if (img_idx >= images.size()) continue;
            
            // Copy image into grid
            for (int y = 0; y < img_h; y++) {
                for (int x = 0; x < img_w; x++) {
                    int src_idx = (y * img_w + x) * channels;
                    int dst_y = r * img_h + y;
                    int dst_x = c * img_w + x;
                    int dst_idx = (dst_y * grid_w + dst_x) * channels;
                    
                    for (int ch = 0; ch < channels; ch++) {
                        grid[dst_idx + ch] = images[img_idx][src_idx + ch];
                    }
                }
            }
        }
    }
    
    return grid;
}
```

### Image Logging Best Practices

1. **Log sparingly**: Images take up space; log every N steps
2. **Resize large images**: Keep dimensions reasonable (e.g., 224x224)
3. **Use grids**: Combine multiple images into one for comparison
4. **Normalize values**: Ensure proper visualization of feature maps
5. **Limit quantity**: Log only a few representative images

## Writing Histogram Summaries

Histograms visualize the distribution of values, essential for monitoring weights, gradients, and activations during training.

### Implementation

```cpp
#include <algorithm>
#include <cmath>

void WriteHistogramSummary(const std::string& log_dir,
                          const std::string& tag,
                          const std::vector<float>& values,
                          int64_t step,
                          int num_bins = 30) {
    
    if (values.empty()) return;
    
    std::string filename = log_dir + "/events.out.tfevents." + 
                          std::to_string(time(nullptr)) + ".hostname";
    
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    
    // Create the event
    tensorflow::Event event;
    event.set_wall_time(GetCurrentTimeMicros() / 1e6);
    event.set_step(step);
    
    // Create summary
    tensorflow::Summary* summary = event.mutable_summary();
    tensorflow::Summary::Value* summary_value = summary->add_value();
    summary_value->set_tag(tag);
    
    // Create histogram proto
    tensorflow::HistogramProto* hist = summary_value->mutable_histo();
    
    // Calculate statistics
    std::vector<float> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    
    float min_val = sorted_values.front();
    float max_val = sorted_values.back();
    float sum = 0.0f;
    float sum_squares = 0.0f;
    
    for (float val : values) {
        sum += val;
        sum_squares += val * val;
    }
    
    hist->set_min(min_val);
    hist->set_max(max_val);
    hist->set_num(values.size());
    hist->set_sum(sum);
    hist->set_sum_squares(sum_squares);
    
    // Create bins
    if (max_val > min_val) {
        float bin_width = (max_val - min_val) / num_bins;
        std::vector<int> bin_counts(num_bins, 0);
        
        // Count values in each bin
        for (float val : values) {
            int bin_idx = static_cast<int>((val - min_val) / bin_width);
            if (bin_idx >= num_bins) bin_idx = num_bins - 1;
            bin_counts[bin_idx]++;
        }
        
        // Add bins to histogram
        for (int i = 0; i < num_bins; i++) {
            float edge = min_val + i * bin_width;
            hist->add_bucket_limit(edge);
            hist->add_bucket(bin_counts[i]);
        }
        
        // Add final edge
        hist->add_bucket_limit(max_val);
    } else {
        // All values are the same
        hist->add_bucket_limit(min_val);
        hist->add_bucket(values.size());
        hist->add_bucket_limit(max_val);
    }
    
    // Serialize and write
    std::string serialized;
    event.SerializeToString(&serialized);
    WriteRecord(file, serialized);
    
    file.close();
}
```

### Advanced: Custom Bin Edges

For more control over histogram bins:

```cpp
void WriteHistogramCustomBins(const std::string& log_dir,
                             const std::string& tag,
                             const std::vector<float>& values,
                             const std::vector<float>& bin_edges,
                             int64_t step) {
    
    std::string filename = log_dir + "/events.out.tfevents." + 
                          std::to_string(time(nullptr)) + ".hostname";
    
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    
    tensorflow::Event event;
    event.set_wall_time(GetCurrentTimeMicros() / 1e6);
    event.set_step(step);
    
    tensorflow::Summary* summary = event.mutable_summary();
    tensorflow::Summary::Value* summary_value = summary->add_value();
    summary_value->set_tag(tag);
    
    tensorflow::HistogramProto* hist = summary_value->mutable_histo();
    
    // Calculate statistics
    float min_val = *std::min_element(values.begin(), values.end());
    float max_val = *std::max_element(values.begin(), values.end());
    float sum = 0.0f;
    float sum_squares = 0.0f;
    
    for (float val : values) {
        sum += val;
        sum_squares += val * val;
    }
    
    hist->set_min(min_val);
    hist->set_max(max_val);
    hist->set_num(values.size());
    hist->set_sum(sum);
    hist->set_sum_squares(sum_squares);
    
    // Count values in custom bins
    std::vector<int> bin_counts(bin_edges.size() - 1, 0);
    
    for (float val : values) {
        for (size_t i = 0; i < bin_edges.size() - 1; i++) {
            if (val >= bin_edges[i] && val < bin_edges[i + 1]) {
                bin_counts[i]++;
                break;
            }
            // Handle last bin edge inclusively
            if (i == bin_edges.size() - 2 && val == bin_edges[i + 1]) {
                bin_counts[i]++;
                break;
            }
        }
    }
    
    // Add bins
    for (size_t i = 0; i < bin_edges.size(); i++) {
        hist->add_bucket_limit(bin_edges[i]);
        if (i < bin_counts.size()) {
            hist->add_bucket(bin_counts[i]);
        }
    }
    
    std::string serialized;
    event.SerializeToString(&serialized);
    WriteRecord(file, serialized);
    
    file.close();
}
```

### Usage Examples

```cpp
// Example 1: Monitor weight distributions
void LogLayerWeights(const std::string& log_dir,
                    const std::vector<std::vector<float>>& layer_weights,
                    int step) {
    for (size_t i = 0; i < layer_weights.size(); i++) {
        std::string tag = "weights/layer_" + std::to_string(i);
        WriteHistogramSummary(log_dir, tag, layer_weights[i], step);
    }
}

// Example 2: Monitor gradients
void LogGradients(const std::string& log_dir,
                 const std::vector<std::vector<float>>& gradients,
                 int step) {
    for (size_t i = 0; i < gradients.size(); i++) {
        std::string tag = "gradients/layer_" + std::to_string(i);
        WriteHistogramSummary(log_dir, tag, gradients[i], step, 50);
    }
}

// Example 3: Monitor activations with custom bins
void LogActivations(const std::string& log_dir,
                   const std::vector<float>& activations,
                   const std::string& layer_name,
                   int step) {
    // Use logarithmic bins for activations
    std::vector<float> bin_edges = {0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0};
    std::string tag = "activations/" + layer_name;
    WriteHistogramCustomBins(log_dir, tag, activations, bin_edges, step);
}

// Example 4: Detect gradient problems
void AnalyzeGradients(const std::vector<float>& gradients) {
    float sum_abs = 0.0f;
    int near_zero = 0;
    
    for (float grad : gradients) {
        sum_abs += std::abs(grad);
        if (std::abs(grad) < 1e-7) near_zero++;
    }
    
    float mean_abs = sum_abs / gradients.size();
    float zero_ratio = static_cast<float>(near_zero) / gradients.size();
    
    if (mean_abs < 1e-5) {
        std::cout << "Warning: Vanishing gradients detected!" << std::endl;
    }
    if (mean_abs > 1.0) {
        std::cout << "Warning: Exploding gradients detected!" << std::endl;
    }
    if (zero_ratio > 0.5) {
        std::cout << "Warning: Many dead neurons detected!" << std::endl;
    }
}
```

### Histogram Best Practices

1. **Monitor all layers**: Track weights, biases, gradients for each layer
2. **Watch for problems**: Vanishing/exploding gradients, dead neurons
3. **Appropriate bins**: 30-50 bins for most cases, custom for special distributions
4. **Log frequency**: Every few steps is sufficient (histograms are expensive)
5. **Compare distributions**: Use consistent binning across layers for comparison

## Complete Example: Training Loop with All Metrics

Here's a complete example integrating all three types of summaries:

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// Assume all previous helper functions are included

class SimpleNN {
public:
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<float>> gradients;
    std::vector<std::vector<float>> activations;
    
    SimpleNN(int num_layers, int layer_size) {
        for (int i = 0; i < num_layers; i++) {
            weights.push_back(std::vector<float>(layer_size));
            gradients.push_back(std::vector<float>(layer_size));
            activations.push_back(std::vector<float>(layer_size));
            
            // Initialize with random values
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dist(0.0f, 0.1f);
            
            for (int j = 0; j < layer_size; j++) {
                weights[i][j] = dist(gen);
            }
        }
    }
    
    void SimulateForwardPass() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& layer : activations) {
            for (auto& val : layer) {
                val = std::max(0.0f, dist(gen));  // ReLU-like
            }
        }
    }
    
    void SimulateBackwardPass(float learning_rate) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.01f);
        
        for (size_t i = 0; i < weights.size(); i++) {
            for (size_t j = 0; j < weights[i].size(); j++) {
                gradients[i][j] = dist(gen);
                weights[i][j] -= learning_rate * gradients[i][j];
            }
        }
    }
};

std::vector<uint8_t> GenerateSampleImage(int size, int step) {
    std::vector<uint8_t> image(size * size * 3);
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int idx = (y * size + x) * 3;
            // Create a pattern that changes with step
            image[idx + 0] = (x * 255 / size + step * 5) % 256;
            image[idx + 1] = (y * 255 / size) % 256;
            image[idx + 2] = ((x + y) * 255 / (2 * size)) % 256;
        }
    }
    
    return image;
}

int main() {
    std::string log_dir = "./logs/training_run";
    
    // Create a simple neural network
    SimpleNN model(5, 1000);  // 5 layers, 1000 neurons each
    
    // Training hyperparameters
    float initial_lr = 0.001f;
    int num_epochs = 200;
    
    std::cout << "Starting training..." << std::endl;
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Simulate forward and backward pass
        model.SimulateForwardPass();
        
        float learning_rate = initial_lr * std::pow(0.95f, epoch / 10);
        model.SimulateBackwardPass(learning_rate);
        
        // Calculate simulated metrics
        float train_loss = 2.0f * std::exp(-0.03f * epoch) + 0.1f;
        float val_loss = train_loss * 1.15f;
        float train_acc = 1.0f - train_loss / 2.0f;
        float val_acc = train_acc * 0.92f;
        
        // Log scalar metrics
        WriteScalarSummary(log_dir, "loss/train", train_loss, epoch);
        WriteScalarSummary(log_dir, "loss/validation", val_loss, epoch);
        WriteScalarSummary(log_dir, "accuracy/train", train_acc, epoch);
        WriteScalarSummary(log_dir, "accuracy/validation", val_acc, epoch);
        WriteScalarSummary(log_dir, "hyperparameters/learning_rate", learning_rate, epoch);
        
        // Log histograms every 5 epochs
        if (epoch % 5 == 0) {
            for (size_t i = 0; i < model.weights.size(); i++) {
                std::string weight_tag = "weights/layer_" + std::to_string(i);
                std::string grad_tag = "gradients/layer_" + std::to_string(i);
                std::string act_tag = "activations/layer_" + std::to_string(i);
                
                WriteHistogramSummary(log_dir, weight_tag, model.weights[i], epoch);
                WriteHistogramSummary(log_dir, grad_tag, model.gradients[i], epoch);
                WriteHistogramSummary(log_dir, act_tag, model.activations[i], epoch);
            }
        }
        
        // Log images every 10 epochs
        if (epoch % 10 == 0) {
            std::vector<uint8_t> sample = GenerateSampleImage(64, epoch);
            WriteImageSummaryPNG(log_dir, "samples/generated", sample, 64, 64, 3, epoch);
        }
        
        // Progress update
        if (epoch % 20 == 0) {
            std::cout << "Epoch " << epoch << "/" << num_epochs 
                     << " - Loss: " << train_loss 
                     << " - Acc: " << train_acc << std::endl;
        }
    }
    
    std::cout << "Training complete! View results with:" << std::endl;
    std::cout << "tensorboard --logdir=" << log_dir << std::endl;
    
    return 0;
}
```

## Building and Running

### Compilation

```bash
# Compile proto files (one-time setup)
protoc --cpp_out=. tensorflow/core/util/event.proto
protoc --cpp_out=. tensorflow/core/framework/summary.proto
protoc --cpp_out=. tensorflow/core/framework/tensor.proto
protoc --cpp_out=. tensorflow/core/framework/types.proto
protoc --cpp_out=. tensorflow/core/framework/resource_handle.proto
protoc --cpp_out=. tensorflow/core/framework/tensor_shape.proto

# Compile your application
g++ -std=c++14 -O2 \
    tensorboard_writer.cpp \
    event.pb.cc \
    summary.pb.cc \
    tensor.pb.cc \
    types.pb.cc \
    resource_handle.pb.cc \
    tensor_shape.pb.cc \
    -lprotobuf \
    -o tensorboard_writer

# Run
./tensorboard_writer

# View in TensorBoard
tensorboard --logdir=./logs
```

### CMake Build System

Create a `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.10)
project(TensorBoardWriter)

set(CMAKE_CXX_STANDARD 14)

# Find Protobuf
find_package(Protobuf REQUIRED)

# Proto files
set(PROTO_FILES
    tensorflow/core/util/event.proto
    tensorflow/core/framework/summary.proto
    tensorflow/core/framework/tensor.proto
    tensorflow/core/framework/types.proto
    tensorflow/core/framework/resource_handle.proto
    tensorflow/core/framework/tensor_shape.proto
)

# Generate proto sources
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# Executable
add_executable(tensorboard_writer
    tensorboard_writer.cpp
    ${PROTO_SRCS}
)

target_include_directories(tensorboard_writer PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(tensorboard_writer
    ${Protobuf_LIBRARIES}
)
```

Build with:

```bash
mkdir build && cd build
cmake ..
make
./tensorboard_writer
```

## Performance Considerations

### 1. Batched Writing

For high-frequency logging, batch your writes:

```cpp
class BatchedEventWriter {
private:
    std::string log_dir_;
    std::vector<tensorflow::Event> event_buffer_;
    size_t max_buffer_size_;
    
public:
    BatchedEventWriter(const std::string& log_dir, size_t buffer_size = 100)
        : log_dir_(log_dir), max_buffer_size_(buffer_size) {}
    
    void AddEvent(const tensorflow::Event& event) {
        event_buffer_.push_back(event);
        if (event_buffer_.size() >= max_buffer_size_) {
            Flush();
        }
    }
    
    void Flush() {
        if (event_buffer_.empty()) return;
        
        std::string filename = log_dir_ + "/events.out.tfevents." + 
                              std::to_string(time(nullptr)) + ".hostname";
        std::ofstream file(filename, std::ios::binary | std::ios::app);
        
        for (const auto& event : event_buffer_) {
            std::string serialized;
            event.SerializeToString(&serialized);
            WriteRecord(file, serialized);
        }
        
        file.close();
        event_buffer_.clear();
    }
    
    ~BatchedEventWriter() {
        Flush();
    }
};
```

### 2. Async Writing

For minimal impact on training:

```cpp
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

class AsyncEventWriter {
private:
    std::string log_dir_;
    std::queue<tensorflow::Event> event_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::thread writer_thread_;
    bool shutdown_ = false;
    
    void WriterLoop() {
        while (true) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] { 
                return !event_queue_.empty() || shutdown_; 
            });
            
            if (shutdown_ && event_queue_.empty()) break;
            
            if (!event_queue_.empty()) {
                tensorflow::Event event = event_queue_.front();
                event_queue_.pop();
                lock.unlock();
                
                // Write event
                std::string filename = log_dir_ + "/events.out.tfevents." + 
                                      std::to_string(time(nullptr)) + ".hostname";
                std::ofstream file(filename, std::ios::binary | std::ios::app);
                std::string serialized;
                event.SerializeToString(&serialized);
                WriteRecord(file, serialized);
                file.close();
            }
        }
    }
    
public:
    AsyncEventWriter(const std::string& log_dir) : log_dir_(log_dir) {
        writer_thread_ = std::thread(&AsyncEventWriter::WriterLoop, this);
    }
    
    void WriteEvent(const tensorflow::Event& event) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        event_queue_.push(event);
        cv_.notify_one();
    }
    
    ~AsyncEventWriter() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            shutdown_ = true;
        }
        cv_.notify_one();
        if (writer_thread_.joinable()) {
            writer_thread_.join();
        }
    }
};
```

### 3. Reduce Logging Frequency

```cpp
class ThrottledLogger {
private:
    int log_every_n_steps_;
    int current_step_ = 0;
    
public:
    ThrottledLogger(int log_every_n) : log_every_n_steps_(log_every_n) {}
    
    bool ShouldLog() {
        current_step_++;
        return (current_step_ % log_every_n_steps_) == 0;
    }
};

// Usage
ThrottledLogger scalar_logger(1);      // Log scalars every step
ThrottledLogger histogram_logger(10);  // Log histograms every 10 steps
ThrottledLogger image_logger(50);      // Log images every 50 steps
```

## Troubleshooting

### Common Issues

**1. TensorBoard shows no data**
- Check file permissions in log directory
- Verify proto files are correctly compiled
- Ensure event files are being created
- Check TensorBoard is pointing to correct directory

**2. Images not displaying**
- Verify PNG encoding is working
- Check image dimensions and channel count
- Ensure colorspace parameter is correct (1, 3, or 4)
- Try with raw data first, then add encoding

**3. Histograms look wrong**
- Verify bin calculation logic
- Check for NaN or Inf values in data
- Ensure min/max are set correctly
- Use appropriate number of bins

**4. Protobuf compilation errors**
- Ensure all dependent proto files are compiled
- Check protobuf version compatibility
- Verify include paths are correct

### Debugging Tips

```cpp
// Add debug output
void DebugEvent(const tensorflow::Event& event) {
    std::cout << "Event step: " << event.step() << std::endl;
    std::cout << "Wall time: " << event.wall_time() << std::endl;
    std::cout << "Has summary: " << event.has_summary() << std::endl;
    
    if (event.has_summary()) {
        std::cout << "Summary values: " << event.summary().value_size() << std::endl;
        for (const auto& value : event.summary().value()) {
            std::cout << "  Tag: " << value.tag() << std::endl;
        }
    }
}

// Validate written files
void ValidateEventFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    
    int record_count = 0;
    while (file.peek() != EOF) {
        uint64_t length;
        file.read(reinterpret_cast<char*>(&length), sizeof(length));
        file.seekg(4 + length + 4, std::ios::cur);  // Skip CRCs and data
        record_count++;
    }
    
    std::cout << "File contains " << record_count << " records" << std::endl;
}
```

## Advanced Topics

### Multiple Experiment Tracking

```cpp
class ExperimentTracker {
private:
    std::string base_log_dir_;
    std::map<std::string, std::string> experiment_dirs_;
    
public:
    ExperimentTracker(const std::string& base_dir) : base_log_dir_(base_dir) {}
    
    std::string CreateExperiment(const std::string& name, 
                                const std::map<std::string, std::string>& config) {
        // Create unique directory with timestamp
        std::string exp_dir = base_log_dir_ + "/" + name + "_" + 
                             std::to_string(time(nullptr));
        
        // Create directory
        system(("mkdir -p " + exp_dir).c_str());
        
        // Save config
        std::ofstream config_file(exp_dir + "/config.txt");
        for (const auto& [key, value] : config) {
            config_file << key << ": " << value << "\n";
        }
        config_file.close();
        
        experiment_dirs_[name] = exp_dir;
        return exp_dir;
    }
    
    std::string GetExperimentDir(const std::string& name) {
        return experiment_dirs_[name];
    }
};
```

### Text Summaries

```cpp
void WriteTextSummary(const std::string& log_dir,
                     const std::string& tag,
                     const std::string& text,
                     int64_t step) {
    
    std::string filename = log_dir + "/events.out.tfevents." + 
                          std::to_string(time(nullptr)) + ".hostname";
    
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    
    tensorflow::Event event;
    event.set_wall_time(GetCurrentTimeMicros() / 1e6);
    event.set_step(step);
    
    tensorflow::Summary* summary = event.mutable_summary();
    tensorflow::Summary::Value* summary_value = summary->add_value();
    summary_value->set_tag(tag);
    
    // Create tensor proto for text
    tensorflow::TensorProto* tensor = summary_value->mutable_tensor();
    tensor->set_dtype(tensorflow::DT_STRING);
    tensor->mutable_tensor_shape()->add_dim()->set_size(1);
    tensor->add_string_val(text);
    
    std::string serialized;
    event.SerializeToString(&serialized);
    WriteRecord(file, serialized);
    
    file.close();
}
```

## Conclusion

Writing TensorFlow events from C++ gives you full control over your logging pipeline while maintaining compatibility with TensorBoard's powerful visualization tools. Key takeaways:

1. **Scalars** are lightweight and should be logged frequently
2. **Images** are expensive; log selectively and resize appropriately
3. **Histograms** help monitor training health; focus on weights and gradients
4. **Performance matters**: Use batching and async writing for production
5. **Organization is key**: Use hierarchical tags and separate experiments

This approach is particularly valuable for:
- High-performance C++ training frameworks
- Real-time applications with ML components
- Systems where Python overhead is unacceptable
- Custom ML pipelines in game engines or embedded systems

The complete code examples provided can serve as a foundation for your own TensorBoard logging infrastructure in C++.

## Resources

- [TensorFlow Protocol Buffers](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [Protocol Buffers C++ Guide](https://developers.google.com/protocol-buffers/docs/cpptutorial)
- [stb_image_write](https://github.com/nothings/stb)
- [TensorBoard Event File Format](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/event.proto)

