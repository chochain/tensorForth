#include <fstream>
#include <string>
#include <ctime>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>

#if 0
# Compile the proto files first
protoc --cpp_out=. tensorflow/core/util/event.proto
protoc --cpp_out=. tensorflow/core/framework/summary.proto

# Compile your code
g++ -std=c++11 your_code.cpp event.pb.cc summary.pb.cc \
    -lprotobuf -o event_writer

# Run
./event_writer

# View in TensorBoard
tensorboard --logdir=./logs
#endif
    
#include "tensorflow/core/util/event.pb.h"
#include "tensorflow/core/framework/summary.pb.h"

// Helper function to get current timestamp in microseconds
int64_t GetCurrentTimeMicros() {
    return static_cast<int64_t>(time(nullptr)) * 1000000;
}

// Helper to write a length-delimited record (TensorBoard format)
void WriteRecord(std::ofstream& file, const std::string& data) {
    // Write length as uint64_t
    uint64_t length = data.size();
    file.write(reinterpret_cast<const char*>(&length), sizeof(length));
    
    // Write CRC of length (simplified - use proper CRC32C in production)
    uint32_t length_crc = 0;  // Should compute CRC32C of length
    file.write(reinterpret_cast<const char*>(&length_crc), sizeof(length_crc));
    
    // Write data
    file.write(data.c_str(), data.size());
    
    // Write CRC of data (simplified - use proper CRC32C in production)
    uint32_t data_crc = 0;  // Should compute CRC32C of data
    file.write(reinterpret_cast<const char*>(&data_crc), sizeof(data_crc));
}

// Write a scalar summary to TensorBoard
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

// Encode raw RGB data to PNG format (simplified - use a proper PNG encoder)
// This is a placeholder - you should use a library like stb_image_write
std::string EncodePNG(const std::vector<uint8_t>& image_data, 
                      int height, int width, int channels) {
    // In practice, use stb_image_write.h or similar:
    // stbi_write_png_to_func(callback, context, width, height, channels, data, stride);
    
    // For now, we'll just return the raw data
    // TensorBoard also accepts raw encoded images
    return std::string(image_data.begin(), image_data.end());
}

// Write an image summary to TensorBoard
// image_data: raw pixel data in RGB or RGBA format
// height, width: image dimensions
// channels: 1 (grayscale), 3 (RGB), or 4 (RGBA)
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
    
    // Encode image to PNG (or JPEG)
    std::string encoded_image = EncodePNG(image_data, height, width, channels);
    image->set_encoded_image_string(encoded_image);
    
    // Serialize and write
    std::string serialized;
    event.SerializeToString(&serialized);
    WriteRecord(file, serialized);
    
    file.close();
}

// Write a histogram summary to TensorBoard
// values: the data to create histogram from
// num_bins: number of histogram bins (default: 30)
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
            // Handle edge case where val == max_val
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

// Alternative: Write histogram with custom bins
void WriteHistogramSummaryCustom(const std::string& log_dir,
                                const std::string& tag,
                                const std::vector<float>& bin_edges,
                                const std::vector<int>& bin_counts,
                                const std::vector<float>& values,
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
    
    // Calculate statistics from values
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
    
    // Add custom bins
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

// Helper function to create a test image (gradient)
std::vector<uint8_t> CreateTestImage(int height, int width, int step) {
    std::vector<uint8_t> image(height * width * 3);  // RGB
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            image[idx + 0] = (x * 255) / width;           // R: horizontal gradient
            image[idx + 1] = (y * 255) / height;          // G: vertical gradient
            image[idx + 2] = ((step * 10) % 255);         // B: changes with step
        }
    }
    
    return image;
}

// Helper to generate random normal distribution
std::vector<float> GenerateNormalDistribution(int n, float mean, float stddev) {
    std::vector<float> values(n);
    
    // Simple Box-Muller transform for normal distribution
    for (int i = 0; i < n; i += 2) {
        float u1 = static_cast<float>(rand()) / RAND_MAX;
        float u2 = static_cast<float>(rand()) / RAND_MAX;
        
        float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
        values[i] = mean + z0 * stddev;
        
        if (i + 1 < n) {
            float z1 = std::sqrt(-2.0f * std::log(u1)) * std::sin(2.0f * M_PI * u2);
            values[i + 1] = mean + z1 * stddev;
        }
    }
    
    return values;
}

int main() {
    std::string log_dir = "./logs";
    srand(time(nullptr));
    
    // Example: Log training metrics, images, and histograms
    for (int step = 0; step < 100; step++) {
        // Log scalars
        float loss = 1.0f / (step + 1);
        WriteScalarSummary(log_dir, "train/loss", loss, step);
        
        float accuracy = 1.0f - loss;
        WriteScalarSummary(log_dir, "train/accuracy", accuracy, step);
        
        // Log an image every 10 steps
        if (step % 10 == 0) {
            std::vector<uint8_t> image = CreateTestImage(128, 128, step);
            WriteImageSummary(log_dir, "visualizations/gradient", 
                            image, 128, 128, 3, step);
        }
        
        // Log histogram of weights (simulated)
        // Generate weights with changing distribution
        float mean = 0.0f;
        float stddev = 1.0f / (1.0f + step * 0.01f);  // Decreasing variance
        std::vector<float> weights = GenerateNormalDistribution(1000, mean, stddev);
        
        WriteHistogramSummary(log_dir, "weights/layer1", weights, step);
        
        // Log histogram of gradients (simulated)
        std::vector<float> gradients = GenerateNormalDistribution(1000, 0.0f, 0.1f);
        WriteHistogramSummary(log_dir, "gradients/layer1", gradients, step, 50);  // 50 bins
        
        // Log histogram of activations (simulated with ReLU-like distribution)
        std::vector<float> activations = GenerateNormalDistribution(1000, 0.5f, 0.3f);
        for (auto& val : activations) {
            if (val < 0) val = 0;  // ReLU
        }
        WriteHistogramSummary(log_dir, "activations/layer1", activations, step);
    }
    
    return 0;
}

