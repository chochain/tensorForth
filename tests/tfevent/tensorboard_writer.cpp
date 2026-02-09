#include <fstream>
#include <string>
#include <ctime>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

// Include generated proto headers
#include "event.pb.h"
#include "summary.pb.h"
#include "histogram.pb.h"

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

// Get event filename for log directory
std::string GetEventFilename(const std::string& log_dir) {
    static std::string cached_filename;
    if (cached_filename.empty()) {
        cached_filename = log_dir + "/events.out.tfevents." + 
                         std::to_string(time(nullptr)) + ".gnii";
    }
    return cached_filename;
}

// Write a scalar summary to TensorBoard
void WriteScalarSummary(const std::string& log_dir, 
                        const std::string& tag,
                        float value,
                        int64_t step) {
    std::string filename = GetEventFilename(log_dir);
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

// Write an image summary (expects raw RGB/RGBA data)
void WriteImageSummary(const std::string& log_dir,
                       const std::string& tag,
                       const std::vector<uint8_t>& image_data,
                       int height,
                       int width,
                       int channels,
                       int64_t step) {
    
    std::string filename = GetEventFilename(log_dir);
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
    
    // For simplicity, store raw image data
    // In production, encode as PNG using stb_image_write
    std::string raw_data(image_data.begin(), image_data.end());
    image->set_encoded_image_string(raw_data);
    
    // Serialize and write
    std::string serialized;
    event.SerializeToString(&serialized);
    WriteRecord(file, serialized);
    
    file.close();
}

// Write a histogram summary
void WriteHistogramSummary(const std::string& log_dir,
                          const std::string& tag,
                          const std::vector<float>& values,
                          int64_t step,
                          int num_bins = 30) {
    
    if (values.empty()) return;
    
    std::string filename = GetEventFilename(log_dir);
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
    double sum = 0.0;
    double sum_squares = 0.0;
    
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
            double edge = min_val + i * bin_width;
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

// Helper to generate test image
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
    std::cout << "=== TensorBoard C++ Event Writer ===" << std::endl;
    std::cout << std::endl;
    
    // Initialize protocol buffers
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    
    std::cout << "Protobuf version: " << GOOGLE_PROTOBUF_VERSION 
              << " (" << (GOOGLE_PROTOBUF_VERSION / 1000000) << "."
              << ((GOOGLE_PROTOBUF_VERSION / 1000) % 1000) << "."
              << (GOOGLE_PROTOBUF_VERSION % 1000) << ")" << std::endl;
    std::cout << std::endl;
    
    std::string log_dir = "/u01/runs";
    
    // Create log directory
    system(("mkdir -p " + log_dir).c_str());
    
    std::cout << "Writing TensorBoard events to: " << log_dir << std::endl;
    std::cout << std::endl;
    
    srand(time(nullptr));
    
    // Simulate training loop
    for (int step = 0; step < 100; step++) {
        // Log scalars
        float loss = 2.0f * std::exp(-0.05f * step) + 0.1f;
        float accuracy = 1.0f - loss / 2.0f;
        
        WriteScalarSummary(log_dir, "train/loss", loss, step);
        WriteScalarSummary(log_dir, "train/accuracy", accuracy, step);
        
        // Log an image every 10 steps
        if (step % 10 == 0) {
            std::vector<uint8_t> image = CreateTestImage(128, 128, step);
            WriteImageSummary(log_dir, "visualizations/gradient", 
                            image, 128, 128, 3, step);
        }
        
        // Log histogram of weights (simulated)
        float mean = 0.0f;
        float stddev = 1.0f / (1.0f + step * 0.01f);
        std::vector<float> weights = GenerateNormalDistribution(1000, mean, stddev);
        
        WriteHistogramSummary(log_dir, "weights/layer1", weights, step);
        
        // Log histogram of gradients (simulated)
        std::vector<float> gradients = GenerateNormalDistribution(1000, 0.0f, 0.1f);
        WriteHistogramSummary(log_dir, "gradients/layer1", gradients, step, 50);
        
        if (step % 10 == 0) {
            std::cout << "Step " << step << " - Loss: " << loss 
                     << ", Accuracy: " << accuracy << std::endl;
        }
    }
    
    std::cout << std::endl;
    std::cout << "=== Success! ===" << std::endl;
    std::cout << std::endl;
    std::cout << "View results with:" << std::endl;
    std::cout << "  tensorboard --logdir=" << log_dir << std::endl;
    std::cout << std::endl;
    
    // Clean up protocol buffers
    google::protobuf::ShutdownProtobufLibrary();
    
    return 0;
}
