/*
 * main.cpp - TensorBoard FlatBuffers Demo
 *
 * Demonstrates writing a .tfevents file containing:
 *   1. Scalar summaries   (loss curve, accuracy curve)
 *   2. Image summaries    (procedurally generated test patterns)
 *   3. Histogram summaries (weight distributions)
 *
 * The FlatBuffers library encodes the plugin metadata embedded in each
 * SummaryMetadata protobuf message, which is how TensorBoard identifies
 * which plugin should handle each data type.
 *
 * Usage:
 *   ./tb_demo [output_dir]
 *   tensorboard --logdir <output_dir>
 */

#include "tensorboard_writer.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <sys/stat.h>

// ─── Helpers ──────────────────────────────────────────────────────────────────

static void ensure_dir(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

static std::string make_event_path(const std::string& dir, const std::string& run_name) {
    std::string run_dir = dir + "/" + run_name;
    ensure_dir(run_dir);
    // FIX 3: use hostname + PID in filename as TensorBoard 2.x requires
    return tensorboard::MakeEventFilePath(run_dir);
}

// Banner printer
static void print_section(const std::string& title) {
    std::cout << "\n";
    std::cout << "  ┌─────────────────────────────────────────────┐\n";
    std::cout << "  │  " << std::left << std::setw(43) << title << " │\n";
    std::cout << "  └─────────────────────────────────────────────┘\n";
}

// ─── Demo 1: Scalar Summaries ─────────────────────────────────────────────────
void demo_scalars(const std::string& logdir) {
    print_section("Demo 1: Scalar Summaries");

    std::string path = make_event_path(logdir, "scalars");
    tensorboard::EventWriter writer(path);
    std::cout << "  Writing to: " << path << "\n";

    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.02f);

    int num_steps = 100;

    for (int step = 0; step < num_steps; ++step) {
        // Simulated training loss (exponential decay + noise)
        float t     = static_cast<float>(step) / num_steps;
        float loss  = 2.0f * std::exp(-3.0f * t) + 0.1f + noise(rng);
        float acc   = 1.0f - std::exp(-4.0f * t) * 0.9f + noise(rng) * 0.5f;
        acc         = std::max(0.0f, std::min(1.0f, acc));

        // Learning rate schedule (cosine decay)
        float lr = 0.001f * (1.0f + std::cos(3.14159f * t)) * 0.5f;

        writer.WriteScalar("train/loss",     loss, step);
        writer.WriteScalar("train/accuracy", acc,  step);
        writer.WriteScalar("train/lr",       lr,   step);

        // Validation metrics (added every 10 steps with more noise)
        if (step % 10 == 0) {
            float val_loss = loss + std::abs(noise(rng)) * 0.3f;
            float val_acc  = acc  - std::abs(noise(rng)) * 0.05f;
            writer.WriteScalar("val/loss",     val_loss, step);
            writer.WriteScalar("val/accuracy", val_acc,  step);
        }

        if (step % 25 == 0) {
            std::cout << "  Step " << std::setw(3) << step
                      << " | loss=" << std::fixed << std::setprecision(4) << loss
                      << " | acc="  << std::fixed << std::setprecision(4) << acc
                      << " | lr="   << std::scientific << std::setprecision(2) << lr
                      << "\n";
        }
    }
    std::cout << "  ✓ Wrote " << num_steps
              << " steps of scalar data (train/loss, train/accuracy, train/lr, val/*)\n";
}

// ─── Demo 2: Image Summaries ──────────────────────────────────────────────────

// Generate a colorful test pattern (sine waves)
static std::vector<uint8_t> make_sine_pattern(int w, int h, float phase) {
    std::vector<uint8_t> px(w * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float fx = static_cast<float>(x) / w;
            float fy = static_cast<float>(y) / h;
            float r  = 0.5f + 0.5f * std::sin(2.0f * 3.14159f * fx * 4 + phase);
            float g  = 0.5f + 0.5f * std::sin(2.0f * 3.14159f * fy * 3 + phase * 1.3f);
            float b  = 0.5f + 0.5f * std::sin(2.0f * 3.14159f * (fx + fy) * 5 + phase * 0.7f);
            px[(y * w + x) * 3 + 0] = static_cast<uint8_t>(r * 255);
            px[(y * w + x) * 3 + 1] = static_cast<uint8_t>(g * 255);
            px[(y * w + x) * 3 + 2] = static_cast<uint8_t>(b * 255);
        }
    }
    return px;
}

// Generate a gradient heatmap
static std::vector<uint8_t> make_gradient(int w, int h, float step_f) {
    std::vector<uint8_t> px(w * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float v  = static_cast<float>(x + y * w) / (w * h);
            float hue = v * 360.0f + step_f * 30.0f;
            // HSV to RGB (simple)
            float h_  = std::fmod(hue, 360.0f) / 60.0f;
            int   i_  = static_cast<int>(h_);
            float f   = h_ - i_;
            float p   = 0.0f, q = 1.0f - f, t2 = f;
            float r = 0, g = 0, b = 0;
            switch (i_ % 6) {
                case 0: r=1;  g=t2; b=p;  break;
                case 1: r=q;  g=1;  b=p;  break;
                case 2: r=p;  g=1;  b=t2; break;
                case 3: r=p;  g=q;  b=1;  break;
                case 4: r=t2; g=p;  b=1;  break;
                case 5: r=1;  g=p;  b=q;  break;
            }
            px[(y * w + x) * 3 + 0] = static_cast<uint8_t>(r * 255);
            px[(y * w + x) * 3 + 1] = static_cast<uint8_t>(g * 255);
            px[(y * w + x) * 3 + 2] = static_cast<uint8_t>(b * 255);
        }
    }
    return px;
}

// Generate a checkerboard
static std::vector<uint8_t> make_checkerboard(int w, int h, int tile, float t_) {
    std::vector<uint8_t> px(w * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            bool chk = ((x / tile) + (y / tile)) % 2 == 0;
            float bright = chk ? 0.9f : 0.1f;
            float r = bright + 0.1f * std::sin(t_ + x * 0.1f);
            float g = bright + 0.1f * std::sin(t_ * 1.3f + y * 0.1f);
            float b = bright + 0.1f * std::sin(t_ * 0.7f + (x + y) * 0.05f);
            px[(y * w + x) * 3 + 0] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, r * 255)));
            px[(y * w + x) * 3 + 1] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, g * 255)));
            px[(y * w + x) * 3 + 2] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, b * 255)));
        }
    }
    return px;
}

void demo_images(const std::string& logdir) {
    print_section("Demo 2: Image Summaries");

    std::string path = make_event_path(logdir, "images");
    tensorboard::EventWriter writer(path);
    std::cout << "  Writing to: " << path << "\n";

    int W = 128, H = 128;
    int num_steps = 10;

    for (int step = 0; step < num_steps; ++step) {
        float t = static_cast<float>(step) * 0.3f;

        // Image 1: Animated sine wave pattern
        auto sine = make_sine_pattern(W, H, t);
        writer.WriteImage("images/sine_wave", W, H, sine, step);

        // Image 2: Rotating gradient heatmap
        auto grad = make_gradient(W, H, static_cast<float>(step));
        writer.WriteImage("images/gradient_heatmap", W, H, grad, step);

        // Image 3: Evolving checkerboard
        auto chk = make_checkerboard(W, H, 16, t);
        writer.WriteImage("images/checkerboard", W, H, chk, step);

        std::cout << "  Step " << std::setw(2) << step
                  << " | wrote 3 images (" << W << "×" << H << " RGB PNG)\n";
    }
    std::cout << "  ✓ Wrote " << num_steps
              << " steps of image data (sine_wave, gradient_heatmap, checkerboard)\n";
}

// ─── Demo 3: Histogram Summaries ──────────────────────────────────────────────

// Generate normally distributed random values
static std::vector<double> normal_samples(std::mt19937& rng, int n,
                                           double mean, double stddev) {
    std::normal_distribution<double> dist(mean, stddev);
    std::vector<double> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

void demo_histograms(const std::string& logdir) {
    print_section("Demo 3: Histogram Summaries");

    std::string path = make_event_path(logdir, "histograms");
    tensorboard::EventWriter writer(path);
    std::cout << "  Writing to: " << path << "\n";

    std::mt19937 rng(123);
    int num_steps = 50;

    for (int step = 0; step < num_steps; ++step) {
        float t = static_cast<float>(step) / num_steps;

        // Layer 1 weights: gradually tightening distribution
        double w1_std = 1.0 - 0.7 * t;  // 1.0 → 0.3
        auto w1 = normal_samples(rng, 512, 0.0, w1_std);
        writer.WriteHistogram("weights/layer1", w1, step, 40);

        // Layer 2 weights: bimodal → unimodal
        double mix = t; // 0 = bimodal, 1 = unimodal
        std::vector<double> w2;
        auto half_a = normal_samples(rng, 256, -1.0 * (1.0 - mix), 0.5);
        auto half_b = normal_samples(rng, 256,  1.0 * (1.0 - mix), 0.5);
        w2.insert(w2.end(), half_a.begin(), half_a.end());
        w2.insert(w2.end(), half_b.begin(), half_b.end());
        writer.WriteHistogram("weights/layer2", w2, step, 40);

        // Activations: shifting mean over training
        double act_mean = -2.0 + 4.0 * t; // -2 → +2
        auto acts = normal_samples(rng, 1024, act_mean, 0.8);
        writer.WriteHistogram("activations/relu", acts, step, 40);

        // Gradients: shrinking magnitude (gradient vanishing demo)
        double grad_std = 0.5 * std::exp(-3.0 * t);
        auto grads = normal_samples(rng, 256, 0.0, grad_std);
        writer.WriteHistogram("gradients/layer1", grads, step, 30);

        if (step % 10 == 0) {
            std::cout << "  Step " << std::setw(2) << step
                      << " | w1_std=" << std::fixed << std::setprecision(3) << w1_std
                      << " | act_mean=" << std::fixed << std::setprecision(3) << act_mean
                      << " | grad_std=" << std::scientific << std::setprecision(2) << grad_std
                      << "\n";
        }
    }
    std::cout << "  ✓ Wrote " << num_steps
              << " steps of histogram data (weights, activations, gradients)\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::string logdir = (argc > 1) ? argv[1] : "/tmp/tb_flatbuffers_demo";
    ensure_dir(logdir);

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║  TensorBoard FlatBuffers Event Writer Demo        ║\n";
    std::cout << "║  Writing .tfevents files to: " << std::left << std::setw(20) << logdir << " ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n";

    try {
        demo_scalars(logdir);
        demo_images(logdir);
        demo_histograms(logdir);
    } catch (const std::exception& e) {
        std::cerr << "\n  ERROR: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║  All demos complete!                                      ║\n";
    std::cout << "║                                                           ║\n";
    std::cout << "║  To visualize, run:                                       ║\n";
    std::cout << "║    pip install tensorboard                                ║\n";
    std::cout << "║    tensorboard --logdir " << std::left << std::setw(34) << logdir << " ║\n";
    std::cout << "║    # Open http://localhost:6006                           ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    return 0;
}
