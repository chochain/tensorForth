/*
 * example_usage.cpp - Demonstrates text and hparams functionality
 */

#include "hparam.h"
#include <iostream>

int main() {
    tensorboard::EventWriter writer("./runs/texts/events.out.tfevents.123.gnii.1");
    tensorboard::HParamWriter hparam("./runs/hparams/events.out.tfevents.123.gnii.1");
    
    // ── Example 1: Text logging ─────────────────────────────────────────────
    std::cout << "Writing text examples...\n";
    
    writer.add_text("model/description", "Hello World", 0);
    
    writer.add_text("training/notes", 
                   "Epoch 1: Loss decreased significantly\n"
                   "Learning rate: 0.001", 1);
    
    writer.add_text("results/summary",
                   "Final accuracy: 92.5%\n"
                   "Test loss: 0.234", 100);
    // ── Example 2: HParams with metrics ─────────────────────────────────────
    std::cout << "Writing hparams examples...\n";

    // Define hyperparameters and metrics for the experiment
    std::map<STR, tensorboard::HParamValue> hparam_defaults;
    hparam_defaults["learning_rate"] = 0.001;
    hparam_defaults["batch_size"] = 32;
    hparam_defaults["optimizer"] = "adam";
    hparam_defaults["dropout"] = 0.5;
    hparam_defaults["use_batch_norm"] = true;
    
    std::vector<std::string> metric_tags = {"Accuracy", "Loss"};
    
    // Initialize hparams experiment configuration
//    hparam.add_config(hparam_defaults, metric_tags);

    // You can also log scalars during training
    float loss, acc;
    for (int step = 0; step < 100; ++step) {
        loss = 1.0f / (step + 1);  // Dummy decreasing loss
        acc = 1.0f - loss;          // Dummy increasing accuracy
        
        writer.add_scalar("accuracy", acc, step);
        writer.add_scalar("loss", loss, step);
        
        if (step % 10 == 0) {
            std::ostringstream ss;
            ss << "<pre>step " << step << ": accuracy=" << acc 
               << ", loss=" << loss << "</pre>";
            writer.add_text("training/progress", ss.str(), step);
        }
    }
    // ── Run 1: Default hyperparameters ──────────────────────────────────────
    std::map<STR, tensorboard::HParamValue> hparams;
    hparams["batch_size"] = 32;
    hparams["optimizer"] = "adam";
    hparams["use_batch_norm"] = true;
    hparams["batch_size"] = 32;
    hparams["learning_rate"] = 0.01;
    hparams["dropout"] = 0.5;

    tensorboard::HParamWriter hp0("./runs/hparams/run0/events.out.tfevents.456.gnii.1");
    hparams["batch_size"] = 64;
    hparams["learning_rate"] = 0.01;
    hparams["dropout"] = 0.5;
    std::map<std::string, double> metrics;
    metrics["Accuracy"] = 0.1;
    metrics["Loss"]     = 0.9;
    hp0.add_hparams(hparams, metrics, "test0", 17000000);

    tensorboard::HParamWriter hp1("./runs/hparams/run1/events.out.tfevents.456.gnii.1");
    hparams["batch_size"] = 32;
    hparams["learning_rate"] = 0.02;
    hparams["dropout"] = 0.75;
    metrics["Accuracy"] = 0.2;
    metrics["Loss"]     = 0.8;
    hp1.add_hparams(hparams, metrics, "test1", 17000001);
    
    tensorboard::HParamWriter hp2("./runs/hparams/run2/events.out.tfevents.456.gnii.1");
    hparams["batch_size"] = 32;
    hparams["learning_rate"] = 0.03;
    hparams["dropout"] = 0.90;
    metrics["Accuracy"] = 0.3;
    metrics["Loss"]     = 0.7;
    hp2.add_hparams(hparams, metrics, "test2", 17000002);
    
    tensorboard::HParamWriter hp3("./runs/hparams/run3/events.out.tfevents.456.gnii.1");
    hparams["batch_size"] = 64;
    hparams["learning_rate"] = 0.03;
    hparams["dropout"] = 0.90;
    metrics["Accuracy"] = 0.4;
    metrics["Loss"]     = 0.6;
    hp3.add_hparams(hparams, metrics, "test3", 17000003);

    std::cout << "Done! View results with:\n";
    std::cout << "  tensorboard --logdir=./logs\n";

    return 0;
}

/*
 * ── EXPECTED OUTPUT IN TENSORBOARD ─────────────────────────────────────────
 * 
 * TEXT TAB:
 *   - model/description: Shows formatted markdown text
 *   - training/notes: Shows multi-line progress notes
 *   - training/progress: Shows step-by-step training updates
 *   - results/summary: Shows final results
 * 
 * HPARAMS TAB:
 *   - Table view: Shows all hyperparameter combinations
 *   - Parallel coordinates: Interactive visualization
 *   - Scatter plot matrix: Correlation between hparams and metrics
 *   
 * SCALARS TAB:
 *   - accuracy: Smooth curve showing improvement
 *   - loss: Smooth curve showing decrease
 * 
 * ───────────────────────────────────────────────────────────────────────────
 */

/*
 * ── ADVANCED USAGE: Multiple HParam Runs ───────────────────────────────────
 */
#if 0
void hyperparameter_search_example() {
    tensorboard::HParamWriter hparam("./hparam_search");
    
    // Define search space
    std::vector<double> learning_rates = {0.001, 0.01, 0.1};
    std::vector<int> batch_sizes = {16, 32, 64};
    std::vector<std::string> optimizers = {"adam", "sgd"};
    
    // Initialize config
    std::map<std::string, tensorboard::HParamValue> defaults;
    defaults["learning_rate"] = 0.001;
    defaults["batch_size"] = 32;
    defaults["optimizer"] = "adam";
    
    std::vector<std::string> metrics = {"val_accuracy", "val_loss", "train_time"};
    writer.add_config(defaults, metrics);
    
    // Run grid search
    int run_id = 0;
    for (double lr : learning_rates) {
        for (int bs : batch_sizes) {
            for (const auto& opt : optimizers) {
                std::cout << "Running experiment " << run_id << "...\n";
                
                // Create new writer for this run
                std::ostringstream path;
                path << "./hparam_search/run_" << run_id;
                tensorboard::EventWriter run_writer(path.str());
                
                // Set hyperparameters
                std::map<std::string, tensorboard::HParamValue> hparams;
                hparams["learning_rate"] = lr;
                hparams["batch_size"] = bs;
                hparams["optimizer"] = opt;
                
                // Simulate training and get metrics
                double val_acc = 0.8 + (1.0 / (1.0 + lr)) * 0.15;  // Dummy
                double val_loss = 0.5 - val_acc * 0.3;              // Dummy
                double train_time = bs * 0.1 + lr * 10.0;          // Dummy
                
                std::map<std::string, double> metrics_result;
                metrics_result["val_accuracy"] = val_acc;
                metrics_result["val_loss"] = val_loss;
                metrics_result["train_time"] = train_time;
                
                // Log results
                run_writer.add_hparams(hparams, metrics_result, run_id);
                
                ++run_id;
            }
        }
    }
    
    std::cout << "Grid search complete! " << run_id << " runs.\n";
    std::cout << "View with: tensorboard --logdir=./hparam_search\n";
}
#endif 
