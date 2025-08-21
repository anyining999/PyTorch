#include "transformer_lib.h"
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/init.h>
#include <torch/nn/parallel/distributed.h>
#include <iostream>
#include <memory>
#include <cstdlib>

// Helper to get environment variables
int get_env_var(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::stoi(value) : -1;
}

int main() {
    // --- Distributed Setup ---
    int rank = get_env_var("RANK");
    int world_size = get_env_var("WORLD_SIZE");
    int local_rank = get_env_var("LOCAL_RANK");

    if (rank == -1 || world_size == -1 || local_rank == -1) {
        std::cerr << "RANK, WORLD_SIZE, and LOCAL_RANK environment variables must be set." << std::endl;
        // For non-distributed runs, we can default to single-GPU mode.
        rank = 0;
        world_size = 1;
        local_rank = 0;
        std::cout << "Running in single-GPU mode." << std::endl;
    } else {
        torch::distributed::init_process_group("nccl", "env://", std::chrono::seconds(30));
    }

    const bool is_distributed = world_size > 1;

    try {
        // --- Configuration ---
        torch::Device device(local_rank);
        const int vocab_size = 30000;
        const int d_model = 512;
        const int n_layers = 6;
        const int n_heads = 8;
        const int min_bs = 8;
        const int max_bs = 64;
        const int epochs = 1; // Keep it short for a test run
        const int total_steps_for_test = 20;

        // --- Initialization ---
        auto model = std::make_shared<Transformer>(vocab_size, d_model, n_layers, n_heads);
        model->to(device);

        // Wrap the model for distributed training if applicable
        if (is_distributed) {
            model = std::make_shared<torch::nn::parallel::DistributedDataParallel>(
                model, torch::nn::parallel::DistributedDataParallelOptions().device_ids({local_rank})
            );
        }

        auto optimizer = std::make_shared<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(0.0001));

        GradScaler scaler;
        DynamicBatchScheduler batch_scheduler(min_bs, max_bs);

        // --- Training Loop ---
        if (rank == 0) {
            std::cout << "Starting training on " << world_size << " GPUs." << std::endl;
        }

        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Using a mock dataset for this example
            auto dataset = torch::randint(0, vocab_size, {1000, 128});

            int step = 0;
            for (long i = 0; i < dataset.size(0); ) {
                int batch_size = batch_scheduler.get_batch_size();
                if (i + batch_size > dataset.size(0)) break;

                auto batch = dataset.slice(0, i, i + batch_size).to(device);
                i += batch_size;

                auto output = model->forward(batch);
                auto targets = batch.clone().detach();

                auto loss = torch::nn::functional::cross_entropy(
                    output.view({-1, vocab_size}),
                    targets.view(-1)
                );

                optimizer->zero_grad();
                scaler.scale(loss).backward();

                torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);

                // GradScaler handles checking for finite gradients, optimizer step, and scale updates
                scaler.step(*optimizer);

                batch_scheduler.step();

                if (rank == 0 && step % 10 == 0) {
                    std::cout << "Epoch: " << epoch
                              << " | Step: " << step
                              << " | Loss: " << loss.item<float>()
                              << " | Batch Size: " << batch_size
                              << std::endl;
                }
                step++;
                if (step >= total_steps_for_test) break;
            }
        }

        if (rank == 0) {
            std::cout << "Training finished successfully." << std::endl;
        }

    } catch (const c10::Error& e) {
        std::cerr << "Caught a c10::Error on rank " << rank << ": " << e.msg() << std::endl;
        if (is_distributed) torch::distributed::destroy_process_group();
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Caught an exception on rank " << rank << ": " << e.what() << std::endl;
        if (is_distributed) torch::distributed::destroy_process_group();
        return -1;
    }

    if (is_distributed) {
        torch::distributed::destroy_process_group();
    }

    return 0;
}
