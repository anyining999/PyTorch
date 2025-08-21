#include "transformer_lib.h"
#include <torch/torch.h>
#include <cmath>
#include <iostream>

// --- DynamicBatchScheduler Implementation ---
DynamicBatchScheduler::DynamicBatchScheduler(int min_bs, int max_bs)
    : min_bs_(min_bs), max_bs_(max_bs), phase_(0.0f) {
    // A reasonable step to cycle through sizes over a few hundred steps
    phase_step_ = 2 * M_PI / 500.0f;
}

int DynamicBatchScheduler::get_batch_size() const {
    float ratio = (1.0f + std::cos(phase_)) / 2.0f; // Varies between 0 and 1
    return min_bs_ + static_cast<int>(ratio * (max_bs_ - min_bs_));
}

void DynamicBatchScheduler::step() {
    phase_ = std::fmod(phase_ + phase_step_, 2 * M_PI);
}


// --- FlashAttention Implementation ---
torch::Tensor flash_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // NOTE: This is a placeholder for a true FlashAttention-2 integration.
    // A real implementation would require linking against the FlashAttention-2
    // library and calling its custom C++ operator here. The process involves:
    // 1. Compiling the FlashAttention-2 library from its source.
    // 2. Creating a C++ wrapper or custom operator recognized by LibTorch.
    // 3. Modifying the CMakeLists.txt to find and link the compiled library.
    //
    // For demonstration purposes, we use a standard scaled dot-product attention.
    return torch::nn::functional::scaled_dot_product_attention(Q, K, V);
}


// --- TransformerBlock Implementation ---
TransformerBlock::TransformerBlock(int d_model, int n_heads) :
    attn(register_module("attn",
        torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(d_model, n_heads)
                .bias(false)
        )
    )),
    ff(register_module("ff", torch::nn::Sequential(
        torch::nn::Linear(d_model, 4 * d_model),
        torch::nn::GELU(),
        torch::nn::Linear(4 * d_model, d_model)
    ))),
    norm1(register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})))),
    norm2(register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}))))
{}

torch::Tensor TransformerBlock::forward(torch::Tensor x) {
    auto attn_out = flash_attention(x, x, x);
    x = norm1(x + attn_out);

    auto ff_out = ff->forward(x);
    x = norm2(x + ff_out);

    return x;
}


// --- Transformer Implementation ---
Transformer::Transformer(int vocab_size, int d_model, int n_layers, int n_heads) {
    embedding = register_module("embedding",
        torch::nn::Embedding(vocab_size, d_model));

    pos_encoding = register_buffer("pos_encoding",
        create_positional_encoding(512, d_model));

    layers = register_module("layers", torch::nn::ModuleList());
    for (int i = 0; i < n_layers; ++i) {
        layers->push_back(register_module("layer" + std::to_string(i),
            std::make_shared<TransformerBlock>(d_model, n_heads)));
    }

    output = register_module("output",
        torch::nn::Linear(d_model, vocab_size));
}

torch::Tensor Transformer::forward(torch::Tensor input) {
    auto seq_len = input.size(1);
    auto x = embedding(input);
    x = x + pos_encoding.narrow(0, 0, seq_len);

    for (auto& layer : *layers) {
        x = layer->as<TransformerBlock>()->forward(x);
    }

    return output(x);
}

torch::Tensor Transformer::create_positional_encoding(int max_len, int d_model) {
    auto pe = torch::zeros({max_len, d_model});
    auto position = torch::arange(0, max_len).unsqueeze(1);
    auto div_term = torch::exp(torch::arange(0, d_model, 2) *
                             (-std::log(10000.0) / d_model));

    pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)},
                  torch::sin(position * div_term));
    pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)},
                  torch::cos(position * div_term));

    return pe;
}


// --- GradScaler Implementation ---
GradScaler::GradScaler(double init_scale, double growth_factor, double backoff_factor, int growth_interval)
    : scale_(init_scale),
      growth_factor_(growth_factor),
      backoff_factor_(backoff_factor),
      growth_interval_(growth_interval),
      successful_steps_(0) {}

torch::Tensor GradScaler::scale(torch::Tensor loss) {
    return loss * scale_;
}

void GradScaler::unscale_(torch::optim::Optimizer& optimizer) {
    for (auto& group : optimizer.param_groups()) {
        for (auto& p : group.params()) {
            if (p.grad().defined()) {
                p.grad().div_(scale_);
            }
        }
    }
}

bool GradScaler::has_finite_grads(const torch::optim::Optimizer& optimizer) {
    for (const auto& group : optimizer.param_groups()) {
        for (const auto& p : group.params()) {
            if (p.grad().defined()) {
                if (!torch::isfinite(p.grad()).all().item<bool>()) {
                    return false;
                }
            }
        }
    }
    return true;
}

void GradScaler::step(torch::optim::Optimizer& optimizer) {
    unscale_(optimizer);

    if (has_finite_grads(optimizer)) {
        optimizer.step();
        successful_steps_++;
        if (successful_steps_ >= growth_interval_) {
            scale_ *= growth_factor_;
            successful_steps_ = 0;
        }
    } else {
        // Gradients were not finite. Skip optimizer step and reduce scale.
        successful_steps_ = 0;
        scale_ *= backoff_factor_;
    }
}
