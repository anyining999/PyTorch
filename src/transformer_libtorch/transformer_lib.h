#ifndef TRANSFORMER_LIB_H
#define TRANSFORMER_LIB_H

#include <torch/torch.h>
#include <cmath>

// 动态批处理调度器
class DynamicBatchScheduler {
public:
    DynamicBatchScheduler(int min_bs, int max_bs);
    int get_batch_size() const;
    void step();

private:
    int min_bs_;
    int max_bs_;
    float phase_step_;
    float phase_;
};

// 集成FlashAttention-2的高效注意力层
torch::Tensor flash_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

// Transformer块实现
struct TransformerBlock : torch::nn::Module {
    TransformerBlock(int d_model, int n_heads);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::MultiheadAttention attn;
    torch::nn::Sequential ff;
    torch::nn::LayerNorm norm1, norm2;
};

// 完整Transformer模型
struct Transformer : torch::nn::Module {
    Transformer(int vocab_size, int d_model, int n_layers, int n_heads);
    torch::Tensor forward(torch::Tensor input);
    static torch::Tensor create_positional_encoding(int max_len, int d_model);

    torch::nn::Embedding embedding;
    torch::Tensor pos_encoding;
    torch::nn::ModuleList layers;
    torch::nn::Linear output;
};

// 混合精度梯度缩放器
// 实现了动态损失缩放，以防止梯度下溢并处理梯度上溢（inf/NaN）。
class GradScaler {
public:
    GradScaler(
        double init_scale = 65536.0,
        double growth_factor = 2.0,
        double backoff_factor = 0.5,
        int growth_interval = 2000);

    // 将损失乘以当前的缩放因子
    torch::Tensor scale(torch::Tensor loss);

    // 在优化器更新参数之前，将梯度反缩放
    void unscale_(torch::optim::Optimizer& optimizer);

    // 更新参数并调整下一次迭代的缩放因子
    void step(torch::optim::Optimizer& optimizer);

private:
    // 检查所有梯度是否都是有限的（非inf, 非NaN）
    bool has_finite_grads(const torch::optim::Optimizer& optimizer);

    double scale_;
    double growth_factor_;
    double backoff_factor_;
    int growth_interval_;
    int successful_steps_;
};

#endif // TRANSFORMER_LIB_H
