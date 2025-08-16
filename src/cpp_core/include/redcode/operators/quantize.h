#pragma once

#include "redcode/red_tensor.h"
#include "redcode/quantization/common.h"
#include "redcode/quantization/packing.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace redcode {
namespace operators {

// INT8 Kernels
inline std::pair<RedTensor<int8_t>, quantization::QuantizationParams>
quantize_symmetric(const RedTensor<float>& input_tensor) {
    if (input_tensor.size() == 0) return {RedTensor<int8_t>(), quantization::QuantizationParams()};
    float abs_max = 0.0f;
    for (size_t i = 0; i < input_tensor.size(); ++i) abs_max = std::max(abs_max, std::abs(input_tensor[i]));
    quantization::QuantizationParams params;
    const float q_max = 127.0f;
    params.scale = (abs_max == 0) ? 1.0f : abs_max / q_max;
    RedTensor<int8_t> output_tensor(input_tensor.shape());
    for (size_t i = 0; i < input_tensor.size(); ++i) {
        float scaled_val = input_tensor[i] / params.scale;
        output_tensor[i] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, std::round(scaled_val))));
    }
    return {std::move(output_tensor), params};
}

inline RedTensor<float>
dequantize_symmetric(const RedTensor<int8_t>& input_tensor, const quantization::QuantizationParams& params) {
    if (input_tensor.size() == 0) return RedTensor<float>();
    RedTensor<float> output_tensor(input_tensor.shape());
    for (size_t i = 0; i < input_tensor.size(); ++i) {
        output_tensor[i] = static_cast<float>(input_tensor[i]) * params.scale;
    }
    return output_tensor;
}

// INT4 Kernels
inline std::pair<RedTensor<uint8_t>, quantization::QuantizationParams>
quantize_symmetric_int4(const RedTensor<float>& input_tensor) {
    if (input_tensor.size() == 0) return {RedTensor<uint8_t>(), quantization::QuantizationParams()};
    float abs_max = 0.0f;
    for (size_t i = 0; i < input_tensor.size(); ++i) abs_max = std::max(abs_max, std::abs(input_tensor[i]));
    quantization::QuantizationParams params;
    const float q_max = 7.0f;
    params.scale = (abs_max == 0) ? 1.0f : abs_max / q_max;
    size_t packed_size = (input_tensor.size() + 1) / 2;
    RedTensor<uint8_t> output_tensor({packed_size});
    for (size_t i = 0; i < packed_size; ++i) {
        size_t idx1 = i * 2;
        int8_t q1 = static_cast<int8_t>(std::max(-8.0f, std::min(7.0f, std::round(input_tensor[idx1] / params.scale))));
        int8_t q2 = 0;
        if (idx1 + 1 < input_tensor.size()) {
            q2 = static_cast<int8_t>(std::max(-8.0f, std::min(7.0f, std::round(input_tensor[idx1 + 1] / params.scale))));
        }
        output_tensor[i] = quantization::pack_int4(q1, q2);
    }
    return {std::move(output_tensor), params};
}

inline RedTensor<float>
dequantize_symmetric_int4(const RedTensor<uint8_t>& packed_tensor,
                          const std::vector<size_t>& original_shape,
                          const quantization::QuantizationParams& params) {
    RedTensor<float> output_tensor(original_shape);
    if (output_tensor.size() == 0) return output_tensor;
    for (size_t i = 0; i < packed_tensor.size(); ++i) {
        auto [q1, q2] = quantization::unpack_int4(packed_tensor[i]);
        size_t idx1 = i * 2;
        if (idx1 < output_tensor.size()) output_tensor[idx1] = static_cast<float>(q1) * params.scale;
        if (idx1 + 1 < output_tensor.size()) output_tensor[idx1 + 1] = static_cast<float>(q2) * params.scale;
    }
    return output_tensor;
}

// INT2 Kernels
inline std::pair<RedTensor<uint8_t>, quantization::QuantizationParams>
quantize_symmetric_int2(const RedTensor<float>& input_tensor) {
    if (input_tensor.size() == 0) return {RedTensor<uint8_t>(), quantization::QuantizationParams()};
    float abs_max = 0.0f;
    for (size_t i = 0; i < input_tensor.size(); ++i) abs_max = std::max(abs_max, std::abs(input_tensor[i]));
    quantization::QuantizationParams params;
    const float q_range = 2.0f;
    params.scale = (abs_max == 0) ? 1.0f : abs_max / q_range;
    size_t packed_size = (input_tensor.size() + 3) / 4;
    RedTensor<uint8_t> output_tensor({packed_size});
    for (size_t i = 0; i < packed_size; ++i) {
        int8_t q_vals[4] = {0, 0, 0, 0};
        for (int j = 0; j < 4; ++j) {
            size_t float_idx = i * 4 + j;
            if (float_idx < input_tensor.size()) {
                q_vals[j] = static_cast<int8_t>(std::max(-2.0f, std::min(1.0f, std::round(input_tensor[float_idx] / params.scale))));
            }
        }
        output_tensor[i] = quantization::pack_int2(q_vals[0], q_vals[1], q_vals[2], q_vals[3]);
    }
    return {std::move(output_tensor), params};
}

inline RedTensor<float>
dequantize_symmetric_int2(const RedTensor<uint8_t>& packed_tensor,
                          const std::vector<size_t>& original_shape,
                          const quantization::QuantizationParams& params) {
    RedTensor<float> output_tensor(original_shape);
    if (output_tensor.size() == 0) return output_tensor;
    for (size_t i = 0; i < packed_tensor.size(); ++i) {
        auto [q1, q2, q3, q4] = quantization::unpack_int2(packed_tensor[i]);
        int8_t q_vals[4] = {q1, q2, q3, q4};
        for (int j = 0; j < 4; ++j) {
            size_t float_idx = i * 4 + j;
            if (float_idx < output_tensor.size()) {
                output_tensor[float_idx] = static_cast<float>(q_vals[j]) * params.scale;
            }
        }
    }
    return output_tensor;
}

} // namespace operators
} // namespace redcode
