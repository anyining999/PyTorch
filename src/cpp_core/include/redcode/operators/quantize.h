#pragma once

#include "redcode/red_tensor.h"
#include "redcode/quantization/common.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>

namespace redcode {
namespace operators {

/**
 * @brief Performs symmetric per-tensor quantization on a float tensor.
 *
 * This function finds the maximum absolute value in the input tensor to determine a
 * uniform scaling factor. Each float value is then scaled, rounded, and clamped
 * to the target integer type's range.
 *
 * @param input_tensor The input tensor with float values.
 * @return A pair containing the new quantized tensor (int8_t) and the quantization parameters (scale).
 */
inline std::pair<RedTensor<int8_t>, quantization::QuantizationParams>
quantize_symmetric(const RedTensor<float>& input_tensor) {
    if (input_tensor.size() == 0) {
        return {RedTensor<int8_t>(), quantization::QuantizationParams()};
    }

    // 1. Find absolute maximum value
    float abs_max = 0.0f;
    for (size_t i = 0; i < input_tensor.size(); ++i) {
        abs_max = std::max(abs_max, std::abs(input_tensor[i]));
    }

    // 2. Calculate scale
    quantization::QuantizationParams params;
    const float q_max = 127.0f;
    params.scale = (abs_max == 0) ? 1.0f : abs_max / q_max;
    params.zero_point = 0;

    // 3. Create output tensor and quantize values
    RedTensor<int8_t> output_tensor(input_tensor.shape());
    for (size_t i = 0; i < input_tensor.size(); ++i) {
        float scaled_val = input_tensor[i] / params.scale;
        float rounded_val = std::round(scaled_val);
        // Clamp the value to the int8_t range [-127, 127]
        output_tensor[i] = static_cast<int8_t>(
            std::max(-127.0f, std::min(127.0f, rounded_val))
        );
    }

    return {std::move(output_tensor), params};
}

/**
 * @brief Performs symmetric per-tensor dequantization.
 *
 * Converts a quantized integer tensor back to a float tensor using the provided
 * scaling factor.
 *
 * @param input_tensor The quantized input tensor (int8_t).
 * @param params The quantization parameters (scale) used for the original quantization.
 * @return The dequantized float tensor.
 */
inline RedTensor<float>
dequantize_symmetric(const RedTensor<int8_t>& input_tensor, const quantization::QuantizationParams& params) {
    if (input_tensor.size() == 0) {
        return RedTensor<float>();
    }

    RedTensor<float> output_tensor(input_tensor.shape());
    for (size_t i = 0; i < input_tensor.size(); ++i) {
        output_tensor[i] = static_cast<float>(input_tensor[i]) * params.scale;
    }

    return output_tensor;
}

} // namespace operators
} // namespace redcode
