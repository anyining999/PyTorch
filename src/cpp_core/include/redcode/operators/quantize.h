#pragma once

#include "redcode/red_tensor.h"
#include "redcode/quantization/common.h"
#include "redcode/quantization/packing.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>

namespace redcode {
namespace operators {

/**
 * @brief Performs symmetric per-tensor quantization on a float tensor to int8.
 *
 * This function finds the maximum absolute value in the input tensor to determine a
 * uniform scaling factor. Each float value is then scaled, rounded, and clamped
 * to the int8 range [-127, 127].
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
 * @brief Performs symmetric per-tensor dequantization from int8.
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

/**
 * @brief Performs symmetric per-tensor quantization on a float tensor to packed int4.
 *
 * @param input_tensor The input tensor with float values.
 * @return A pair containing the new packed quantized tensor (uint8_t) and the quantization parameters.
 */
inline std::pair<RedTensor<uint8_t>, quantization::QuantizationParams>
quantize_symmetric_int4(const RedTensor<float>& input_tensor) {
    if (input_tensor.size() == 0) {
        return {RedTensor<uint8_t>(), quantization::QuantizationParams()};
    }

    // 1. Find absolute maximum value
    float abs_max = 0.0f;
    for (size_t i = 0; i < input_tensor.size(); ++i) {
        abs_max = std::max(abs_max, std::abs(input_tensor[i]));
    }

    // 2. Calculate scale for int4 [-8, 7] range
    quantization::QuantizationParams params;
    const float q_max = 7.0f;
    params.scale = (abs_max == 0) ? 1.0f : abs_max / q_max;
    params.zero_point = 0;

    // 3. Create output tensor and pack values
    size_t packed_size = (input_tensor.size() + 1) / 2;
    RedTensor<uint8_t> output_tensor({packed_size});

    for (size_t i = 0; i < packed_size; ++i) {
        size_t float_idx1 = i * 2;
        size_t float_idx2 = float_idx1 + 1;

        // Quantize the first value (high nibble)
        float scaled_val1 = input_tensor[float_idx1] / params.scale;
        int8_t quantized_val1 = static_cast<int8_t>(std::round(scaled_val1));
        quantized_val1 = std::max((int8_t)-8, std::min((int8_t)7, quantized_val1));

        // Quantize the second value (low nibble), handle odd-sized tensors
        int8_t quantized_val2 = 0;
        if (float_idx2 < input_tensor.size()) {
            float scaled_val2 = input_tensor[float_idx2] / params.scale;
            quantized_val2 = static_cast<int8_t>(std::round(scaled_val2));
            quantized_val2 = std::max((int8_t)-8, std::min((int8_t)7, quantized_val2));
        }

        output_tensor[i] = quantization::pack_int4(quantized_val1, quantized_val2);
    }

    return {std::move(output_tensor), params};
}

/**
 * @brief Performs symmetric per-tensor dequantization from packed int4.
 *
 * @param packed_tensor The quantized input tensor with packed int4 data (uint8_t).
 * @param original_shape The original shape of the tensor before quantization.
 * @param params The quantization parameters used for the original quantization.
 * @return The dequantized float tensor.
 */
inline RedTensor<float>
dequantize_symmetric_int4(const RedTensor<uint8_t>& packed_tensor,
                          const std::vector<size_t>& original_shape,
                          const quantization::QuantizationParams& params) {
    RedTensor<float> output_tensor(original_shape);
    if (output_tensor.size() == 0) {
        return output_tensor;
    }

    for (size_t i = 0; i < packed_tensor.size(); ++i) {
        auto [high_val, low_val] = quantization::unpack_int4(packed_tensor[i]);

        size_t float_idx1 = i * 2;
        size_t float_idx2 = float_idx1 + 1;

        if (float_idx1 < output_tensor.size()) {
            output_tensor[float_idx1] = static_cast<float>(high_val) * params.scale;
        }
        if (float_idx2 < output_tensor.size()) {
            output_tensor[float_idx2] = static_cast<float>(low_val) * params.scale;
        }
    }

    return output_tensor;
}

} // namespace operators
} // namespace redcode
