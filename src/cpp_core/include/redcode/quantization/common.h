#pragma once

#include <cstdint>

namespace redcode {
namespace quantization {

/**
 * @struct QuantizationParams
 * @brief Holds the parameters required for quantizing and dequantizing a tensor.
 *
 * This struct is designed to be extensible to support various quantization schemes.
 */
struct QuantizationParams {
    /// @brief The scaling factor used to map floating-point values to quantized values.
    /// For symmetric quantization, dequantized_value = quantized_value * scale.
    float scale = 1.0f;

    /// @brief The quantized value that corresponds to a floating-point value of 0.
    /// For symmetric quantization, this is typically 0. For asymmetric, it may vary.
    int32_t zero_point = 0;
};

} // namespace quantization
} // namespace redcode
