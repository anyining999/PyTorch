#pragma once

#include <cstdint>
#include <cmath>
#include <limits>
#include <cstring> // For memcpy
#include <stdexcept>

namespace redcode {
namespace quantization {

/**
 * @class fp8_e4m3_t
 * @brief Represents an 8-bit E4M3 floating-point number.
 *
 * E4M3 (4-bit exponent, 3-bit mantissa) is a format often used in AI.
 * This class handles the bit-level conversions to and from a standard 32-bit float.
 * Note: This is a simplified implementation focusing on common cases. It does not
 * handle subnormal-to-subnormal conversions and uses round-to-zero.
 */
class fp8_e4m3_t {
public:
    uint8_t bits;

    /// Default constructor initializes to zero.
    fp8_e4m3_t() : bits(0) {}

    /// Conversion from float. This is a complex operation involving bit manipulation.
    explicit fp8_e4m3_t(float f) {
        union {
            float f;
            uint32_t u;
        } converter = {f};

        uint32_t sign = (converter.u >> 31);

        // Handle NaN and Inf
        if (std::isnan(f)) {
            bits = 0b01111101; // Canonical quiet NaN
            return;
        }
        if (std::isinf(f)) {
            bits = (sign << 7) | 0b01111000; // Sign | 1111 (exp) | 000 (mant)
            return;
        }

        int32_t exp = ((converter.u >> 23) & 0xFF) - 127;
        uint32_t mant = converter.u & 0x7FFFFF;

        // E4M3 constants
        const int32_t fp8_exp_bias = 7;
        int32_t fp8_exp = exp + fp8_exp_bias;

        // Handle overflow and underflow
        if (fp8_exp >= 15) { // Overflow to Inf
            bits = (sign << 7) | 0b01111000;
            return;
        }
        if (fp8_exp <= 0) { // Underflow to zero (simplified)
            bits = (sign << 7);
            return;
        }

        uint32_t fp8_mant = mant >> 20;

        bits = (sign << 7) | (static_cast<uint8_t>(fp8_exp) << 3) | static_cast<uint8_t>(fp8_mant);
    }

    /// Conversion to float.
    explicit operator float() const {
        uint8_t sign = (bits >> 7);
        uint8_t exp_bits = (bits >> 3) & 0x0F;
        uint8_t mant_bits = bits & 0x07;

        // Handle special values based on E4M3 standard
        if (exp_bits == 15) {
            if (mant_bits != 0) return std::numeric_limits<float>::quiet_NaN();
            return (sign) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        }

        // Handle zero
        if (exp_bits == 0 && mant_bits == 0) {
            return (sign) ? -0.0f : 0.0f;
        }

        // Handle normal numbers (simplified, no subnormals)
        int32_t float_exp = exp_bits - 7 + 127;
        uint32_t float_mant = mant_bits << 20;

        union {
            uint32_t u;
            float f;
        } converter = {(sign << 31) | (static_cast<uint32_t>(float_exp) << 23) | float_mant};

        return converter.f;
    }
};

} // namespace quantization
} // namespace redcode
