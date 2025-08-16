#pragma once

#include <cstdint>
#include <utility>

namespace redcode {
namespace quantization {

/**
 * @brief Packs two signed 4-bit integers into a single 8-bit unsigned integer.
 * @param high The 4-bit integer to be stored in the most significant 4 bits. Must be in [-8, 7].
 * @param low The 4-bit integer to be stored in the least significant 4 bits. Must be in [-8, 7].
 * @return A uint8_t containing the two packed int4 values.
 */
inline uint8_t pack_int4(int8_t high, int8_t low) {
    // Mask to get the 4-bit representation and cast to unsigned to avoid sign extension issues
    uint8_t high_bits = (static_cast<uint8_t>(high) & 0x0F) << 4;
    uint8_t low_bits = static_cast<uint8_t>(low) & 0x0F;
    return high_bits | low_bits;
}

/**
 * @brief Unpacks two signed 4-bit integers from a single 8-bit unsigned integer.
 * @param packed_val The uint8_t containing the packed data.
 * @return A pair where the first element is the high int4 value and the second is the low int4 value.
 */
inline std::pair<int8_t, int8_t> unpack_int4(uint8_t packed_val) {
    // Extract the high and low 4-bit patterns
    uint8_t high_nibble = packed_val >> 4;
    uint8_t low_nibble = packed_val & 0x0F;

    // Perform sign extension to convert the 4-bit patterns to signed 8-bit integers.
    // If the 4th bit (sign bit) is 1, the number is negative.
    int8_t high_val = (high_nibble & 0x08) ? (high_nibble | 0xF0) : high_nibble;
    int8_t low_val = (low_nibble & 0x08) ? (low_nibble | 0xF0) : low_nibble;

    return {high_val, low_val};
}

} // namespace quantization
} // namespace redcode
