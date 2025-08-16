#pragma once

#include <cstdint>
#include <utility>
#include <tuple>

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

/**
 * @brief Packs four signed 2-bit integers into a single 8-bit unsigned integer.
 * @param v1 Most significant 2 bits. Must be in [-2, 1].
 * @param v2 Second most significant 2 bits. Must be in [-2, 1].
 * @param v3 Third most significant 2 bits. Must be in [-2, 1].
 * @param v4 Least significant 2 bits. Must be in [-2, 1].
 * @return A uint8_t containing the four packed int2 values.
 */
inline uint8_t pack_int2(int8_t v1, int8_t v2, int8_t v3, int8_t v4) {
    uint8_t b1 = (static_cast<uint8_t>(v1) & 0x03) << 6;
    uint8_t b2 = (static_cast<uint8_t>(v2) & 0x03) << 4;
    uint8_t b3 = (static_cast<uint8_t>(v3) & 0x03) << 2;
    uint8_t b4 = static_cast<uint8_t>(v4) & 0x03;
    return b1 | b2 | b3 | b4;
}

/**
 * @brief Unpacks four signed 2-bit integers from a single 8-bit unsigned integer.
 * @param packed_val The uint8_t containing the packed data.
 * @return A tuple containing the four unpacked int2 values in order from most to least significant.
 */
inline std::tuple<int8_t, int8_t, int8_t, int8_t> unpack_int2(uint8_t packed_val) {
    // Extract the 2-bit patterns
    uint8_t p1 = (packed_val >> 6) & 0x03;
    uint8_t p2 = (packed_val >> 4) & 0x03;
    uint8_t p3 = (packed_val >> 2) & 0x03;
    uint8_t p4 = packed_val & 0x03;

    // Sign extend each 2-bit pattern to an 8-bit signed integer
    // For 2's complement int2: 00=0, 01=1, 10=-2, 11=-1
    int8_t v1 = (p1 & 0x02) ? (p1 | 0xFC) : p1;
    int8_t v2 = (p2 & 0x02) ? (p2 | 0xFC) : p2;
    int8_t v3 = (p3 & 0x02) ? (p3 | 0xFC) : p3;
    int8_t v4 = (p4 & 0x02) ? (p4 | 0xFC) : p4;

    return {v1, v2, v3, v4};
}

} // namespace quantization
} // namespace redcode
