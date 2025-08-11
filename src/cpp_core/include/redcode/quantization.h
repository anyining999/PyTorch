#pragma once

#include <cstddef>
#include <algorithm> // for std::clamp
#include <limits>    // for std::numeric_limits

namespace redcode {

// Provides static methods for performing quantization operations.
template<typename SrcT, typename DstT>
class QuantizationKernel {
public:
    /**
     * @brief Quantizes a batch of source data into a destination format.
     * @param src Pointer to the source data.
     * @param dst Pointer to the destination buffer.
     * @param count The number of elements to quantize.
     * @param scale The quantization scale factor.
     * @param zero_point The quantization zero point.
     */
    static void quantize_batch(const SrcT* src, DstT* dst,
                              size_t count, float scale, DstT zero_point);
};

} // namespace redcode
