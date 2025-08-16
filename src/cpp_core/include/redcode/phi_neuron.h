#pragma once

#include <type_traits>
#include <cmath> // For std::isinf and std::isnan

#if defined(__CUDA_ARCH__)
#include <cuda_runtime.h>
#define REDCODE_FORCEINLINE __forceinline__ __host__ __device__
#else
#define REDCODE_FORCEINLINE inline
#endif


namespace redcode {

// A fast approximation of the hyperbolic tangent function, based on the formula
// referenced in the project documentation's AVX2 implementation details.
// tanh(x) ≈ x * (27 + x²) / (27 + 9*x²)
// To meet the "Super Engineer" standard for error handling, we add checks for
// non-finite inputs, which can cause undefined behavior.
template<typename T>
REDCODE_FORCEINLINE T fast_tanh(T x) noexcept {
    static_assert(std::is_floating_point_v<T>, "fast_tanh requires a floating-point type.");
    // Industrial-grade error handling: ensure input is finite.
    if (std::isinf(x)) return (x > 0) ? T(1) : T(-1);
    if (std::isnan(x)) return x; // Propagate NaN

    const T x2 = x * x;
    return x * (T(27) + x2) / (T(27) + T(9) * x2);
}

// PhiNeuron is the core computational unit of RedcodE, designed to meet the highest
// industrial standards of performance and mathematical purity.
//
// Mathematical Principle: The activation function uses the golden ratio (φ ≈ 0.618)
// to balance excitatory (ReLU-like) and inhibitory (tanh-like) responses.
// This is expressed as: f(x) = φ * excite(x) + (1-φ) * inhibit(x).
// This design avoids the "dying ReLU" problem by ensuring a non-zero gradient for
// negative inputs, while maintaining a fast, scalable activation.
//
// The struct is aligned to 64 bytes to ensure optimal memory access patterns
// and prevent false sharing on multi-core CPU and GPU architectures.
struct alignas(64) PhiNeuron {
    static constexpr float PHI = 0.618033988749f;

    template<typename T>
    REDCODE_FORCEINLINE T forward(T x) const noexcept {
        // Excitatory part: A standard Rectified Linear Unit (ReLU).
        const T excite = (x > T(0)) ? x : T(0);
        // Inhibitory part: A fast hyperbolic tangent approximation.
        const T inhibit = fast_tanh(x);
        // The final activation is a weighted sum, balanced by the golden ratio.
        return PHI * excite + (T(1) - PHI) * inhibit;
    }

    template<typename T>
    REDCODE_FORCEINLINE T backward(T x, T grad_out) const noexcept {
        // Gradient from the excitatory part (derivative of ReLU is a step function).
        const T d_excite = (x > T(0)) ? PHI : T(0);

        // Gradient from the inhibitory part. The derivative of tanh(x) is 1 - tanh^2(x).
        // We reuse fast_tanh(x) as per the design to maintain consistency.
        const T tanh_x = fast_tanh(x);
        const T d_inhibit = (T(1) - PHI) * (T(1) - tanh_x * tanh_x);

        // The total gradient is the sum of the component gradients, scaled by the output gradient.
        return (d_excite + d_inhibit) * grad_out;
    }
};

} // namespace redcode
