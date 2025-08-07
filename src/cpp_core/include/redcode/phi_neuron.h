#pragma once

#include <type_traits>

namespace redcode {

// A fast approximation of the hyperbolic tangent function, based on the formula
// referenced in the project documentation's AVX2 implementation details.
// tanh(x) ≈ x * (27 + x²) / (27 + 9*x²)
template<typename T>
inline T fast_tanh(T x) noexcept {
    static_assert(std::is_floating_point_v<T>, "fast_tanh requires a floating-point type.");
    const T x2 = x * x;
    return x * (T(27) + x2) / (T(27) + T(9) * x2);
}

// PhiNeuron is the core computational unit of RedcodE, based on the golden ratio.
// This is the initial CPU-only implementation based on the design document.
// CUDA-specific keywords and alignment specifiers are omitted for now.
struct PhiNeuron {
    static constexpr float PHI = 0.618033988749f;

    template<typename T>
    inline T forward(T x) const noexcept {
        const T excite = (x > T(0)) ? x : T(0);      // ReLU activation
        const T inhibit = fast_tanh(x);
        return PHI * excite + (T(1) - PHI) * inhibit;
    }

    template<typename T>
    inline T backward(T x, T grad_out) const noexcept {
        const T d_excite = (x > T(0)) ? PHI : T(0);
        // The derivative of tanh(x) is 1 - tanh^2(x).
        // We reuse fast_tanh(x) here as intended by the design document.
        const T tanh_x = fast_tanh(x);
        const T d_inhibit = (T(1) - PHI) * (T(1) - tanh_x * tanh_x);
        return (d_excite + d_inhibit) * grad_out;
    }
};

} // namespace redcode
