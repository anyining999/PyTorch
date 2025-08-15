#pragma once

#include <vector>
#include <numeric> // for size_t
#include <type_traits>
#include <utility> // for std::move

namespace redcode {

// Defines the target device for tensor operations and data storage.
enum class Device {
    CPU,
    GPU
};

// A foundational tensor class for the RedcodE framework.
// This initial implementation provides the core data structure.
// Advanced features like SIMD optimizations and fused kernels will be added later.
template<typename T>
class RedTensor {
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type");

public:
    // A zero-copy constructor that takes a raw pointer to existing data.
    RedTensor(T* data, std::vector<size_t> shape, Device device) noexcept
        : data_(data), shape_(std::move(shape)), device_(device) {}

    // Provides read-only access to tensor elements.
    const T& operator[](size_t idx) const noexcept {
        return data_[idx];
    }

    // Provides read/write access to tensor elements.
    T& operator[](size_t idx) noexcept {
        return data_[idx];
    }

    // -- Accessors --

    T* data() const noexcept { return data_; }
    const std::vector<size_t>& shape() const noexcept { return shape_; }
    Device device() const noexcept { return device_; }

private:
    T* data_;
    std::vector<size_t> shape_;
    Device device_;
};

} // namespace redcode
