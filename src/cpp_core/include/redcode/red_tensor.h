#pragma once

#include <vector>
#include <numeric>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace redcode {

/// @brief Defines the target device for tensor operations and data storage.
enum class Device {
    CPU,
    GPU
};

/**
 * @class RedTensor
 * @brief A foundational N-dimensional array (tensor) for the RedcodE framework.
 *
 * This class is designed with RAII (Resource Acquisition Is Initialization) principles
 * to ensure robust memory management. It can either own its memory, allocating and
 * deallocating it automatically, or act as a non-owning view of external data.
 * All accessors perform strict bounds checking to meet the "Zero Defect" standard.
 *
 * @tparam T The arithmetic type of the elements in the tensor (e.g., float, int).
 */
template<typename T>
class RedTensor {
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type");

private:
    T* data_ = nullptr;
    std::vector<size_t> shape_;
    size_t size_ = 0;
    Device device_;
    bool owns_data_ = false;

    /// @brief Helper to calculate total number of elements from a shape vector.
    static size_t calculate_size(const std::vector<size_t>& shape) {
        if (shape.empty()) {
            return 0;
        }
        size_t size = 1;
        for (const auto& dim : shape) {
            if (dim == 0) return 0; // A dimension of zero means zero total size
            size *= dim;
        }
        return size;
    }

public:
    /**
     * @brief Default constructor. Creates an empty, non-owning tensor.
     * @example redcode::RedTensor<float> empty_tensor;
     */
    RedTensor() noexcept : device_(Device::CPU) {}

    /**
     * @brief Constructs an owning tensor with a given shape. Memory is allocated and managed by the tensor.
     * @param shape A vector defining the size of each dimension.
     * @param device The target device (CPU or GPU) where the data should reside.
     * @example redcode::RedTensor<float> tensor({2, 3, 4}); // Creates a 2x3x4 tensor
     */
    explicit RedTensor(const std::vector<size_t>& shape, Device device = Device::CPU)
        : shape_(shape), size_(calculate_size(shape)), device_(device), owns_data_(true) {
        if (size_ > 0) {
            data_ = new T[size_];
        }
    }

    /**
     * @brief Constructs a non-owning tensor that acts as a view of external data.
     * @warning The caller is responsible for managing the lifetime of the external data.
     * @param external_data A raw pointer to the external data buffer.
     * @param shape A vector defining the shape of the external data.
     * @param device The device where the external data resides.
     * @example float my_data[10]; redcode::RedTensor<float> view(my_data, {2, 5}, redcode::Device::CPU);
     */
    RedTensor(T* external_data, std::vector<size_t> shape, Device device) noexcept
        : data_(external_data), shape_(std::move(shape)), size_(calculate_size(shape_)), device_(device), owns_data_(false) {}

    /**
     * @brief Destructor. If the tensor is owning, it deallocates the managed memory.
     */
    ~RedTensor() {
        if (owns_data_ && data_ != nullptr) {
            delete[] data_;
        }
    }

    /**
     * @brief Copy constructor. Performs a deep copy of the other tensor.
     * The new tensor will always own its memory.
     */
    RedTensor(const RedTensor& other)
        : shape_(other.shape_), size_(other.size_), device_(other.device_), owns_data_(true) {
        if (size_ > 0) {
            data_ = new T[size_];
            std::copy(other.data_, other.data_ + size_, data_);
        } else {
            data_ = nullptr;
        }
    }

    /**
     * @brief Copy assignment operator. Performs a deep copy.
     */
    RedTensor& operator=(const RedTensor& other) {
        if (this == &other) {
            return *this;
        }
        if (owns_data_ && data_ != nullptr) {
            delete[] data_;
        }
        shape_ = other.shape_;
        size_ = other.size_;
        device_ = other.device_;
        owns_data_ = true; // A copy assignment always results in an owning tensor
        if (size_ > 0) {
            data_ = new T[size_];
            std::copy(other.data_, other.data_ + size_, data_);
        } else {
            data_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Move constructor. Takes ownership of the other tensor's resources.
     */
    RedTensor(RedTensor&& other) noexcept
        : data_(other.data_), shape_(std::move(other.shape_)), size_(other.size_), device_(other.device_), owns_data_(other.owns_data_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.owns_data_ = false;
        other.shape_.clear();
    }

    /**
     * @brief Move assignment operator. Takes ownership of the other tensor's resources.
     */
    RedTensor& operator=(RedTensor&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (owns_data_ && data_ != nullptr) {
            delete[] data_;
        }
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        device_ = other.device_;
        owns_data_ = other.owns_data_;

        other.data_ = nullptr;
        other.size_ = 0;
        other.owns_data_ = false;
        other.shape_.clear();

        return *this;
    }

    /**
     * @brief Provides 1D read/write access to tensor elements with bounds checking.
     * @param idx The flat 1D index of the element.
     * @throws std::out_of_range if idx is out of bounds.
     * @return A reference to the element at the specified index.
     */
    T& operator[](size_t idx) {
        if (idx >= size_) {
            throw std::out_of_range("1D index " + std::to_string(idx) + " is out of range for tensor with size " + std::to_string(size_));
        }
        return data_[idx];
    }

    /// @copydoc operator[](size_t)
    const T& operator[](size_t idx) const {
        if (idx >= size_) {
            throw std::out_of_range("1D index " + std::to_string(idx) + " is out of range for tensor with size " + std::to_string(size_));
        }
        return data_[idx];
    }

    /**
     * @brief Provides safe, multi-dimensional read/write access with bounds checking.
     * @param indices An initializer list of indices for each dimension.
     * @throws std::out_of_range if the number of indices is incorrect or any index is out of bounds.
     * @return A reference to the element at the specified coordinates.
     * @example tensor.at({row, col}) = 5;
     */
    T& at(std::initializer_list<size_t> indices) {
        return const_cast<T&>(static_cast<const RedTensor&>(*this).at(indices));
    }

    /// @copydoc at(std::initializer_list<size_t>)
    const T& at(std::initializer_list<size_t> indices) const {
        if (indices.size() != shape_.size()) {
            throw std::out_of_range("Incorrect number of indices provided. Expected " + std::to_string(shape_.size()) + ", got " + std::to_string(indices.size()));
        }

        size_t flat_index = 0;
        size_t stride = 1;

        auto idx_it = indices.end();
        for (int i = shape_.size() - 1; i >= 0; --i) {
            --idx_it;
            size_t dim_index = *idx_it;

            if (dim_index >= shape_[i]) {
                throw std::out_of_range("Index " + std::to_string(dim_index) +
                                      " is out of range for dimension " + std::to_string(i) +
                                      " with size " + std::to_string(shape_[i]));
            }
            flat_index += dim_index * stride;
            stride *= shape_[i];
        }
        return data_[flat_index];
    }

    // -- Accessors --

    /// @brief Returns a raw pointer to the tensor's data buffer.
    T* data() const noexcept { return data_; }
    /// @brief Returns the shape of the tensor as a vector of dimension sizes.
    const std::vector<size_t>& shape() const noexcept { return shape_; }
    /// @brief Returns the total number of elements in the tensor.
    size_t size() const noexcept { return size_; }
    /// @brief Returns the device where the tensor data is stored.
    Device device() const noexcept { return device_; }
    /// @brief Returns true if the tensor owns its data, false if it's a view.
    bool is_owning() const noexcept { return owns_data_; }
};

} // namespace redcode
