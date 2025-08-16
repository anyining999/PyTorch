#include <gtest/gtest.h>
#include "redcode/red_tensor.h"
#include <vector>

// Test fixture for RedTensor tests
class RedTensorTest : public ::testing::Test {};

// Test default constructor
TEST_F(RedTensorTest, DefaultConstructor) {
    redcode::RedTensor<float> tensor;
    EXPECT_EQ(tensor.size(), 0);
    EXPECT_EQ(tensor.data(), nullptr);
    EXPECT_FALSE(tensor.is_owning());
    EXPECT_TRUE(tensor.shape().empty());
}

// Test owning constructor
TEST_F(RedTensorTest, OwningConstructor) {
    redcode::RedTensor<float> tensor({2, 3, 4});
    EXPECT_EQ(tensor.size(), 24);
    EXPECT_NE(tensor.data(), nullptr);
    EXPECT_TRUE(tensor.is_owning());
    ASSERT_EQ(tensor.shape().size(), 3);
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 3);
    EXPECT_EQ(tensor.shape()[2], 4);
}

// Test non-owning (view) constructor
TEST_F(RedTensorTest, NonOwningConstructor) {
    float data_buffer[10];
    redcode::RedTensor<float> tensor(data_buffer, {2, 5}, redcode::Device::CPU);
    EXPECT_EQ(tensor.size(), 10);
    EXPECT_EQ(tensor.data(), data_buffer);
    EXPECT_FALSE(tensor.is_owning());
}

// Test copy constructor for deep copy
TEST_F(RedTensorTest, CopyConstructor) {
    redcode::RedTensor<float> tensor_a({2, 2});
    for (size_t i = 0; i < tensor_a.size(); ++i) {
        tensor_a[i] = static_cast<float>(i);
    }

    redcode::RedTensor<float> tensor_b(tensor_a); // Copy constructor called

    EXPECT_TRUE(tensor_b.is_owning());
    EXPECT_NE(tensor_a.data(), tensor_b.data()); // Must have different data pointers
    EXPECT_EQ(tensor_a.size(), tensor_b.size());
    for (size_t i = 0; i < tensor_b.size(); ++i) {
        EXPECT_EQ(tensor_a[i], tensor_b[i]);
    }
}

// Test move constructor
TEST_F(RedTensorTest, MoveConstructor) {
    redcode::RedTensor<float> tensor_a({3, 3});
    float* original_data_ptr = tensor_a.data();
    size_t original_size = tensor_a.size();

    redcode::RedTensor<float> tensor_b(std::move(tensor_a)); // Move constructor

    // Check that tensor_b has taken ownership
    EXPECT_EQ(tensor_b.data(), original_data_ptr);
    EXPECT_EQ(tensor_b.size(), original_size);
    EXPECT_TRUE(tensor_b.is_owning());

    // Check that tensor_a is now empty and non-owning
    EXPECT_EQ(tensor_a.data(), nullptr);
    EXPECT_EQ(tensor_a.size(), 0);
    EXPECT_FALSE(tensor_a.is_owning());
}

// Test bounds checking for 1D operator[]
TEST_F(RedTensorTest, AccessOperator1DBounds) {
    redcode::RedTensor<float> tensor({5});
    EXPECT_NO_THROW(tensor[4] = 1.0f);
    EXPECT_THROW(tensor[5], std::out_of_range);
}

// Test bounds checking for multi-dimensional at()
TEST_F(RedTensorTest, AccessAtMultiDimBounds) {
    redcode::RedTensor<int> tensor({2, 3}); // 2 rows, 3 columns

    // Valid access
    EXPECT_NO_THROW(tensor.at({1, 2}) = 42);
    EXPECT_EQ(tensor.at({1, 2}), 42);

    // Invalid number of indices
    EXPECT_THROW(tensor.at({1}), std::out_of_range);
    EXPECT_THROW(tensor.at({1, 2, 0}), std::out_of_range);

    // Index out of bounds for a dimension
    EXPECT_THROW(tensor.at({2, 0}), std::out_of_range); // Row index too high
    EXPECT_THROW(tensor.at({0, 3}), std::out_of_range); // Col index too high
}
