#include <gtest/gtest.h>
#include "redcode/operators/quantize.h"
#include "redcode/red_tensor.h"
#include <vector>
#include <cmath>
#include <limits>

// Test fixture for Quantization tests
class QuantizationTest : public ::testing::Test {};

// Helper function to calculate Root Mean Square Error (RMSE)
template<typename T>
float calculate_rmse(const redcode::RedTensor<T>& a, const redcode::RedTensor<T>& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<float>::max();
    }
    if (a.size() == 0) {
        return 0.0f;
    }
    float mse = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        mse += diff * diff;
    }
    mse /= a.size();
    return std::sqrt(mse);
}

// Tests the full symmetric quantization and dequantization cycle.
TEST_F(QuantizationTest, SymmetricQuantizationCycle) {
    // 1. Create original float tensor with values from -1.0 to 1.0
    redcode::RedTensor<float> original_tensor({2, 5});
    for (size_t i = 0; i < original_tensor.size(); ++i) {
        original_tensor[i] = (static_cast<float>(i) / 5.0f) - 1.0f;
    }

    // 2. Quantize the tensor
    auto [quantized_tensor, params] = redcode::operators::quantize_symmetric(original_tensor);

    // 3. Verify quantization parameters and output
    // abs_max of original tensor is 1.0. scale = 1.0 / 127.0
    EXPECT_NEAR(params.scale, 1.0f / 127.0f, 1e-6);
    EXPECT_EQ(params.zero_point, 0);
    EXPECT_EQ(quantized_tensor.shape(), original_tensor.shape());

    // Check a few quantized values
    // original_tensor[5] is 0.0, so quantized should be 0
    EXPECT_EQ(quantized_tensor[5], 0);
    // original_tensor[9] is 0.8, scaled is 0.8 / (1/127) = 101.6, rounded is 102
    EXPECT_EQ(quantized_tensor[9], 102);

    // 4. Dequantize the tensor
    redcode::RedTensor<float> dequantized_tensor = redcode::operators::dequantize_symmetric(quantized_tensor, params);

    // 5. Verify the error (RMSE) is within an acceptable tolerance
    float rmse = calculate_rmse(original_tensor, dequantized_tensor);
    // The tolerance is chosen based on the expected error from quantization.
    // For int8, this is a reasonable starting point.
    EXPECT_LT(rmse, 0.01f);
}

// Test quantization of a tensor with all zero values
TEST_F(QuantizationTest, AllZerosTensor) {
    redcode::RedTensor<float> original_tensor({2, 2});
    for (size_t i = 0; i < original_tensor.size(); ++i) {
        original_tensor[i] = 0.0f;
    }

    auto [quantized_tensor, params] = redcode::operators::quantize_symmetric(original_tensor);

    // Scale should be 1.0 to avoid division by zero, all values should be 0
    EXPECT_EQ(params.scale, 1.0f);
    for (size_t i = 0; i < quantized_tensor.size(); ++i) {
        EXPECT_EQ(quantized_tensor[i], 0);
    }

    redcode::RedTensor<float> dequantized_tensor = redcode::operators::dequantize_symmetric(quantized_tensor, params);
    float rmse = calculate_rmse(original_tensor, dequantized_tensor);
    EXPECT_EQ(rmse, 0.0f);
}
