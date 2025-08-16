#include <gtest/gtest.h>
#include "redcode/quantization/fp8.h"
#include "redcode/operators/quantize.h"
#include "redcode/red_tensor.h"
#include <vector>
#include <cmath>
#include <limits>

// Test fixture for FP8 tests
class FP8Test : public ::testing::Test {};

// Test individual float to FP8 to float conversions
TEST_F(FP8Test, FloatToFp8Conversion) {
    using fp8 = redcode::quantization::fp8_e4m3_t;

    // Test zero
    EXPECT_EQ(static_cast<float>(fp8(0.0f)), 0.0f);
    EXPECT_EQ(static_cast<float>(fp8(-0.0f)), -0.0f);

    // Test a simple value
    EXPECT_NEAR(static_cast<float>(fp8(1.0f)), 1.0f, 1e-1);
    EXPECT_NEAR(static_cast<float>(fp8(-2.5f)), -2.5f, 1e-1);

    // Test Infinity
    EXPECT_TRUE(std::isinf(static_cast<float>(fp8(std::numeric_limits<float>::infinity()))));
    EXPECT_TRUE(static_cast<float>(fp8(std::numeric_limits<float>::infinity())) > 0);
    EXPECT_TRUE(std::isinf(static_cast<float>(fp8(-std::numeric_limits<float>::infinity()))));
    EXPECT_TRUE(static_cast<float>(fp8(-std::numeric_limits<float>::infinity())) < 0);

    // Test NaN
    EXPECT_TRUE(std::isnan(static_cast<float>(fp8(std::numeric_limits<float>::quiet_NaN()))));
}

// Helper to calculate RMSE
static float calculate_fp8_rmse(const redcode::RedTensor<float>& a, const redcode::RedTensor<float>& b) {
    if (a.size() != b.size()) return std::numeric_limits<float>::max();
    if (a.size() == 0) return 0.0f;
    float mse = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        mse += diff * diff;
    }
    mse /= a.size();
    return std::sqrt(mse);
}


// Tests the tensor-level round trip conversion
TEST_F(FP8Test, TensorRoundTrip) {
    redcode::RedTensor<float> original_tensor({2, 4});
    for(size_t i = 0; i < original_tensor.size(); ++i) {
        original_tensor[i] = static_cast<float>(i) - 4.0f;
    }

    auto fp8_tensor = redcode::operators::convert_to_fp8_e4m3(original_tensor);
    auto dequantized_tensor = redcode::operators::convert_from_fp8_e4m3(fp8_tensor);

    EXPECT_EQ(original_tensor.shape(), dequantized_tensor.shape());

    float rmse = calculate_fp8_rmse(original_tensor, dequantized_tensor);
    // FP8 has low precision, so the tolerance is relatively high.
    EXPECT_LT(rmse, 0.5f);
}
