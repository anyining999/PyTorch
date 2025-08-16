#include <gtest/gtest.h>
#include "redcode/phi_neuron.h"
#include <cmath>

// Test fixture for PhiNeuron tests to reuse the neuron object.
class PhiNeuronTest : public ::testing::Test {
protected:
    redcode::PhiNeuron neuron;
};

// Tests the forward pass of the PhiNeuron against its mathematical definition.
TEST_F(PhiNeuronTest, ForwardPassIsCorrect) {
    const float phi = redcode::PhiNeuron::PHI;

    // Test case 1: Positive input
    // For x > 0, forward(x) = PHI * x + (1 - PHI) * fast_tanh(x)
    const float input1 = 1.5f;
    const float tanh_val1 = redcode::fast_tanh(input1);
    const float expected1 = phi * input1 + (1.0f - phi) * tanh_val1;
    EXPECT_NEAR(neuron.forward(input1), expected1, 1e-6);

    // Test case 2: Negative input
    // For x < 0, forward(x) = (1 - PHI) * fast_tanh(x)
    const float input2 = -2.0f;
    const float tanh_val2 = redcode::fast_tanh(input2);
    const float expected2 = (1.0f - phi) * tanh_val2;
    EXPECT_NEAR(neuron.forward(input2), expected2, 1e-6);

    // Test case 3: Zero input
    // For x = 0, forward(x) = 0
    const float input3 = 0.0f;
    EXPECT_NEAR(neuron.forward(input3), 0.0f, 1e-6);
}

// Tests the backward pass of the PhiNeuron against its mathematical definition.
TEST_F(PhiNeuronTest, BackwardPassIsCorrect) {
    const float phi = redcode::PhiNeuron::PHI;
    const float grad_out = 1.0f; // Assume gradient output is 1 for simplicity

    // Test case 1: Positive input
    const float input1 = 1.5f;
    const float tanh_val1 = redcode::fast_tanh(input1);
    const float d_inhibit1 = (1.0f - phi) * (1.0f - tanh_val1 * tanh_val1);
    const float expected1 = (phi + d_inhibit1) * grad_out;
    EXPECT_NEAR(neuron.backward(input1, grad_out), expected1, 1e-6);

    // Test case 2: Negative input
    const float input2 = -2.0f;
    const float tanh_val2 = redcode::fast_tanh(input2);
    const float d_inhibit2 = (1.0f - phi) * (1.0f - tanh_val2 * tanh_val2);
    const float expected2 = (0.0f + d_inhibit2) * grad_out;
    EXPECT_NEAR(neuron.backward(input2, grad_out), expected2, 1e-6);

    // Test case 3: Zero input
    const float input3 = 0.0f;
    const float tanh_val3 = redcode::fast_tanh(input3);
    const float d_inhibit3 = (1.0f - phi) * (1.0f - tanh_val3 * tanh_val3);
    const float expected3 = (0.0f + d_inhibit3) * grad_out;
    EXPECT_NEAR(neuron.backward(input3, grad_out), expected3, 1e-6);
}

// Test fixture for edge cases to align with "Super Engineer" quality standards.
class PhiNeuronEdgeCases : public ::testing::Test {
protected:
    redcode::PhiNeuron neuron;
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float phi = redcode::PhiNeuron::PHI;
};

// Tests the forward and backward passes with non-finite inputs (inf, -inf, NaN).
TEST_F(PhiNeuronEdgeCases, HandlesNonFiniteInputs) {
    // Test forward pass with infinity
    EXPECT_TRUE(std::isinf(neuron.forward(inf)) && neuron.forward(inf) > 0);

    // Test forward pass with negative infinity
    const float expected_neg_inf = (1.0f - phi) * -1.0f;
    EXPECT_NEAR(neuron.forward(-inf), expected_neg_inf, 1e-6);

    // Test forward pass with NaN
    EXPECT_TRUE(std::isnan(neuron.forward(nan)));

    // Test backward pass with non-finite inputs
    const float grad_out = 1.0f;
    EXPECT_TRUE(std::isnan(neuron.backward(nan, grad_out)));
    EXPECT_NEAR(neuron.backward(inf, grad_out), phi * grad_out, 1e-6);
    EXPECT_NEAR(neuron.backward(-inf, grad_out), 0.0f, 1e-6);
}
