#include <gtest/gtest.h>
#include "redcode/rust_bridge.h"

// This test suite verifies the connection between C++ and Rust.
TEST(FFIBridgeTest, CanCallRustFunction) {
    // This test calls the function that is implemented in the Rust crate.
    // The primary purpose is to ensure that the C++ code can successfully
    // link against and call the Rust library.
    // The Rust function will print a message to the console.
    // We use ASSERT_NO_THROW to confirm the function call itself doesn't cause a crash.

    // Redirect stdout to a buffer to check the output could be a future improvement.
    ASSERT_NO_THROW(hello_from_rust());
}
