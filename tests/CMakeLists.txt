# Define the test executable from the source files
add_executable(redcode_tests
    test_phi_neuron.cpp
    test_ffi_bridge.cpp
)

# Link the test executable against GoogleTest, our core library, and the Rust scheduler library
target_link_libraries(redcode_tests
    PRIVATE
    redcode_core
    GTest::gtest_main
    rust_scheduler
)

# Discover tests from the executable so they can be run by CTest
include(GoogleTest)
gtest_discover_tests(redcode_tests)
