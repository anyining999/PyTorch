# Define the test executable from the source file
add_executable(redcode_tests
    test_phi_neuron.cpp
)

# Link the test executable against GoogleTest and our core library
target_link_libraries(redcode_tests
    PRIVATE
    redcode_core
    GTest::gtest_main
)

# Discover tests from the executable so they can be run by CTest
include(GoogleTest)
gtest_discover_tests(redcode_tests)
