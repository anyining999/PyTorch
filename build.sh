#!/bin/bash

# This script builds the entire RedcodE project.
# It is designed to be run inside the Docker container defined in Dockerfile.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting RedcodE Project Build ---"

# 1. Create a clean build directory
echo "--- Preparing build directory... ---"
rm -rf build
mkdir build
cd build

# 2. Configure the project with CMake
# The Docker environment has all dependencies (CUDA, Rust, etc.) in standard paths,
# so CMake should find everything automatically.
echo "--- Configuring CMake... ---"
cmake ..

# 3. Build the project
# This will compile all C++, CUDA, and Rust code. The --verbose flag provides detailed output.
echo "--- Building Project... ---"
cmake --build . --verbose

# 4. Run the tests
# This will execute the C++ test suite, which should include the CUDA tests.
echo "--- Running Tests... ---"
ctest --verbose

echo "--- RedcodE Project Build and Test Successful ---"
