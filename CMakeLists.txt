cmake_minimum_required(VERSION 3.14)
project(RedcodE LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- GoogleTest ---
# Add GoogleTest to the project
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --- Corrosion (Rust/CMake Integration) ---
include(FetchContent)
FetchContent_Declare(
  Corrosion
  GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
  GIT_TAG v0.4.5 # Using a specific, stable version
)
FetchContent_MakeAvailable(Corrosion)

# Import the Rust scheduler crate.
corrosion_import_crate(MANIFEST_PATH ${CMAKE_SOURCE_DIR}/src/rust_scheduler/Cargo.toml)

add_subdirectory(src/cpp_core)

# --- Testing ---
# Enable testing and add the tests directory
enable_testing()
add_subdirectory(tests)
