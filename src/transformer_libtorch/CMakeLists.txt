# This CMake script builds the Transformer C++ library and a standalone executable.

# --- LibTorch Dependency ---
# Find the LibTorch package. This is a critical dependency.
# The user must point CMake to their LibTorch installation via CMAKE_PREFIX_PATH.
find_package(Torch REQUIRED)

# --- Transformer Library ---
# Define a static library from the core transformer implementation files.
# This library can be linked against by other parts of the project (e.g., tests).
add_library(transformer_libtorch_lib STATIC
    transformer_lib.cpp
    transformer_lib.h
)

# The library needs to know where its own headers are.
target_include_directories(transformer_libtorch_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link the library against LibTorch.
# PUBLIC ensures that consumers of our library also get the Torch dependencies.
target_link_libraries(transformer_libtorch_lib PUBLIC "${TORCH_LIBRARIES}")


# --- Transformer Executable ---
# Define a standalone executable from the main.cpp source file.
# This allows us to run the training loop directly for validation.
add_executable(transformer_main main.cpp)

# Link the executable against our transformer library.
# This transitively links against LibTorch as well.
target_link_libraries(transformer_main PRIVATE transformer_libtorch_lib)
