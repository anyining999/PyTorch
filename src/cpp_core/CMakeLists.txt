# Define the library as STATIC since it now has source files to compile.
add_library(redcode_core STATIC
    src/phi_neuron.cu
    src/quantization.cu
)

# The core library needs to know where its own headers are.
target_include_directories(redcode_core
    PUBLIC # Use PUBLIC so consumers of this library also get the include path.
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
