add_library(redcode_core INTERFACE)

# The core library needs to know where its own headers are.
# The generated headers are now in a subdirectory of the include path.
target_include_directories(redcode_core
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
