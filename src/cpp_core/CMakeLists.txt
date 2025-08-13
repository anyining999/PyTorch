add_library(redcode_core INTERFACE)

# The core library needs to know where its own headers are.
target_include_directories(redcode_core
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
