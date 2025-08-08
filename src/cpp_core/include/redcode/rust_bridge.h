#pragma once

// This header provides the C++ declarations for functions implemented in Rust
// and exposed via FFI.

// The `extern "C"` linkage specification is crucial for ensuring that the C++
// compiler uses the C naming convention, which is necessary to link
// correctly with the Rust `#[no_mangle]` function.
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief A simple function implemented in Rust to verify the C++/Rust FFI bridge.
 *
 * This function prints a message to the console from the Rust side.
 */
void hello_from_rust();

/**
 * @brief Schedules a training job from a JSON configuration string.
 * @param config_json A null-terminated C string containing the JSON
 *                    representation of a TrainingConfig.
 * @return 0 on success, a negative integer on failure.
 */
int schedule_training_from_json(const char* config_json);

#ifdef __cplusplus
}
#endif
