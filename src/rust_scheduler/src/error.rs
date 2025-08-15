use thiserror::Error;

// Placeholders for specific error types from the design document.
// These can be fleshed out with more details as the project grows.
#[derive(Error, Debug)]
#[error("a computation error occurred")]
pub struct ComputationError;

#[derive(Error, Debug)]
#[error("a resource allocation error occurred")]
pub struct ResourceError;

#[derive(Error, Debug)]
#[error("a network communication error occurred")]
pub struct NetworkError;

#[derive(Error, Debug)]
#[error("a quantization error occurred")]
pub struct QuantizationError;

/// The main error enum for the RedcodE scheduler, consolidating all possible errors.
#[derive(Error, Debug)]
pub enum RedError {
    #[error("Computation error: {0}")]
    Computation(#[from] ComputationError),

    #[error("Resource allocation error: {0}")]
    Resource(#[from] ResourceError),

    #[error("Network communication error: {0}")]
    Network(#[from] NetworkError),

    #[error("Quantization error: {0}")]
    Quantization(#[from] QuantizationError),

    // A variant for general I/O errors (e.g., reading a config file).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    // A variant for errors during serialization or deserialization.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// A specialized `Result` type for scheduler operations to simplify error handling.
pub type Result<T> = std::result::Result<T, RedError>;
