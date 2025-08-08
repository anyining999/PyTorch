use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// A placeholder to represent a C++ RedTensor on the Rust side.
// In a real implementation, this would likely be a handle or an ID
// that can be passed across the FFI boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustRedTensor {
    // For now, just a placeholder ID and shape
    pub id: u64,
    pub shape: Vec<usize>,
}

pub type ModelId = String;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelMetadata {
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub learning_rate: f32,
}

// Defines the layers of a model, based on the design document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Layer {
    PhiLinear {
        input_dim: usize,
        output_dim: usize,
        weights: RustRedTensor,
        bias: Option<RustRedTensor>,
    },
    PhiConv2d {
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        weights: RustRedTensor,
        bias: Option<RustRedTensor>,
    },
    PhiActivation,
    BatchNorm, // Simplified from doc
    Dropout { rate: f32 },
}

// The unified model representation in Rust.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedModel {
    pub id: ModelId,
    pub layers: Vec<Layer>,
    pub config: ModelConfig,
    pub metadata: ModelMetadata,
}

// Placeholders for configuration structs needed by TrainingConfig.
// Defined as empty structs rather than unit structs to allow for
// deserialization from empty JSON objects `{}`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizerConfig {}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SchedulerConfig {}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantizationConfig {}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DistributedConfig {}

// The main configuration for a training job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub model: RedModel,
    pub optimizer: OptimizerConfig,
    pub scheduler: SchedulerConfig,
    pub quantization: QuantizationConfig,
    pub distributed: DistributedConfig,
}

// Represents the allocated resources for a training job.
#[derive(Debug)]
pub struct Resources;

// A handle to a running training job, which allows for awaiting its completion.
#[derive(Debug)]
pub struct TrainingHandle {
    // The JoinHandle from tokio gives us a way to await the task's completion.
    // The task is expected to return a Result to indicate success or failure.
    pub handle: tokio::task::JoinHandle<crate::error::Result<()>>,
}

impl TrainingHandle {
    pub fn new(handle: tokio::task::JoinHandle<crate::error::Result<()>>) -> Self {
        Self { handle }
    }
}
