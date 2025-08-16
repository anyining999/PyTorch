use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

// Use a type alias for clarity
pub type NodeId = String;
pub type JobId = String;
pub type TaskId = String;

/// Represents the health and status of a worker node in the cluster.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeStatus {
    Healthy,
    Unresponsive,
    Failed,
}

/// Represents a single worker node in the distributed training cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub address: String,
    pub status: NodeStatus,
    pub last_heartbeat: DateTime<Utc>,
    pub total_gpus: u32,
    pub available_gpus: u32,
}

/// Represents a single, schedulable unit of work within a larger Job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    pub job_id: JobId,
    // Dependencies: this task cannot start until all tasks in this list are complete.
    pub dependencies: Vec<TaskId>,
    // The actual work to be done (e.g., a specific model layer to process).
    pub work_spec: String, // Placeholder for the actual work definition
}

/// Represents a full training job, composed of multiple interdependent tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: JobId,
    pub name: String,
    pub tasks: Vec<Task>,
    pub created_at: DateTime<Utc>,
}

/// Represents the overall state of the distributed cluster at a point in time.
/// The scheduler uses this as its view of the world to make decisions.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClusterState {
    pub nodes: HashMap<NodeId, Node>,
    pub job_queue: Vec<JobId>,
    pub running_tasks: HashMap<TaskId, NodeId>, // Map from TaskId to the NodeId it's running on
}
