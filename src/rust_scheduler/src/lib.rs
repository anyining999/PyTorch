pub mod models;
pub mod error;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use crate::error::{Result, RedError, ResourceError};
use crate::models::{
    TrainingConfig, Resources, ResourcePool,
    DeviceResource, ResourceState, Task
};

// A placeholder for collecting performance and usage metrics.
#[derive(Debug)]
pub struct AtomicMetrics;

// The main scheduler for RedcodE, responsible for managing and dispatching tasks.
#[derive(Debug)]
pub struct RedScheduler {
    resource_pool: Arc<RwLock<ResourcePool>>,
    task_queue: Arc<Mutex<VecDeque<Task>>>,
    metrics: Arc<AtomicMetrics>,
    next_task_id: AtomicU64,
    _worker_handle: tokio::task::JoinHandle<()>,
}

impl RedScheduler {
    // Creates a new `RedScheduler` and spawns its background worker task.
    pub fn new() -> Arc<Self> {
        let devices = vec![
            DeviceResource { id: 0, memory_mb: 8192, state: ResourceState::Available },
            DeviceResource { id: 1, memory_mb: 16384, state: ResourceState::Available },
        ];
        let resource_pool = ResourcePool { devices };

        let scheduler = Arc::new(Self {
            resource_pool: Arc::new(RwLock::new(resource_pool)),
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            metrics: Arc::new(AtomicMetrics),
            next_task_id: AtomicU64::new(1),
            _worker_handle: tokio::spawn(async {}), // dummy handle
        });

        let worker_handle = Self::spawn_worker(scheduler.clone());

        unsafe {
            let mutable_scheduler = Arc::as_ptr(&scheduler) as *mut Self;
            (*mutable_scheduler)._worker_handle = worker_handle;
        }

        scheduler
    }

    fn spawn_worker(scheduler: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            println!("Scheduler worker started.");
            loop {
                let task = {
                    scheduler.task_queue.lock().unwrap().pop_front()
                };

                if let Some(task) = task {
                    println!("Worker picked up task {} for model: {}", task.id, task.config.model.id);
                }

                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
        })
    }

    /// Schedules a new training job by pushing it to the queue. Returns a unique task ID.
    pub async fn schedule_training(&self, config: TrainingConfig) -> Result<u64> {
        println!("Scheduling training for model: {}", config.model.id);

        let resources = self.allocate_optimal_resources(&config).await?;
        println!("Resources allocated successfully.");

        let task_id = self.next_task_id.fetch_add(1, Ordering::SeqCst);
        let task = Task { id: task_id, config, resources };

        {
            let mut queue = self.task_queue.lock().unwrap();
            queue.push_back(task);
            println!("Task {} added to the queue. Current queue size: {}", task_id, queue.len());
        }

        Ok(task_id)
    }

    /// A skeleton method for the resource allocation logic.
    async fn allocate_optimal_resources(&self, config: &TrainingConfig) -> Result<Resources> {
        println!("Attempting to lock resource pool and allocate...");
        // For testing purposes, we'll use the learning_rate as the required memory.
        let required_memory = config.model.config.learning_rate as u64;

        let mut pool = self.resource_pool.write().unwrap();
        match pool.allocate(required_memory) {
            Ok(device) => {
                println!("Allocated device {} with {} MB memory.", device.id, device.memory_mb);
                Ok(Resources)
            }
            Err(e) => {
                eprintln!("Resource allocation failed: {}", e);
                Err(RedError::Resource(ResourceError))
            }
        }
    }
}

impl Default for RedScheduler {
    fn default() -> Self {
        let devices = vec![];
        let resource_pool = ResourcePool { devices };
        Self {
            resource_pool: Arc::new(RwLock::new(resource_pool)),
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            metrics: Arc::new(AtomicMetrics),
            next_task_id: AtomicU64::new(1),
            _worker_handle: tokio::spawn(async {}), // dummy handle
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn scheduler_can_be_created_and_worker_runs() {
        let scheduler = RedScheduler::new();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        println!("{:?}", scheduler);
    }

    #[test]
    fn resource_pool_allocation_logic_works() {
        let devices = vec![
            DeviceResource { id: 0, memory_mb: 100, state: ResourceState::Available },
            DeviceResource { id: 1, memory_mb: 200, state: ResourceState::Available },
        ];
        let mut pool = ResourcePool { devices };

        // Allocate first device
        let res1 = pool.allocate(50).unwrap();
        assert_eq!(res1.id, 0);
        assert_eq!(res1.state, ResourceState::InUse);

        // Allocate second device
        let res2 = pool.allocate(150).unwrap();
        assert_eq!(res2.id, 1);
        assert_eq!(res2.state, ResourceState::InUse);

        // Try to allocate a third, which should fail
        let res3 = pool.allocate(100);
        assert!(res3.is_err());

        // Check that the pool state is correct
        assert_eq!(pool.devices[0].state, ResourceState::InUse);
        assert_eq!(pool.devices[1].state, ResourceState::InUse);
    }
}

use std::ffi::CStr;
use std::os::raw::c_char;

/// A simple function to be called from C++ to verify the FFI bridge.
#[unsafe(no_mangle)]
pub extern "C" fn hello_from_rust() {
    println!("Hello from Rust! The FFI bridge is working.");
}

/// Accepts a JSON string representing a TrainingConfig, deserializes it,
/// and passes it to the scheduler. Returns a task_id (> 0) on success, or a negative error code.
#[unsafe(no_mangle)]
pub extern "C" fn schedule_training_from_json(config_json: *const c_char) -> i64 {
    let c_str = unsafe {
        if config_json.is_null() {
            eprintln!("FFI Error: config_json pointer was null.");
            return -1;
        }
        CStr::from_ptr(config_json)
    };

    let json_str = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FFI Error: Failed to convert C string: {}", e);
            return -2;
        }
    };

    let config: models::TrainingConfig = match serde_json::from_str(json_str) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FFI Error: Failed to deserialize TrainingConfig: {}", e);
            return -3;
        }
    };

    let runtime = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("FFI Error: Failed to create tokio runtime: {}", e);
            return -4;
        }
    };

    runtime.block_on(async {
        let scheduler = RedScheduler::new();

        match scheduler.schedule_training(config).await {
            Ok(task_id) => {
                println!("FFI: Successfully scheduled task with ID: {}", task_id);
                // Give the worker a moment to pick up the task
                tokio::time::sleep(std::time::Duration::from_millis(600)).await;
                task_id as i64
            }
            Err(e) => {
                eprintln!("FFI Error: schedule_training failed: {}", e);
                match e {
                    RedError::Resource(_) => -10,
                    _ => -5,
                }
            }
        }
    })
}
