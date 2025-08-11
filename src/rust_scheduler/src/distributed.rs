use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum NodeStatus {
    Healthy,
    Unresponsive,
    Down,
}

// A function to provide a default value for the `last_heartbeat` field.
fn default_heartbeat() -> Instant {
    Instant::now()
}

// A custom deserialization function for the `last_heartbeat` field.
// It ignores any value in the input and always returns the default.
fn deserialize_heartbeat<'de, D>(_deserializer: D) -> std::result::Result<Instant, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(default_heartbeat())
}

/// Contains all necessary information about a single node in the distributed cluster.
/// This struct is serialized and sent over the network as a heartbeat.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NodeInfo {
    pub id: String,
    pub addr: SocketAddr,
    pub status: NodeStatus,
    // `Instant` cannot be serialized, and we must provide a custom deserializer
    // to construct it, as it does not implement `Deserialize`.
    #[serde(skip_serializing, deserialize_with = "deserialize_heartbeat")]
    pub last_heartbeat: Instant,
}

/// A thread-safe registry of all known nodes in the cluster.
/// The key is the node's unique ID.
#[derive(Debug, Clone, Default)]
pub struct NodeRegistry {
    nodes: Arc<Mutex<HashMap<String, NodeInfo>>>,
}

impl NodeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the registry with info from a heartbeat, or adds a new node.
    pub fn update_node(&self, mut new_node_info: NodeInfo) {
        let mut nodes = self.nodes.lock().unwrap();
        new_node_info.last_heartbeat = Instant::now();
        nodes.insert(new_node_info.id.clone(), new_node_info);
    }

    /// A method to periodically check for unresponsive nodes.
    pub fn prune_unresponsive(&self, timeout: Duration) {
        let mut nodes = self.nodes.lock().unwrap();
        // This is a simplified version. A real implementation would be more complex.
        nodes.retain(|_, node_info| {
            if node_info.status == NodeStatus::Healthy && node_info.last_heartbeat.elapsed() > timeout {
                println!("Node {} timed out. Marking as unresponsive.", node_info.id);
                node_info.status = NodeStatus::Unresponsive;
            }
            // Keep the node in the list even if unresponsive for now.
            true
        });
    }

    /// Returns a copy of all current node information.
    pub fn get_all_nodes(&self) -> Vec<NodeInfo> {
        let nodes = self.nodes.lock().unwrap();
        nodes.values().cloned().collect()
    }
}

use tokio::net::UdpSocket;
use tokio::time;

pub const DISCOVERY_PORT: u16 = 61803;
const BROADCAST_ADDR: &str = "255.255.255.255";
const HEARTBEAT_INTERVAL_SECS: u64 = 5;

/// Periodically broadcasts a node's presence to the network.
pub async fn broadcast_heartbeat(node_info: NodeInfo) {
    // Bind to a random port on all interfaces for sending.
    let socket = UdpSocket::bind("0.0.0.0:0").await.expect("Failed to bind broadcast socket");
    socket.set_broadcast(true).expect("Failed to set broadcast option on socket");

    let broadcast_addr = format!("{}:{}", BROADCAST_ADDR, DISCOVERY_PORT);
    println!("Node {} starting heartbeat broadcasts to {}", node_info.id, broadcast_addr);

    let mut interval = time::interval(Duration::from_secs(HEARTBEAT_INTERVAL_SECS));
    loop {
        interval.tick().await;
        let serialized_info = serde_json::to_vec(&node_info).unwrap();

        if let Err(e) = socket.send_to(&serialized_info, &broadcast_addr).await {
            eprintln!("[Heartbeat] Failed to broadcast presence: {}", e);
        }
    }
}

/// Listens for heartbeats from other nodes and updates the registry.
pub async fn listen_for_heartbeats(registry: NodeRegistry, local_node_id: String) {
    // For production, 0.0.0.0 is correct. For local testing, 127.0.0.1 is more reliable.
    // We can make this configurable later if needed.
    let listen_addr = format!("127.0.0.1:{}", DISCOVERY_PORT);
    let socket = UdpSocket::bind(&listen_addr).await.expect("Failed to bind listen socket");
    println!("Node {} listening for heartbeats on {}", local_node_id, listen_addr);

    let mut buf = [0; 1024];
    loop {
        match socket.recv_from(&mut buf).await {
            Ok((len, _src_addr)) => {
                let node_info: std::result::Result<NodeInfo, _> = serde_json::from_slice(&buf[..len]);

                if let Ok(info) = node_info {
                    // Ignore our own heartbeat
                    if info.id != local_node_id {
                        println!("[Discovery] Received heartbeat from node {}", info.id);
                        registry.update_node(info);
                        println!("[Discovery] Registry updated for node {}.", local_node_id);
                    }
                }
            }
            Err(e) => {
                eprintln!("[Discovery] Failed to receive heartbeat: {}", e);
            }
        }
    }
}
