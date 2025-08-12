use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::net::{UdpSocket, TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time;
use crate::RedScheduler;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum NodeStatus {
    Healthy,
    Unresponsive,
    Down,
}

fn default_heartbeat() -> Instant {
    Instant::now()
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NodeInfo {
    pub id: String,
    pub addr: SocketAddr,
    pub command_port: u16,
    pub status: NodeStatus,
    #[serde(skip_serializing, deserialize_with = "deserialize_heartbeat")]
    pub last_heartbeat: Instant,
}

fn deserialize_heartbeat<'de, D>(_deserializer: D) -> std::result::Result<Instant, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(default_heartbeat())
}

#[derive(Debug, Clone, Default)]
pub struct NodeRegistry {
    nodes: Arc<Mutex<HashMap<String, NodeInfo>>>,
}

impl NodeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update_node(&self, mut new_node_info: NodeInfo) {
        let mut nodes = self.nodes.lock().unwrap();
        new_node_info.last_heartbeat = Instant::now();
        nodes.insert(new_node_info.id.clone(), new_node_info);
    }

    pub fn prune_unresponsive(&self, timeout: Duration) {
        let mut nodes = self.nodes.lock().unwrap();
        nodes.retain(|_, node_info| {
            if node_info.status == NodeStatus::Healthy && node_info.last_heartbeat.elapsed() > timeout {
                println!("Node {} timed out. Marking as unresponsive.", node_info.id);
                node_info.status = NodeStatus::Unresponsive;
            }
            true
        });
    }

    pub fn get_all_nodes(&self) -> Vec<NodeInfo> {
        let nodes = self.nodes.lock().unwrap();
        nodes.values().cloned().collect()
    }

    pub async fn send_message(&self, node_id: &str, message: &Message) -> std::io::Result<()> {
        let node_info = {
            let nodes = self.nodes.lock().unwrap();
            nodes.get(node_id).cloned()
        };

        if let Some(info) = node_info {
            let addr = format!("{}:{}", info.addr.ip(), info.command_port);
            println!("[TCP] Sending message {:?} to node {} at {}", message, node_id, addr);

            let mut stream = TcpStream::connect(&addr).await?;
            let serialized_msg = serde_json::to_vec(message).unwrap();
            stream.write_all(&serialized_msg).await?;
            Ok(())
        } else {
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "Node not found in registry"))
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Message {
    Ping,
    Pong,
    GetStatus,
    Status(NodeInfo),
    SubmitTask { config: crate::models::TrainingConfig },
}

pub const DISCOVERY_PORT: u16 = 61803;
pub const COMMAND_PORT: u16 = 61802;
const BROADCAST_ADDR: &str = "255.255.255.255";
const HEARTBEAT_INTERVAL_SECS: u64 = 5;

pub async fn broadcast_heartbeat(node_info: NodeInfo) {
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

pub async fn run_tcp_listener(scheduler: Arc<RedScheduler>) {
    let listen_addr = format!("0.0.0.0:{}", COMMAND_PORT);
    let listener = match TcpListener::bind(&listen_addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[TCP] Failed to bind listener on {}: {}", listen_addr, e);
            return;
        }
    };
    println!("[TCP] Node {} listening for commands on {}", scheduler.node_id, listen_addr);

    loop {
        match listener.accept().await {
            Ok((mut socket, addr)) => {
                println!("[TCP] Accepted connection from {}", addr);
                let scheduler_clone = scheduler.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    if let Err(e) = socket.read_to_end(&mut buf).await {
                        eprintln!("[TCP] Failed to read data from socket: {}", e);
                        return;
                    }

                    let msg: std::result::Result<Message, _> = serde_json::from_slice(&buf);
                    if let Ok(message) = msg {
                        handle_message(message, &mut socket, &scheduler_clone).await;
                    } else {
                        eprintln!("[TCP] Failed to deserialize message from {}", addr);
                    }
                });
            },
            Err(e) => {
                eprintln!("[TCP] Failed to accept connection: {}", e);
            }
        }
    }
}

async fn handle_message(msg: Message, socket: &mut TcpStream, scheduler: &Arc<RedScheduler>) {
    match msg {
        Message::Ping => {
            println!("[TCP] Responding to Ping with Pong");
            let response = serde_json::to_vec(&Message::Pong).unwrap();
            if let Err(e) = socket.write_all(&response).await {
                eprintln!("[TCP] Failed to send Pong: {}", e);
            }
        },
        Message::GetStatus => {
            println!("[TCP] Responding to GetStatus with own status");
            // We need a way to get the node's own info.
            // Let's add a method to RedScheduler for this.
            // For now, this part is incomplete.
        },
        Message::SubmitTask { config } => {
            println!("[TCP] Received SubmitTask for model {}", config.model.id);
            match scheduler.schedule_training(config).await {
                Ok(task_id) => {
                    println!("[TCP] Successfully queued delegated task as local task {}", task_id);
                    // Optionally send a confirmation back to the caller
                },
                Err(e) => eprintln!("[TCP] Failed to queue delegated task: {}", e),
            }
        },
        _ => {
            println!("[TCP] Received unhandled message type: {:?}", msg);
        }
    }
}

pub async fn listen_for_heartbeats(registry: NodeRegistry, local_node_id: String) {
    let listen_addr = format!("127.0.0.1:{}", DISCOVERY_PORT);
    let socket = UdpSocket::bind(&listen_addr).await.expect("Failed to bind listen socket");
    println!("Node {} listening for heartbeats on {}", local_node_id, listen_addr);

    let mut buf = [0; 1024];
    loop {
        match socket.recv_from(&mut buf).await {
            Ok((len, _src_addr)) => {
                let node_info: std::result::Result<NodeInfo, _> = serde_json::from_slice(&buf[..len]);

                if let Ok(info) = node_info {
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
