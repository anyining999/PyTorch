//! Auditing and logging for security-sensitive events.

use chrono::{DateTime, Utc};

pub struct AuditEvent {
    pub timestamp: DateTime<Utc>,
    pub user: String,
    pub action: String,
    pub outcome: String, // e.g., "success", "failure"
}

/// A trait for logging audit events to a secure, persistent store.
pub trait AuditLogger {
    fn log(&self, event: AuditEvent);
}

/// A placeholder logger that prints to stdout.
pub struct StdoutAuditLogger;

impl AuditLogger for StdoutAuditLogger {
    fn log(&self, event: AuditEvent) {
        println!("[AUDIT] {} | User: {} | Action: {} | Outcome: {}",
            event.timestamp, event.user, event.action, event.outcome);
    }
}
