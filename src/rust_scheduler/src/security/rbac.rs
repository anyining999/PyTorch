//! Role-Based Access Control (RBAC) system.

use std::collections::HashSet;

pub type Role = String;
pub type Permission = String;

/// Manages roles and permissions for access control.
pub struct RbacManager {
    // A map from a role to a set of permissions.
    roles: std::collections::HashMap<Role, HashSet<Permission>>,
}

impl RbacManager {
    pub fn new() -> Self {
        // In a real implementation, this would be loaded from a config.
        Self { roles: std::collections::HashMap::new() }
    }

    /// Checks if a given role has a specific permission.
    pub fn has_permission(&self, role: &Role, permission: &Permission) -> bool {
        self.roles
            .get(role)
            .map_or(false, |perms| perms.contains(permission))
    }
}
