pub mod plugin;
pub mod systems;

pub use plugin::JeodGravityPlugin;
pub use systems::*;

// Re-export GravitySourceC from bevy_jeod_dynamics for convenience.
pub use bevy_jeod_dynamics::GravitySourceC;
