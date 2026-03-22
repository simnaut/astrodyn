use bevy::prelude::*;

pub use bevy_jeod_dynamics::{
    DynamicsConfigC, FrameDerivativesC, GravityAccelerationC, GravityControlsC, GravitySourceC,
    IntegrationFrameRef, JeodDynamicsPlugin, JeodSet, MassPropertiesC, TotalForceC,
    TranslationalStateC,
};
pub use bevy_jeod_frames::{JeodFramesPlugin, RefFrameNameC, RefFrameStateC};
pub use bevy_jeod_gravity::JeodGravityPlugin;

// Re-export core types for convenience.
pub use jeod_dynamics::{DynamicsConfig, MassProperties, TranslationalState};
pub use jeod_gravity::{GravityControl, GravityControls, GravityModel, GravitySource};
pub use jeod_math::{DQuat, DMat3, DVec3, JeodQuat, OrbitalElements};

pub struct JeodPlugin;

impl Plugin for JeodPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((JeodDynamicsPlugin, JeodGravityPlugin, JeodFramesPlugin));
    }
}
