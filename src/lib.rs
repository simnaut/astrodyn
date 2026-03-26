use bevy::prelude::*;

pub use bevy_jeod_dynamics::{
    DynamicsConfigC, FrameDerivativesC, GravityAccelerationC, GravityControlsC, GravitySourceC,
    IntegrationFrameRef, JeodDynamicsPlugin, JeodSet, MassPropertiesC, TotalForceC,
    TranslationalStateC,
};
pub use bevy_jeod_ephemeris::{EphemerisR, JeodEphemerisPlugin};
pub use bevy_jeod_frames::{JeodFramesPlugin, RefFrameNameC, RefFrameStateC};
pub use bevy_jeod_gravity::JeodGravityPlugin;
pub use bevy_jeod_planet::{JeodPlanetPlugin, PlanetC};
pub use bevy_jeod_time::{JeodTimePlugin, SimulationTimeR};

// Re-export core types for convenience.
pub use jeod_dynamics::{DynamicsConfig, MassProperties, TranslationalState};
pub use jeod_gravity::{GravityControl, GravityControls, GravityModel, GravitySource, SphericalHarmonicsData};
pub use jeod_math::{DQuat, DMat3, DVec3, JeodQuat, OrbitalElements};
pub use jeod_planet::{PlanetShape, presets as planet_presets};
pub use jeod_ephemeris::{Ephemeris, EphemerisBody};
pub use jeod_time::SimulationTime;

pub struct JeodPlugin;

impl Plugin for JeodPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            JeodTimePlugin, // Must precede JeodFramesPlugin (provides SimulationTimeR)
            JeodDynamicsPlugin,
            JeodGravityPlugin,
            JeodFramesPlugin,
            JeodEphemerisPlugin,
            JeodPlanetPlugin,
        ));
    }
}
