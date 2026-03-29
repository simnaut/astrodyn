use bevy::prelude::*;

pub use bevy_jeod_atmosphere::{
    AtmosphereConfig, AtmosphereModel, AtmosphereModelR, JeodAtmospherePlugin,
};
pub use bevy_jeod_dynamics::{
    AerodynamicForceC, AtmosphericStateC as AtmosphericStateDynC, DynamicsConfigC,
    FrameDerivativesC, GravityAccelerationC, GravityControlsC, GravitySourceC, GravityTorqueC,
    IntegrationFrameRef, JeodDynamicsPlugin, JeodSet, MassPropertiesC, PlanetFixedRotationC,
    RadiationForceC, RotationalStateC, StructuralTransformC, TotalForceC, TranslationalStateC,
};
pub use bevy_jeod_ephemeris::{EphemerisR, JeodEphemerisPlugin};
pub use bevy_jeod_frames::{JeodFramesPlugin, RefFrameNameC, RefFrameStateC};
pub use bevy_jeod_gravity::JeodGravityPlugin;
pub use bevy_jeod_interactions::{DragConfigC, JeodInteractionsPlugin, SrpConfigC, SunMarker};
pub use bevy_jeod_planet::{JeodPlanetPlugin, PlanetC};
pub use bevy_jeod_time::{JeodTimePlugin, SimulationTimeR};

// Re-export core types for convenience.
pub use jeod_atmosphere::{
    compute_corotation_wind, exponential::ExponentialAtmosphere, AtmosphereState,
};
pub use jeod_dynamics::{DynamicsConfig, MassProperties, TranslationalState};
pub use jeod_ephemeris::{Ephemeris, EphemerisBody};
pub use jeod_gravity::{
    GravityControl, GravityControls, GravityModel, GravitySource, SphericalHarmonicsData,
};
pub use jeod_interactions::{
    compute_gravity_torque, AerodynamicForce, DragConfig, RadiationForce, SrpConfig,
};
pub use jeod_math::{DMat3, DQuat, DVec3, JeodQuat, OrbitalElements};
pub use jeod_planet::{presets as planet_presets, PlanetShape};
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
            JeodAtmospherePlugin,
            JeodInteractionsPlugin,
        ));
    }
}
