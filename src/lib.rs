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
pub use bevy_jeod_interactions::{
    DragConfigC, FlatPlateConfigC, JeodInteractionsPlugin, ShadowBodyC, SunMarker,
};
pub use bevy_jeod_planet::{JeodPlanetPlugin, PlanetC};
pub use bevy_jeod_time::{JeodTimePlugin, SimulationTimeR};

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
