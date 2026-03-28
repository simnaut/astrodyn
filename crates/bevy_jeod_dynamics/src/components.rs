use bevy::prelude::*;
use glam::DVec3;
use jeod_dynamics::{
    DynamicsConfig, FrameDerivatives, GravityAcceleration, MassProperties, RotationalState,
    TotalForce, TranslationalState,
};
use jeod_gravity::{GravityControls, GravitySource};

#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default)]
pub struct TranslationalStateC(pub TranslationalState);

#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct RotationalStateC(pub RotationalState);

#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
pub struct MassPropertiesC(pub MassProperties);

#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default)]
pub struct GravityAccelerationC(pub GravityAcceleration);

#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default)]
pub struct TotalForceC(pub TotalForce);

#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default)]
pub struct FrameDerivativesC(pub FrameDerivatives);

#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct DynamicsConfigC(pub DynamicsConfig);

#[derive(Component, Debug, Clone, Copy)]
pub struct IntegrationFrameRef(pub Entity);

#[derive(Component, Debug, Clone)]
pub struct GravityControlsC(pub GravityControls<Entity>);

#[derive(Component, Debug, Clone, Deref, DerefMut)]
pub struct GravitySourceC(pub GravitySource);

/// Aerodynamic force and torque in the body frame (N, N*m).
///
/// Written by the aerodynamic drag system (`bevy_jeod_interactions`).
/// Read by `force_collection_system` as `Option<&AerodynamicForceC>`.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct AerodynamicForceC {
    pub force: DVec3,
    pub torque: DVec3,
}

/// Solar radiation pressure force in the inertial frame and torque in body frame.
///
/// Written by the radiation pressure system (`bevy_jeod_interactions`).
/// Read by `force_collection_system` as `Option<&RadiationForceC>`.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct RadiationForceC {
    pub force: DVec3,
    pub torque: DVec3,
}

/// Gravity gradient torque in the body frame (N*m).
///
/// Written by the gravity torque system (`bevy_jeod_interactions`).
/// Read by `force_collection_system` as `Option<&GravityTorqueC>`.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct GravityTorqueC(pub DVec3);

/// Atmospheric state at the vehicle's position.
///
/// Written by the atmosphere system (`bevy_jeod_atmosphere`).
/// Read by the aerodynamic drag system.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct AtmosphericStateC {
    pub density: f64,
    pub temperature: f64,
    pub pressure: f64,
    pub wind: DVec3,
}

/// Inertial-to-planet-fixed rotation matrix for a gravity source entity.
///
/// When present on a gravity source entity, `gravity_computation_system` and
/// `integration_system` use this matrix instead of `DMat3::IDENTITY` to rotate
/// the spacecraft position into the body-fixed frame before evaluating
/// spherical-harmonic gravity.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
pub struct PlanetFixedRotationC(pub glam::DMat3);
