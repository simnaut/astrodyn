use bevy::prelude::*;
use glam::DVec3;
use jeod_dynamics::{
    DynamicsConfig, FrameDerivatives, GravityAcceleration, MassProperties, RotationalState,
    TotalForce, TranslationalState,
};
use jeod_gravity::{GravityControls, GravitySource};

// JEOD_INV: DB.24 — default integrated_frame is composite_body (we integrate composite_body state)
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

/// Aerodynamic force and torque in the **structural** frame (N, N*m).
///
/// Written by `aero_drag_system` (`bevy_jeod_interactions`).
/// `force_collection_system` rotates force to inertial and torque to body
/// via `StructuralTransformC`.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct AerodynamicForceC {
    pub force: DVec3,
    pub torque: DVec3,
}

/// Solar radiation pressure force and torque.
///
/// Force frame depends on the model: spherical = inertial, flat-plate = structural.
/// Torque is in the **structural** frame.
/// Written by `radiation_pressure_system` (`bevy_jeod_interactions`).
/// `force_collection_system` rotates torque to body via `StructuralTransformC`.
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

// JEOD_INV: AT.01 — active flag gates computation (presence of AtmosphericStateC = active)
/// Atmospheric state at the vehicle's position.
///
/// Written by the atmosphere system (`bevy_jeod_atmosphere`).
/// Read by the aerodynamic drag system.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct AtmosphericStateC(pub jeod_atmosphere::AtmosphereState);

/// Rotation matrix from structural frame to body (composite_body) frame.
///
/// Matches JEOD `mass.composite_properties.T_parent_this` where parent=structure.
/// Default is identity (structural frame = body frame), which is correct for
/// single-body vehicles with `eigen_angle=0`.
///
/// Used by `force_collection_system` to:
/// - Compute `T_inertial_struct = T_struct_body^T * T_inertial_body`
/// - Rotate structural-frame torques to body frame
// JEOD_INV: DB.28 — forces collected in structural frame, rotated to inertial at root
// JEOD_INV: DB.29 — torques collected in structural frame, rotated to body at root
#[derive(Component, Debug, Clone, Copy)]
pub struct StructuralTransformC(pub glam::DMat3);

impl Default for StructuralTransformC {
    fn default() -> Self {
        Self(glam::DMat3::IDENTITY)
    }
}

/// Inertial-to-planet-fixed rotation matrix for a gravity source entity.
///
/// When present on a gravity source entity, `gravity_computation_system` and
/// `integration_system` use this matrix instead of `DMat3::IDENTITY` to rotate
/// the spacecraft position into the body-fixed frame before evaluating
/// spherical-harmonic gravity.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
pub struct PlanetFixedRotationC(pub glam::DMat3);
