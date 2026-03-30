use bevy::prelude::*;
use glam::DVec3;
use jeod_sim::{
    DragConfig, DynamicsConfig, FlatPlate, FlatPlateParams, FlatPlateThermal, FrameDerivatives,
    GravityAcceleration, GravityControls, GravitySource, MassProperties, PlanetShape,
    RefFrameState, RotationalState, TotalForce, TranslationalState,
};

// ── Dynamics ──

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
/// Written by `aero_drag_system`.
/// `force_collection_system` rotates force to inertial and torque to body
/// via `StructuralTransformC`.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct AerodynamicForceC {
    pub force: DVec3,
    pub torque: DVec3,
}

/// Solar radiation pressure force and torque.
///
/// Force is always in the **inertial** frame (`flat_plate_srp_system` rotates
/// from structural to inertial before writing).
/// Torque is always in the **structural** frame.
/// Written by `flat_plate_srp_system`.
/// `force_collection_system` rotates torque to body via `StructuralTransformC`.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct RadiationForceC {
    pub force: DVec3,
    pub torque: DVec3,
}

/// Gravity gradient torque in the body frame (N*m).
///
/// Written by the gravity torque system.
/// Read by `force_collection_system` as `Option<&GravityTorqueC>`.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct GravityTorqueC(pub DVec3);

// JEOD_INV: AT.01 — active flag gates computation (presence of AtmosphericStateC = active)
/// Atmospheric state at the vehicle's position.
///
/// Written by the atmosphere system. Read by the aerodynamic drag system.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct AtmosphericStateC(pub jeod_sim::AtmosphereState);

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

// ── Interactions ──

/// Vehicle drag configuration (Cd, area).
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
pub struct DragConfigC(pub DragConfig);

/// Flat-plate SRP configuration with thermal state.
///
/// Contains per-plate geometry, optical/thermal properties, and temperature state.
/// Temperatures are integrated via forward Euler each step (matching the
/// Simulation runner's approach).
#[derive(Component, Debug, Clone)]
pub struct FlatPlateConfigC {
    /// Per-plate geometry, optical, and thermal properties.
    pub plates: Vec<(FlatPlate, FlatPlateParams, FlatPlateThermal)>,
    /// Per-plate temperatures (K). Same length as `plates`.
    pub temperatures: Vec<f64>,
    /// Cached T^4 per plate from previous step (for thermal emission).
    pub t_pow4_cached: Vec<f64>,
}

/// Marker for an entity that casts shadows (e.g., Earth).
///
/// The shadow detection system queries all entities with this component
/// and computes the illumination factor for SRP. Place on any planet
/// entity along with `TranslationalStateC`.
#[derive(Component, Debug, Clone, Copy)]
pub struct ShadowBodyC {
    /// Body radius (m) for conical shadow computation.
    pub radius: f64,
}

/// Marker component for the Sun entity (used by SRP system to find Sun position).
#[derive(Component)]
pub struct SunMarker;

// ── Frames ──

#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default)]
pub struct RefFrameStateC(pub RefFrameState);

#[derive(Component, Debug, Clone)]
pub struct RefFrameNameC(pub String);

// ── Planet ──

/// Bevy component wrapping `PlanetShape`.
#[derive(Component, Debug, Clone, Deref, DerefMut)]
pub struct PlanetC(pub PlanetShape);
