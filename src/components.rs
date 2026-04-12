use bevy::prelude::*;
use glam::DVec3;
use jeod_sim::{
    DragConfig, DynamicsConfig, FrameDerivatives, GravityAcceleration, GravityControls,
    GravitySource, MassProperties, PlanetShape, RotationalState, TotalForce, TranslationalState,
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

/// Integration method for this body. Defaults to RK4 when absent.
///
/// When present on a dynamic body entity, the integration system dispatches
/// to the specified method. When absent, `IntegratorType::Rk4` is used.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct IntegratorTypeC(pub jeod_sim::IntegratorType);

/// Persistent Gauss-Jackson (Störmer-Cowell) integrator state.
///
/// Required on entities using `IntegratorType::GaussJackson`. Created once
/// with `GaussJacksonState::new(config)` and maintained across steps.
/// When absent, `integration_system` will panic if `IntegratorTypeC` is GJ.
#[derive(Component, Debug, Clone, Deref, DerefMut)]
pub struct GaussJacksonStateC(pub jeod_sim::GaussJacksonState);

#[derive(Component, Debug, Clone)]
pub struct GravityControlsC(pub GravityControls<Entity>);

#[derive(Component, Debug, Clone, Deref, DerefMut)]
pub struct GravitySourceC(pub GravitySource);

/// Inertial-frame position of a gravity source (m).
///
/// For the central body (e.g., Earth in an Earth-centered sim), this is
/// typically `DVec3::ZERO`. For third bodies (Sun, Moon), this value should
/// be provided and maintained by the application's ephemeris/update logic.
/// Used by the gravity computation to apply differential (third-body)
/// acceleration corrections.
///
/// Required on all gravity source entities. The gravity systems will panic
/// if a source entity referenced by a `GravityControlsC` is missing this
/// component.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct SourceInertialPositionC(pub DVec3);

/// Inertial-frame velocity of a gravity source (m/s).
///
/// For the central body (e.g., Earth in an Earth-centered sim), this is
/// typically `DVec3::ZERO`. For third bodies (Sun, Moon), this value is
/// maintained by the `ephemeris_update_system`.
///
/// Used by the integration system to provide source velocity to the
/// relativistic correction computation. Stored separately from
/// `TranslationalStateC` to avoid Bevy query conflicts (the body's
/// `TranslationalStateC` is already mutably queried by the integration system).
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct SourceInertialVelocityC(pub DVec3);

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

/// Tidal configuration for a gravity source entity.
///
/// When present on a gravity source entity alongside `PlanetFixedRotationC`,
/// the `tidal_update_system` computes ΔC20 each step and writes it to
/// `TidalDeltaC20C`. The application is responsible for updating
/// `tidal_bodies[].position_inertial` each step from ephemeris data.
#[derive(Component, Debug, Clone, Deref, DerefMut)]
pub struct TidalConfigC(pub jeod_sim::TidalConfig);

/// Computed tidal ΔC20 for a gravity source entity.
///
/// Written by `tidal_update_system`. Read by gravity computation and
/// integration systems. Defaults to 0.0 (no tidal effect).
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct TidalDeltaC20C(pub f64);

// ── Interactions ──

/// Vehicle drag configuration (Cd, area).
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
pub struct DragConfigC(pub DragConfig);

/// Flat-plate SRP configuration with thermal state.
///
/// Wraps [`jeod_sim::FlatPlateState`] so the same type (and its
/// `integrate_temperatures` method) is shared with the `Simulation` runner.
#[derive(Component, Debug, Clone, Deref, DerefMut)]
pub struct FlatPlateConfigC(pub jeod_sim::FlatPlateState);

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

/// Per-source rotation model dispatch.
///
/// When present on a gravity source entity alongside `PlanetFixedRotationC`,
/// the `planet_fixed_rotation_system` dispatches to the correct rotation
/// computation based on this value. When absent, `EarthRNP` is assumed
/// for backward compatibility.
#[derive(Component, Debug, Clone, Deref, DerefMut)]
pub struct RotationModelC(pub jeod_sim::RotationModel);

/// Ephemeris body mapping for automatic position updates from DE4xx.
///
/// When present on a gravity source entity, the `ephemeris_update_system`
/// queries the `EphemerisR` resource each step to update the entity's
/// `SourceInertialPositionC` (and optionally `TranslationalStateC`).
#[derive(Component, Debug, Clone, Copy)]
pub struct EphemerisBodyC {
    /// The body this source represents (e.g., `EphemerisBody::Sun`).
    pub target: jeod_sim::EphemerisBody,
    /// The integration frame center (e.g., `EphemerisBody::Earth`).
    pub observer: jeod_sim::EphemerisBody,
}

/// Cannonball SRP configuration using JEOD's `RadiationDefaultSurface` formula.
///
/// Force = (flux/c) * cx_area * [1 + albedo*diffuse*(4/9)] * flux_hat * illum_factor.
/// Mutually exclusive with `FlatPlateConfigC` (use one or the other).
///
/// Requires `SunMarker` entity in the world. Optional `ShadowBodyC` for eclipse.
/// Writes to `RadiationForceC`.
#[derive(Component, Debug, Clone, Copy)]
pub struct CannonballSrpC {
    /// Cross-section area * Cr (m²).
    pub cx_area: f64,
    /// Surface albedo (0–1).
    pub albedo: f64,
    /// Diffuse reflection fraction (0–1).
    pub diffuse: f64,
}

/// Marker component for the Sun entity (used by SRP system to find Sun position).
#[derive(Component)]
pub struct SunMarker;

/// Marker component for the Moon entity (used by earth lighting system).
#[derive(Component)]
pub struct MoonMarker;

// ── Planet ──

/// Bevy component wrapping `PlanetShape`.
#[derive(Component, Debug, Clone, Deref, DerefMut)]
pub struct PlanetC(pub PlanetShape);

// ── Derived State Configuration ──

/// Configuration for orbital elements computation.
///
/// The `gravity_source` entity is queried for `GravitySourceC` to obtain `mu`.
/// Presence of this component + `OrbitalElementsC` on an entity enables
/// per-step orbital elements computation in `JeodSet::DerivedState`.
#[derive(Component, Debug, Clone, Copy)]
pub struct OrbitalElementsConfigC {
    pub gravity_source: Entity,
}

/// Configuration for Euler angle decomposition.
///
/// Presence of this component + `EulerAnglesC` on an entity enables
/// per-step Euler angle computation in `JeodSet::DerivedState`.
#[derive(Component, Debug, Clone, Copy)]
pub struct EulerAnglesConfigC {
    pub sequence: jeod_sim::EulerSequence,
}

/// Configuration for geodetic state computation.
///
/// The `planet` entity is queried for `PlanetFixedRotationC` and `PlanetC`
/// to obtain the rotation matrix and ellipsoid radii.
/// Presence of this component + `GeodeticStateC` on an entity enables
/// per-step geodetic computation in `JeodSet::DerivedState`.
#[derive(Component, Debug, Clone, Copy)]
pub struct GeodeticConfigC {
    pub planet: Entity,
}

// ── Derived State Outputs ──

/// Orbital elements computed each step.
///
/// Written by `orbital_elements_system` for entities that also have
/// `OrbitalElementsConfigC`.
#[derive(Component, Debug, Clone, Default)]
pub struct OrbitalElementsC(pub jeod_sim::OrbitalElements);

/// Euler angles `[phi, theta, psi]` computed each step.
///
/// Written by `euler_angles_system` for entities that also have
/// `EulerAnglesConfigC`.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct EulerAnglesC(pub [f64; 3]);

/// LVLH (Local Vertical Local Horizontal) frame computed each step.
///
/// Presence of this component alone enables computation — no separate
/// config component needed (only requires translational state).
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct LvlhFrameC(pub jeod_sim::LvlhFrame);

/// Geodetic state (latitude, longitude, altitude) computed each step.
///
/// Written by `geodetic_system` for entities that also have `GeodeticConfigC`.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct GeodeticStateC(pub jeod_sim::GeodeticState);

/// Solar beta angle (radians) computed each step.
///
/// Presence of this component alone enables computation — requires a
/// `SunMarker` entity to exist in the world.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct SolarBetaC(pub f64);

/// Configuration for Earth lighting (eclipse/albedo) computation.
///
/// Requires `SunMarker` and `MoonMarker` entities to exist in the world.
/// Presence of this component + `EarthLightingStateC` on an entity enables
/// per-step earth lighting computation in `JeodSet::DerivedState`.
#[derive(Component, Debug, Clone, Copy)]
pub struct EarthLightingConfigC {
    /// Earth equatorial radius (m).
    pub earth_radius: f64,
    /// Moon mean radius (m).
    pub moon_radius: f64,
    /// Sun mean radius (m).
    pub sun_radius: f64,
}

/// Earth lighting state computed each step.
///
/// Written by `earth_lighting_system` for entities that also have
/// `EarthLightingConfigC`.
#[derive(Component, Debug, Clone, Default)]
pub struct EarthLightingStateC(pub jeod_sim::EarthLightingState);

// ── External Loads ──

/// External force in the **inertial** frame (N).
///
/// Added to `TotalForceC.force` each step after force collection.
/// Matches `SimBody.external_force` in `jeod_sim::Simulation`.
///
/// Mutate between steps to implement time-scheduled force injection.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct ExternalForceC(pub DVec3);

/// External torque in the **body** frame (N·m).
///
/// Added to `TotalForceC.torque` each step after force collection.
/// Matches `SimBody.external_torque` in `jeod_sim::Simulation`.
///
/// Mutate between steps to implement time-scheduled torque injection.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct ExternalTorqueC(pub DVec3);
