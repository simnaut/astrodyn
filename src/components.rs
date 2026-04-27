use bevy::prelude::*;
use glam::DVec3;
use jeod_sim::{
    Angle, BodyFrame, DragConfig, DragConfigTyped, DynamicsConfig, FrameDerivatives,
    FrameDerivativesTyped, FrameTransform, GravityAcceleration, GravityAccelerationTyped,
    GravityControls, GravitySource, Inertial, MassProperties, MassPropertiesTyped, PlanetFixed,
    PlanetShape, Position, Ratio, RotationalState, RotationalStateTyped, SelfPlanet, SelfRef,
    StructuralFrame, Torque, TotalForce, TotalForceTyped, TranslationalState,
    TranslationalStateTyped, Velocity,
};

// ── Dynamics ──
//
// Spatial Components wrap the **typed siblings** from `jeod_dynamics`,
// not the raw untyped storage. The frame phantoms (`Inertial`,
// `BodyFrame<SelfRef>`, `StructuralFrame<SelfRef>`) are baked into the
// component at the type level, so systems read typed values directly
// without the per-step `from_raw_si` lifts that the audit's #172 H1
// flagged as the load-bearing failure mode of the typed-quantity
// facade. Mission code that mutates `c.0.position` directly via raw
// `DVec3` is now a compile error — the typed accessor `Position<Inertial>`
// surfaces the convention as a type, not just a comment.
//
// `From<Untyped>` impls are provided on every spatial Component so
// existing test/example code that constructs `TranslationalStateC(state)`
// from an untyped `TranslationalState` switches to
// `TranslationalStateC::from(state)` without other changes.

// JEOD_INV: DB.24 — default integrated_frame is composite_body (we integrate composite_body state)
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct TranslationalStateC(pub TranslationalStateTyped<Inertial>);

impl TranslationalStateC {
    /// Wrap an untyped [`TranslationalState`] as the typed Bevy
    /// Component. The frame is asserted to be `Inertial` — the only
    /// integration frame the Bevy adapter currently supports.
    #[inline]
    pub fn from_untyped(state: TranslationalState) -> Self {
        Self(TranslationalStateTyped::<Inertial>::from_untyped_unchecked(
            &state,
        ))
    }
}

impl From<TranslationalState> for TranslationalStateC {
    #[inline]
    fn from(state: TranslationalState) -> Self {
        Self::from_untyped(state)
    }
}

#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct RotationalStateC(pub RotationalStateTyped<SelfRef>);

impl RotationalStateC {
    /// Wrap an untyped [`RotationalState`] as the typed Bevy Component.
    /// The vehicle phantom is `SelfRef` (the Bevy adapter's wildcard
    /// "this entity's vehicle" tag). Panics if the quaternion is not
    /// unit-norm within `NormalizedQuat::DEFAULT_TOLERANCE` (1e-12) —
    /// the typed `RotationalStateTyped` carries a `NormalizedQuat`
    /// witness, so callers must pass a normalized input. Use
    /// `JeodQuat::normalize()` (or construct via the orbital-init
    /// helpers, which guarantee unit-norm) before constructing.
    #[inline]
    pub fn from_untyped(state: RotationalState) -> Self {
        Self(RotationalStateTyped::<SelfRef>::from_untyped_unchecked(
            &state,
        ))
    }
}

impl From<RotationalState> for RotationalStateC {
    #[inline]
    fn from(state: RotationalState) -> Self {
        Self::from_untyped(state)
    }
}

#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct MassPropertiesC(pub MassPropertiesTyped<SelfRef>);

impl MassPropertiesC {
    /// Wrap an untyped [`MassProperties`] as the typed Bevy Component.
    /// The inertia tensor is asserted to be in `BodyFrame<SelfRef>`
    /// and the center-of-mass position in `StructuralFrame<SelfRef>`.
    #[inline]
    pub fn from_untyped(mp: MassProperties) -> Self {
        Self(MassPropertiesTyped::<SelfRef>::from_untyped_unchecked(&mp))
    }
}

impl From<MassProperties> for MassPropertiesC {
    #[inline]
    fn from(mp: MassProperties) -> Self {
        Self::from_untyped(mp)
    }
}

#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct GravityAccelerationC(pub GravityAccelerationTyped<Inertial>);

impl From<GravityAcceleration> for GravityAccelerationC {
    #[inline]
    fn from(g: GravityAcceleration) -> Self {
        Self(GravityAccelerationTyped::<Inertial>::from_untyped_unchecked(&g))
    }
}

#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct TotalForceC(pub TotalForceTyped<SelfRef, Inertial>);

impl From<TotalForce> for TotalForceC {
    #[inline]
    fn from(t: TotalForce) -> Self {
        Self(TotalForceTyped::<SelfRef, Inertial>::from_untyped_unchecked(&t))
    }
}

#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct FrameDerivativesC(pub FrameDerivativesTyped<Inertial, SelfRef>);

impl From<FrameDerivatives> for FrameDerivativesC {
    #[inline]
    fn from(d: FrameDerivatives) -> Self {
        Self(FrameDerivativesTyped::<Inertial, SelfRef>::from_untyped_unchecked(&d))
    }
}

#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
#[require(FrameDerivativesC)]
pub struct DynamicsConfigC(pub DynamicsConfig);

/// Integration method for this body. Defaults to RK4 when absent.
///
/// When present on a dynamic body entity, the integration system dispatches
/// to the specified method. When absent, `IntegratorType::Rk4` is used.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct IntegratorTypeC(pub jeod_sim::IntegratorType);

/// Persistent Gauss-Jackson (Störmer-Cowell) integrator state.
///
/// Required on entities using `IntegratorType::GaussJackson`. Created once
/// with `GaussJacksonState::new(config)` and maintained across steps.
/// When absent, `integration_system` will panic if `IntegratorTypeC` is GJ.
#[derive(Component, Debug, Clone, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct GaussJacksonStateC(pub jeod_sim::GaussJacksonState);

/// Persistent Adams-Bashforth-Moulton 4 integrator state.
///
/// Required on entities using `IntegratorType::Abm4`. Created once with
/// `Abm4State::new()` and maintained across steps. When absent,
/// `integration_system` will panic if `IntegratorTypeC` is `Abm4`.
#[derive(Component, Debug, Clone, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct Abm4StateC(pub jeod_sim::Abm4State);

#[derive(Component, Debug, Clone, Reflect)]
#[reflect(opaque, Component)]
#[require(GravityAccelerationC, TotalForceC)]
pub struct GravityControlsC(pub GravityControls<Entity>);

#[derive(Component, Debug, Clone, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct GravitySourceC(pub GravitySource);

/// Inertial-frame position of a gravity source.
///
/// For the central body (e.g., Earth in an Earth-centered sim), this is
/// typically `Position::<Inertial>::zero()`. For third bodies (Sun, Moon),
/// this value should be provided and maintained by the application's
/// ephemeris/update logic. Used by the gravity computation to apply
/// differential (third-body) acceleration corrections.
///
/// Required on all gravity source entities. The gravity systems will panic
/// if a source entity referenced by a `GravityControlsC` is missing this
/// component.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct SourceInertialPositionC(pub Position<Inertial>);

/// Inertial-frame velocity of a gravity source.
///
/// Optional component. For the central body (e.g., Earth in an Earth-centered
/// sim), this is typically `Velocity::<Inertial>::zero()`. For third bodies
/// (Sun, Moon), attach this component alongside [`EphemerisBodyC`] and the
/// `ephemeris_update_system` will populate it each step. When absent,
/// relativistic corrections fall back to zero source velocity.
///
/// Used by the gravity and integration systems to provide source velocity to
/// the relativistic correction computation. Stored separately from
/// `TranslationalStateC` to avoid Bevy query conflicts (the body's
/// `TranslationalStateC` is already mutably queried by the integration system).
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct SourceInertialVelocityC(pub Velocity<Inertial>);

/// Aerodynamic force and torque in the **structural** frame (N, N*m).
///
/// Written by `aero_drag_system`.
/// `force_collection_system` rotates force to inertial and torque to body
/// via `StructuralTransformC`.
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(opaque, Component)]
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
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct RadiationForceC {
    pub force: DVec3,
    pub torque: DVec3,
}

/// Gravity gradient torque in the body frame (N·m).
///
/// Written by the gravity torque system.
/// Read by `force_collection_system` as `Option<&GravityTorqueC>`.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct GravityTorqueC(pub Torque<BodyFrame<SelfRef>>);

// JEOD_INV: AT.01 — active flag gates computation (presence of AtmosphericStateC = active)
/// Atmospheric state at the vehicle's position.
///
/// Written by the atmosphere system. Read by the aerodynamic drag system.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct AtmosphericStateC(pub jeod_sim::AtmosphereState);

/// Typed structural→body rotation for a vehicle entity.
///
/// Stores the rotation that maps structural-frame vectors into body-frame
/// vectors (matches JEOD `mass.composite_properties.T_parent_this` where
/// parent=structure). The `FrameTransform`'s phantom `<StructuralFrame<SelfRef>,
/// BodyFrame<SelfRef>>` parameters encode the *direction* — `SelfRef` is the
/// wildcard `Vehicle` marker indicating "this entity's vehicle"; the actual
/// vehicle identity stays at the entity level via Bevy queries.
///
/// Default is identity (structural frame = body frame), which is correct for
/// single-body vehicles with `eigen_angle=0`.
///
/// Used by `force_collection_system` to:
/// - Compute `T_inertial_struct = T_struct_body^T * T_inertial_body`
/// - Rotate structural-frame torques to body frame
// JEOD_INV: DB.28 — forces collected in structural frame, rotated to inertial at root
// JEOD_INV: DB.29 — torques collected in structural frame, rotated to body at root
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
pub struct StructuralTransformC(pub FrameTransform<StructuralFrame<SelfRef>, BodyFrame<SelfRef>>);

impl Default for StructuralTransformC {
    fn default() -> Self {
        Self(FrameTransform::from_matrix(glam::DMat3::IDENTITY))
    }
}

/// Typed inertial→planet-fixed rotation for a gravity source entity.
///
/// Stores the rotation that maps inertial-frame vectors into the planet-fixed
/// frame of the source. The `FrameTransform`'s phantom `<Inertial,
/// PlanetFixed<SelfPlanet>>` parameters encode the *direction* — `SelfPlanet`
/// is the wildcard `Planet` marker indicating "this entity's planet"; the
/// actual planet identity stays at the entity level via the existing
/// `PlanetC` discriminator.
///
/// When present on a gravity source entity, `gravity_computation_system` and
/// `integration_system` use this rotation instead of `DMat3::IDENTITY` to
/// rotate the spacecraft position into the body-fixed frame before evaluating
/// spherical-harmonic gravity.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
pub struct PlanetFixedRotationC(pub FrameTransform<Inertial, PlanetFixed<SelfPlanet>>);

/// Tidal configuration for a gravity source entity.
///
/// When present on a gravity source entity alongside `PlanetFixedRotationC`,
/// the `tidal_update_system` computes ΔC20 each step and writes it to
/// `TidalDeltaC20C`. The application is responsible for updating
/// `tidal_bodies[].position_inertial` each step from ephemeris data.
///
/// Wraps [`jeod_sim::TidalConfigTyped`] (typed sibling of
/// [`jeod_sim::TidalConfig`]) so the untyped → typed conversion happens
/// **once at insertion**, not per tick in `tidal_update_system`. This
/// eliminates the per-frame `TidalConfigTyped::from_untyped` allocation
/// (Vec collect + per-body f64 → typed boxing). Convenience constructors
/// `from_untyped` / `From` impls are provided for callers building from
/// the untyped struct.
#[derive(Component, Debug, Clone, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
#[require(TidalDeltaC20C)]
pub struct TidalConfigC(pub jeod_sim::TidalConfigTyped);

impl TidalConfigC {
    /// Wrap an untyped [`jeod_sim::TidalConfig`] as a typed Bevy component.
    ///
    /// The dimensional lift (`f64` → `Ratio`/`GravParam`/`Length`/`Position`)
    /// happens here at insertion. After that, the wrapped value is already
    /// typed for the lifetime of the component, eliminating per-tick
    /// `from_untyped` calls in `tidal_update_system`.
    #[inline]
    pub fn from_untyped(config: &jeod_sim::TidalConfig) -> Self {
        Self(jeod_sim::TidalConfigTyped::from_untyped(config))
    }
}

impl From<jeod_sim::TidalConfig> for TidalConfigC {
    fn from(config: jeod_sim::TidalConfig) -> Self {
        Self::from_untyped(&config)
    }
}

impl From<jeod_sim::TidalConfigTyped> for TidalConfigC {
    fn from(config: jeod_sim::TidalConfigTyped) -> Self {
        Self(config)
    }
}

/// Computed tidal ΔC20 for a gravity source entity.
///
/// Written by `tidal_update_system`. Read by gravity computation and
/// integration systems. Defaults to zero (no tidal effect).
///
/// Wrapped as a [`Ratio`] (dimensionless) so the value carries unit
/// metadata at the type level — matching `compute_delta_c20_typed`'s
/// return type.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct TidalDeltaC20C(pub Ratio);

// ── Interactions ──

/// Vehicle drag configuration (Cd, area).
///
/// Wraps [`DragConfigTyped`] (typed sibling of [`DragConfig`]) so the
/// untyped → typed conversion happens **once at insertion**, not per tick
/// in `aero_drag_system`. Convenience constructors `from_untyped` /
/// `new` are provided for callers building from raw `f64` fields.
///
/// Auto-inserts [`AtmosphericStateC`] and [`AerodynamicForceC`] when added.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
#[require(AtmosphericStateC, AerodynamicForceC)]
pub struct DragConfigC(pub DragConfigTyped);

impl DragConfigC {
    /// Wrap an untyped [`DragConfig`] as a typed Bevy component.
    ///
    /// The dimensional lift (`f64` → `Ratio`/`Area`/`MassDensity`) happens
    /// here at insertion. After that, the wrapped value is already typed
    /// for the lifetime of the component, eliminating per-tick
    /// per-tick unchecked conversions in `aero_drag_system`. Per #172 H1,
    /// this is the documented insertion-time boundary: DragConfig has no
    /// spatial fields (Cd / area / optional density override only), so
    /// the lift here is the JEOD-CSV-style boundary the audit carves out.
    /// The component then stores DragConfigTyped for the rest of its
    /// lifetime; no per-step re-minting occurs.
    #[inline]
    pub fn from_untyped(config: &DragConfig) -> Self {
        Self(DragConfigTyped::from_untyped_unchecked(config)) // allowed: #172 H1 insertion-time boundary, see docstring
    }
}

impl From<DragConfig> for DragConfigC {
    fn from(config: DragConfig) -> Self {
        Self::from_untyped(&config)
    }
}

impl From<DragConfigTyped> for DragConfigC {
    fn from(config: DragConfigTyped) -> Self {
        Self(config)
    }
}

/// Flat-plate SRP configuration with thermal state.
///
/// Wraps [`jeod_sim::FlatPlateState`] so the same type (and its
/// `integrate_temperatures` method) is shared with the `Simulation` runner.
///
/// Auto-inserts [`RadiationForceC`] when added.
#[derive(Component, Debug, Clone, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
#[require(RadiationForceC)]
pub struct FlatPlateConfigC(pub jeod_sim::FlatPlateState);

/// Marker for an entity that casts shadows (e.g., Earth).
///
/// The shadow detection system queries all entities with this component
/// and computes the illumination factor for SRP. Place on any planet
/// entity along with `TranslationalStateC`.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
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
#[derive(Component, Debug, Clone, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct RotationModelC(pub jeod_sim::RotationModel);

/// Ephemeris body mapping for automatic position updates from DE4xx.
///
/// When present on a gravity source entity, the `ephemeris_update_system`
/// queries the `EphemerisR` resource each step to update the entity's
/// `SourceInertialPositionC` (and optionally `TranslationalStateC`).
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
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
///
/// Auto-inserts [`RadiationForceC`] when added.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
#[require(RadiationForceC)]
pub struct CannonballSrpC {
    /// Cross-section area * Cr (m²).
    pub cx_area: f64,
    /// Surface albedo (0–1).
    pub albedo: f64,
    /// Diffuse reflection fraction (0–1).
    pub diffuse: f64,
}

/// Marker component for the Sun entity (used by SRP system to find Sun position).
#[derive(Component, Reflect, Default, Clone, Copy, Debug)]
#[reflect(opaque, Component)]
pub struct SunMarker;

/// Marker component for the Moon entity (used by earth lighting system).
#[derive(Component, Reflect, Default, Clone, Copy, Debug)]
#[reflect(opaque, Component)]
pub struct MoonMarker;

// ── Planet ──

/// Bevy component wrapping `PlanetShape`.
#[derive(Component, Debug, Clone, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct PlanetC(pub PlanetShape);

// ── Derived State Configuration ──

/// Configuration for orbital elements computation.
///
/// The `gravity_source` entity is queried for `GravitySourceC` to obtain `mu`.
/// Presence of this component + `OrbitalElementsC` on an entity enables
/// per-step orbital elements computation in `JeodSet::DerivedState`.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
#[require(OrbitalElementsC)]
pub struct OrbitalElementsConfigC {
    pub gravity_source: Entity,
}

/// Configuration for Euler angle decomposition.
///
/// Presence of this component + `EulerAnglesC` on an entity enables
/// per-step Euler angle computation in `JeodSet::DerivedState`.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
#[require(EulerAnglesC)]
pub struct EulerAnglesConfigC {
    pub sequence: jeod_sim::EulerSequence,
}

/// Configuration for geodetic state computation.
///
/// The `planet` entity is queried for `PlanetFixedRotationC` and `PlanetC`
/// to obtain the rotation matrix and ellipsoid radii.
/// Presence of this component + `GeodeticStateC` on an entity enables
/// per-step geodetic computation in `JeodSet::DerivedState`.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
#[require(GeodeticStateC)]
pub struct GeodeticConfigC {
    pub planet: Entity,
}

// ── Derived State Outputs ──

/// Orbital elements computed each step.
///
/// Written by `orbital_elements_system` for entities that also have
/// `OrbitalElementsConfigC`.
#[derive(Component, Debug, Clone, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct OrbitalElementsC(pub jeod_sim::OrbitalElements);

/// Euler angles `[phi, theta, psi]` computed each step.
///
/// Written by `euler_angles_system` for entities that also have
/// `EulerAnglesConfigC`. Each component is a [`Angle`] (uom radian-backed
/// scalar) so consumers don't have to remember the radian convention.
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct EulerAnglesC(pub [Angle; 3]);

/// LVLH (Local Vertical Local Horizontal) frame computed each step.
///
/// Presence of this component alone enables computation — no separate
/// config component needed (only requires translational state).
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct LvlhFrameC(pub jeod_sim::LvlhFrame);

/// Geodetic state (latitude, longitude, altitude) computed each step.
///
/// Written by `geodetic_system` for entities that also have `GeodeticConfigC`.
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct GeodeticStateC(pub jeod_sim::GeodeticState);

/// Solar beta angle (radians) computed each step.
///
/// Presence of this component alone enables computation — requires a
/// `SunMarker` entity to exist in the world.
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct SolarBetaC(pub f64);

/// Configuration for Earth lighting (eclipse/albedo) computation.
///
/// Requires `SunMarker` and `MoonMarker` entities to exist in the world.
/// Presence of this component + `EarthLightingStateC` on an entity enables
/// per-step earth lighting computation in `JeodSet::DerivedState`.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
#[require(EarthLightingStateC)]
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
#[derive(Component, Debug, Clone, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct EarthLightingStateC(pub jeod_sim::EarthLightingState);

// ── External Loads ──

/// External force in the **inertial** frame.
///
/// Added to `TotalForceC.force` each step after force collection.
/// Matches `SimBody.external_force` in `jeod_sim::Simulation`.
///
/// Mutate between steps to implement time-scheduled force injection.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct ExternalForceC(pub jeod_sim::Force<Inertial>);

/// External torque in the **body** frame.
///
/// Added to `TotalForceC.torque` each step after force collection.
/// Matches `SimBody.external_torque` in `jeod_sim::Simulation`.
///
/// Mutate between steps to implement time-scheduled torque injection.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct ExternalTorqueC(pub Torque<BodyFrame<SelfRef>>);

// ── Mass Tree (Staging) ──

/// Maps this entity to a node in the shared [`MassTreeR`](crate::MassTreeR) resource.
///
/// Entities with this component participate in the mass tree. After
/// attach/detach events are processed, the entity's [`MassPropertiesC`]
/// is synced from the tree's composite properties.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
pub struct MassBodyIdC(pub jeod_sim::MassBodyId);

/// Message: attach a child body to a parent in the mass tree.
///
/// Both entities must have [`MassBodyIdC`]. Processed by `staging_system`
/// before integration each step.
#[derive(Message, Debug, Clone)]
pub struct AttachEvent {
    /// Entity of the child body.
    pub child: Entity,
    /// Entity of the parent body.
    pub parent: Entity,
    /// Child structural origin in parent's structural frame (m).
    pub offset: DVec3,
    /// Rotation from parent structural frame to child structural frame.
    pub t_parent_child: glam::DMat3,
}

/// Message: detach a child body from its parent in the mass tree.
///
/// The entity must have [`MassBodyIdC`] and be attached to a parent.
/// Processed by `staging_system` before integration each step.
#[derive(Message, Debug, Clone)]
pub struct DetachEvent {
    /// Entity to detach from its parent.
    pub child: Entity,
}
