//! Bevy `Component` newtypes for gravity sources, source-side inertial
//! state, source rotation / ephemeris configuration, the gravity-gradient
//! torque accumulator, tidal configuration / output, and the planet
//! shape + angular-velocity surface.

use bevy::prelude::*;
use jeod_sim::{
    AngularVelocity, BodyFrame, FrameTransform, GravityControls, GravitySource, Planet,
    PlanetFixed, PlanetShape, Position, Ratio, RootInertial, SelfRef, Torque, Velocity,
};

use super::state::{GravityAccelerationC, TotalForceC};

/// Per-body list of gravity controls keyed by source [`Entity`]. Each
/// control selects the model (point-mass / spherical-harmonics) and
/// which body it represents (central, third, etc.).
#[derive(Component, Debug, Clone)]
#[require(GravityAccelerationC, TotalForceC)]
pub struct GravityControlsC(pub GravityControls<Entity>);

/// Gravity source attached to a planet entity (mu plus optional
/// spherical-harmonics coefficients). Queried by gravity controls
/// targeting this entity.
#[derive(Component, Debug, Clone, Deref, DerefMut)]
pub struct GravitySourceC(pub GravitySource);

/// RootInertial-frame position of a gravity source.
///
/// For the central body (e.g., Earth in an Earth-centered sim), this is
/// typically `Position::<RootInertial>::zero()`. For third bodies (Sun, Moon),
/// this value should be provided and maintained by the application's
/// ephemeris/update logic. Used by the gravity computation to apply
/// differential (third-body) acceleration corrections.
///
/// Required on all gravity source entities. The gravity systems will panic
/// if a source entity referenced by a `GravityControlsC` is missing this
/// component.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct SourceInertialPositionC(pub Position<RootInertial>);

/// RootInertial-frame velocity of a gravity source.
///
/// Optional component. For the central body (e.g., Earth in an Earth-centered
/// sim), this is typically `Velocity::<RootInertial>::zero()`. For third bodies
/// (Sun, Moon), attach this component alongside [`EphemerisBodyC`] and the
/// `ephemeris_update_system` will populate it each step. When absent,
/// relativistic corrections fall back to zero source velocity.
///
/// Used by the gravity and integration systems to provide source velocity to
/// the relativistic correction computation. Stored separately from
/// `TranslationalStateC` to avoid Bevy query conflicts (the body's
/// `TranslationalStateC` is already mutably queried by the integration system).
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct SourceInertialVelocityC(pub Velocity<RootInertial>);

/// Gravity gradient torque in the body frame (N·m).
///
/// Written by the gravity torque system.
/// Read by `force_collection_system` as `Option<&GravityTorqueC>`.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct GravityTorqueC(pub Torque<BodyFrame<SelfRef>>);

/// Typed inertial→planet-fixed rotation for a gravity source entity.
///
/// Stores the rotation that maps inertial-frame vectors into the planet-fixed
/// frame of the source. The `FrameTransform`'s phantom `<RootInertial,
/// PlanetFixed<P>>` parameters encode the *direction*. Every call site must
/// pin `P` explicitly — there is no fallback.
///
/// When present on a gravity source entity, `gravity_computation_system` and
/// `integration_system` use this rotation instead of `DMat3::IDENTITY` to
/// rotate the spacecraft position into the body-fixed frame before evaluating
/// spherical-harmonic gravity.
#[derive(Component, Debug, Clone, Copy)]
pub struct PlanetFixedRotationC<P: Planet>(pub FrameTransform<RootInertial, PlanetFixed<P>>);

/// Sidereal rotation rate (rad/s) used by `planet_fixed_rotation_system`
/// to populate [`PlanetAngularVelocityC`] each step. Sourced from
/// [`jeod_sim::PlanetConfig::omega`] at insertion (e.g. from
/// [`PlanetBundle::from_config`](crate::PlanetBundle::from_config)).
///
/// Issue #71 item 1: without this, velocity composition through
/// planet-fixed frames silently uses zero angular velocity, producing
/// the wrong NED-relative or geodetic velocity.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct PlanetOmegaC(pub f64);

/// Angular velocity of the planet-fixed frame relative to its inertial
/// parent, expressed in pfix coordinates. Computed each step by
/// `planet_fixed_rotation_system` as `[0, 0, omega]` matching JEOD's
/// `planet_rnp.cc`.
///
/// The `AngularVelocity<PlanetFixed<P>>` phantom indicates "in the
/// pfix frame of planet `P`". Every call site must pin `P` explicitly
/// — there is no fallback.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
pub struct PlanetAngularVelocityC<P: Planet>(pub AngularVelocity<PlanetFixed<P>>);

impl<P: Planet> Default for PlanetAngularVelocityC<P> {
    #[inline]
    fn default() -> Self {
        Self(AngularVelocity::<PlanetFixed<P>>::zero())
    }
}

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
#[derive(Component, Debug, Clone, Deref, DerefMut)]
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
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct TidalDeltaC20C(pub Ratio);

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

/// Marker component for the Sun entity (used by SRP system to find Sun position).
#[derive(Component, Default, Clone, Copy, Debug)]
pub struct SunMarker;

/// Marker component for the Moon entity (used by earth lighting system).
#[derive(Component, Default, Clone, Copy, Debug)]
pub struct MoonMarker;

/// Marks an entity as the simulation's designated central body.
///
/// [`crate::SourceMutator::set_source_position`] and
/// [`crate::SourceMutator::set_source_state`] panic if the target entity
/// carries this marker. Mission code attaches it to the gravity-source
/// entity it treats as the pinned origin (e.g. Earth in an Earth-centered
/// scenario), opting that entity into the same protection that
/// `jeod_runner::Simulation::set_source_*` enforces against the
/// root-mapped source via `assert_ne!(fid, root_frame_id, …)`.
///
/// At most one entity should carry this marker per simulation; multiple
/// markers are not currently rejected, but `SourceMutator` panics on
/// every call that targets a marked entity, so a well-behaved app
/// attaches it once.
#[derive(Component, Default, Clone, Copy, Debug)]
pub struct CentralSourceMarker;

// ── Planet ──

/// Bevy component wrapping `PlanetShape`.
#[derive(Component, Debug, Clone, Deref, DerefMut)]
pub struct PlanetC(pub PlanetShape);
