// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Vehicle-level configuration types.
//!
//! [`VehicleConfig`] is the user-facing description of a single simulated
//! vehicle: initial state plus all physics configuration. Mission code passes
//! one to `SimulationBuilder::add_body` (or to a Bevy spawn helper in Phase 9).
//!
//! Phase 6 of #101 relocated [`VehicleConfig`] and its companion option
//! structs out of `astrodyn_runner`; the runner and the future Bevy adapter both
//! consume this single description.

use glam::DMat3;

use crate::integrator::IntegratorType;
use crate::interactions::FlatPlateState;
use crate::EulerSequence;
use astrodyn_gravity::GravityControls;
use astrodyn_interactions::DragConfig;

use astrodyn_dynamics::state::TranslationalStateTyped;
use astrodyn_dynamics::{MassPropertiesTyped, RotationalStateTyped};
use astrodyn_quantities::frame::{RootInertial, SelfRef};
use astrodyn_quantities::frame_descriptor::FrameUid;

// ── Frame switching ─────────────────────────────────────────────────────

/// Trigger condition for a frame switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchSense {
    /// Switch when the body approaches the target frame origin.
    OnApproach,
    /// Switch when the body departs from the current frame origin.
    OnDeparture,
}

/// Configuration for a distance-based integration frame switch.
///
/// Port of JEOD's `DynBodyFrameSwitch` body action. When triggered, the
/// body's integration frame is reparented to the target source's inertial
/// frame in the frame tree, and gravity controls are flipped to make the
/// target source non-differential (central body).
///
/// The target is referenced by its inertial-frame [`FrameUid`] (issue
/// #668) — the same value identity in every host. The runner resolves
/// it against its source table; the Bevy adapter resolves it against
/// the registered source entities; a miss fails loudly naming the uid.
#[derive(Debug, Clone)]
pub struct FrameSwitchConfig {
    /// Identity of the gravity source's inertial frame to switch to
    /// (e.g. `FrameUid::of::<PlanetInertial<Moon>>()`). On switch, this
    /// source becomes non-differential and all others become
    /// differential, matching JEOD's `GravityInteraction::set_integ_frame()`.
    pub target: FrameUid,
    /// Whether to switch on approach or departure.
    pub switch_sense: SwitchSense,
    /// Distance threshold (meters).
    pub switch_distance: f64,
    /// Whether this switch is active.
    pub active: bool,
}

// ── Solar radiation pressure ────────────────────────────────────────────

/// Solar radiation pressure model — mutually exclusive variants.
///
/// The `FlatPlate` variant carries `FlatPlateState<SelfRef>`: this
/// adapter-neutral struct is the runtime-resolved boundary where the
/// vehicle phantom is `SelfRef` (the per-entity adapter knows the
/// concrete vehicle at runtime). The underlying
/// [`astrodyn_interactions::FlatPlate<V>`] is `<V: Vehicle>`-parametric so
/// mission code that pins a concrete vehicle (e.g.
/// `FlatPlateState<Iss>`) can demonstrate cross-vehicle compile-time
/// blocking before lowering through the runner; the runner-facing
/// `VehicleConfig` always lands at `<SelfRef>`.
#[derive(Debug, Clone)]
pub enum SrpModel {
    /// Per-plate modeling with thermal emission.
    FlatPlate(FlatPlateState<astrodyn_quantities::frame::SelfRef>),
    /// Simple cannonball model.
    Cannonball {
        /// Effective cross-section area (m²).
        cx_area: f64,
        /// Surface albedo.
        albedo: f64,
        /// Diffuse reflection fraction.
        diffuse: f64,
    },
}

// ── Shadow body ─────────────────────────────────────────────────────────

/// Shadow-casting body for SRP eclipse computation.
#[derive(Debug, Clone)]
pub struct ShadowBody {
    /// The shadow-casting gravity source's inertial-frame identity
    /// (issue #668).
    pub source: FrameUid,
    /// Body radius (m) for eclipse geometry.
    pub radius: f64,
}

// ── Geodetic computation ────────────────────────────────────────────────

/// Geodetic computation configuration.
#[derive(Debug, Clone)]
pub struct GeodeticConfig {
    /// The reference planet's inertial-frame identity (issue #668). The
    /// source must have `t_inertial_pfix` for planet-fixed rotation.
    pub source: FrameUid,
    /// Equatorial radius (m).
    pub r_eq: f64,
    /// Polar radius (m).
    pub r_pol: f64,
}

// ── NED frame (runtime) ─────────────────────────────────────────────────

/// Runtime NED-frame computation configuration (the per-step analog of
/// JEOD's `NedDerivedState`). Needs the same inputs as [`GeodeticConfig`] —
/// the planet-fixed rotation source plus ellipsoid radii — and additionally
/// reads the source's planet-rotation rate for the NED frame's pfix-relative
/// velocity.
#[derive(Debug, Clone)]
pub struct NedConfig {
    /// The reference planet's inertial-frame identity (issue #668). The
    /// source must have `t_inertial_pfix` for the planet-fixed rotation.
    pub source: FrameUid,
    /// Equatorial radius (m).
    pub r_eq: f64,
    /// Polar radius (m).
    pub r_pol: f64,
}

// ── Earth lighting ──────────────────────────────────────────────────────

/// Earth lighting computation configuration.
#[derive(Debug, Clone, Copy)]
pub struct EarthLightingConfig {
    /// Earth mean radius (m) for eclipse geometry.
    pub earth_radius: f64,
    /// Moon mean radius (m) for eclipse geometry.
    pub moon_radius: f64,
    /// Sun mean radius (m) for eclipse geometry.
    pub sun_radius: f64,
}

// ── Derived state requests ──────────────────────────────────────────────

/// All derived-state requests for a vehicle, grouped in one place.
#[derive(Debug, Clone, Default)]
pub struct DerivedStateConfig {
    /// The orbital-elements reference source's inertial-frame identity
    /// (issue #668). `None` = skip.
    pub orbital_elements_source: Option<FrameUid>,
    /// Euler angle decomposition sequence. `None` = skip.
    pub euler_sequence: Option<EulerSequence>,
    /// Whether to compute LVLH frame each step.
    pub lvlh: bool,
    /// Geodetic computation config. `None` = skip.
    pub geodetic: Option<GeodeticConfig>,
    /// Runtime NED-frame computation config. `None` = skip.
    pub ned: Option<NedConfig>,
    /// Whether to compute solar beta angle. Requires `sun_source` on Simulation.
    pub solar_beta: bool,
    /// Earth lighting config. Requires `sun_source` and `moon_source`.
    pub earth_lighting: Option<EarthLightingConfig>,
}

// ── Vehicle configuration ───────────────────────────────────────────────

/// User-facing vehicle configuration.
///
/// Passed to `SimulationBuilder::add_body` to create a simulated
/// vehicle. Contains initial state plus all physics configuration.
/// `VehicleConfig` is adapter-neutral: it has no output fields, and
/// results are read back via the adapter's own output view (the
/// standalone runner exposes one; the Bevy adapter reads components).
pub struct VehicleConfig {
    /// Mission-supplied runtime identity for this vehicle's composite-body
    /// frame. **Required — no default**: a defaulted identity would be
    /// identity-by-accident and would let two bodies silently collide.
    /// Construct configs via [`VehicleConfig::for_vehicle`] (typed,
    /// `FrameUid::of::<BodyFrame<V>>()`), [`VehicleConfig::named`]
    /// (mission-named value identity), [`VehicleConfig::for_uid`], or the
    /// `VehicleBuilder` identity stage (`.vehicle::<V>()` /
    /// `.vehicle_named(..)`).
    pub frame_uid: FrameUid,
    // ── Initial state ──
    /// Translational state in the root-inertial frame, typed
    /// end-to-end. The runner re-tags as `<IntegrationFrame>` at
    /// `SimBody::new`; the Bevy adapter relabels to
    /// `<PlanetInertial<P>>` via the `From<TranslationalStateTyped<RootInertial>>`
    /// component impl. Mission code that constructs `VehicleConfig`
    /// directly via struct literal can pass an untyped
    /// `TranslationalState` via `.into()` (the
    /// `From<TranslationalState> for TranslationalStateTyped<F>` impl
    /// in `astrodyn_dynamics` lifts at the boundary).
    pub trans: TranslationalStateTyped<RootInertial>,
    /// Rotational state (typed). `None` for 3-DOF bodies. The vehicle
    /// phantom is the runtime-resolved wildcard `<SelfRef>` (JEOD_INV
    /// `TS.01`); the runner / Bevy adapter drops to raw at the
    /// construction boundary. Mission code can pass an untyped
    /// `RotationalState` via `.into()` (the
    /// `From<RotationalState> for RotationalStateTyped<V>` impl in
    /// `astrodyn_dynamics` lifts at the boundary).
    pub rot: Option<RotationalStateTyped<SelfRef>>,
    /// Mass properties (typed). `None` for massless test particles
    /// (gravity-only). Phantom is `<SelfRef>` (JEOD_INV `TS.01`);
    /// mission code can pass an untyped `MassProperties` via `.into()`.
    pub mass: Option<MassPropertiesTyped<SelfRef>>,

    // ── Dynamics ──
    /// Integration method. Defaults to `IntegratorType::Rk4`.
    pub integrator: IntegratorType,
    /// Structural-to-body rotation matrix. `DMat3::IDENTITY` when structure = body.
    pub t_struct_body: DMat3,

    // ── Gravity ──
    /// Gravity controls referencing sources by inertial-frame identity
    /// (issue #668).
    pub gravity_controls: GravityControls,
    /// Whether to compute gravity gradient (needed for gravity torque).
    pub compute_gravity_gradient: bool,

    // ── Interactions ──
    /// Drag configuration. `None` disables drag.
    pub drag: Option<DragConfig>,
    /// Solar radiation pressure model. `None` disables SRP.
    pub srp: Option<SrpModel>,
    /// Shadow-casting body for SRP eclipse. `None` = full illumination.
    pub shadow_body: Option<ShadowBody>,

    // ── Derived state requests ──
    /// Derived state computation requests.
    pub derived: DerivedStateConfig,

    // ── External loads ──
    /// External force in the root-inertial frame, typed end-to-end.
    pub external_force:
        astrodyn_quantities::aliases::Force<astrodyn_quantities::frame::RootInertial>,
    /// External torque in the body frame, typed against the wildcard
    /// vehicle phantom `<SelfRef>` at this storage boundary
    /// (per-vehicle phantom is runtime-resolved by the runner / Bevy
    /// adapter; documented under JEOD_INV `TS.01`).
    pub external_torque: astrodyn_quantities::aliases::Torque<
        astrodyn_quantities::frame::BodyFrame<astrodyn_quantities::frame::SelfRef>,
    >,

    // ── Frame switching ──
    /// Gravity source whose inertial frame is used for integration,
    /// referenced by inertial-frame identity (issue #668). `None` means
    /// the root frame. `Some(uid)` means the inertial frame of the
    /// source carrying that identity.
    ///
    /// **Non-root caveat (issue #263).** When `Some(...)`, the
    /// integrated translational state is integ-frame-relative rather
    /// than root-inertial, but downstream Bevy storage
    /// (`TranslationalStateC<RootInertial>`) is still tagged
    /// `RootInertial` — issue #263 Section A.1. Derived-state
    /// consumers that read the state as absolute root-inertial
    /// (geodetic vs. another planet, solar-beta, SRP relative to a Sun
    /// position not in the integ frame) will silently produce wrong
    /// answers. Until #263 closes, mission code should either avoid
    /// non-root integration or restrict derived states to ones
    /// evaluated in the same source's frame.
    pub integ_source: Option<FrameUid>,
    /// Distance-based frame switch triggers.
    pub frame_switches: Vec<FrameSwitchConfig>,
}

impl VehicleConfig {
    /// Base constructor carrying a caller-supplied frame identity; every
    /// other field takes its neutral default. The identity-bearing
    /// replacement for the removed `impl Default` (a defaulted identity
    /// would be identity-by-accident).
    pub fn for_uid(frame_uid: FrameUid) -> Self {
        Self {
            frame_uid,
            trans: TranslationalStateTyped::<RootInertial>::default(),
            rot: None,
            mass: None,
            integrator: IntegratorType::default(),
            t_struct_body: DMat3::IDENTITY,
            gravity_controls: GravityControls::default(),
            compute_gravity_gradient: false,
            drag: None,
            srp: None,
            shadow_body: None,
            derived: DerivedStateConfig::default(),
            external_force: astrodyn_quantities::aliases::Force::<
                astrodyn_quantities::frame::RootInertial,
            >::zero(),
            external_torque: astrodyn_quantities::aliases::Torque::<
                astrodyn_quantities::frame::BodyFrame<astrodyn_quantities::frame::SelfRef>,
            >::zero(),
            integ_source: None,
            frame_switches: Vec::new(),
        }
    }

    /// Typed base constructor: identity derived from the compile-time
    /// vehicle marker (`FrameUid::of::<BodyFrame<V>>()`, namespace LOCAL).
    pub fn for_vehicle<V: astrodyn_quantities::frame::Vehicle>() -> Self {
        Self::for_uid(FrameUid::of::<astrodyn_quantities::frame::BodyFrame<V>>())
    }

    /// Named base constructor: mission-named value identity in
    /// [`crate::MISSION_NAMED_NS`] via the shared
    /// [`crate::named_body_frame_uid`] mint.
    pub fn named(name: impl Into<String>) -> Self {
        Self::for_uid(crate::named_body_frame_uid(&name.into()))
    }
}
