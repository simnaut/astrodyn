//! Vehicle-level configuration types.
//!
//! [`VehicleConfig`] is the user-facing description of a single simulated
//! vehicle: initial state plus all physics configuration. Mission code passes
//! one to `SimulationBuilder::add_body` (or to a Bevy spawn helper in Phase 9).
//!
//! Phase 6 of #101 relocated [`VehicleConfig`] and its companion option
//! structs out of `jeod_runner`; the runner and the future Bevy adapter both
//! consume this single description.

use glam::{DMat3, DVec3};

use crate::interactions::FlatPlateState;
use crate::EulerSequence;
use jeod_dynamics::IntegratorType;
use jeod_gravity::GravityControls;
use jeod_interactions::DragConfig;

use jeod_dynamics::{MassProperties, RotationalState, TranslationalState};

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
/// Generic over `SourceId` to mirror [`jeod_gravity::GravityControls`]:
/// `jeod_runner::Simulation` uses the default `SourceId = usize` (sources
/// are identified by their registration order); the Bevy adapter uses
/// `SourceId = bevy::ecs::entity::Entity` (sources are identified by
/// their ECS entity). The lifted
/// [`crate::evaluate_and_apply_frame_switch`] helper is generic over the
/// same type so both consumers share one implementation.
#[derive(Debug, Clone)]
pub struct FrameSwitchConfig<SourceId = usize> {
    /// Identifier of the gravity source whose inertial frame to switch to.
    /// On switch, this source becomes non-differential and all others become
    /// differential, matching JEOD's `GravityInteraction::set_integ_frame()`.
    pub target_source: SourceId,
    /// Whether to switch on approach or departure.
    pub switch_sense: SwitchSense,
    /// Distance threshold (meters).
    pub switch_distance: f64,
    /// Whether this switch is active.
    pub active: bool,
}

// ── Solar radiation pressure ────────────────────────────────────────────

/// Solar radiation pressure model — mutually exclusive variants.
#[derive(Debug, Clone)]
pub enum SrpModel {
    /// Per-plate modeling with thermal emission.
    FlatPlate(FlatPlateState),
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
#[derive(Debug, Clone, Copy)]
pub struct ShadowBody {
    /// Index into the gravity source table.
    pub source_idx: usize,
    /// Body radius (m) for eclipse geometry.
    pub radius: f64,
}

// ── Geodetic computation ────────────────────────────────────────────────

/// Geodetic computation configuration.
#[derive(Debug, Clone, Copy)]
pub struct GeodeticConfig {
    /// Gravity source index (must have `t_inertial_pfix` for planet-fixed rotation).
    pub source_idx: usize,
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
    /// Gravity source index for orbital elements. `None` = skip.
    pub orbital_elements_source: Option<usize>,
    /// Euler angle decomposition sequence. `None` = skip.
    pub euler_sequence: Option<EulerSequence>,
    /// Whether to compute LVLH frame each step.
    pub lvlh: bool,
    /// Geodetic computation config. `None` = skip.
    pub geodetic: Option<GeodeticConfig>,
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
    // ── Initial state ──
    /// Translational state: position and velocity in the inertial frame.
    pub trans: TranslationalState,
    /// Rotational state: quaternion and angular velocity. `None` for 3-DOF bodies.
    pub rot: Option<RotationalState>,
    /// Mass properties. `None` for massless test particles (gravity-only).
    pub mass: Option<MassProperties>,

    // ── Dynamics ──
    /// Integration method. Defaults to `IntegratorType::Rk4`.
    pub integrator: IntegratorType,
    /// Structural-to-body rotation matrix. `DMat3::IDENTITY` when structure = body.
    pub t_struct_body: DMat3,

    // ── Gravity ──
    /// Gravity controls referencing sources by index.
    pub gravity_controls: GravityControls<usize>,
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
    /// External force in the inertial frame (N). Defaults to zero.
    pub external_force: DVec3,
    /// External torque in the body frame (N·m). Defaults to zero.
    pub external_torque: DVec3,

    // ── Frame switching ──
    /// Gravity source whose inertial frame is used for integration.
    /// `None` means the root frame (Earth.inertial). `Some(idx)` means
    /// the inertial frame of the source at that index.
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
    pub integ_source: Option<usize>,
    /// Distance-based frame switch triggers.
    pub frame_switches: Vec<FrameSwitchConfig>,
}

impl Default for VehicleConfig {
    fn default() -> Self {
        Self {
            trans: TranslationalState::default(),
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
            external_force: DVec3::ZERO,
            external_torque: DVec3::ZERO,
            integ_source: None,
            frame_switches: Vec::new(),
        }
    }
}
