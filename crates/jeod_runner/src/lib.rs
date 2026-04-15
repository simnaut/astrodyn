//! Standalone simulation runner for JEOD physics.
//!
//! Provides a [`Simulation`] struct for batch propagation, scripting, and
//! Tier 3 cross-validation tests. Owns all state and runs the `jeod_sim`
//! pipeline internally.
//!
//! ECS adapters should **not** depend on this crate — use the per-body
//! functions from `jeod_sim` directly instead.
//!
//! # Example
//! ```ignore
//! use jeod_runner::{Simulation, VehicleConfig, GravitySourceEntry};
//! use jeod_sim::{SimulationTime, EARTH};
//!
//! let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
//! let mut sim = Simulation::new(time, 10.0);
//! let earth = sim.add_source(GravitySourceEntry::central_body(&EARTH));
//! let vehicle = sim.add_body(VehicleConfig {
//!     trans: initial_state,
//!     gravity_controls: controls,
//!     ..Default::default()
//! });
//! sim.validate().unwrap();
//! sim.step_n(100);
//! let output = sim.body(vehicle);
//! println!("{:?}", output.trans.position);
//! ```

use glam::{DMat3, DVec3};

use jeod_frames::{FrameTree, RefFrameKind, RefFrameRot, RefFrameState, RefFrameTrans};
use jeod_sim::atmosphere::{evaluate_atmosphere, AtmosphereConfig};
use jeod_sim::forces::collect_and_resolve_forces;
use jeod_sim::gravity::accumulate_gravity;
use jeod_sim::integration::integrate_body;
use jeod_sim::validation::ValidationError;
use jeod_sim::{
    AerodynamicForce, AtmosphereState, DragConfig, DynamicsConfig, EulerSequence, FrameDerivatives,
    GeodeticState, GravityAcceleration, GravityControls, GravitySource, JeodQuat, LvlhFrame,
    MassProperties, OrbitalElements, PlanetConfig, RadiationForce, RotationalState, SimulationTime,
    TotalForce, TranslationalState,
};

pub mod builder;

// Re-export jeod_sim so downstream tests can access types through either path.
pub use jeod_sim;
pub use jeod_sim::RotationModel;

// Re-export builder types for ergonomic use.
pub use builder::{SimulationBuilder, VehicleBuilder};

// Re-export FrameId for downstream API.
pub use jeod_frames::FrameId;

// ══════════════════════════════════════════════════════════════════════════════
// Integration frame switching
// ══════════════════════════════════════════════════════════════════════════════

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
#[derive(Debug, Clone)]
pub struct FrameSwitchConfig {
    /// Index of the gravity source whose inertial frame to switch to.
    /// On switch, this source becomes non-differential and all others become
    /// differential, matching JEOD's `GravityInteraction::set_integ_frame()`.
    pub target_source: usize,
    /// Whether to switch on approach or departure.
    pub switch_sense: SwitchSense,
    /// Distance threshold (meters).
    pub switch_distance: f64,
    /// Whether this switch is active.
    pub active: bool,
}

// ══════════════════════════════════════════════════════════════════════════════
// Frame tree source registry
// ══════════════════════════════════════════════════════════════════════════════

/// Maps a gravity source to its frame tree nodes.
struct SourceFrameIds {
    /// Inertial frame for this source (e.g., "Earth.inertial").
    inertial: FrameId,
    /// Planet-fixed frame (e.g., "Earth.pfix"), if the source has a rotation model.
    pfix: Option<FrameId>,
}

/// Gravity-specific data associated with a source (decoupled from frame tree).
///
/// The frame tree stores position/velocity/rotation state; this struct stores
/// the physical gravity model data that lives alongside it. The `velocity`
/// field stores source velocity for relativistic corrections — for central
/// bodies at the root frame, the tree node has zero velocity but the source
/// may still have physical velocity (e.g., Sun orbiting the barycenter).
struct GravityData {
    /// Physical gravity source (mu, model: PointMass or SphericalHarmonics).
    source: GravitySource,
    /// Source velocity in the inertial frame (m/s). Used for relativistic
    /// corrections. Stored here rather than in the tree because the root
    /// frame's velocity must be zero (it's the reference origin).
    velocity: DVec3,
    /// Tidal ΔC20 to add to the base C20 coefficient. Updated each step.
    delta_c20: f64,
    /// Tidal configuration. When `Some`, the simulation computes ΔC20 each step.
    tidal_config: Option<jeod_gravity::tides::TidalConfig>,
    /// Rotation model for updating planet-fixed frame each step.
    rotation_model: RotationModel,
}

// ══════════════════════════════════════════════════════════════════════════════
// Gravity source
// ══════════════════════════════════════════════════════════════════════════════

/// Entry in the gravity source table.
///
/// Gravity sources are referenced by index (`usize`) from body gravity controls.
pub struct GravitySourceEntry {
    /// Physical gravity source (mu, model).
    pub source: GravitySource,
    /// Position in the inertial frame (m). For Earth-centered sims, Earth is at origin.
    pub position: DVec3,
    /// Velocity in the inertial frame (m/s). Required for relativistic corrections.
    /// Zero for stationary sources (e.g., central body at origin).
    pub velocity: DVec3,
    /// Inertial-to-planet-fixed rotation matrix. Updated each step when
    /// `rotation_model` is not `None`. If `None`, no rotation is applied
    /// (point-mass only).
    pub t_inertial_pfix: Option<DMat3>,
    /// Rotation model for updating `t_inertial_pfix` each step.
    pub rotation_model: RotationModel,
    /// Tidal ΔC20 to add to the base C20 coefficient before spherical harmonics
    /// evaluation. Updated each step by the environment stage if tidal effects
    /// are configured. Zero when no tides.
    pub delta_c20: f64,
    /// Tidal configuration. When `Some`, the simulation computes ΔC20 each step.
    pub tidal_config: Option<jeod_gravity::tides::TidalConfig>,
    /// Whether this is the central body (uses the root frame in the tree).
    /// Set automatically by [`central_body`](Self::central_body) and
    /// [`central_body_sh`](Self::central_body_sh).
    pub central: bool,
}

impl GravitySourceEntry {
    /// Create a new gravity source entry without tidal effects.
    ///
    /// `rotation_model` defaults to `None`. Set it explicitly after construction
    /// (or use struct literal syntax) to enable per-step rotation updates.
    pub fn new(source: GravitySource, position: DVec3, t_inertial_pfix: Option<DMat3>) -> Self {
        Self {
            source,
            position,
            velocity: DVec3::ZERO,
            t_inertial_pfix,
            rotation_model: RotationModel::None,
            delta_c20: 0.0,
            tidal_config: None,
            central: false,
        }
    }

    /// Central body at the origin with point-mass gravity and rotation from a
    /// [`PlanetConfig`] preset.
    ///
    /// Sets rotation model and initial identity rotation matrix (if the planet
    /// has a rotation model). Position and velocity are zero (central body).
    pub fn central_body(planet: &PlanetConfig) -> Self {
        Self {
            source: GravitySource {
                mu: planet.shape.mu,
                model: jeod_sim::GravityModel::PointMass,
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: if planet.rotation_model != RotationModel::None {
                Some(DMat3::IDENTITY)
            } else {
                None
            },
            rotation_model: planet.rotation_model,
            delta_c20: 0.0,
            tidal_config: None,
            central: true,
        }
    }

    /// Central body at the origin with spherical harmonics gravity and rotation
    /// from a [`PlanetConfig`] preset.
    ///
    /// Uses `mu` from the spherical harmonics data (which may differ slightly
    /// from the planet preset's geodetic mu).
    pub fn central_body_sh(
        planet: &PlanetConfig,
        sh_data: jeod_gravity::SphericalHarmonicsData,
    ) -> Self {
        Self {
            source: GravitySource {
                mu: sh_data.mu,
                model: jeod_sim::GravityModel::SphericalHarmonics(Box::new(sh_data)),
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: if planet.rotation_model != RotationModel::None {
                Some(DMat3::IDENTITY)
            } else {
                None
            },
            rotation_model: planet.rotation_model,
            delta_c20: 0.0,
            tidal_config: None,
            central: true,
        }
    }

    /// Third body (perturbation source) at a given position.
    ///
    /// Point-mass only, no rotation. Typical use: Sun or Moon as a third-body
    /// perturbation in Earth-centered integration.
    pub fn third_body(planet: &PlanetConfig, position: DVec3) -> Self {
        Self {
            source: GravitySource {
                mu: planet.shape.mu,
                model: jeod_sim::GravityModel::PointMass,
            },
            position,
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            rotation_model: RotationModel::None,
            delta_c20: 0.0,
            tidal_config: None,
            central: false,
        }
    }

    /// Add tidal configuration (builder-style, consumes and returns self).
    pub fn with_tidal(mut self, config: jeod_gravity::tides::TidalConfig) -> Self {
        self.tidal_config = Some(config);
        self
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Vehicle configuration (public, user-facing)
// ══════════════════════════════════════════════════════════════════════════════

/// Solar radiation pressure model — mutually exclusive variants.
#[derive(Debug, Clone)]
pub enum SrpModel {
    /// Per-plate modeling with thermal emission.
    FlatPlate(jeod_sim::FlatPlateState),
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

/// Shadow-casting body for SRP eclipse computation.
#[derive(Debug, Clone, Copy)]
pub struct ShadowBody {
    /// Index into the gravity source table.
    pub source_idx: usize,
    /// Body radius (m) for eclipse geometry.
    pub radius: f64,
}

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

/// User-facing vehicle configuration.
///
/// Passed to [`Simulation::add_body`] to create a simulated vehicle.
/// Contains initial state plus all physics configuration. No output fields —
/// results are accessed via [`Simulation::body`] which returns [`VehicleOutput`].
///
/// Use the builder ([`VehicleConfig::builder`]) for ergonomic construction, or
/// struct literal syntax with `..Default::default()` for direct access.
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
    pub integrator: jeod_dynamics::IntegratorType,
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
            integrator: jeod_dynamics::IntegratorType::default(),
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

// ══════════════════════════════════════════════════════════════════════════════
// Vehicle output (public, read-only view of results after step)
// ══════════════════════════════════════════════════════════════════════════════

/// Read-only view of vehicle state after stepping.
///
/// Returned by [`Simulation::body`]. Contains the current integrated state
/// plus any derived states that were configured.
#[derive(Debug, Clone)]
pub struct VehicleOutput {
    /// Current translational state (position, velocity) in the integration frame.
    pub trans: TranslationalState,
    /// Frame ID of the current integration frame in the simulation's frame tree.
    pub integ_frame_id: FrameId,
    /// Current rotational state (quaternion, angular velocity). `None` for 3-DOF.
    pub rot: Option<RotationalState>,
    /// Orbital elements from the latest step.
    pub orbital_elements: Option<OrbitalElements>,
    /// Euler angles `[phi, theta, psi]` from the latest step.
    pub euler_angles: Option<[f64; 3]>,
    /// LVLH frame from the latest step.
    pub lvlh_frame: Option<LvlhFrame>,
    /// Geodetic state (latitude, longitude, altitude).
    pub geodetic_state: Option<GeodeticState>,
    /// Solar beta angle (radians).
    pub solar_beta: Option<f64>,
    /// Earth lighting state (sun/moon occlusion, albedo).
    pub earth_lighting: Option<jeod_interactions::earth_lighting::EarthLightingState>,
}

// ══════════════════════════════════════════════════════════════════════════════
// Internal body state (private — not part of public API)
// ══════════════════════════════════════════════════════════════════════════════

/// Internal per-body simulation state. Combines user config with bookkeeping
/// and output fields. Not exposed publicly — users interact through
/// [`VehicleConfig`] (input) and [`VehicleOutput`] (output).
struct SimBody {
    // ── Config (from VehicleConfig) ──
    trans: TranslationalState,
    rot: Option<RotationalState>,
    mass: Option<MassProperties>,
    /// If this body participates in a mass tree, its node ID.
    mass_body_id: Option<jeod_dynamics::MassBodyId>,
    config: DynamicsConfig,
    gravity_controls: GravityControls<usize>,
    integrator: jeod_dynamics::IntegratorType,
    drag: Option<DragConfig>,
    flat_plate_state: Option<jeod_sim::FlatPlateState>,
    cannonball_srp: Option<(f64, f64, f64)>,
    shadow_body: Option<(usize, f64)>,
    t_struct_body: DMat3,
    compute_gravity_torque: bool,
    atmospheric_state: Option<AtmosphereState>,
    external_force: DVec3,
    external_torque: DVec3,

    // ── Frame switching ──
    integ_frame_id: FrameId,
    body_frame_id: FrameId,
    frame_switches: Vec<FrameSwitchConfig>,

    // ── Bookkeeping (written each step, not user-visible) ──
    gravity_accel: GravityAcceleration,
    total_force: TotalForce,
    frame_derivs: FrameDerivatives,
    aero_force: Option<AerodynamicForce>,
    radiation_force: Option<RadiationForce>,
    gravity_torque: Option<DVec3>,

    // ── Derived state config ──
    orbital_elements_source: Option<usize>,
    euler_sequence: Option<EulerSequence>,
    compute_lvlh: bool,
    geodetic_planet: Option<(usize, f64, f64)>,
    compute_solar_beta: bool,
    earth_lighting_config: Option<(f64, f64, f64)>,

    // ── Derived state outputs ──
    orbital_elements: Option<OrbitalElements>,
    euler_angles: Option<[f64; 3]>,
    lvlh_frame: Option<LvlhFrame>,
    geodetic_state: Option<GeodeticState>,
    solar_beta: Option<f64>,
    earth_lighting: Option<jeod_interactions::earth_lighting::EarthLightingState>,

    // ── Integrator state ──
    gj_state: Option<jeod_dynamics::GaussJacksonState>,
}

impl SimBody {
    /// Convert a user-facing VehicleConfig into an internal SimBody.
    fn from_config(config: VehicleConfig, integ_frame_id: FrameId, body_frame_id: FrameId) -> Self {
        let has_rot = config.rot.is_some();
        let dynamics_config = DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: has_rot,
            three_dof: !has_rot,
        };

        let (flat_plate_state, cannonball_srp) = match config.srp {
            Some(SrpModel::FlatPlate(fps)) => (Some(fps), None),
            Some(SrpModel::Cannonball {
                cx_area,
                albedo,
                diffuse,
            }) => (None, Some((cx_area, albedo, diffuse))),
            None => (None, None),
        };

        let shadow_body = config.shadow_body.map(|sb| (sb.source_idx, sb.radius));

        let has_drag = config.drag.is_some();
        let atmospheric_state = if has_drag {
            Some(AtmosphereState::default())
        } else {
            None
        };

        Self {
            trans: config.trans,
            rot: config.rot,
            mass: config.mass,
            mass_body_id: None,
            config: dynamics_config,
            gravity_controls: config.gravity_controls,
            integrator: config.integrator,
            drag: config.drag,
            flat_plate_state,
            cannonball_srp,
            shadow_body,
            t_struct_body: config.t_struct_body,
            compute_gravity_torque: config.compute_gravity_gradient,
            atmospheric_state,
            external_force: config.external_force,
            external_torque: config.external_torque,

            integ_frame_id,
            body_frame_id,
            frame_switches: config.frame_switches,

            gravity_accel: GravityAcceleration::default(),
            total_force: TotalForce::default(),
            frame_derivs: FrameDerivatives::default(),
            aero_force: None,
            radiation_force: None,
            gravity_torque: None,

            orbital_elements_source: config.derived.orbital_elements_source,
            euler_sequence: config.derived.euler_sequence,
            compute_lvlh: config.derived.lvlh,
            geodetic_planet: config
                .derived
                .geodetic
                .map(|g| (g.source_idx, g.r_eq, g.r_pol)),
            compute_solar_beta: config.derived.solar_beta,
            earth_lighting_config: config
                .derived
                .earth_lighting
                .map(|e| (e.earth_radius, e.moon_radius, e.sun_radius)),

            orbital_elements: None,
            euler_angles: None,
            lvlh_frame: None,
            geodetic_state: None,
            solar_beta: None,
            earth_lighting: None,

            gj_state: None,
        }
    }

    /// Create a VehicleOutput view of the current state.
    fn output(&self) -> VehicleOutput {
        VehicleOutput {
            trans: self.trans,
            integ_frame_id: self.integ_frame_id,
            rot: self.rot,
            orbital_elements: self.orbital_elements.clone(),
            euler_angles: self.euler_angles,
            lvlh_frame: self.lvlh_frame,
            geodetic_state: self.geodetic_state,
            solar_beta: self.solar_beta,
            earth_lighting: self.earth_lighting.clone(),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Simulation
// ══════════════════════════════════════════════════════════════════════════════

/// ECS-agnostic simulation runner.
///
/// Owns all simulation state and runs the JEOD pipeline in `step()`.
/// This is the **non-ECS** path — ECS adapters should call the per-body
/// functions (`accumulate_gravity`, `evaluate_atmosphere`, etc.) directly
/// from their system functions.
///
/// # Example
/// ```ignore
/// let mut sim = Simulation::new(time, 10.0);
/// let earth = sim.add_source("Earth", GravitySourceEntry::central_body(&EARTH));
/// let vehicle = sim.add_body(VehicleConfig {
///     trans: initial_state,
///     gravity_controls: controls,
///     ..Default::default()
/// });
/// sim.validate().unwrap();
/// sim.step_n(100);
/// let output = sim.body(vehicle);
/// ```
pub struct Simulation {
    /// Simulation time (TAI, UTC, TDB, GMST, etc.).
    pub time: SimulationTime,
    /// Dynamic bodies (internal, private).
    // JEOD_INV: DS.01 — private to prevent runtime mutation of derived-state config
    bodies: Vec<SimBody>,
    /// Reference frame tree — single source of truth for celestial body positions,
    /// velocities, and rotations. Updated each step from ephemeris data.
    /// Private to protect invariants; use [`frame_tree()`](Self::frame_tree) for
    /// read-only access.
    frame_tree: FrameTree,
    /// Root inertial frame ID for this simulation. This is the integration-origin
    /// frame to which all positions are relative, and it is not necessarily
    /// `Earth.inertial` (for example, it may be renamed to match the configured
    /// central body, such as `Mars.inertial`).
    pub root_frame_id: FrameId,
    /// Per-source frame tree node IDs (parallel to `gravity_data`).
    source_frame_ids: Vec<SourceFrameIds>,
    /// Per-source gravity model data (parallel to `source_frame_ids`).
    gravity_data: Vec<GravityData>,
    /// Per-source ephemeris body mapping (parallel to `source_frame_ids`).
    source_ephem_bodies: Vec<Option<(jeod_sim::EphemerisBody, jeod_sim::EphemerisBody)>>,
    /// Atmosphere configuration. `None` disables atmosphere for all bodies.
    pub atmosphere: Option<AtmosphereConfig>,
    /// Source index for the planet whose rotation is used for atmosphere.
    pub atmosphere_planet_source: Option<usize>,
    /// Source index for the Sun (used by SRP and earth lighting).
    pub sun_source: Option<usize>,
    /// Source index for the Moon (used by earth lighting).
    pub moon_source: Option<usize>,
    /// Polar motion parameters (xp, yp) in radians. When `Some`, the RNP
    /// composition includes polar motion: W(xp,yp) × R(GAST) × N × P.
    /// When `None`, polar motion is omitted (matches JEOD `enable_polar=false`).
    pub polar_motion: Option<(f64, f64)>,
    /// Integration timestep (seconds).
    pub dt: f64,
    /// Optional ephemeris for per-step source position updates.
    pub ephemeris: Option<jeod_sim::Ephemeris>,
    /// Optional mass tree for multi-body vehicles (attach/detach/staging).
    /// Bodies participating in the tree have `SimBody::mass_body_id` set.
    pub mass_tree: Option<jeod_dynamics::MassTree>,
}

impl Simulation {
    /// Create a new simulation with the given initial time and timestep.
    ///
    /// Creates a frame tree with a root "Earth.inertial" frame (all positions
    /// are relative to this).
    pub fn new(time: SimulationTime, dt: f64) -> Self {
        let mut frame_tree = FrameTree::new();
        let root_frame_id = frame_tree.add_root("Earth.inertial".into(), RefFrameKind::Inertial);
        Self {
            time,
            bodies: Vec::new(),
            frame_tree,
            root_frame_id,
            source_frame_ids: Vec::new(),
            gravity_data: Vec::new(),
            source_ephem_bodies: Vec::new(),
            atmosphere: None,
            atmosphere_planet_source: None,
            sun_source: None,
            moon_source: None,
            polar_motion: None,
            dt,
            ephemeris: None,
            mass_tree: None,
        }
    }

    /// Add a gravity source. Returns its index for use in `GravityControls`.
    ///
    /// Sources with `central: true` (set by [`GravitySourceEntry::central_body`]
    /// and [`GravitySourceEntry::central_body_sh`]) are mapped to the root frame.
    /// Non-central sources get child inertial frames under the root.
    ///
    /// Only one central source may be added; a second will panic.
    ///
    /// If the source has a rotation model, a planet-fixed child frame is also
    /// created under the source's inertial frame.
    pub fn add_source(&mut self, name: impl Into<String>, entry: GravitySourceEntry) -> usize {
        let idx = self.gravity_data.len();
        let name = name.into();

        // Central bodies map to the root frame; third bodies get child frames.
        // Only one central source is allowed (the root can't be shared).
        let inertial_name = format!("{name}.inertial");
        let inertial_id = if entry.central {
            assert!(
                !self
                    .source_frame_ids
                    .iter()
                    .any(|sf| sf.inertial == self.root_frame_id),
                "add_source: a central source already maps to root_frame_id. \
                 Only one central source is allowed per simulation."
            );
            assert!(
                entry.position == DVec3::ZERO,
                "add_source: central sources must have zero position because they map \
                 directly to root_frame_id."
            );
            // Central body: use the root frame directly. Rename to match.
            // `entry.velocity` is stored in `gravity_data` for relativistic
            // corrections, but is not applied as root-frame kinematics.
            self.frame_tree.get_mut(self.root_frame_id).name = inertial_name;
            self.root_frame_id
        } else {
            self.frame_tree.add_child(
                self.root_frame_id,
                inertial_name,
                RefFrameKind::Inertial,
                RefFrameState {
                    trans: RefFrameTrans {
                        position: entry.position,
                        velocity: entry.velocity,
                    },
                    rot: RefFrameRot::default(),
                },
            )
        };

        // Create a planet-fixed child when the source has a rotation model or
        // an explicit inertial-to-pfix transform. This ensures a fixed initial
        // orientation is not silently ignored when rotation_model is None.
        let pfix_id =
            if entry.rotation_model != RotationModel::None || entry.t_inertial_pfix.is_some() {
                let pfix_name = format!("{name}.pfix");
                let rot = if let Some(t) = entry.t_inertial_pfix {
                    RefFrameRot {
                        q_parent_this: JeodQuat::left_quat_from_transformation(&t),
                        t_parent_this: t,
                        ang_vel_this: DVec3::ZERO,
                    }
                } else {
                    RefFrameRot::default()
                };
                Some(self.frame_tree.add_child(
                    inertial_id,
                    pfix_name,
                    RefFrameKind::PlanetFixed,
                    RefFrameState {
                        trans: RefFrameTrans::default(),
                        rot,
                    },
                ))
            } else {
                None
            };

        // Tidal ΔC20 requires a planet-fixed frame for the rotation matrix.
        assert!(
            entry.tidal_config.is_none() || pfix_id.is_some(),
            "add_source: tidal_config requires a planet-fixed frame \
             (set rotation_model or t_inertial_pfix on the source)."
        );

        self.source_frame_ids.push(SourceFrameIds {
            inertial: inertial_id,
            pfix: pfix_id,
        });
        self.gravity_data.push(GravityData {
            source: entry.source,
            velocity: entry.velocity,
            delta_c20: entry.delta_c20,
            tidal_config: entry.tidal_config,
            rotation_model: entry.rotation_model,
        });
        self.source_ephem_bodies.push(None);
        idx
    }

    /// Configure ephemeris-based position updates for a source.
    /// Each step, the source's position and velocity will be updated from DE4xx.
    ///
    /// `target` is the body this source represents (e.g., `EphemerisBody::Sun`).
    /// `observer` is the integration frame center (e.g., `EphemerisBody::Earth`).
    pub fn set_source_ephemeris(
        &mut self,
        source_idx: usize,
        target: jeod_sim::EphemerisBody,
        observer: jeod_sim::EphemerisBody,
    ) {
        assert!(
            source_idx < self.source_ephem_bodies.len(),
            "set_source_ephemeris: source_idx {source_idx} out of bounds (len = {})",
            self.source_ephem_bodies.len()
        );
        // Root-frame conflict is caught by validate() → EphemerisOnRootSource.
        // We don't panic here so that all misconfiguration errors are reported
        // together in a single validate() pass rather than aborting on the first.
        self.source_ephem_bodies[source_idx] = Some((target, observer));
    }

    /// Add a dynamic body from a [`VehicleConfig`]. Returns its index.
    ///
    /// The config is consumed and converted into internal state. Creates a
    /// body frame in the frame tree under the integration frame. Use
    /// [`body`](Simulation::body) to access results after stepping.
    pub fn add_body(&mut self, config: VehicleConfig) -> usize {
        let idx = self.bodies.len();

        // Resolve integration frame from source index.
        let integ_frame_id = config
            .integ_source
            .map(|src| {
                self.source_frame_ids
                    .get(src)
                    .unwrap_or_else(|| {
                        panic!(
                            "VehicleConfig::integ_source index {src} is out of range; \
                             {} source frame(s) configured",
                            self.source_frame_ids.len()
                        )
                    })
                    .inertial
            })
            .unwrap_or(self.root_frame_id);

        // Create body frame in tree under the integration frame.
        let body_frame_id = self.frame_tree.add_child(
            integ_frame_id,
            format!("body_{idx}.integ"),
            RefFrameKind::Body,
            RefFrameState {
                trans: RefFrameTrans {
                    position: config.trans.position,
                    velocity: config.trans.velocity,
                },
                rot: RefFrameRot::default(),
            },
        );

        self.bodies
            .push(SimBody::from_config(config, integ_frame_id, body_frame_id));
        idx
    }

    /// Read-only access to the reference frame tree.
    pub fn frame_tree(&self) -> &FrameTree {
        &self.frame_tree
    }

    /// Number of gravity sources.
    pub fn num_sources(&self) -> usize {
        self.gravity_data.len()
    }

    /// Get the inertial frame ID for a gravity source.
    pub fn source_frame(&self, source_idx: usize) -> FrameId {
        self.source_frame_ids
            .get(source_idx)
            .unwrap_or_else(|| {
                panic!(
                    "source_frame: source index {source_idx} is out of range; \
                     {} source frame(s) configured",
                    self.num_sources()
                )
            })
            .inertial
    }

    /// Get the current position of a gravity source in the inertial frame.
    pub fn source_position(&self, source_idx: usize) -> DVec3 {
        let fid = self.source_frame(source_idx);
        if fid == self.root_frame_id {
            DVec3::ZERO
        } else {
            self.frame_tree.get(fid).state.trans.position
        }
    }

    /// Set the position of a gravity source in the inertial frame.
    pub fn set_source_position(&mut self, source_idx: usize, position: DVec3) {
        assert!(
            source_idx < self.source_frame_ids.len(),
            "set_source_position: source index {source_idx} out of range; \
             {} source(s) configured",
            self.source_frame_ids.len()
        );
        let fid = self.source_frame_ids[source_idx].inertial;
        assert_ne!(
            fid, self.root_frame_id,
            "Cannot set position of the root (central body) source"
        );
        self.frame_tree.get_mut(fid).state.trans.position = position;
    }

    /// Set the position and velocity of a gravity source in the inertial frame.
    ///
    /// Prefer this over [`set_source_position`](Simulation::set_source_position)
    /// when velocity is also available, to keep position and velocity consistent.
    pub fn set_source_state(&mut self, source_idx: usize, position: DVec3, velocity: DVec3) {
        assert!(
            source_idx < self.source_frame_ids.len(),
            "set_source_state: source index {source_idx} out of range; \
             {} source(s) configured",
            self.source_frame_ids.len()
        );
        let fid = self.source_frame_ids[source_idx].inertial;
        assert_ne!(
            fid, self.root_frame_id,
            "Cannot set state of the root (central body) source"
        );
        let node = self.frame_tree.get_mut(fid);
        node.state.trans.position = position;
        node.state.trans.velocity = velocity;
    }

    /// Get the planet-fixed rotation matrix for a gravity source. Returns `None`
    /// if the source has no rotation model (no pfix frame).
    pub fn source_pfix_rotation(&self, source_idx: usize) -> Option<DMat3> {
        self.source_frame_ids
            .get(source_idx)
            .unwrap_or_else(|| {
                panic!(
                    "source_pfix_rotation: source index {source_idx} out of range; \
                     {} source(s) configured",
                    self.source_frame_ids.len()
                )
            })
            .pfix
            .map(|pfix_id| self.frame_tree.get(pfix_id).state.rot.t_parent_this)
    }

    /// Get mutable access to a source's tidal configuration.
    pub fn source_tidal_config_mut(
        &mut self,
        source_idx: usize,
    ) -> Option<&mut jeod_gravity::tides::TidalConfig> {
        let len = self.gravity_data.len();
        self.gravity_data
            .get_mut(source_idx)
            .unwrap_or_else(|| {
                panic!(
                    "source_tidal_config_mut: source index {source_idx} out of range; \
                     {len} source(s) configured",
                )
            })
            .tidal_config
            .as_mut()
    }

    /// Get the current ΔC20 tidal correction for a gravity source.
    pub fn source_delta_c20(&self, source_idx: usize) -> f64 {
        assert!(
            source_idx < self.gravity_data.len(),
            "source_delta_c20: source index {source_idx} out of range; \
             {} source(s) configured",
            self.gravity_data.len()
        );
        self.gravity_data[source_idx].delta_c20
    }

    /// Validate all bodies against JEOD invariants and apply auto-corrections.
    ///
    /// Call once before the first `step()`. Returns `Ok(())` if all bodies are
    /// valid, or `Err(errors)` with all validation errors found.
    ///
    /// Also runs `GravityControl::check_validity()` on each control to
    /// auto-correct degree/order (matching JEOD's `initialize_gravity_controls()`
    /// and the Bevy adapter's startup validation).
    // JEOD_INV: GV.03 — check_validity() called at startup (auto-corrections applied in-place)
    pub fn validate(&mut self) -> Result<(), Vec<ValidationError>> {
        let num_sources = self.gravity_data.len();
        let mut all_errors = Vec::new();
        for (body_idx, body) in self.bodies.iter_mut().enumerate() {
            let plate_counts = body.flat_plate_state.as_ref().map(|fps| {
                (
                    fps.plates.len(),
                    fps.temperatures.len(),
                    fps.t_pow4_cached.len(),
                )
            });
            let errors = jeod_sim::validate_body(
                &body.config,
                &body.gravity_controls,
                true, // SimBody always has gravity_accel field
                body.mass.as_ref(),
                body.rot.is_some(),
                Some(&body.trans),
                |source_id: usize| self.gravity_data.get(source_id).map(|g| &g.source),
                plate_counts,
            );
            all_errors.extend(errors);

            // Validate shadow_body index
            if let Some((idx, _radius)) = body.shadow_body {
                if idx >= num_sources {
                    all_errors.push(ValidationError::ShadowBodyOutOfRange {
                        index: idx,
                        num_sources,
                    });
                }
            }

            // Validate geodetic_planet index
            if let Some((idx, _, _)) = body.geodetic_planet {
                if idx >= num_sources {
                    all_errors.push(ValidationError::GeodeticPlanetOutOfRange {
                        index: idx,
                        num_sources,
                    });
                }
            }

            // Validate orbital_elements_source index
            if let Some(idx) = body.orbital_elements_source {
                if idx >= num_sources {
                    all_errors.push(ValidationError::OrbitalElementsSourceOutOfRange {
                        index: idx,
                        num_sources,
                    });
                }
            }

            // Validate force producers have mass (JEOD_INV: MA.01 — MassBody always present)
            if (body.drag.is_some() || body.flat_plate_state.is_some()) && body.mass.is_none() {
                all_errors.push(ValidationError::ForceProducerWithoutMass { body_idx });
            }

            // GaussJackson is translational-only (6-DOF not yet supported)
            if matches!(
                body.integrator,
                jeod_dynamics::IntegratorType::GaussJackson(..)
            ) && body.config.rotational_dynamics
            {
                all_errors.push(ValidationError::GaussJacksonWith6Dof { body_idx });
            }

            // GaussJackson config validation — delegates to GaussJacksonConfig::check()
            // so the predicate is defined in one place.
            if let jeod_dynamics::IntegratorType::GaussJackson(ref config) = body.integrator {
                for detail in config.check() {
                    all_errors
                        .push(ValidationError::GaussJacksonConfigInvalid { body_idx, detail });
                }
            }

            // Atmospheric state requires atmosphere config on the simulation
            if body.atmospheric_state.is_some() && self.atmosphere.is_none() {
                all_errors.push(ValidationError::AtmosphericStateWithoutAtmosphere { body_idx });
            }

            // Solar beta requires sun_source on the simulation
            if body.compute_solar_beta && self.sun_source.is_none() {
                all_errors.push(ValidationError::SolarBetaWithoutSunSource { body_idx });
            }

            // Earth lighting requires both sun_source and moon_source
            if body.earth_lighting_config.is_some() {
                if self.sun_source.is_none() {
                    all_errors.push(ValidationError::EarthLightingWithoutSunSource { body_idx });
                }
                if self.moon_source.is_none() {
                    all_errors.push(ValidationError::EarthLightingWithoutMoonSource { body_idx });
                }
            }

            // Gravity torque requires both mass and rotational state
            if body.compute_gravity_torque && (body.mass.is_none() || body.rot.is_none()) {
                all_errors.push(ValidationError::GravityTorqueWithoutMassOrRot { body_idx });
            }

            // Frame switch target_source must be a valid source index AND
            // present in the body's gravity controls (so the post-switch
            // differential flip actually takes effect).
            // Only validate active switches — JEOD only evaluates active switches.
            for sw in &body.frame_switches {
                if sw.active {
                    let central = sw.target_source;
                    if central >= num_sources {
                        all_errors.push(ValidationError::FrameSwitchCentralSourceOutOfRange {
                            body_idx,
                            central_source: central,
                            num_sources,
                        });
                    } else if !body
                        .gravity_controls
                        .controls
                        .iter()
                        .any(|c| c.source_name == central)
                    {
                        all_errors.push(ValidationError::FrameSwitchCentralSourceNotInControls {
                            body_idx,
                            central_source: central,
                        });
                    }
                }
            }

            // Warn when body uses a non-root integration frame with features
            // that assume root-inertial coordinates. JEOD evaluates these
            // derived states in the central-body inertial frame; they will
            // produce incorrect results in other frames.
            {
                let non_eci_integ = body.integ_frame_id != self.root_frame_id;
                let non_eci_switch = body.frame_switches.iter().any(|sw| {
                    sw.active
                        && self
                            .source_frame_ids
                            .get(sw.target_source)
                            .is_some_and(|frame| frame.inertial != self.root_frame_id)
                });
                if non_eci_integ || non_eci_switch {
                    let has_eci_feature = body.drag.is_some()
                        || body.flat_plate_state.is_some()
                        || body.cannonball_srp.is_some()
                        || body.orbital_elements_source.is_some()
                        || body.euler_sequence.is_some()
                        || body.compute_lvlh
                        || body.geodetic_planet.is_some()
                        || body.compute_solar_beta
                        || body.earth_lighting_config.is_some();
                    if has_eci_feature {
                        all_errors.push(ValidationError::NonEciFrameWithEciDependentFeatures {
                            body_idx,
                        });
                    }
                }
            }

            // Apply gravity control auto-corrections (degree/order clamping).
            // JEOD_INV: GV.03 — check_validity() auto-corrects out-of-range settings
            for ctrl in &mut body.gravity_controls.controls {
                if let Some(grav) = self.gravity_data.get(ctrl.source_name) {
                    ctrl.check_validity(&grav.source);
                }
            }
        }

        // Validate sun_source index (simulation-level, outside body loop)
        if let Some(idx) = self.sun_source {
            if idx >= num_sources {
                all_errors.push(ValidationError::SunSourceOutOfRange {
                    index: idx,
                    num_sources,
                });
            }
        }

        // Validate moon_source index
        if let Some(idx) = self.moon_source {
            if idx >= num_sources {
                all_errors.push(ValidationError::MoonSourceOutOfRange {
                    index: idx,
                    num_sources,
                });
            }
        }

        // Validate atmosphere_planet_source index
        if let Some(idx) = self.atmosphere_planet_source {
            if idx >= num_sources {
                all_errors.push(ValidationError::AtmospherePlanetOutOfRange {
                    index: idx,
                    num_sources,
                });
            }
        }

        // Ephemeris mapping on root-frame sources — would silently discard position.
        for (i, ephem) in self.source_ephem_bodies.iter().enumerate() {
            if ephem.is_some() && self.source_frame_ids[i].inertial == self.root_frame_id {
                all_errors.push(ValidationError::EphemerisOnRootSource { source_idx: i });
            }
        }

        // Separate warnings from fatal errors — warnings are logged, not returned.
        let mut fatal = Vec::new();
        for error in all_errors {
            if error.is_warning() {
                log::warn!("{error}");
            } else {
                fatal.push(error);
            }
        }
        if !fatal.is_empty() {
            return Err(fatal);
        }

        // Auto-initialize Gauss-Jackson state for bodies that need it.
        // Check config consistency for pre-existing states.
        for (body_idx, body) in self.bodies.iter_mut().enumerate() {
            if let jeod_dynamics::IntegratorType::GaussJackson(ref config) = body.integrator {
                match &body.gj_state {
                    None => {
                        body.gj_state = Some(jeod_dynamics::GaussJacksonState::new(*config));
                    }
                    Some(state) if state.config() != config => {
                        fatal.push(ValidationError::GaussJacksonConfigInvalid {
                            body_idx,
                            detail: format!(
                                "existing gj_state config does not match IntegratorType config \
                                 (initial_order {}/{}, final_order {}/{}). \
                                 Remove gj_state or recreate from the same config.",
                                state.config().initial_order,
                                config.initial_order,
                                state.config().final_order,
                                config.final_order,
                            ),
                        });
                    }
                    Some(_) => {} // config matches, keep existing state
                }
            }
        }
        if !fatal.is_empty() {
            return Err(fatal);
        }

        Ok(())
    }

    /// Advance the simulation by one timestep.
    ///
    /// Runs the full JEOD pipeline in order:
    /// 1. Time update
    /// 2. Ephemeris update (planet-fixed rotations + frame tree sync)
    /// 3. Mass update (recompute derived quantities)
    /// 4. Gravity computation
    /// 5. Atmosphere evaluation
    /// 6. Interaction computation (drag, SRP, gravity torque)
    /// 7. Force collection and frame derivative computation
    /// 8. State integration (RK4, with sub-stage tree updates)
    /// 9. Derived state computation
    pub fn step(&mut self) {
        self.step_internal(self.dt);
    }

    /// Get the position and velocity of a frame relative to the root inertial frame.
    pub fn frame_origin(&self, frame_id: FrameId) -> (DVec3, DVec3) {
        if frame_id == self.root_frame_id {
            return (DVec3::ZERO, DVec3::ZERO);
        }
        let state = self
            .frame_tree
            .compute_relative_state(self.root_frame_id, frame_id);
        (state.trans.position, state.trans.velocity)
    }

    /// Internal step with explicit dt (avoids temporary mutation of `self.dt`
    /// in `step_until`).
    fn step_internal(&mut self, dt: f64) {
        // ── 1. Time update ──
        self.time.advance(dt);

        // ── 2. Ephemeris update — planet-fixed rotations + frame tree sync ──
        // JEOD_INV: DM.13 — ephemeris updated before gravity
        // Per-source rotation dispatch: each source has its own rotation model.
        // Lazy-compute Earth RNP only if needed (most common case).
        let mut earth_rotation: Option<DMat3> = Option::None;
        for (i, grav) in self.gravity_data.iter_mut().enumerate() {
            match grav.rotation_model {
                RotationModel::None => {}
                RotationModel::EarthRNP => {
                    let rotation = *earth_rotation.get_or_insert_with(|| {
                        jeod_sim::compute_t_parent_this_from_tjt_with_polar(
                            self.time.gmst_seconds,
                            self.time.tt_tjt(),
                            self.polar_motion,
                        )
                    });
                    // Sync to frame tree pfix node.
                    if let Some(pfix_id) = self.source_frame_ids[i].pfix {
                        let node = self.frame_tree.get_mut(pfix_id);
                        node.state.rot.t_parent_this = rotation;
                        node.state.rot.q_parent_this =
                            JeodQuat::left_quat_from_transformation(&rotation);
                    }
                }
                RotationModel::MarsIAU => {
                    // JEOD's RNPMars receives TT seconds since J2000 (time_tt.seconds).
                    let tt_s_since_j2000 = (self.time.tt_tjt() - jeod_time::epoch::J2000_TT_TJT)
                        * jeod_time::epoch::SECONDS_PER_DAY;
                    let rotation =
                        jeod_frames::rotation_mars::compute_mars_rotation(tt_s_since_j2000);
                    if let Some(pfix_id) = self.source_frame_ids[i].pfix {
                        let node = self.frame_tree.get_mut(pfix_id);
                        node.state.rot.t_parent_this = rotation;
                        node.state.rot.q_parent_this =
                            JeodQuat::left_quat_from_transformation(&rotation);
                    }
                }
                RotationModel::MoonIAU => {
                    let tdb_jd = self.time.tdb_julian_date();
                    let tdb_s_since_j2000 = (tdb_jd - jeod_time::epoch::J2000_TT_JD)
                        * jeod_time::epoch::SECONDS_PER_DAY;
                    let rotation =
                        jeod_frames::rotation_moon::compute_moon_rotation(tdb_s_since_j2000);
                    if let Some(pfix_id) = self.source_frame_ids[i].pfix {
                        let node = self.frame_tree.get_mut(pfix_id);
                        node.state.rot.t_parent_this = rotation;
                        node.state.rot.q_parent_this =
                            JeodQuat::left_quat_from_transformation(&rotation);
                    }
                }
                RotationModel::MoonDE421 => {
                    let eph = self.ephemeris.as_ref().expect(
                        "MoonDE421 rotation requires ephemeris with BPC. \
                         Set sim.ephemeris = Some(eph) after calling eph.load_bpc().",
                    );
                    let tdb_jd = self.time.tdb_julian_date();
                    let rotation = eph
                        .get_body_rotation(jeod_sim::EphemerisBody::Moon, tdb_jd)
                        .expect("Moon DE421 BPC rotation query failed");
                    if let Some(pfix_id) = self.source_frame_ids[i].pfix {
                        let node = self.frame_tree.get_mut(pfix_id);
                        node.state.rot.t_parent_this = rotation;
                        node.state.rot.q_parent_this =
                            JeodQuat::left_quat_from_transformation(&rotation);
                    }
                }
            }
            // Compute tidal ΔC20 if configured; otherwise clear any stale value.
            if let Some(ref config) = grav.tidal_config {
                let pfix_id = self.source_frame_ids[i]
                    .pfix
                    .expect("tidal_config requires a planet-fixed frame (set rotation_model or t_inertial_pfix).");
                let rotation = self.frame_tree.get(pfix_id).state.rot.t_parent_this;
                grav.delta_c20 = jeod_gravity::tides::compute_delta_c20(config, &rotation);
            } else {
                grav.delta_c20 = 0.0;
            }
        }

        // ── 2b. Ephemeris update — source positions from DE4xx ──
        // Update source positions from ephemeris each step and sync to frame tree.
        if let Some(ref eph) = self.ephemeris {
            let tdb_jd = self.time.tdb_julian_date();
            for i in 0..self.source_ephem_bodies.len() {
                if let Some(Some((target, observer))) = self.source_ephem_bodies.get(i) {
                    let (pos, vel) =
                        eph.get_state(*target, *observer, tdb_jd)
                            .unwrap_or_else(|e| {
                                panic!(
                                    "Ephemeris lookup failed for source {i} \
                                 ({target:?} wrt {observer:?}) at TDB JD {tdb_jd}: {e}"
                                )
                            });
                    // Root-mapped sources cannot consume ephemeris position updates:
                    // the root frame must remain identity, so accepting such a
                    // mapping would silently ignore `pos` and yield an incorrect
                    // source position.
                    let fid = self.source_frame_ids[i].inertial;
                    assert!(
                        fid != self.root_frame_id,
                        "Invalid ephemeris mapping for source {i} \
                         ({target:?} wrt {observer:?}): source inertial frame is the root frame, \
                         whose state must remain identity. Root-mapped sources cannot use \
                         ephemeris position updates."
                    );
                    // Update frame tree node with ephemeris position/velocity.
                    let node = self.frame_tree.get_mut(fid);
                    node.state.trans.position = pos;
                    node.state.trans.velocity = vel;
                    // Also update gravity_data velocity for relativistic corrections.
                    self.gravity_data[i].velocity = vel;
                }
            }
        }

        // ── 3. Mass update — recompute inverse_mass/inverse_inertia ──
        for body in &mut self.bodies {
            if let Some(ref mut mass) = body.mass {
                mass.recompute_derived();
            }
        }

        // Precompute frame origins from the tree for all body integration frames.
        let body_integ_origins: Vec<(DVec3, DVec3)> = self
            .bodies
            .iter()
            .map(|b| self.frame_origin(b.integ_frame_id))
            .collect();

        // ── 4. Environment — gravity ──
        // Helper: resolve source to gravity data via frame tree.
        let gravity_data = &self.gravity_data;
        let source_frame_ids = &self.source_frame_ids;
        let frame_tree = &self.frame_tree;
        let root_fid = self.root_frame_id;
        let resolve_source = |source_id: usize| -> Option<jeod_sim::ResolvedSource<'_>> {
            let grav = gravity_data.get(source_id)?;
            let sfids = &source_frame_ids[source_id];
            let src_node = frame_tree.get(sfids.inertial);
            let position = if sfids.inertial == root_fid {
                DVec3::ZERO
            } else {
                src_node.state.trans.position
            };
            let rotation = sfids
                .pfix
                .map(|pfix_id| &frame_tree.get(pfix_id).state.rot.t_parent_this);
            Some(jeod_sim::ResolvedSource {
                source: &grav.source,
                rotation,
                position,
                delta_c20: grav.delta_c20,
                has_delta_coeffs: grav.tidal_config.is_some(),
            })
        };

        for (body_idx, body) in self.bodies.iter_mut().enumerate() {
            let integ_origin = body_integ_origins[body_idx].0;
            body.gravity_accel = accumulate_gravity(
                body.trans.position + integ_origin,
                &body.gravity_controls,
                integ_origin,
                resolve_source,
            );
        }

        // ── 4b. Relativistic corrections ──
        // After Newtonian gravity, apply post-Newtonian PPN correction for
        // any source with `relativistic: true`. Folkner eq 27 (β=γ=1).
        // PPN uses inertial coordinates — convert from integration frame.
        let resolve_rel_source =
            |source_id: usize| -> Option<jeod_sim::ResolvedRelativisticSource> {
                let grav = gravity_data.get(source_id)?;
                let sfids = &source_frame_ids[source_id];
                let position = if sfids.inertial == root_fid {
                    DVec3::ZERO
                } else {
                    frame_tree.get(sfids.inertial).state.trans.position
                };
                Some(jeod_sim::ResolvedRelativisticSource {
                    mu: grav.source.mu,
                    position,
                    // Use velocity from gravity_data, not the tree node, because
                    // central bodies at the root frame have zero tree velocity
                    // but may have physical velocity for relativistic corrections.
                    velocity: grav.velocity,
                })
            };

        for (body_idx, body) in self.bodies.iter_mut().enumerate() {
            let (origin, origin_vel) = body_integ_origins[body_idx];
            body.gravity_accel.grav_accel += jeod_sim::accumulate_relativistic_corrections(
                body.trans.position + origin,
                body.trans.velocity + origin_vel,
                &body.gravity_controls,
                resolve_rel_source,
            );
        }

        // ── 5. Environment — atmosphere ──
        if let Some(ref atmos_config) = self.atmosphere {
            let t_pfix = self
                .atmosphere_planet_source
                .and_then(|idx| self.source_frame_ids.get(idx))
                .and_then(|sfids| sfids.pfix)
                .map(|pfix_id| &self.frame_tree.get(pfix_id).state.rot.t_parent_this);
            let tai_tjt = Some(self.time.tai_tjt);

            for body in &mut self.bodies {
                if body.atmospheric_state.is_some() {
                    body.atmospheric_state = Some(evaluate_atmosphere(
                        atmos_config,
                        body.trans.position,
                        t_pfix,
                        tai_tjt,
                    ));
                }
            }
        }

        // ── 6. Interactions — drag, SRP, gravity torque ──
        // sun_pos is also used in stage 9 (solar beta, earth lighting); compute once here.
        let sun_pos = self.sun_source.map(|idx| self.source_position(idx));
        let moon_pos = self.moon_source.map(|idx| self.source_position(idx));
        let source_frame_ids = &self.source_frame_ids;
        let frame_tree = &self.frame_tree;
        let root_fid = self.root_frame_id;

        for body in &mut self.bodies {
            // Compute structural transform once (shared by drag and flat-plate SRP)
            let t_inertial_body = body.rot.as_ref().map_or(DMat3::IDENTITY, |r| {
                r.quaternion.left_quat_to_transformation()
            });
            let t_inertial_struct =
                jeod_sim::compute_t_inertial_struct(&body.t_struct_body, &t_inertial_body);

            // Aerodynamic drag
            body.aero_force = None;
            if let (Some(ref drag_config), Some(ref atmos)) = (&body.drag, &body.atmospheric_state)
            {
                body.aero_force = Some(jeod_sim::compute_drag(
                    drag_config,
                    atmos,
                    body.trans.velocity,
                    body.rot.as_ref(),
                    body.t_struct_body,
                ));
            }

            // Solar radiation pressure (flat-plate)
            body.radiation_force = None;
            if let Some(sun_position) = sun_pos {
                if let Some(ref mut fps) = body.flat_plate_state {
                    // Flat-plate SRP with thermal emission
                    let sun_to_vehicle = body.trans.position - sun_position;
                    let distance = sun_to_vehicle.length();
                    // Skip SRP (not the whole body) if too close to Sun
                    if distance >= 1.0 {
                        let flux_inertial_hat = sun_to_vehicle / distance;
                        let flux_mag = jeod_sim::solar_flux_at_distance(distance);

                        // Shadow fraction
                        let illum_factor = body
                            .shadow_body
                            .map(|(idx, radius)| {
                                jeod_sim::compute_shadow_fraction(
                                    body.trans.position,
                                    sun_position,
                                    {
                                        let fid = source_frame_ids[idx].inertial;
                                        if fid == root_fid {
                                            DVec3::ZERO
                                        } else {
                                            frame_tree.get(fid).state.trans.position
                                        }
                                    },
                                    radius,
                                    jeod_sim::SOLAR_RADIUS,
                                )
                            })
                            .unwrap_or(1.0);

                        // Rotate flux direction to structural frame
                        let flux_struct_hat = t_inertial_struct * flux_inertial_hat;

                        let center_grav = body.mass.as_ref().map_or(DVec3::ZERO, |m| m.position);

                        let srp_result = jeod_sim::compute_flat_plate_srp_thermal(
                            &fps.plates,
                            &fps.t_pow4_cached,
                            flux_struct_hat,
                            flux_mag,
                            center_grav,
                            illum_factor,
                        );

                        // Force: rotate from structural to inertial. Torque: stays structural.
                        let force_inertial = t_inertial_struct.transpose() * srp_result.force;
                        body.radiation_force = Some(RadiationForce {
                            force: force_inertial,
                            torque: srp_result.torque,
                        });

                        fps.integrate_temperatures(&srp_result.temp_dots, dt);
                    }
                } else if let Some((cx_area, albedo, diffuse)) = body.cannonball_srp {
                    let illum_factor = body
                        .shadow_body
                        .map(|(idx, radius)| {
                            jeod_sim::compute_shadow_fraction(
                                body.trans.position,
                                sun_position,
                                {
                                    let fid = source_frame_ids[idx].inertial;
                                    if fid == root_fid {
                                        DVec3::ZERO
                                    } else {
                                        frame_tree.get(fid).state.trans.position
                                    }
                                },
                                radius,
                                jeod_sim::SOLAR_RADIUS,
                            )
                        })
                        .unwrap_or(1.0);

                    let force = jeod_sim::compute_cannonball_srp(
                        body.trans.position,
                        sun_position,
                        cx_area,
                        albedo,
                        diffuse,
                        illum_factor,
                    );
                    if force != DVec3::ZERO {
                        body.radiation_force = Some(RadiationForce {
                            force,
                            torque: DVec3::ZERO,
                        });
                    }
                }
            }

            // Gravity gradient torque
            body.gravity_torque = None;
            if body.compute_gravity_torque {
                if let (Some(ref rot), Some(ref mass)) = (&body.rot, &body.mass) {
                    body.gravity_torque = Some(jeod_sim::compute_gravity_torque(
                        &body.gravity_accel.grav_grad,
                        rot,
                        &mass.inertia,
                    ));
                }
            }
        }

        // ── 7. Force collection ──
        for body in &mut self.bodies {
            let (total, derivs) = collect_and_resolve_forces(
                body.aero_force.as_ref(),
                body.radiation_force.as_ref(),
                body.gravity_torque,
                body.rot.as_ref(),
                body.t_struct_body,
                body.mass.as_ref(),
                body.gravity_accel.grav_accel,
            );
            body.total_force = total;
            body.frame_derivs = derivs;

            // Apply external force/torque (set by caller between steps).
            // Recompute frame derivatives so they stay consistent with total_force.
            body.total_force.force += body.external_force;
            body.total_force.torque += body.external_torque;
            if body.external_force != DVec3::ZERO {
                if let Some(mass) = &body.mass {
                    body.frame_derivs.trans_accel += body.external_force * mass.inverse_mass;
                }
            }
            if body.external_torque != DVec3::ZERO {
                if let Some(mass) = &body.mass {
                    body.frame_derivs.rot_accel += mass.inverse_inertia * body.external_torque;
                }
            }
        }

        // ── 8. Integration ──
        // Gravity (including relativistic corrections) is recomputed at each
        // RK4 intermediate state for 4th-order accuracy, matching JEOD's
        // DynamicsIntegrationGroup where the derivative function calls gravity
        // at every stage with the current intermediate position and velocity.
        //
        // For RK4 sub-stage evaluations, source positions are derived from a
        // linear interpolation of their base inertial position using
        // velocity * (time_frac * dt), matching JEOD's behavior of evaluating
        // gravity using the current sub-stage source state.
        //
        // Snapshot base source positions and velocities for sub-stage interpolation.
        let base_positions: Vec<DVec3> = self
            .source_frame_ids
            .iter()
            .map(|sfids| {
                if sfids.inertial == self.root_frame_id {
                    DVec3::ZERO
                } else {
                    self.frame_tree.get(sfids.inertial).state.trans.position
                }
            })
            .collect();
        let base_velocities: Vec<DVec3> = self
            .source_frame_ids
            .iter()
            .map(|sfids| {
                if sfids.inertial == self.root_frame_id {
                    DVec3::ZERO
                } else {
                    self.frame_tree.get(sfids.inertial).state.trans.velocity
                }
            })
            .collect();

        let gravity_data = &self.gravity_data;
        let source_frame_ids = &self.source_frame_ids;
        let frame_tree = &self.frame_tree;
        let root_fid = self.root_frame_id;

        for (body_idx, body) in self.bodies.iter_mut().enumerate() {
            let (integ_origin, integ_vel) = body_integ_origins[body_idx];
            let controls = &body.gravity_controls;

            // Precompute relativistic "other source" lists outside the closure
            // to avoid heap allocation at every RK4 stage. Source positions are
            // constant within a single step. NOTE: for non-ECI frames, Newtonian
            // gravity interpolates source positions per sub-stage but relativistic
            // corrections use these frozen positions. The PPN correction is ~1e-8
            // of Newtonian gravity, so the sub-stage shift is negligible.
            let rel_data: Vec<_> = controls
                .controls
                .iter()
                .filter(|c| c.relativistic)
                .filter_map(|ctrl| {
                    let grav = gravity_data.get(ctrl.source_name)?;
                    let sfids = &source_frame_ids[ctrl.source_name];
                    let src_pos = if sfids.inertial == root_fid {
                        DVec3::ZERO
                    } else {
                        frame_tree.get(sfids.inertial).state.trans.position
                    };
                    let src_vel = grav.velocity;
                    let other: Vec<_> = controls
                        .controls
                        .iter()
                        .filter(|c| c.source_name != ctrl.source_name)
                        .filter_map(|c| {
                            let g = gravity_data.get(c.source_name)?;
                            let sf = &source_frame_ids[c.source_name];
                            let pos = if sf.inertial == root_fid {
                                DVec3::ZERO
                            } else {
                                frame_tree.get(sf.inertial).state.trans.position
                            };
                            Some(jeod_gravity::relativistic::RelativisticSource {
                                mu: g.source.mu,
                                position: pos,
                            })
                        })
                        .collect();
                    Some((grav.source.mu, src_pos, src_vel, other))
                })
                .collect();

            integrate_body(
                &body.config,
                &mut body.trans,
                body.rot.as_mut(),
                body.mass.as_ref(),
                |pos, vel, time_frac| {
                    // Sub-stage interpolation for the integration frame origin.
                    let origin = integ_origin + integ_vel * (time_frac * dt);
                    // Source position interpolation: JEOD's `deriv_ephem_update`
                    // (DynamicsIntegrationGroup, default false) controls whether
                    // ephemerides are updated at each derivative evaluation.
                    // With default=false, source positions are frozen within a
                    // step — we match this for root-frame integration where
                    // integ_vel == ZERO. For non-root frames the frame origin
                    // moves within the step, so we must interpolate source
                    // positions to keep gravity evaluation consistent with the
                    // interpolated integration-frame origin.
                    let sub_dt = if integ_vel != DVec3::ZERO {
                        time_frac * dt
                    } else {
                        0.0
                    };
                    let mut accel =
                        accumulate_gravity(pos + origin, controls, origin, |source_id: usize| {
                            let grav = gravity_data.get(source_id)?;
                            let sfids = &source_frame_ids[source_id];
                            let position =
                                base_positions[source_id] + base_velocities[source_id] * sub_dt;
                            let rotation = sfids
                                .pfix
                                .map(|pfix_id| &frame_tree.get(pfix_id).state.rot.t_parent_this);
                            Some(jeod_sim::ResolvedSource {
                                source: &grav.source,
                                rotation,
                                position,
                                delta_c20: grav.delta_c20,
                                has_delta_coeffs: grav.tidal_config.is_some(),
                            })
                        })
                        .grav_accel;
                    // Relativistic corrections use inertial coordinates.
                    // Convert from integration frame to ECI for the PPN formula.
                    let pos_eci = pos + origin;
                    let vel_eci = vel + integ_vel;
                    for &(mu, src_pos, src_vel, ref other) in &rel_data {
                        accel += jeod_gravity::relativistic::compute_relativistic_correction(
                            mu, src_pos, pos_eci, vel_eci, src_vel, other,
                        );
                    }
                    accel
                },
                body.total_force.force,
                body.total_force.torque,
                dt,
                self.time.time_scale_factor,
                body.integrator,
                body.gj_state.as_mut(),
            );
        }

        // Sync body positions back to frame tree after integration.
        for body in &self.bodies {
            let node = self.frame_tree.get_mut(body.body_frame_id);
            node.state.trans.position = body.trans.position;
            node.state.trans.velocity = body.trans.velocity;
        }

        // ── 8b. Frame switch (body actions) ──
        // Applied AFTER integration, matching JEOD's pipeline where
        // DynBodyFrameSwitch is a body action evaluated post-integration.
        // The body has already been integrated in its current frame for this
        // step; the switch transforms to the new frame for the NEXT step.
        // Uses frame tree reparenting for structural correctness.
        // Use index-based loop to avoid borrow conflict with self.frame_tree.
        for body_idx in 0..self.bodies.len() {
            if self.bodies[body_idx].frame_switches.is_empty() {
                continue;
            }
            let mut switch_idx = None;
            for (idx, sw) in self.bodies[body_idx].frame_switches.iter().enumerate() {
                if !sw.active {
                    continue;
                }
                let target_fid = self
                    .source_frame_ids
                    .get(sw.target_source)
                    .unwrap_or_else(|| {
                        panic!(
                            "frame switch evaluation: target_source {} out of range; \
                             {} source(s) configured. Run validate() before step().",
                            sw.target_source,
                            self.source_frame_ids.len()
                        )
                    })
                    .inertial;
                let (target_origin, _) = self.frame_origin(target_fid);
                let (current_origin, _) = self.frame_origin(self.bodies[body_idx].integ_frame_id);
                let body_pos_eci = self.bodies[body_idx].trans.position + current_origin;
                let threshold_sq = sw.switch_distance * sw.switch_distance;

                // JEOD dyn_body_frame_switch.cc:173-182:
                // OnApproach: compute_position_from(*integ_frame) → distance to target
                // OnDeparture: state.trans.position magnitude → distance from current origin
                let triggered = match sw.switch_sense {
                    SwitchSense::OnApproach => {
                        (body_pos_eci - target_origin).length_squared() < threshold_sq
                    }
                    SwitchSense::OnDeparture => {
                        self.bodies[body_idx].trans.position.length_squared() > threshold_sq
                    }
                };
                if triggered {
                    switch_idx = Some(idx);
                    break;
                }
            }
            if let Some(idx) = switch_idx {
                let target_source = self.bodies[body_idx].frame_switches[idx].target_source;
                self.bodies[body_idx].frame_switches[idx].active = false;

                let new_integ_fid = self.source_frame_ids[target_source].inertial; // bounds already checked above
                let body_fid = self.bodies[body_idx].body_frame_id;

                // Reparent body frame in tree (preserves absolute state).
                self.frame_tree.reparent(body_fid, new_integ_fid);
                let new_state = self.frame_tree.get(body_fid).state;
                self.bodies[body_idx].trans.position = new_state.trans.position;
                self.bodies[body_idx].trans.velocity = new_state.trans.velocity;
                self.bodies[body_idx].integ_frame_id = new_integ_fid;

                // Flip gravity controls: target source becomes non-differential
                // (central body), all others become differential.
                for ctrl in &mut self.bodies[body_idx].gravity_controls.controls {
                    ctrl.differential = ctrl.source_name != target_source;
                }
            }
        }

        // ── 9. Derived states ──
        let gravity_data = &self.gravity_data;

        for body in &mut self.bodies {
            // Orbital elements
            if let Some(src_idx) = body.orbital_elements_source {
                if let Some(mu) = gravity_data.get(src_idx).map(|g| g.source.mu) {
                    body.orbital_elements = jeod_sim::compute_orbital_elements(
                        mu,
                        body.trans.position,
                        body.trans.velocity,
                    )
                    .ok();
                } else {
                    body.orbital_elements = None;
                }
            }

            // Euler angles
            if let Some(seq) = body.euler_sequence {
                if let Some(ref rot) = body.rot {
                    body.euler_angles = Some(jeod_sim::compute_body_euler_angles(rot, seq));
                } else {
                    body.euler_angles = None;
                }
            }

            // LVLH frame
            if body.compute_lvlh {
                body.lvlh_frame = Some(jeod_sim::compute_body_lvlh_frame(
                    body.trans.position,
                    body.trans.velocity,
                ));
            }

            // Geodetic state
            if let Some((src_idx, r_eq, r_pol)) = body.geodetic_planet {
                let pfix_rot = self
                    .source_frame_ids
                    .get(src_idx)
                    .and_then(|sfids| sfids.pfix)
                    .map(|pfix_id| self.frame_tree.get(pfix_id).state.rot.t_parent_this);
                if let Some(t_pfix) = pfix_rot {
                    body.geodetic_state = Some(jeod_sim::compute_body_geodetic(
                        body.trans.position,
                        &t_pfix,
                        r_eq,
                        r_pol,
                    ));
                } else {
                    body.geodetic_state = None;
                }
            }

            // Solar beta
            if body.compute_solar_beta {
                if let Some(sp) = sun_pos {
                    body.solar_beta = Some(jeod_sim::compute_body_solar_beta(
                        body.trans.position,
                        body.trans.velocity,
                        sp,
                    ));
                } else {
                    body.solar_beta = None;
                }
            }

            // Earth lighting
            if let Some((earth_r, moon_r, sun_r)) = body.earth_lighting_config {
                if let (Some(sp), Some(mp)) = (sun_pos, moon_pos) {
                    body.earth_lighting =
                        Some(jeod_interactions::earth_lighting::compute_earth_lighting(
                            body.trans.position,
                            sp,
                            mp,
                            sun_r,
                            earth_r,
                            moon_r,
                        ));
                } else {
                    body.earth_lighting = None;
                }
            }
        }
    }

    /// Advance the simulation by `n` timesteps.
    pub fn step_n(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Advance the simulation until `target_time` (in simulation seconds).
    ///
    /// Steps at `self.dt` until the remaining time is less than `dt`,
    /// then takes a final fractional step if the remainder exceeds 1 ms.
    pub fn step_until(&mut self, target_time: f64) {
        while self.time.simtime + self.dt <= target_time + 0.001 {
            self.step();
        }
        let remainder = target_time - self.time.simtime;
        if remainder > 0.001 {
            // Fractional steps corrupt Gauss-Jackson history (the Störmer-Cowell
            // coefficients and delinv accumulators assume constant dt).
            let has_gj = self.bodies.iter().any(|b| {
                matches!(
                    b.integrator,
                    jeod_dynamics::IntegratorType::GaussJackson(..)
                )
            });
            assert!(
                !has_gj,
                "step_until() would take a fractional step ({remainder:.6}s vs dt={:.6}s). \
                 GaussJackson requires constant dt. Ensure target_time is \
                 an integer multiple of dt.",
                self.dt
            );
            self.step_internal(remainder);
        }
    }

    // JEOD_INV: DS.01 — derived state config immutable after init; read-only access only
    /// Get the current output state of a body by index.
    ///
    /// Returns a [`VehicleOutput`] containing the current integrated state
    /// plus any derived states that were configured.
    pub fn body(&self, idx: usize) -> VehicleOutput {
        self.bodies[idx].output()
    }

    /// Set the externally applied force (inertial frame, N) for a body.
    ///
    /// Added to `total_force.force` each step after force collection.
    pub fn set_body_external_force(&mut self, idx: usize, force: DVec3) {
        self.bodies[idx].external_force = force;
    }

    /// Set the externally applied torque (body frame, N*m) for a body.
    ///
    /// Added to `total_force.torque` each step after force collection.
    pub fn set_body_external_torque(&mut self, idx: usize, torque: DVec3) {
        self.bodies[idx].external_torque = torque;
    }

    /// Set a body's translational position (inertial frame, m).
    ///
    /// Used for prescribed-motion tests where position is set externally
    /// at each timestep (e.g., SIM_2A_SHADOW_CALC).
    pub fn set_body_position(&mut self, idx: usize, position: DVec3) {
        self.bodies[idx].trans.position = position;
        let fid = self.bodies[idx].body_frame_id;
        self.frame_tree.get_mut(fid).state.trans.position = position;
    }

    /// Set a body's translational velocity (inertial frame, m/s).
    ///
    /// Used for impulsive maneuvers (e.g., Apollo TLI delta-V).
    pub fn set_body_velocity(&mut self, idx: usize, velocity: DVec3) {
        self.bodies[idx].trans.velocity = velocity;
        let fid = self.bodies[idx].body_frame_id;
        self.frame_tree.get_mut(fid).state.trans.velocity = velocity;
    }

    /// Replace a body's mass properties.
    ///
    /// Used for discrete mass changes (e.g., post-burn fuel consumption,
    /// stage separation). Recomputes `inverse_mass` and `inverse_inertia`.
    ///
    /// **Warning:** If the body is registered in the mass tree, calling this
    /// method will desynchronize the body's mass from the tree's copy. Use
    /// [`sync_body_mass_from_tree`](Self::sync_body_mass_from_tree) instead
    /// when the mass tree has been modified via `attach`/`detach`.
    pub fn set_body_mass(&mut self, idx: usize, mut mass: MassProperties) {
        mass.dirty = true;
        mass.recompute_derived();
        self.bodies[idx].mass = Some(mass);
    }

    /// Sync a body's mass properties from the mass tree's composite.
    ///
    /// After modifying the mass tree via `attach`/`detach`, call this to
    /// update the body's mass from the tree's composite properties.
    ///
    /// # Panics
    /// Panics if the body is not registered in the mass tree.
    pub fn sync_body_mass_from_tree(&mut self, idx: usize) {
        let id = self.bodies[idx]
            .mass_body_id
            .expect("sync_body_mass_from_tree requires body registered in mass tree");
        let tree = self
            .mass_tree
            .as_ref()
            .expect("sync_body_mass_from_tree requires a mass tree");
        let mut composite = tree.get(id).composite_properties;
        composite.dirty = true;
        composite.recompute_derived();
        self.bodies[idx].mass = Some(composite);
    }

    /// Register a body in the simulation's mass tree.
    ///
    /// Creates (or reuses) a `MassTree` and adds the body's mass as a node.
    /// Returns the `MassBodyId` for use with [`attach`](Self::attach) and
    /// [`detach`](Self::detach). The body's `mass` field must be `Some`.
    ///
    /// # Panics
    /// Panics if the body has no mass properties.
    pub fn add_body_to_tree(
        &mut self,
        body_idx: usize,
        name: impl Into<String>,
    ) -> jeod_dynamics::MassBodyId {
        let mass = self.bodies[body_idx]
            .mass
            .expect("add_body_to_tree requires mass properties");
        let tree = self
            .mass_tree
            .get_or_insert_with(jeod_dynamics::MassTree::new);
        let id = tree.add_body(name.into(), mass);
        self.bodies[body_idx].mass_body_id = Some(id);
        id
    }

    /// Attach a child body to a parent body in the mass tree.
    ///
    /// Both bodies must have been registered via [`add_body_to_tree`](Self::add_body_to_tree).
    /// After attachment, the parent's composite mass properties are updated
    /// automatically. The parent body's `mass` is synced from the tree.
    ///
    /// # Panics
    /// Panics if either body is not in the tree, or if the child already has a parent.
    pub fn attach(
        &mut self,
        child_idx: usize,
        parent_idx: usize,
        offset: DVec3,
        t_parent_child: DMat3,
    ) {
        let child_id = self.bodies[child_idx]
            .mass_body_id
            .expect("child not in mass tree");
        let parent_id = self.bodies[parent_idx]
            .mass_body_id
            .expect("parent not in mass tree");
        let tree = self.mass_tree.as_mut().expect("no mass tree");
        tree.attach(child_id, parent_id, offset, t_parent_child);
        // Sync parent's composite mass from tree
        self.bodies[parent_idx].mass = Some(tree.get(parent_id).composite_properties);
    }

    /// Detach a child body from its parent in the mass tree.
    ///
    /// After detachment, both the former parent's and the child's mass
    /// properties are updated from the tree's recomputed composites.
    ///
    /// # Panics
    /// Panics if the body is not in the tree or has no parent.
    pub fn detach(&mut self, child_idx: usize) {
        let child_id = self.bodies[child_idx]
            .mass_body_id
            .expect("child not in mass tree");
        let tree = self.mass_tree.as_mut().expect("no mass tree");
        let parent_id = tree
            .parent(child_id)
            .expect("detach called on body with no parent in tree");
        tree.detach(child_id);
        // Sync both bodies' mass from tree
        self.bodies[child_idx].mass = Some(tree.get(child_id).composite_properties);
        // Find parent body index and sync
        for body in &mut self.bodies {
            if body.mass_body_id == Some(parent_id) {
                body.mass = Some(tree.get(parent_id).composite_properties);
                break;
            }
        }
    }

    /// Number of bodies in the simulation.
    pub fn num_bodies(&self) -> usize {
        self.bodies.len()
    }

    /// Set the integration timestep (must be positive).
    ///
    /// For JEOD-style time reversal, use `sim.time.time_scale_factor = -1.0`
    /// instead of negative dt. This keeps `simtime` monotonically increasing
    /// while reversing dynamic time (TAI, TDB, etc.) and integration direction.
    ///
    /// # Panics
    /// Panics if `dt` is not finite or not positive.
    pub fn set_dt(&mut self, dt: f64) {
        assert!(
            dt.is_finite() && dt > 0.0,
            "dt must be finite and > 0, got {dt}"
        );
        self.dt = dt;
    }

    /// Current simulation elapsed time in seconds.
    pub fn elapsed(&self) -> f64 {
        self.time.simtime
    }
}
