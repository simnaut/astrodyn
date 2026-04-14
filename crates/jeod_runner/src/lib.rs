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

use jeod_sim::atmosphere::{evaluate_atmosphere, AtmosphereConfig};
use jeod_sim::forces::collect_and_resolve_forces;
use jeod_sim::gravity::accumulate_gravity;
use jeod_sim::integration::integrate_body;
use jeod_sim::validation::ValidationError;
use jeod_sim::{
    AerodynamicForce, AtmosphereState, DragConfig, DynamicsConfig, EulerSequence, FrameDerivatives,
    GeodeticState, GravityAcceleration, GravityControls, GravitySource, LvlhFrame, MassProperties,
    OrbitalElements, PlanetConfig, RadiationForce, RotationalState, SimulationTime, TotalForce,
    TranslationalState,
};

pub mod builder;

// Re-export jeod_sim so downstream tests can access types through either path.
pub use jeod_sim;
pub use jeod_sim::RotationModel;

// Re-export builder types for ergonomic use.
pub use builder::{SimulationBuilder, VehicleBuilder};

// ══════════════════════════════════════════════════════════════════════════════
// Integration frame switching
// ══════════════════════════════════════════════════════════════════════════════

/// Which celestial body's inertial frame is used for integration.
///
/// Body position/velocity are stored relative to this frame's origin.
/// The frame's origin position in canonical (Earth-centered) coordinates
/// is derived from ephemeris at each timestep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntegrationFrame {
    /// Earth-centered inertial (J2000 ICRF). Origin at (0,0,0).
    #[default]
    EarthInertial,
    /// Moon-centered inertial. Origin = Moon ephemeris position.
    MoonInertial,
    /// Sun-centered inertial. Origin = Sun ephemeris position.
    SunInertial,
}

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
/// Port of JEOD's `DynBodyFrameSwitch` body action.
#[derive(Debug, Clone)]
pub struct FrameSwitchConfig {
    /// Target integration frame to switch to.
    pub target_frame: IntegrationFrame,
    /// Whether to switch on approach or departure.
    pub switch_sense: SwitchSense,
    /// Distance threshold (meters).
    pub switch_distance: f64,
    /// Whether this switch is active.
    pub active: bool,
    /// Index of the gravity source that is the central body in the target frame.
    /// On switch, this source becomes non-differential and all others become
    /// differential, matching JEOD's `GravityInteraction::set_integ_frame()`.
    pub central_source: Option<usize>,
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
    /// Initial integration frame. Defaults to [`IntegrationFrame::EarthInertial`].
    pub integ_frame: IntegrationFrame,
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
            integ_frame: IntegrationFrame::default(),
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
    /// Current integration frame (for converting to inertial if needed).
    pub integ_frame: IntegrationFrame,
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
    integ_frame: IntegrationFrame,
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
    fn from_config(config: VehicleConfig) -> Self {
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

            integ_frame: config.integ_frame,
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
            integ_frame: self.integ_frame,
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
/// let earth = sim.add_source(GravitySourceEntry::central_body(&EARTH));
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
    /// Gravity sources.
    pub sources: Vec<GravitySourceEntry>,
    /// Atmosphere configuration. `None` disables atmosphere for all bodies.
    pub atmosphere: Option<AtmosphereConfig>,
    /// Index into `sources` for the planet whose rotation is used for atmosphere.
    pub atmosphere_planet_source: Option<usize>,
    /// Index into `sources` for the Sun (used by SRP and earth lighting).
    pub sun_source: Option<usize>,
    /// Index into `sources` for the Moon (used by earth lighting).
    pub moon_source: Option<usize>,
    /// Polar motion parameters (xp, yp) in radians. When `Some`, the RNP
    /// composition includes polar motion: W(xp,yp) × R(GAST) × N × P.
    /// When `None`, polar motion is omitted (matches JEOD `enable_polar=false`).
    pub polar_motion: Option<(f64, f64)>,
    /// Integration timestep (seconds).
    pub dt: f64,
    /// Optional ephemeris for per-step source position updates.
    pub ephemeris: Option<jeod_sim::Ephemeris>,
    /// Per-source ephemeris body mapping. Index matches `sources` vector.
    pub source_ephem_bodies: Vec<Option<(jeod_sim::EphemerisBody, jeod_sim::EphemerisBody)>>,
    /// Optional mass tree for multi-body vehicles (attach/detach/staging).
    /// Bodies participating in the tree have `SimBody::mass_body_id` set.
    pub mass_tree: Option<jeod_dynamics::MassTree>,
}

impl Simulation {
    /// Create a new simulation with the given initial time and timestep.
    pub fn new(time: SimulationTime, dt: f64) -> Self {
        Self {
            time,
            bodies: Vec::new(),
            sources: Vec::new(),
            atmosphere: None,
            atmosphere_planet_source: None,
            sun_source: None,
            moon_source: None,
            polar_motion: None,
            dt,
            ephemeris: None,
            source_ephem_bodies: Vec::new(),
            mass_tree: None,
        }
    }

    /// Add a gravity source. Returns its index for use in `GravityControls`.
    pub fn add_source(&mut self, entry: GravitySourceEntry) -> usize {
        let idx = self.sources.len();
        self.sources.push(entry);
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
        self.source_ephem_bodies[source_idx] = Some((target, observer));
    }

    /// Add a dynamic body from a [`VehicleConfig`]. Returns its index.
    ///
    /// The config is consumed and converted into internal state. Use
    /// [`body`](Simulation::body) to access results after stepping.
    pub fn add_body(&mut self, config: VehicleConfig) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(SimBody::from_config(config));
        idx
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
                |source_id: usize| self.sources.get(source_id).map(|s| &s.source),
                plate_counts,
            );
            all_errors.extend(errors);

            // Validate shadow_body index
            if let Some((idx, _radius)) = body.shadow_body {
                if idx >= self.sources.len() {
                    all_errors.push(ValidationError::ShadowBodyOutOfRange {
                        index: idx,
                        num_sources: self.sources.len(),
                    });
                }
            }

            // Validate geodetic_planet index
            if let Some((idx, _, _)) = body.geodetic_planet {
                if idx >= self.sources.len() {
                    all_errors.push(ValidationError::GeodeticPlanetOutOfRange {
                        index: idx,
                        num_sources: self.sources.len(),
                    });
                }
            }

            // Validate orbital_elements_source index
            if let Some(idx) = body.orbital_elements_source {
                if idx >= self.sources.len() {
                    all_errors.push(ValidationError::OrbitalElementsSourceOutOfRange {
                        index: idx,
                        num_sources: self.sources.len(),
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

            // Frame switch central_source must be a valid source index AND
            // present in the body's gravity controls (so the post-switch
            // differential flip actually takes effect).
            for sw in &body.frame_switches {
                if let Some(central) = sw.central_source {
                    let in_range = central < self.sources.len();
                    let in_controls = body
                        .gravity_controls
                        .controls
                        .iter()
                        .any(|c| c.source_name == central);
                    if !in_range || !in_controls {
                        all_errors.push(ValidationError::FrameSwitchCentralSourceOutOfRange {
                            body_idx,
                            central_source: central,
                            num_sources: self.sources.len(),
                        });
                    }
                }
            }

            // Apply gravity control auto-corrections (degree/order clamping).
            // JEOD_INV: GV.03 — check_validity() auto-corrects out-of-range settings
            for ctrl in &mut body.gravity_controls.controls {
                if let Some(source_entry) = self.sources.get(ctrl.source_name) {
                    ctrl.check_validity(&source_entry.source);
                }
            }
        }

        // Validate sun_source index (simulation-level, outside body loop)
        if let Some(idx) = self.sun_source {
            if idx >= self.sources.len() {
                all_errors.push(ValidationError::SunSourceOutOfRange {
                    index: idx,
                    num_sources: self.sources.len(),
                });
            }
        }

        // Validate moon_source index
        if let Some(idx) = self.moon_source {
            if idx >= self.sources.len() {
                all_errors.push(ValidationError::MoonSourceOutOfRange {
                    index: idx,
                    num_sources: self.sources.len(),
                });
            }
        }

        // Validate atmosphere_planet_source index
        if let Some(idx) = self.atmosphere_planet_source {
            if idx >= self.sources.len() {
                all_errors.push(ValidationError::AtmospherePlanetOutOfRange {
                    index: idx,
                    num_sources: self.sources.len(),
                });
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
    /// 2. Ephemeris update (planet-fixed rotations)
    /// 3. Mass update (recompute derived quantities)
    /// 4. Gravity computation
    /// 5. Atmosphere evaluation
    /// 6. Interaction computation (drag, SRP, gravity torque)
    /// 7. Force collection and frame derivative computation
    /// 8. State integration (RK4)
    /// 9. Derived state computation
    pub fn step(&mut self) {
        self.step_internal(self.dt);
    }

    /// Resolve the origin position and velocity of an integration frame
    /// in canonical (Earth-centered inertial) coordinates.
    pub fn resolve_frame_origin(&self, frame: IntegrationFrame) -> (DVec3, DVec3) {
        match frame {
            IntegrationFrame::EarthInertial => (DVec3::ZERO, DVec3::ZERO),
            IntegrationFrame::MoonInertial => {
                if let Some(ref eph) = self.ephemeris {
                    let tdb_jd = self.time.tdb_julian_date();
                    eph.get_state(
                        jeod_sim::EphemerisBody::Moon,
                        jeod_sim::EphemerisBody::Earth,
                        tdb_jd,
                    )
                    .expect("Moon ephemeris lookup failed for frame switch")
                } else {
                    panic!("MoonInertial frame requires ephemeris");
                }
            }
            IntegrationFrame::SunInertial => {
                if let Some(ref eph) = self.ephemeris {
                    let tdb_jd = self.time.tdb_julian_date();
                    eph.get_state(
                        jeod_sim::EphemerisBody::Sun,
                        jeod_sim::EphemerisBody::Earth,
                        tdb_jd,
                    )
                    .expect("Sun ephemeris lookup failed for frame switch")
                } else {
                    panic!("SunInertial frame requires ephemeris");
                }
            }
        }
    }

    /// Internal step with explicit dt (avoids temporary mutation of `self.dt`
    /// in `step_until`).
    fn step_internal(&mut self, dt: f64) {
        // ── 1. Time update ──
        self.time.advance(dt);

        // ── 2. Ephemeris update — planet-fixed rotations ──
        // JEOD_INV: DM.13 — ephemeris updated before gravity
        // Per-source rotation dispatch: each source has its own rotation model.
        // Lazy-compute Earth RNP only if needed (most common case).
        let mut earth_rotation: Option<DMat3> = Option::None;
        for source in &mut self.sources {
            match source.rotation_model {
                RotationModel::None => {}
                RotationModel::EarthRNP => {
                    let rotation = *earth_rotation.get_or_insert_with(|| {
                        jeod_sim::compute_t_parent_this_from_tjt_with_polar(
                            self.time.gmst_seconds,
                            self.time.tt_tjt(),
                            self.polar_motion,
                        )
                    });
                    source.t_inertial_pfix = Some(rotation);
                }
                RotationModel::MarsIAU => {
                    // JEOD's RNPMars receives TT seconds since J2000 (time_tt.seconds).
                    let tt_s_since_j2000 = (self.time.tt_tjt() - jeod_time::epoch::J2000_TT_TJT)
                        * jeod_time::epoch::SECONDS_PER_DAY;
                    let rotation =
                        jeod_frames::rotation_mars::compute_mars_rotation(tt_s_since_j2000);
                    source.t_inertial_pfix = Some(rotation);
                }
                RotationModel::MoonIAU => {
                    let tdb_jd = self.time.tdb_julian_date();
                    let tdb_s_since_j2000 = (tdb_jd - jeod_time::epoch::J2000_TT_JD)
                        * jeod_time::epoch::SECONDS_PER_DAY;
                    let rotation =
                        jeod_frames::rotation_moon::compute_moon_rotation(tdb_s_since_j2000);
                    source.t_inertial_pfix = Some(rotation);
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
                    source.t_inertial_pfix = Some(rotation);
                }
            }
            // Compute tidal ΔC20 if configured; otherwise clear any stale value.
            // Uses whatever rotation is current (Earth RNP for Earth sources).
            if let Some(ref config) = source.tidal_config {
                let rotation = source.t_inertial_pfix.expect(
                    "tidal_config requires t_inertial_pfix (planet-fixed rotation). \
                     Set a rotation_model or provide an initial t_inertial_pfix.",
                );
                source.delta_c20 = jeod_gravity::tides::compute_delta_c20(config, &rotation);
            } else {
                source.delta_c20 = 0.0;
            }
        }

        // ── 2b. Ephemeris update — source positions from DE4xx ──
        // Update source positions from ephemeris each step (for 3rd-body accuracy).
        if let Some(ref eph) = self.ephemeris {
            let tdb_jd = self.time.tdb_julian_date();
            for (i, source) in self.sources.iter_mut().enumerate() {
                if let Some(Some((target, observer))) = self.source_ephem_bodies.get(i) {
                    let (pos, vel) =
                        eph.get_state(*target, *observer, tdb_jd)
                            .unwrap_or_else(|e| {
                                panic!(
                                    "Ephemeris lookup failed for source {i} \
                                 ({target:?} wrt {observer:?}) at TDB JD {tdb_jd}: {e}"
                                )
                            });
                    source.position = pos;
                    source.velocity = vel;
                }
            }
        }

        // ── 3. Mass update — recompute inverse_mass/inverse_inertia ──
        for body in &mut self.bodies {
            if let Some(ref mut mass) = body.mass {
                mass.recompute_derived();
            }
        }

        // Precompute frame origins only for frames actually in use.
        // Avoids unnecessary ephemeris lookups and panics early if ephemeris
        // is absent but a non-Earth frame is needed.
        let (needs_moon, needs_sun) =
            self.bodies
                .iter()
                .fold((false, false), |(moon, sun), body| {
                    let (m, s) = match body.integ_frame {
                        IntegrationFrame::MoonInertial => (true, sun),
                        IntegrationFrame::SunInertial => (moon, true),
                        IntegrationFrame::EarthInertial => (moon, sun),
                    };
                    body.frame_switches
                        .iter()
                        .filter(|sw| sw.active)
                        .fold((m, s), |(m, s), sw| match sw.target_frame {
                            IntegrationFrame::MoonInertial => (true, s),
                            IntegrationFrame::SunInertial => (m, true),
                            IntegrationFrame::EarthInertial => (m, s),
                        })
                });

        let earth_origin = (DVec3::ZERO, DVec3::ZERO);
        let moon_origin = if needs_moon {
            if self.ephemeris.is_some() {
                self.resolve_frame_origin(IntegrationFrame::MoonInertial)
            } else {
                panic!("MoonInertial frame requires ephemeris. Set sim.ephemeris = Some(...).")
            }
        } else {
            (DVec3::ZERO, DVec3::ZERO)
        };
        let sun_origin = if needs_sun {
            if self.ephemeris.is_some() {
                self.resolve_frame_origin(IntegrationFrame::SunInertial)
            } else {
                panic!("SunInertial frame requires ephemeris. Set sim.ephemeris = Some(...).")
            }
        } else {
            (DVec3::ZERO, DVec3::ZERO)
        };
        let resolve = |frame: IntegrationFrame| -> (DVec3, DVec3) {
            match frame {
                IntegrationFrame::EarthInertial => earth_origin,
                IntegrationFrame::MoonInertial => moon_origin,
                IntegrationFrame::SunInertial => sun_origin,
            }
        };

        // ── 4. Environment — gravity ──
        // Split borrows: sources (immutable) vs bodies (mutable)
        let sources = &self.sources;
        for body in &mut self.bodies {
            let integ_origin = resolve(body.integ_frame).0;
            body.gravity_accel = accumulate_gravity(
                body.trans.position + integ_origin,
                &body.gravity_controls,
                integ_origin,
                |source_id: usize| {
                    sources.get(source_id).map(|s| jeod_sim::ResolvedSource {
                        source: &s.source,
                        rotation: s.t_inertial_pfix.as_ref(),
                        position: s.position,
                        delta_c20: s.delta_c20,
                        has_delta_coeffs: s.tidal_config.is_some(),
                    })
                },
            );
        }

        // ── 4b. Relativistic corrections ──
        // After Newtonian gravity, apply post-Newtonian PPN correction for
        // any source with `relativistic: true`. Folkner eq 27 (β=γ=1).
        for body in &mut self.bodies {
            body.gravity_accel.grav_accel += jeod_sim::accumulate_relativistic_corrections(
                body.trans.position,
                body.trans.velocity,
                &body.gravity_controls,
                |source_id: usize| {
                    sources
                        .get(source_id)
                        .map(|s| jeod_sim::ResolvedRelativisticSource {
                            mu: s.source.mu,
                            position: s.position,
                            velocity: s.velocity,
                        })
                },
            );
        }

        // ── 5. Environment — atmosphere ──
        if let Some(ref atmos_config) = self.atmosphere {
            let t_pfix = self
                .atmosphere_planet_source
                .and_then(|idx| self.sources.get(idx))
                .and_then(|s| s.t_inertial_pfix.as_ref());
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
        let sun_pos = self
            .sun_source
            .and_then(|idx| self.sources.get(idx).map(|s| s.position));
        let moon_pos = self
            .moon_source
            .and_then(|idx| self.sources.get(idx).map(|s| s.position));
        let sources = &self.sources;

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
                                    sources[idx].position,
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
                                sources[idx].position,
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
        // Precompute per-body integration frame origins and velocities.
        // For non-ECI bodies, the origin velocity is used to linearly
        // interpolate the frame origin at each RK4 sub-stage, matching
        // JEOD's behavior of updating the reference frame tree at each
        // derivative evaluation (even with deriv_ephem_update=false).
        let body_integ_data: Vec<(DVec3, DVec3)> =
            self.bodies.iter().map(|b| resolve(b.integ_frame)).collect();
        let sources = &self.sources;
        for (body_idx, body) in self.bodies.iter_mut().enumerate() {
            let (integ_origin, integ_vel) = body_integ_data[body_idx];
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
                    let src = sources.get(ctrl.source_name)?;
                    let other: Vec<_> = controls
                        .controls
                        .iter()
                        .filter(|c| c.source_name != ctrl.source_name)
                        .filter_map(|c| {
                            sources.get(c.source_name).map(|s| {
                                jeod_gravity::relativistic::RelativisticSource {
                                    mu: s.source.mu,
                                    position: s.position,
                                }
                            })
                        })
                        .collect();
                    Some((src.source.mu, src.position, src.velocity, other))
                })
                .collect();

            integrate_body(
                &body.config,
                &mut body.trans,
                body.rot.as_mut(),
                body.mass.as_ref(),
                |pos, vel, time_frac| {
                    // For non-ECI frames, linearly interpolate the frame origin
                    // at the current integrator sub-stage time. This matches JEOD's
                    // behavior of updating its reference frame tree at each
                    // derivative evaluation.
                    let origin = integ_origin + integ_vel * (time_frac * dt);
                    // Interpolate source positions at the sub-stage time,
                    // matching JEOD's continuous frame tree updates.
                    // Only for non-ECI frames; ECI with deriv_ephem_update=false
                    // freezes source positions per step (matching JEOD).
                    let sub_dt = if integ_vel != DVec3::ZERO {
                        time_frac * dt
                    } else {
                        0.0
                    };
                    let mut accel =
                        accumulate_gravity(pos + origin, controls, origin, |source_id| {
                            sources.get(source_id).map(|s| jeod_sim::ResolvedSource {
                                source: &s.source,
                                rotation: s.t_inertial_pfix.as_ref(),
                                position: s.position + s.velocity * sub_dt,
                                delta_c20: s.delta_c20,
                                has_delta_coeffs: s.tidal_config.is_some(),
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

        // ── 8b. Frame switch (body actions) ──
        // Applied AFTER integration, matching JEOD's pipeline where
        // DynBodyFrameSwitch is a body action evaluated post-integration.
        // The body has already been integrated in its current frame for this
        // step; the switch transforms to the new frame for the NEXT step.
        for body in &mut self.bodies {
            if body.frame_switches.is_empty() {
                continue;
            }
            let mut switch_idx = None;
            for (idx, sw) in body.frame_switches.iter().enumerate() {
                if !sw.active {
                    continue;
                }
                let (target_origin, _) = resolve(sw.target_frame);
                let (current_origin, _) = resolve(body.integ_frame);
                let body_pos_eci = body.trans.position + current_origin;
                let dist_sq = (body_pos_eci - target_origin).length_squared();
                let threshold_sq = sw.switch_distance * sw.switch_distance;

                let triggered = match sw.switch_sense {
                    SwitchSense::OnApproach => dist_sq < threshold_sq,
                    SwitchSense::OnDeparture => dist_sq > threshold_sq,
                };
                if triggered {
                    switch_idx = Some(idx);
                    break;
                }
            }
            if let Some(idx) = switch_idx {
                let target_frame = body.frame_switches[idx].target_frame;
                let central_source = body.frame_switches[idx].central_source;
                body.frame_switches[idx].active = false;

                let (old_origin, old_vel) = resolve(body.integ_frame);
                let (new_origin, new_vel) = resolve(target_frame);
                body.trans.position = body.trans.position + old_origin - new_origin;
                body.trans.velocity = body.trans.velocity + old_vel - new_vel;
                body.integ_frame = target_frame;

                if let Some(central) = central_source {
                    for ctrl in &mut body.gravity_controls.controls {
                        ctrl.differential = ctrl.source_name != central;
                    }
                }
            }
        }

        // ── 9. Derived states ──
        // NOTE: Derived states, atmosphere evaluation, SRP/shadow geometry, solar
        // beta, and earth lighting all assume body.trans.position is in ECI.
        // When integ_frame != EarthInertial, these results will be incorrect.
        // Currently frame switching is only supported for gravity + integration;
        // non-Earth frames disable these other pipeline stages in practice
        // (no existing test configures both frame switching and interactions).
        // A proper fix requires converting to ECI at each call site (#67).
        let sources = &self.sources;

        for body in &mut self.bodies {
            // Orbital elements
            if let Some(src_idx) = body.orbital_elements_source {
                if let Some(mu) = sources.get(src_idx).map(|s| s.source.mu) {
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
                if let Some(src) = sources.get(src_idx) {
                    if let Some(t_pfix) = src.t_inertial_pfix.as_ref() {
                        body.geodetic_state = Some(jeod_sim::compute_body_geodetic(
                            body.trans.position,
                            t_pfix,
                            r_eq,
                            r_pol,
                        ));
                    } else {
                        body.geodetic_state = None;
                    }
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
    }

    /// Set a body's translational velocity (inertial frame, m/s).
    ///
    /// Used for impulsive maneuvers (e.g., Apollo TLI delta-V).
    pub fn set_body_velocity(&mut self, idx: usize, velocity: DVec3) {
        self.bodies[idx].trans.velocity = velocity;
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
