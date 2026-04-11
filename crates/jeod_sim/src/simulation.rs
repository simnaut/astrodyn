use glam::{DMat3, DVec3};

use crate::atmosphere::{evaluate_atmosphere, AtmosphereConfig};
use crate::forces::collect_and_resolve_forces;
use crate::gravity::accumulate_gravity;
use crate::integration::integrate_body;
use crate::validation::ValidationError;
use crate::{
    AerodynamicForce, AtmosphereState, DragConfig, DynamicsConfig, EulerSequence, FrameDerivatives,
    GeodeticState, GravityAcceleration, GravityControls, GravitySource, LvlhFrame, MassProperties,
    OrbitalElements, RadiationForce, RotationalState, SimulationTime, TotalForce,
    TranslationalState,
};

/// Rotation model for a gravity source's planet-fixed frame.
///
/// Determines how `t_inertial_pfix` is updated each step. Each planet has its
/// own rotation model; point-mass sources use `None`.
#[derive(Debug, Clone, Default)]
pub enum RotationModel {
    /// No rotation — point-mass source or body without a planet-fixed frame.
    #[default]
    None,
    /// Earth rotation via IAU 2000A precession-nutation + GAST + optional polar
    /// motion. Uses the simulation's `gmst_seconds`, `tt_tjt`, and `polar_motion`.
    EarthRNP,
    /// Mars rotation via IAU pole orientation + spin + nutation Fourier series.
    /// Uses the simulation's TDB Julian date.
    MarsIAU,
    /// Moon rotation via IAU 2009 pole + prime meridian model.
    /// Uses the simulation's TDB seconds.
    MoonIAU,
}

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
    /// Inertial-to-planet-fixed rotation matrix. If `Some`, the ephemeris stage
    /// updates it each step. If `None`, no rotation is applied (point-mass only).
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
    pub fn new(source: GravitySource, position: DVec3, t_inertial_pfix: Option<DMat3>) -> Self {
        // Infer rotation model from t_inertial_pfix presence for backward compat
        let rotation_model = if t_inertial_pfix.is_some() {
            RotationModel::EarthRNP
        } else {
            RotationModel::None
        };
        Self {
            source,
            position,
            velocity: DVec3::ZERO,
            t_inertial_pfix,
            rotation_model,
            delta_c20: 0.0,
            tidal_config: None,
        }
    }
}

/// Per-body simulation state and configuration.
///
/// Combines dynamic state (integrated), configuration (fixed), and computed
/// intermediates (written each step). In an ECS, these would be separate
/// components; here they are co-located for standalone use.
pub struct SimBody {
    // ── Dynamic state (integrated each step) ──
    /// Translational state: position and velocity in the inertial frame.
    pub trans: TranslationalState,
    /// Rotational state: quaternion and angular velocity. `None` for 3-DOF bodies.
    pub rot: Option<RotationalState>,
    /// Mass properties. `None` for massless test particles (gravity-only).
    pub mass: Option<MassProperties>,

    // ── Configuration (fixed between steps) ──
    /// Dynamics flags: translational/rotational/three_dof.
    pub config: DynamicsConfig,
    /// Gravity controls referencing sources by index.
    pub gravity_controls: GravityControls<usize>,
    /// Integration method. Defaults to `IntegratorType::Rk4`.
    pub integrator: jeod_dynamics::IntegratorType,
    /// Drag configuration. `None` disables drag.
    pub drag: Option<DragConfig>,
    /// Flat-plate SRP configuration with thermal state. `None` disables SRP.
    pub flat_plate_state: Option<crate::FlatPlateState>,
    /// Shadow-casting body: `(source_index, body_radius_m)`.
    /// Used by flat-plate SRP for eclipse computation.
    pub shadow_body: Option<(usize, f64)>,
    /// Structural-to-body rotation matrix. `DMat3::IDENTITY` when structure = body.
    pub t_struct_body: DMat3,
    /// Whether to compute gravity gradient torque for this body.
    pub compute_gravity_torque: bool,
    /// Set to `Some(Default)` to enable atmosphere computation for this body.
    /// `None` means no atmosphere (JEOD_INV: AT.01 — absence = inactive).
    pub atmospheric_state: Option<AtmosphereState>,

    // ── Computed intermediates (written each step, readable after) ──
    /// Accumulated gravitational acceleration, gradient, and potential.
    pub gravity_accel: GravityAcceleration,
    /// Total non-gravity force (inertial) and torque (body).
    pub total_force: TotalForce,
    /// Frame derivatives (translational and rotational acceleration).
    pub frame_derivs: FrameDerivatives,
    /// Aerodynamic force and torque in structural frame.
    pub aero_force: Option<AerodynamicForce>,
    /// Radiation force (inertial) and torque (structural).
    pub radiation_force: Option<RadiationForce>,
    /// Gravity gradient torque in body frame.
    pub gravity_torque: Option<DVec3>,

    // ── Derived state configuration (optional per-body) ──
    /// Gravity source index for orbital elements computation. `None` = skip.
    /// `mu` is read from the corresponding `GravitySourceEntry` at runtime,
    /// ensuring consistency with the dynamics gravity model.
    pub orbital_elements_source: Option<usize>,
    /// Euler angle decomposition sequence. `None` = skip.
    pub euler_sequence: Option<EulerSequence>,
    /// Whether to compute LVLH frame each step.
    pub compute_lvlh: bool,
    /// Planet source for geodetic: `(source_idx, r_eq, r_pol)`. `None` = skip.
    pub geodetic_planet: Option<(usize, f64, f64)>,
    /// Whether to compute solar beta angle each step. Requires `sun_source` on Simulation.
    pub compute_solar_beta: bool,

    // ── Derived state outputs (written each step if configured) ──
    /// Orbital elements from latest translational state.
    pub orbital_elements: Option<OrbitalElements>,
    /// Euler angles `[phi, theta, psi]` from latest rotational state.
    pub euler_angles: Option<[f64; 3]>,
    /// LVLH frame from latest translational state.
    pub lvlh_frame: Option<LvlhFrame>,
    /// Geodetic state (latitude, longitude, altitude).
    pub geodetic_state: Option<GeodeticState>,
    /// Solar beta angle (radians).
    pub solar_beta: Option<f64>,

    // ── Stateful integrator state ──
    /// Gauss-Jackson (Störmer-Cowell) integrator state. `None` for non-GJ bodies.
    /// Auto-initialized by `Simulation::validate()` when `integrator` is
    /// `IntegratorType::GaussJackson(config)`.
    pub gj_state: Option<jeod_dynamics::GaussJacksonState>,
}

impl Default for SimBody {
    fn default() -> Self {
        Self {
            trans: TranslationalState::default(),
            rot: None,
            mass: None,
            config: DynamicsConfig::default(),
            integrator: jeod_dynamics::IntegratorType::default(),
            gravity_controls: GravityControls::default(),
            drag: None,
            flat_plate_state: None,
            shadow_body: None,
            t_struct_body: DMat3::IDENTITY,
            compute_gravity_torque: false,
            atmospheric_state: None,
            gravity_accel: GravityAcceleration::default(),
            total_force: TotalForce::default(),
            frame_derivs: FrameDerivatives::default(),
            aero_force: None,
            radiation_force: None,
            gravity_torque: None,
            orbital_elements_source: None,
            euler_sequence: None,
            compute_lvlh: false,
            geodetic_planet: None,
            compute_solar_beta: false,
            orbital_elements: None,
            euler_angles: None,
            lvlh_frame: None,
            geodetic_state: None,
            solar_beta: None,
            gj_state: None,
        }
    }
}

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
/// let earth = sim.add_source(earth_source);
/// let vehicle = sim.add_body(vehicle_body);
/// sim.validate().unwrap();
/// sim.step_n(100);
/// println!("{:?}", sim.body(vehicle).trans.position);
/// ```
pub struct Simulation {
    /// Simulation time (TAI, UTC, TDB, GMST, etc.).
    pub time: SimulationTime,
    /// Dynamic bodies.
    // JEOD_INV: DS.01 — private to prevent runtime mutation of derived-state config
    bodies: Vec<SimBody>,
    /// Gravity sources.
    pub sources: Vec<GravitySourceEntry>,
    /// Atmosphere configuration. `None` disables atmosphere for all bodies.
    pub atmosphere: Option<AtmosphereConfig>,
    /// Index into `sources` for the planet whose rotation is used for atmosphere.
    pub atmosphere_planet_source: Option<usize>,
    /// Index into `sources` for the Sun (used by SRP).
    pub sun_source: Option<usize>,
    /// Polar motion parameters (xp, yp) in radians. When `Some`, the RNP
    /// composition includes polar motion: W(xp,yp) × R(GAST) × N × P.
    /// When `None`, polar motion is omitted (matches JEOD `enable_polar=false`).
    ///
    /// For static simulations, set this once. For time-varying polar motion,
    /// update before each step from IERS EOP data (table interpolation).
    pub polar_motion: Option<(f64, f64)>,
    /// Integration timestep (seconds).
    pub dt: f64,
    /// Optional ephemeris for per-step source position updates.
    /// When set, sources with `ephemeris_body` configured will have their
    /// position (and velocity) updated from DE421 each step.
    pub ephemeris: Option<crate::Ephemeris>,
    /// Per-source ephemeris body mapping. Index matches `sources` vector.
    /// `None` means the source position is static (not updated from ephemeris).
    pub source_ephem_bodies: Vec<Option<(crate::EphemerisBody, crate::EphemerisBody)>>,
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
            polar_motion: None,
            dt,
            ephemeris: None,
            source_ephem_bodies: Vec::new(),
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
    /// `target` is the body this source represents (e.g., `EphemerisBody::Sun`).
    /// `observer` is the integration frame center (e.g., `EphemerisBody::Earth`).
    pub fn set_source_ephemeris(
        &mut self,
        source_idx: usize,
        target: crate::EphemerisBody,
        observer: crate::EphemerisBody,
    ) {
        assert!(
            source_idx < self.source_ephem_bodies.len(),
            "set_source_ephemeris: source_idx {source_idx} out of bounds (len = {})",
            self.source_ephem_bodies.len()
        );
        self.source_ephem_bodies[source_idx] = Some((target, observer));
    }

    /// Add a dynamic body. Returns its index.
    pub fn add_body(&mut self, body: SimBody) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(body);
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
            let errors = crate::validate_body(
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

            // Gravity torque requires both mass and rotational state
            if body.compute_gravity_torque && (body.mass.is_none() || body.rot.is_none()) {
                all_errors.push(ValidationError::GravityTorqueWithoutMassOrRot { body_idx });
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
                        crate::compute_t_parent_this_from_tjt_with_polar(
                            self.time.gmst_seconds,
                            self.time.tt_tjt(),
                            self.polar_motion,
                        )
                    });
                    source.t_inertial_pfix = Some(rotation);
                }
                RotationModel::MarsIAU => {
                    // JEOD's RNPMars receives TT seconds since J2000 (time_tt.seconds).
                    // Compute absolute TT seconds since J2000 from TT TJT:
                    //   tt_seconds = (tt_tjt - J2000_TT_TJT) * 86400
                    // J2000 TT TJT = 11544.5 (2000-01-01 12:00:00 TT)
                    let tt_s_since_j2000 = (self.time.tt_tjt() - 11544.5) * 86400.0;
                    let rotation =
                        jeod_frames::rotation_mars::compute_mars_rotation(tt_s_since_j2000);
                    source.t_inertial_pfix = Some(rotation);
                }
                RotationModel::MoonIAU => {
                    let tdb_jd = self.time.tdb_julian_date();
                    let tdb_s_since_j2000 = (tdb_jd - 2_451_545.0) * 86400.0;
                    let rotation =
                        jeod_frames::rotation_moon::compute_moon_rotation(tdb_s_since_j2000);
                    source.t_inertial_pfix = Some(rotation);
                }
            }
            // Compute tidal ΔC20 if configured; otherwise clear any stale value.
            // Uses whatever rotation is current (Earth RNP for Earth sources).
            if let Some(ref config) = source.tidal_config {
                let rotation = source.t_inertial_pfix.unwrap_or(DMat3::IDENTITY);
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
                    if let Ok((pos, vel)) = eph.get_state(*target, *observer, tdb_jd) {
                        source.position = pos;
                        source.velocity = vel;
                    }
                }
            }
        }

        // ── 3. Mass update — recompute inverse_mass/inverse_inertia ──
        for body in &mut self.bodies {
            if let Some(ref mut mass) = body.mass {
                mass.recompute_derived();
            }
        }

        // ── 4. Environment — gravity ──
        // Split borrows: sources (immutable) vs bodies (mutable)
        let sources = &self.sources;
        for body in &mut self.bodies {
            body.gravity_accel = accumulate_gravity(
                body.trans.position,
                &body.gravity_controls,
                DVec3::ZERO,
                |source_id: usize| {
                    sources.get(source_id).map(|s| crate::ResolvedSource {
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
            for ctrl in &body.gravity_controls.controls {
                if !ctrl.relativistic {
                    continue;
                }
                if let Some(src) = sources.get(ctrl.source_name) {
                    // Build "other sources" list for potential computation
                    let other: Vec<jeod_gravity::relativistic::RelativisticSource> = body
                        .gravity_controls
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

                    let correction = jeod_gravity::relativistic::compute_relativistic_correction(
                        src.source.mu,
                        src.position,
                        body.trans.position,
                        body.trans.velocity,
                        src.velocity,
                        &other,
                    );
                    body.gravity_accel.grav_accel += correction;
                }
            }
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
        // sun_pos is also used in stage 9 (solar beta); compute once here.
        let sun_pos = self
            .sun_source
            .and_then(|idx| self.sources.get(idx).map(|s| s.position));
        let sources = &self.sources;

        for body in &mut self.bodies {
            // Compute structural transform once (shared by drag and flat-plate SRP)
            let t_inertial_body = body.rot.as_ref().map_or(DMat3::IDENTITY, |r| {
                r.quaternion.left_quat_to_transformation()
            });
            let t_inertial_struct =
                crate::compute_t_inertial_struct(&body.t_struct_body, &t_inertial_body);

            // Aerodynamic drag
            body.aero_force = None;
            if let (Some(ref drag_config), Some(ref atmos)) = (&body.drag, &body.atmospheric_state)
            {
                body.aero_force = Some(crate::compute_drag(
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
                        let flux_mag = crate::solar_flux_at_distance(distance);

                        // Shadow fraction
                        let illum_factor = body
                            .shadow_body
                            .map(|(idx, radius)| {
                                crate::compute_shadow_fraction(
                                    body.trans.position,
                                    sun_position,
                                    sources[idx].position,
                                    radius,
                                    crate::SOLAR_RADIUS,
                                )
                            })
                            .unwrap_or(1.0);

                        // Rotate flux direction to structural frame
                        let flux_struct_hat = t_inertial_struct * flux_inertial_hat;

                        let center_grav = body.mass.as_ref().map_or(DVec3::ZERO, |m| m.position);

                        let srp_result = crate::compute_flat_plate_srp_thermal(
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
                }
            }

            // Gravity gradient torque
            body.gravity_torque = None;
            if body.compute_gravity_torque {
                if let (Some(ref rot), Some(ref mass)) = (&body.rot, &body.mass) {
                    body.gravity_torque = Some(crate::compute_gravity_torque(
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
        }

        // ── 8. Integration ──
        // Gravity (including relativistic corrections) is recomputed at each
        // RK4 intermediate state for 4th-order accuracy, matching JEOD's
        // DynamicsIntegrationGroup where the derivative function calls gravity
        // at every stage with the current intermediate position and velocity.
        let sources = &self.sources;
        for body in &mut self.bodies {
            let controls = &body.gravity_controls;
            integrate_body(
                &body.config,
                &mut body.trans,
                body.rot.as_mut(),
                body.mass.as_ref(),
                |pos, vel| {
                    let mut accel = accumulate_gravity(pos, controls, DVec3::ZERO, |source_id| {
                        sources.get(source_id).map(|s| crate::ResolvedSource {
                            source: &s.source,
                            rotation: s.t_inertial_pfix.as_ref(),
                            position: s.position,
                            delta_c20: s.delta_c20,
                            has_delta_coeffs: s.tidal_config.is_some(),
                        })
                    })
                    .grav_accel;
                    // Apply relativistic corrections inside the gravity closure
                    // so they're evaluated at each RK4 substep position.
                    for ctrl in &controls.controls {
                        if ctrl.relativistic {
                            if let Some(src) = sources.get(ctrl.source_name) {
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
                                accel +=
                                    jeod_gravity::relativistic::compute_relativistic_correction(
                                        src.source.mu,
                                        src.position,
                                        pos,
                                        vel,
                                        src.velocity,
                                        &other,
                                    );
                            }
                        }
                    }
                    accel
                },
                body.total_force.force,
                body.total_force.torque,
                dt,
                body.integrator,
                body.gj_state.as_mut(),
            );
        }

        // ── 9. Derived states ──
        let sources = &self.sources;

        for body in &mut self.bodies {
            // Orbital elements
            if let Some(src_idx) = body.orbital_elements_source {
                if let Some(mu) = sources.get(src_idx).map(|s| s.source.mu) {
                    body.orbital_elements = crate::compute_orbital_elements(
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
                    body.euler_angles = Some(crate::compute_body_euler_angles(rot, seq));
                } else {
                    body.euler_angles = None;
                }
            }

            // LVLH frame
            if body.compute_lvlh {
                body.lvlh_frame = Some(crate::compute_body_lvlh_frame(
                    body.trans.position,
                    body.trans.velocity,
                ));
            }

            // Geodetic state
            if let Some((src_idx, r_eq, r_pol)) = body.geodetic_planet {
                if let Some(src) = sources.get(src_idx) {
                    if let Some(t_pfix) = src.t_inertial_pfix.as_ref() {
                        body.geodetic_state = Some(crate::compute_body_geodetic(
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
                    body.solar_beta = Some(crate::compute_body_solar_beta(
                        body.trans.position,
                        body.trans.velocity,
                        sp,
                    ));
                } else {
                    body.solar_beta = None;
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
    /// Access a body by index (read-only).
    pub fn body(&self, idx: usize) -> &SimBody {
        &self.bodies[idx]
    }

    /// Read-only slice of all bodies.
    pub fn bodies(&self) -> &[SimBody] {
        &self.bodies
    }

    /// Number of bodies in the simulation.
    pub fn num_bodies(&self) -> usize {
        self.bodies.len()
    }

    /// Set the integration timestep. Use to reverse time direction mid-simulation
    /// (JEOD's `scale_factor = -1` mode). All integrators (RK4, RKF45) work
    /// with negative dt; Gauss-Jackson requires the same absolute step size.
    ///
    /// # Panics
    /// Panics if `dt` is not finite.
    pub fn set_dt(&mut self, dt: f64) {
        assert!(dt.is_finite(), "dt must be finite, got {dt}");
        self.dt = dt;
    }

    /// Current simulation elapsed time in seconds.
    pub fn elapsed(&self) -> f64 {
        self.time.simtime
    }
}
