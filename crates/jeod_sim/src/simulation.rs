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

/// Entry in the gravity source table.
///
/// Gravity sources are referenced by index (`usize`) from body gravity controls.
pub struct GravitySourceEntry {
    /// Physical gravity source (mu, model).
    pub source: GravitySource,
    /// Position in the inertial frame (m). For Earth-centered sims, Earth is at origin.
    pub position: DVec3,
    /// Inertial-to-planet-fixed rotation matrix. If `Some`, the ephemeris stage
    /// updates it each step. If `None`, no rotation is applied (point-mass only).
    pub t_inertial_pfix: Option<DMat3>,
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
}

impl Default for SimBody {
    fn default() -> Self {
        Self {
            trans: TranslationalState::default(),
            rot: None,
            mass: None,
            config: DynamicsConfig::default(),
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
    /// Integration timestep (seconds).
    pub dt: f64,
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
            dt,
        }
    }

    /// Add a gravity source. Returns its index for use in `GravityControls`.
    pub fn add_source(&mut self, entry: GravitySourceEntry) -> usize {
        let idx = self.sources.len();
        self.sources.push(entry);
        idx
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
        for body in &mut self.bodies {
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
                all_errors.push(ValidationError::ForceProducerWithoutMass);
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
        if fatal.is_empty() {
            Ok(())
        } else {
            Err(fatal)
        }
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
        let dt = self.dt;

        // ── 1. Time update ──
        self.time.advance(dt);

        // ── 2. Ephemeris update — planet-fixed rotations ──
        // JEOD_INV: DM.13 — ephemeris updated before gravity
        // NOTE: Currently applies the same Earth RNP rotation to ALL rotating
        // sources. Multi-planet sims (Moon, Mars) would need per-source rotation
        // parameters. This is a Phase 5 limitation.
        let rotation =
            crate::compute_t_parent_this_from_tjt(self.time.gmst_seconds, self.time.tt_tjt());
        for source in &mut self.sources {
            if source.t_inertial_pfix.is_some() {
                source.t_inertial_pfix = Some(rotation);
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
                |source_id: usize| {
                    sources
                        .get(source_id)
                        .map(|s| (&s.source, s.t_inertial_pfix.as_ref()))
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
        // Gravity is recomputed at each RK4 intermediate state for 4th-order
        // accuracy, matching JEOD's DynamicsIntegrationGroup where the
        // derivative function calls gravity at every stage.
        let sources = &self.sources;
        for body in &mut self.bodies {
            let controls = &body.gravity_controls;
            integrate_body(
                &body.config,
                &mut body.trans,
                body.rot.as_mut(),
                body.mass.as_ref(),
                |pos| {
                    accumulate_gravity(pos, controls, |source_id| {
                        sources
                            .get(source_id)
                            .map(|s| (&s.source, s.t_inertial_pfix.as_ref()))
                    })
                    .grav_accel
                },
                body.total_force.force,
                body.total_force.torque,
                dt,
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
            let saved_dt = self.dt;
            self.dt = remainder;
            self.step();
            self.dt = saved_dt;
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

    /// Current simulation elapsed time in seconds.
    pub fn elapsed(&self) -> f64 {
        self.time.simtime
    }
}
