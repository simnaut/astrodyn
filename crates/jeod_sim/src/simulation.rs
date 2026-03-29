use glam::{DMat3, DVec3};

use crate::atmosphere::{evaluate_atmosphere, AtmosphereConfig};
use crate::forces::collect_and_resolve_forces;
use crate::gravity::accumulate_gravity;
use crate::integration::integrate_body;
use crate::validation::ValidationError;
use crate::{
    AerodynamicForce, AtmosphereState, DragConfig, DynamicsConfig, FlatPlate, FlatPlateParams,
    FlatPlateThermal, FrameDerivatives, GravityAcceleration, GravityControls, GravitySource,
    MassProperties, RadiationForce, RotationalState, SimulationTime, TotalForce,
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
    /// Flat-plate SRP configuration (plate geometry, optical, thermal properties). `None` disables SRP.
    /// Requires `plate_temperatures`
    /// and `plate_t_pow4_cached` to be initialized with matching lengths.
    pub flat_plates: Option<Vec<(FlatPlate, FlatPlateParams, FlatPlateThermal)>>,
    /// Per-plate temperatures (K). Same length as `flat_plates`.
    pub plate_temperatures: Vec<f64>,
    /// Cached T⁴ per plate (K⁴) from previous step. Same length as `flat_plates`.
    /// Used for thermal emission force computation (JEOD convention: emission uses
    /// previous-step temperature, not current).
    pub plate_t_pow4_cached: Vec<f64>,
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
    pub bodies: Vec<SimBody>,
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
            let errors = crate::validate_body(
                &body.config,
                &body.gravity_controls,
                true, // SimBody always has gravity_accel field
                body.mass.as_ref(),
                body.rot.is_some(),
                Some(&body.trans),
                |source_id: usize| self.sources.get(source_id).map(|s| &s.source),
            );
            all_errors.extend(errors);

            // Validate plate_temperatures / plate_t_pow4_cached lengths match flat_plates
            if let Some(ref plates) = body.flat_plates {
                let n = plates.len();
                if body.plate_temperatures.len() != n || body.plate_t_pow4_cached.len() != n {
                    all_errors.push(ValidationError::PlateTemperatureLengthMismatch {
                        num_plates: n,
                        num_temperatures: body.plate_temperatures.len(),
                        num_t_pow4: body.plate_t_pow4_cached.len(),
                    });
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
        if all_errors.is_empty() {
            Ok(())
        } else {
            Err(all_errors)
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
    pub fn step(&mut self) {
        let dt = self.dt;

        // ── 1. Time update ──
        self.time.advance(dt);

        // ── 2. Ephemeris update — planet-fixed rotations ──
        // JEOD_INV: DM.13 — ephemeris updated before gravity
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
        let sun_pos = self.sun_source.map(|idx| self.sources[idx].position);
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

            // Solar radiation pressure (flat-plate or spherical)
            body.radiation_force = None;
            if let Some(sun_position) = sun_pos {
                if let Some(ref flat_plates) = body.flat_plates {
                    // Flat-plate SRP with thermal emission
                    let sun_to_vehicle = body.trans.position - sun_position;
                    let distance = sun_to_vehicle.length();
                    if distance < 1.0 {
                        continue;
                    }
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
                        flat_plates,
                        &body.plate_t_pow4_cached,
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

                    // Integrate plate temperatures (forward Euler)
                    for (i, temp) in body.plate_temperatures.iter_mut().enumerate() {
                        *temp += srp_result.temp_dots[i] * dt;
                        if *temp < 0.0 {
                            *temp = 0.0;
                        }
                    }
                    body.plate_t_pow4_cached =
                        body.plate_temperatures.iter().map(|t| t.powi(4)).collect();
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

    /// Access a body by index.
    pub fn body(&self, idx: usize) -> &SimBody {
        &self.bodies[idx]
    }

    /// Mutably access a body by index.
    pub fn body_mut(&mut self, idx: usize) -> &mut SimBody {
        &mut self.bodies[idx]
    }

    /// Current simulation elapsed time in seconds.
    pub fn elapsed(&self) -> f64 {
        self.time.simtime
    }
}
