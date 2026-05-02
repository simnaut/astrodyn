//! Per-step integration pipeline for [`super::Simulation`].
//!
//! Houses `step`, `step_internal` (the ~1,000-line RK4 orchestrator
//! covering JEOD's 9 integration stages), `step_n`, `step_until`,
//! `frame_origin`, and `sync_pfix_rotation`. A future split per JEOD
//! integration stage is tracked under issue #253 (Task B).

use glam::{DMat3, DVec3};

use jeod_frames::FrameId;
use jeod_sim::atmosphere::evaluate_atmosphere;
use jeod_sim::forces::collect_and_resolve_forces;
use jeod_sim::gravity::accumulate_gravity;
use jeod_sim::integration::{integrate_bodies_contact_coupled, integrate_body, CoupledBodyInput};
use jeod_sim::{
    evaluate_contact_pair, integrate_body_coupled, CoupledStageEval, GravityControls, JeodQuat,
    MassProperties, RadiationForce, RotationModel, RotationalState, SwitchSense,
    TranslationalState,
};

use super::types::ContactPairConfig;
use super::Simulation;
use crate::error::StepError;

impl Simulation {
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
    pub fn step(&mut self) -> Result<(), StepError> {
        self.step_internal(self.dt)
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

    /// Sync a planet-fixed frame node's rotation state from a computed matrix.
    ///
    /// Sets `t_parent_this`, derives `q_parent_this` from it, and sets
    /// `ang_vel_this = [0, 0, planet_omega]` matching JEOD's `planet_rnp.cc`.
    /// The `planet_omega` value comes from [`PlanetConfig::omega`] via
    /// [`GravityData::planet_omega`].
    fn sync_pfix_rotation(
        frame_tree: &mut jeod_frames::FrameTree,
        pfix_id: jeod_frames::FrameId,
        rotation: DMat3,
        planet_omega: f64,
    ) {
        let node = frame_tree.get_mut(pfix_id);
        node.state.rot.t_parent_this = rotation;
        node.state.rot.q_parent_this = JeodQuat::left_quat_from_transformation(&rotation);
        // JEOD sets ang_vel_this = [0, 0, planet_omega] in planet_rnp.cc.
        // This is used by compute_relative_state velocity composition.
        node.state.rot.ang_vel_this = DVec3::new(0.0, 0.0, planet_omega);
    }

    /// Internal step with explicit dt (avoids temporary mutation of `self.dt`
    /// in `step_until`).
    fn step_internal(&mut self, dt: f64) -> Result<(), StepError> {
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
                        Self::sync_pfix_rotation(
                            &mut self.frame_tree,
                            pfix_id,
                            rotation,
                            grav.planet_omega,
                        );
                    }
                }
                RotationModel::MarsIAU => {
                    // JEOD's RNPMars receives TT seconds since J2000 (time_tt.seconds).
                    let tt_s_since_j2000 = (self.time.tt_tjt() - jeod_time::epoch::J2000_TT_TJT)
                        * jeod_time::epoch::SECONDS_PER_DAY;
                    let rotation =
                        jeod_frames::rotation_mars::compute_mars_rotation(tt_s_since_j2000);
                    if let Some(pfix_id) = self.source_frame_ids[i].pfix {
                        Self::sync_pfix_rotation(
                            &mut self.frame_tree,
                            pfix_id,
                            rotation,
                            grav.planet_omega,
                        );
                    }
                }
                RotationModel::MoonIAU => {
                    let tdb_jd = self.time.tdb_julian_date();
                    let tdb_s_since_j2000 = (tdb_jd - jeod_time::epoch::J2000_TT_JD)
                        * jeod_time::epoch::SECONDS_PER_DAY;
                    let rotation =
                        jeod_frames::rotation_moon::compute_moon_rotation(tdb_s_since_j2000);
                    if let Some(pfix_id) = self.source_frame_ids[i].pfix {
                        Self::sync_pfix_rotation(
                            &mut self.frame_tree,
                            pfix_id,
                            rotation,
                            grav.planet_omega,
                        );
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
                        Self::sync_pfix_rotation(
                            &mut self.frame_tree,
                            pfix_id,
                            rotation,
                            grav.planet_omega,
                        );
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
                    let (pos_typed, vel_typed) = eph
                        .get_state_typed(*target, *observer, tdb_jd)
                        .map_err(|e| StepError::EphemerisLookup {
                            source_idx: i,
                            target: *target,
                            observer: *observer,
                            tdb_jd,
                            message: e.to_string(),
                        })?;
                    let (pos, vel) = (pos_typed.raw_si(), vel_typed.raw_si());
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
            if let Some(ref mut fps) = body.flat_plate_state {
                fps.stage_inputs = None;
            }
            if let Some(sun_position) = sun_pos {
                if let Some(ref mut fps) = body.flat_plate_state {
                    // Flat-plate SRP with thermal emission
                    let sun_to_vehicle = body.trans.position - sun_position;
                    let distance = sun_to_vehicle.length();
                    // Skip SRP (not the whole body) if too close to Sun
                    if distance >= 1.0 {
                        let flux_inertial_hat = sun_to_vehicle / distance;
                        let flux_mag = jeod_sim::solar_flux_at_distance(distance);

                        // Shadow fraction (step-constant — matches JEOD's
                        // scheduled-class shadow evaluation in SIM_3_ORBIT).
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

                        let center_grav = body.mass.as_ref().map_or(DVec3::ZERO, |m| m.position);

                        match fps.integration_order {
                            jeod_sim::ThermalIntegrationOrder::Scheduled => {
                                // Scheduled-class: compute SRP force + Euler T
                                // update once per step (JEOD SIM_3_ORBIT).
                                let flux_struct_hat = t_inertial_struct * flux_inertial_hat;
                                let srp_result = jeod_sim::compute_flat_plate_srp_thermal(
                                    &fps.plates,
                                    &fps.t_pow4_cached,
                                    flux_struct_hat,
                                    flux_mag,
                                    center_grav,
                                    illum_factor,
                                );

                                // Force: structural → inertial. Torque: stays structural.
                                let force_inertial =
                                    t_inertial_struct.transpose() * srp_result.force;
                                body.radiation_force = Some(RadiationForce {
                                    force: force_inertial,
                                    torque: srp_result.torque,
                                });

                                fps.integrate_temperatures(&srp_result.temp_dots, dt);
                            }
                            jeod_sim::ThermalIntegrationOrder::DerivativeFirstOrder
                            | jeod_sim::ThermalIntegrationOrder::DerivativeRk4 => {
                                // Derivative-class: SRP force (and optionally T)
                                // recomputed per RK4 stage. Stash the step-start
                                // inputs on the plate state; Stage 8 consumes
                                // them via `integrate_body_coupled` below.
                                fps.stage_inputs = Some(jeod_sim::FlatPlateStageInputs {
                                    sun_position,
                                    illum_factor,
                                    center_grav,
                                });
                                // `radiation_force` is left None here; Stage 8
                                // writes a representative final-stage value so
                                // `VehicleOutput` still reports SRP force.
                            }
                        }
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

        // Precompute per-body relativistic "other source" lists outside the
        // closures to avoid heap allocation at every RK4 stage.
        // Indexed by body_idx.
        type RelDatum = (
            f64,
            DVec3,
            DVec3,
            Vec<jeod_gravity::relativistic::RelativisticSource>,
        );
        let per_body_rel_data: Vec<Vec<RelDatum>> = self
            .bodies
            .iter()
            .map(|body| {
                let controls = &body.gravity_controls;
                controls
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
                    .collect()
            })
            .collect();

        // Dynamic timestep: JEOD's `integ_dyndt = sim_dt * time_scale_factor`
        // (`standard_integration_controls.cc:80-82`). Integrators step in
        // dynamic time, so stage-time interpolations of the integration
        // frame origin and source positions must scale by `integ_dt`, not
        // by the raw `dt` — otherwise reversed/scaled time produces
        // inconsistent gravity during coupled integration.
        let integ_dt = dt * self.time.time_scale_factor;

        // Helper: evaluate gravity (Newtonian + relativistic) at an
        // intermediate (pos, vel) for the given body. Takes `controls` as
        // a parameter so the non-contact path can borrow
        // `&body.gravity_controls` directly (no per-step clone), while
        // the contact-coupled path can pass slices of its cloned
        // `per_body_gravity_controls` snapshot.
        let eval_body_gravity = |controls: &GravityControls<usize>,
                                 body_idx: usize,
                                 pos: DVec3,
                                 vel: DVec3,
                                 time_frac: f64|
         -> DVec3 {
            let (integ_origin, integ_vel) = body_integ_origins[body_idx];
            let stage_dt = time_frac * integ_dt;
            let origin = integ_origin + integ_vel * stage_dt;
            let sub_dt = if integ_vel != DVec3::ZERO {
                stage_dt
            } else {
                0.0
            };
            let mut accel =
                accumulate_gravity(pos + origin, controls, origin, |source_id: usize| {
                    let grav = gravity_data.get(source_id)?;
                    let sfids = &source_frame_ids[source_id];
                    let position = base_positions[source_id] + base_velocities[source_id] * sub_dt;
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
            let pos_eci = pos + origin;
            let vel_eci = vel + integ_vel;
            for &(mu, src_pos, src_vel, ref other) in &per_body_rel_data[body_idx] {
                accel += jeod_gravity::relativistic::compute_relativistic_correction(
                    mu, src_pos, pos_eci, vel_eci, src_vel, other,
                );
            }
            accel
        };

        if self.contact_pairs.is_empty() {
            // ── Standard path: per-body RK4 / GJ integration ──
            // No clone of gravity_controls: the outer iter_mut gives us
            // a &mut SimBody, and Rust's disjoint-field split borrow lets
            // the closure capture &body.gravity_controls while other
            // fields of `body` are borrowed mutably for the integrator.
            let time_scale_factor = self.time.time_scale_factor;
            for (body_idx, body) in self.bodies.iter_mut().enumerate() {
                let stage_inputs_and_order = body
                    .flat_plate_state
                    .as_ref()
                    .and_then(|fps| fps.stage_inputs.map(|si| (si, fps.integration_order)));
                if let Some((srp_inputs, thermal_order)) = stage_inputs_and_order {
                    // JEOD_INV: IN.32 — derivative-class thermal: SRP force
                    // (and temp_dots for DerivativeRk4) recomputed at each
                    // RK4 stage from the intermediate orbital + thermal
                    // state. Aero / external / gravity-gradient torque are
                    // step-constant (scheduled-class) and captured once
                    // from total_force.
                    assert!(
                        matches!(body.integrator, jeod_dynamics::IntegratorType::Rk4),
                        "ThermalIntegrationOrder::{thermal_order:?} requires \
                         jeod_dynamics::IntegratorType::Rk4 for body {body_idx}; \
                         switch the body integrator to Rk4, or choose \
                         ThermalIntegrationOrder::Scheduled to avoid the coupled \
                         RK4 thermal path.",
                    );
                    let t_struct_body = body.t_struct_body;
                    let non_grav_non_srp_force = body.total_force.force;
                    let constant_torque = body.total_force.torque;
                    let config = body.config;
                    let controls = &body.gravity_controls;
                    // Stash the final-stage SRP result so we can write a
                    // representative `radiation_force` for `VehicleOutput`.
                    let mut final_srp_inertial_force = DVec3::ZERO;
                    let mut final_srp_torque = DVec3::ZERO;
                    // For DerivativeFirstOrder: capture stage-1 temp_dots and
                    // feed them back at stages 2-4, collapsing the RK4 thermal
                    // combine to Euler (matches JEOD's ER7_Utils first-order
                    // integrator behavior while still evaluating SRP per stage
                    // for the orbital RK4).
                    let mut k1_temp_dots: Option<Vec<f64>> = None;
                    let mass_copy = body.mass;

                    integrate_body_coupled(
                        &config,
                        &mut body.trans,
                        body.rot.as_mut(),
                        mass_copy.as_ref(),
                        |stage_trans, stage_rot, stage_thermal, time_frac| {
                            let gravity_accel = eval_body_gravity(
                                controls,
                                body_idx,
                                stage_trans.position,
                                stage_trans.velocity,
                                time_frac,
                            );
                            let t_inertial_body = stage_rot.map_or(DMat3::IDENTITY, |r| {
                                r.quaternion.left_quat_to_transformation()
                            });
                            let t_inertial_struct = jeod_sim::compute_t_inertial_struct(
                                &t_struct_body,
                                &t_inertial_body,
                            );
                            // Per-stage flux recompute from intermediate vehicle
                            // position — matches JEOD's derivative-class
                            // `RadiationSource::calculate_flux`. Sun position is
                            // step-constant (ephemeris is scheduled-class).
                            let sun_to_vehicle = stage_trans.position - srp_inputs.sun_position;
                            let distance = sun_to_vehicle.length().max(1.0);
                            let stage_flux_inertial_hat = sun_to_vehicle / distance;
                            let stage_flux_mag = jeod_sim::solar_flux_at_distance(distance);
                            let flux_struct_hat = t_inertial_struct * stage_flux_inertial_hat;
                            let srp_result = jeod_sim::compute_flat_plate_srp_thermal(
                                &stage_thermal.plates,
                                &stage_thermal.t_pow4_cached,
                                flux_struct_hat,
                                stage_flux_mag,
                                srp_inputs.center_grav,
                                srp_inputs.illum_factor,
                            );
                            let srp_force_inertial =
                                t_inertial_struct.transpose() * srp_result.force;
                            // Stage 4 (time_frac == 1.0) is the representative
                            // final-state SRP — cached for writeback below.
                            final_srp_inertial_force = srp_force_inertial;
                            final_srp_torque = srp_result.torque;
                            let temp_dots = match thermal_order {
                                jeod_sim::ThermalIntegrationOrder::DerivativeRk4 => {
                                    srp_result.temp_dots
                                }
                                jeod_sim::ThermalIntegrationOrder::DerivativeFirstOrder => {
                                    // Capture at stage 1 (time_frac == 0.0);
                                    // reuse at stages 2-4 so RK4 combine
                                    // collapses to Euler over k1.
                                    if time_frac == 0.0 {
                                        k1_temp_dots = Some(srp_result.temp_dots.clone());
                                        srp_result.temp_dots
                                    } else {
                                        k1_temp_dots
                                            .as_ref()
                                            .expect("stage 1 runs before stages 2-4")
                                            .clone()
                                    }
                                }
                                jeod_sim::ThermalIntegrationOrder::Scheduled => {
                                    unreachable!(
                                        "Scheduled thermal bodies do not enter the coupled path"
                                    )
                                }
                            };
                            // `srp_result.torque` is structural-frame per
                            // `FlatPlateSrpResult` docs; `constant_torque`
                            // is body-frame (from `collect_and_resolve_forces`).
                            // Rotate to body frame before summing so the
                            // coupled integrator's rotational dynamics
                            // are correct when t_struct_body != IDENTITY.
                            let srp_torque_body = t_struct_body * srp_result.torque;
                            CoupledStageEval {
                                gravity_accel,
                                non_grav_force: non_grav_non_srp_force + srp_force_inertial,
                                torque: constant_torque + srp_torque_body,
                                temp_dots,
                            }
                        },
                        body.flat_plate_state
                            .as_mut()
                            .expect("srp_stage_inputs implies flat_plate_state"),
                        dt,
                        time_scale_factor,
                    );

                    body.radiation_force = Some(RadiationForce {
                        force: final_srp_inertial_force,
                        torque: final_srp_torque,
                    });
                    // Backfill `TotalForce` and `FrameDerivatives` with the
                    // final-stage SRP contribution so downstream observers
                    // reading these see SRP-inclusive values — matching the
                    // Scheduled-mode invariant that `total_force` reflects
                    // every applied force and `frame_derivs` the resulting
                    // accelerations. In derivative modes this is a
                    // "representative stage" (stage 4) snapshot, same as
                    // `radiation_force` above.
                    body.total_force.force += final_srp_inertial_force;
                    let final_srp_torque_body = t_struct_body * final_srp_torque;
                    body.total_force.torque += final_srp_torque_body;
                    if let Some(mass) = body.mass {
                        body.frame_derivs.trans_accel +=
                            final_srp_inertial_force * mass.inverse_mass;
                        body.frame_derivs.rot_accel += mass.inverse_inertia * final_srp_torque_body;
                    }
                } else {
                    let controls = &body.gravity_controls;
                    integrate_body(
                        &body.config,
                        &mut body.trans,
                        body.rot.as_mut(),
                        body.mass.as_ref(),
                        |pos, vel, time_frac| {
                            eval_body_gravity(controls, body_idx, pos, vel, time_frac)
                        },
                        body.total_force.force,
                        body.total_force.torque,
                        dt,
                        time_scale_factor,
                        body.integrator,
                        body.gj_state.as_mut(),
                        body.abm4_state.as_mut(),
                    );
                }
            }
        } else {
            // ── Contact-coupled path: multi-body RK4 where contact forces
            //    are recomputed at each stage from all bodies' intermediate
            //    states (matching JEOD's `check_contact()` derivative job).
            // JEOD_INV: IN.31 — contact evaluated at every derivative evaluation
            //
            // Enforce preconditions: all bodies participating in contact
            // pairs must use RK4 + 6-DOF. We integrate ALL bodies through
            // the coupled path; bodies without contact pairs just pass
            // their constant forces through the same RK4 kernel.
            assert!(
                self.bodies
                    .iter()
                    .all(|b| matches!(b.integrator, jeod_dynamics::IntegratorType::Rk4)),
                "contact pairs require RK4 integrator on all bodies"
            );
            assert!(
                self.bodies
                    .iter()
                    .all(|b| b.rot.is_some() && b.mass.is_some()),
                "contact pairs require 6-DOF (rotational state + mass) on all bodies"
            );
            // Derivative-class thermal (DerivativeFirstOrder /
            // DerivativeRk4) is not extended to the contact-coupled kernel.
            // JEOD's `DynamicsIntegrationGroup` handles this case natively,
            // but our `integrate_bodies_contact_coupled` has no per-stage
            // SRP/thermal hook yet; opt such bodies into
            // `ThermalIntegrationOrder::Scheduled` or disable contact pairs.
            assert!(
                self.bodies.iter().all(|b| b
                    .flat_plate_state
                    .as_ref()
                    .is_none_or(|fps| fps.stage_inputs.is_none())),
                "Derivative-class thermal integration is not yet supported with contact pairs; \
                 use ThermalIntegrationOrder::Scheduled on flat-plate SRP bodies \
                 when contact pairs are active",
            );
            // Contact pair states must share the root inertial frame, since
            // the coupled contact evaluator uses each body's stage state
            // directly without any per-step frame transform. `validate()`
            // catches this at config time; the assert is defense-in-depth
            // for callers that skip validation.
            assert!(
                self.contact_pairs.iter().all(|p| {
                    let fa = self.bodies[p.body_a].integ_frame_id;
                    let fb = self.bodies[p.body_b].integ_frame_id;
                    fa == fb && fa == self.root_frame_id
                }),
                "contact pair bodies must share the root inertial integration frame"
            );

            // `integ_dt` (dynamic timestep) is defined above the gravity
            // closure; reuse it here for the coupled integrator call.

            // Split disjoint `self` fields up front so we can keep mutable
            // access to bodies (and to coupled_integ_scratch below) while
            // borrowing contact pairs immutably — no per-step clone of the
            // facet/material data.
            let contact_pairs: &Vec<ContactPairConfig> = &self.contact_pairs;
            let bodies_mut = &mut self.bodies;

            // Gather per-body immutable data (t_struct_body, mass, constant
            // forces/torques, gravity_controls) BEFORE the per-body
            // `iter_mut()` projection below builds the `CoupledBodyInput`
            // vector. Once that vector is live, holding any shared borrow
            // into `bodies_mut` (even for a disjoint field on a different
            // body) would conflict with the &mut field projections, so
            // snapshot everything up front. Cloning `gravity_controls`
            // happens here (and only here, since contact_pairs is
            // non-empty) — the non-contact path above borrows directly.
            let t_struct_body_vec: Vec<DMat3> =
                bodies_mut.iter().map(|b| b.t_struct_body).collect();
            let mass_vec: Vec<MassProperties> = bodies_mut
                .iter()
                .map(|b| b.mass.expect("validated"))
                .collect();
            let non_grav_non_contact_vec: Vec<DVec3> =
                bodies_mut.iter().map(|b| b.total_force.force).collect();
            let non_contact_torque_vec: Vec<DVec3> =
                bodies_mut.iter().map(|b| b.total_force.torque).collect();
            let per_body_gravity_controls: Vec<GravityControls<usize>> = bodies_mut
                .iter()
                .map(|b| b.gravity_controls.clone())
                .collect();

            // Build coupled inputs. `CoupledBodyInput` needs separate &mut
            // borrows for `trans` and `rot.as_mut()` on each `SimBody`. The
            // borrow checker accepts this when both come out of a single
            // `iter_mut()` chain — each closure invocation receives a
            // distinct `&mut SimBody` (the iterator hands them out one at
            // a time), and disjoint-field projection within that body
            // produces the two split mutable borrows.
            //
            // Earlier revisions used raw pointers in an `unsafe` block to
            // simulate the same effect, but the immutable-snapshot
            // pattern above (mass_vec, non_grav_*_vec, …) means no
            // shared borrow into `bodies_mut` is held while the &muts
            // are live, so the safe projection compiles cleanly.
            //
            // The `expect`s match the validated invariants — the
            // contact-coupled path requires 6-DOF + 3-component mass on
            // every body (enforced by `Self::validate`).
            let mut inputs: Vec<CoupledBodyInput<'_>> = bodies_mut
                .iter_mut()
                .enumerate()
                .map(|(i, body)| CoupledBodyInput {
                    trans: &mut body.trans,
                    rot: body
                        .rot
                        .as_mut()
                        .expect("validated: 6-DOF required for contact-coupled path"),
                    mass: &mass_vec[i],
                    non_grav_non_contact_force: non_grav_non_contact_vec[i],
                    non_contact_torque_body: non_contact_torque_vec[i],
                })
                .collect();

            integrate_bodies_contact_coupled(
                &mut inputs,
                &mut self.coupled_integ_scratch,
                |body_idx: usize, pos: DVec3, vel: DVec3, time_frac: f64| {
                    eval_body_gravity(
                        &per_body_gravity_controls[body_idx],
                        body_idx,
                        pos,
                        vel,
                        time_frac,
                    )
                },
                |stage_trans: &[TranslationalState],
                 stage_rot: &[RotationalState],
                 out: &mut [(DVec3, DVec3)]| {
                    // Evaluate every registered contact pair at the stage
                    // states and accumulate force/torque on each body. The
                    // integrator (`eval_stage` in jeod_sim::integration)
                    // zeroes `out` before calling us, so this closure just
                    // accumulates.
                    for pair in contact_pairs {
                        if let Some(eval) = evaluate_contact_pair(
                            &pair.facet_a,
                            &pair.facet_b,
                            &stage_trans[pair.body_a],
                            &stage_trans[pair.body_b],
                            Some(&stage_rot[pair.body_a]),
                            Some(&stage_rot[pair.body_b]),
                            t_struct_body_vec[pair.body_a],
                            t_struct_body_vec[pair.body_b],
                            Some(&mass_vec[pair.body_a]),
                            Some(&mass_vec[pair.body_b]),
                        ) {
                            out[pair.body_a].0 += eval.force_on_a;
                            out[pair.body_b].0 -= eval.force_on_a;
                            out[pair.body_a].1 += eval.torque_a_body;
                            out[pair.body_b].1 += eval.torque_b_body;
                        }
                    }
                },
                integ_dt,
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
                let num_sources = self.source_frame_ids.len();
                let target_fid = self
                    .source_frame_ids
                    .get(sw.target_source)
                    .ok_or(StepError::FrameSwitchTargetMissing {
                        body_idx,
                        target_source: sw.target_source,
                        num_sources,
                    })?
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

        // Advance any free-flying detached subtrees ballistically. This
        // matches JEOD's behavior for tree roots whose grav_interaction
        // is empty (the common case for staging — no force applied to
        // separated stages between detach and reattach). Use the dynamic
        // timestep `dt * time_scale_factor` so the ballistic propagation
        // stays consistent with the integrated bodies under time reversal
        // / scaling (matches `integ_dt` used elsewhere in this step).
        if !self.detached_subtrees.is_empty() {
            self.step_detached_subtrees(dt * self.time.time_scale_factor);
        }

        Ok(())
    }

    /// Advance the simulation by `n` timesteps.
    pub fn step_n(&mut self, n: usize) -> Result<(), StepError> {
        for _ in 0..n {
            self.step()?;
        }
        Ok(())
    }

    /// Advance the simulation until `target_time` (in simulation seconds).
    ///
    /// Steps at `self.dt` until the remaining time is less than `dt`,
    /// then takes a final fractional step if the remainder exceeds 1 ms.
    pub fn step_until(&mut self, target_time: f64) -> Result<(), StepError> {
        while self.time.simtime + self.dt <= target_time + 0.001 {
            self.step()?;
        }
        let remainder = target_time - self.time.simtime;
        if remainder > 0.001 {
            // Fractional steps corrupt multi-step history arrays (GJ's
            // Störmer-Cowell coefficients and delinv accumulators, ABM4's
            // Adams history) — both methods assume constant dt.
            let has_multistep = self.bodies.iter().any(|b| {
                matches!(
                    b.integrator,
                    jeod_dynamics::IntegratorType::GaussJackson(..)
                        | jeod_dynamics::IntegratorType::Abm4
                )
            });
            assert!(
                !has_multistep,
                "step_until() would take a fractional step ({remainder:.6}s vs dt={:.6}s). \
                 Multi-step integrators (GaussJackson, ABM4) require constant dt. \
                 Ensure target_time is an integer multiple of dt.",
                self.dt
            );
            self.step_internal(remainder)?;
        }
        Ok(())
    }
}
