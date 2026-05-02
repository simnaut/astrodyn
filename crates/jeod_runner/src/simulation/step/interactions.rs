//! Stage 6 of [`super::super::Simulation::step_internal`]: drag, SRP,
//! and gravity-gradient-torque interactions. Reads atmospheric state
//! (stage 5), gravity acceleration (stage 4), and source positions
//! (stages 2-2b); writes per-body force/torque outputs and returns
//! `sun_pos` / `moon_pos` so stages 7 and 9 don't have to re-resolve
//! them.

use glam::{DMat3, DVec3};

use jeod_sim::{IntegOrigin, Position, RadiationForce, RootInertial};

use super::super::Simulation;

impl Simulation {
    /// Stage 6 — drag, SRP (flat-plate or cannonball), gravity gradient
    /// torque. Returns the resolved Sun and Moon inertial positions in
    /// root-inertial coordinates (computed once at the top so subsequent
    /// stages can reuse them). `sun_pos` / `moon_pos` are typed
    /// `Position<RootInertial>` so SRP / solar-beta / earth-lighting
    /// callers cannot silently mix them with integration-frame body
    /// state — RF.10.
    pub(super) fn compute_interactions(
        &mut self,
        dt: f64,
        body_integ_origins: &[IntegOrigin],
    ) -> (
        Option<Position<RootInertial>>,
        Option<Position<RootInertial>>,
    ) {
        // sun_pos is also used in stage 9 (solar beta, earth lighting); compute once here.
        let sun_pos = self
            .sun_source
            .map(|idx| Position::<RootInertial>::from_raw_si(self.source_position(idx)));
        let moon_pos = self
            .moon_source
            .map(|idx| Position::<RootInertial>::from_raw_si(self.source_position(idx)));
        let source_frame_ids = &self.source_frame_ids;
        let frame_tree = &self.frame_tree;
        let root_fid = self.root_frame_id;

        for (body_idx, body) in self.bodies.iter_mut().enumerate() {
            // JEOD_INV: RF.10 — SRP / shadow consume root-inertial position
            // (because `sun_position` is root-inertial). Drag, by contrast,
            // is *not* in the shift class: `compute_drag` subtracts
            // `atmos.wind` (computed via `omega × atmosphere_position` in
            // the atmosphere planet's frame) from the vehicle velocity,
            // so the velocity must match the same planet-centered frame —
            // which is the body's integration frame in realistic configs.
            //
            // The body position is held typed (`Position<RootInertial>`) so
            // every subtraction `body_pos - sun_position` typechecks only
            // when both sides agree on `RootInertial`. Mixing
            // `body.trans.position` (`Position<IntegrationFrame>`) with
            // `sun_position` (`Position<RootInertial>`) is a compile error.
            let body_pos_typed = body
                .trans
                .to_inertial(&body_integ_origins[body_idx])
                .position;
            let inertial_pos = body_pos_typed.raw_si();

            // Compute structural transform once (shared by drag and flat-plate SRP)
            let t_inertial_body = body.rot.as_ref().map_or(DMat3::IDENTITY, |r| {
                r.quaternion.left_quat_to_transformation()
            });
            let t_inertial_struct =
                jeod_sim::compute_t_inertial_struct(&body.t_struct_body, &t_inertial_body);

            // Aerodynamic drag — uses planet-centered velocity (NOT a
            // shift site; see comment above).
            body.aero_force = None;
            if let (Some(ref drag_config), Some(ref atmos)) = (&body.drag, &body.atmospheric_state)
            {
                body.aero_force = Some(jeod_sim::compute_drag(
                    drag_config,
                    atmos,
                    body.trans.velocity.raw_si(),
                    body.rot.as_ref(),
                    body.t_struct_body,
                ));
            }

            // Solar radiation pressure (flat-plate)
            body.radiation_force = None;
            if let Some(ref mut fps) = body.flat_plate_state {
                fps.stage_inputs = None;
            }
            if let Some(sun_position_typed) = sun_pos {
                // Local raw alias for the legacy DVec3-arg consumers
                // (`compute_shadow_fraction`, `compute_cannonball_srp`,
                // `compute_flat_plate_srp_thermal`). The typed value
                // remains in scope for the structural guards below.
                let sun_position = sun_position_typed.raw_si();
                if let Some(ref mut fps) = body.flat_plate_state {
                    // Flat-plate SRP with thermal emission. Typed
                    // subtraction enforces matching frames (RF.10):
                    // `body_pos_typed - sun_position_typed` only typechecks
                    // when both are `Position<RootInertial>`, structurally
                    // catching the integration-frame mixing bug.
                    let sun_to_vehicle: Position<RootInertial> =
                        body_pos_typed - sun_position_typed;
                    let distance = sun_to_vehicle.raw_si().length();
                    let sun_to_vehicle = sun_to_vehicle.raw_si();
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
                                    inertial_pos,
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
                                    // Typed `sun_position` carries the
                                    // RootInertial frame phantom into the
                                    // RK4 derivative closure, where it
                                    // can only be subtracted from a
                                    // typed `Position<RootInertial>` —
                                    // structurally refusing the bug
                                    // shape `stage_trans.position
                                    // - sun_position` as DVec3-DVec3.
                                    sun_position: sun_position_typed,
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
                                inertial_pos,
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
                        inertial_pos,
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

        (sun_pos, moon_pos)
    }
}
