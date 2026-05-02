//! Stages 7 + 8 + 8b of [`super::super::Simulation::step_internal`]:
//! force collection, RK4 integration (with contact-pair coupling), and
//! frame-switch handling. The bulk of `step_internal` lives here. The
//! contact-pair branch interleaves stage-7 force collection with the
//! per-RK4-stage derivative recomputation, so the three stages stay
//! together rather than getting carved further.

use glam::{DMat3, DVec3};

use jeod_sim::forces::collect_and_resolve_forces;
use jeod_sim::frame_orchestration::{evaluate_and_apply_frame_switch, FrameSwitchTargetMissing};
use jeod_sim::gravity::accumulate_gravity;
use jeod_sim::integration::{integrate_bodies_contact_coupled, integrate_body, CoupledBodyInput};
use jeod_sim::{
    evaluate_contact_pair, integrate_body_coupled, CoupledStageEval, GravityControls,
    MassProperties, RadiationForce, RotationalState, TranslationalState,
};

use super::super::types::ContactPairConfig;
use super::super::Simulation;
use crate::error::StepError;

impl Simulation {
    /// Stages 7 + 8 + 8b — force collection, RK4 integration, and
    /// frame-switch handling.
    pub(super) fn run_integration(
        &mut self,
        dt: f64,
        body_integ_origins: &[(DVec3, DVec3)],
    ) -> Result<(), StepError> {
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
        // The lifted helper in `jeod_sim::frame_orchestration` performs the
        // distance check, reparent, state copy-out, and gravity-controls
        // flip — same logic that previously lived inline here, now shared
        // with ECS adapters (issue #71). Phase C made the helper generic
        // over the source-id type via a closure-based source lookup; the
        // runner uses the default `usize` instantiation.
        let inertial_fids: Vec<jeod_frames::FrameId> =
            self.source_frame_ids.iter().map(|sf| sf.inertial).collect();
        let num_sources = inertial_fids.len();
        let root_frame_id = self.root_frame_id;
        for body_idx in 0..self.bodies.len() {
            let body = &mut self.bodies[body_idx];
            evaluate_and_apply_frame_switch(
                &mut self.frame_tree,
                root_frame_id,
                body.body_frame_id,
                &mut body.integ_frame_id,
                &mut body.trans,
                &mut body.frame_switches,
                &mut body.gravity_controls,
                |idx| inertial_fids.get(*idx).copied(),
                num_sources,
                body_idx,
            )
            .map_err(
                |FrameSwitchTargetMissing {
                     body_idx,
                     target_source,
                     num_sources,
                 }| StepError::FrameSwitchTargetMissing {
                    body_idx,
                    target_source,
                    num_sources,
                },
            )?;
        }

        Ok(())
    }
}
