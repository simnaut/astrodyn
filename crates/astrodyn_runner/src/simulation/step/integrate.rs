//! Stages 7 + 8 + 8b of [`super::super::Simulation::step_internal`]:
//! force collection, RK4 integration (with contact-pair coupling), and
//! frame-switch handling. The bulk of `step_internal` lives here. The
//! contact-pair branch interleaves stage-7 force collection with the
//! per-RK4-stage derivative recomputation, so the three stages stay
//! together rather than getting carved further.

use glam::{DMat3, DVec3};

use astrodyn::forces::collect_and_resolve_forces;
use astrodyn::frame_orchestration::{evaluate_and_apply_frame_switch, FrameSwitchTargetMissing};
use astrodyn::gravity::accumulate_gravity;
use astrodyn::integration::{
    integrate_bodies_contact_coupled_typed, integrate_body_coupled_typed, integrate_body_typed,
    CoupledBodyInputTyped,
};
use astrodyn::{
    aggregate_wrenches_via_storage, evaluate_contact_pair, evaluate_ground_contact_pair,
    Acceleration, BodyFrame, CoupledStageEval, EdgeGeometry, Force, GravityControls,
    MassProperties, MassStorage, Position, RadiationForce, RotationalState, SelfRef, Torque,
    TranslationalState, Velocity,
};
use astrodyn_dynamics::wrench::Wrench;
use astrodyn_dynamics::MassBodyId;
use astrodyn_interactions::Phase;
use astrodyn_quantities::frame::IntegrationFrame;
use astrodyn_quantities::IntegOrigin;

use std::collections::HashMap;

use super::super::types::{ContactPairConfig, GroundContactPairConfig};
use super::super::Simulation;
use crate::error::StepError;

impl Simulation {
    /// Stages 7 + 8 + 8b — force collection, RK4 integration, and
    /// frame-switch handling.
    pub(super) fn run_integration(
        &mut self,
        dt: f64,
        body_integ_origins: &[IntegOrigin],
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
                // allowed: typed→raw boundary — `collect_and_resolve_forces`
                // is the untyped force-collection kernel; the runner stores
                // gravity acceleration typed against `RootInertial`.
                body.gravity_accel.grav_accel.raw_si(),
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

            // Struct-frame external force/torque: rotate to inertial /
            // body using the body's current attitude. Mirrors JEOD's
            // `dyn_body_collect.cc:219-221` where `extern_forc_inrtl =
            // T_inertial_struct^T · extern_forc_struct`. JEOD recomputes
            // this at every derivative call (each RK4 stage); we evaluate
            // once at force collection — the per-stage drift is
            // `O(ω · dt)` per step on the inertial-frame magnitude, which
            // is negligible for the Tier 3 step sizes
            // `SIM_verif_attach_detach` exercises.
            // JEOD_INV: DB.28 — forces collected in structural frame, rotated to inertial at root
            if body.external_force_struct != DVec3::ZERO
                || body.external_torque_struct != DVec3::ZERO
            {
                let t_inertial_body = body.rot.as_ref().map_or(DMat3::IDENTITY, |r| {
                    r.quaternion.left_quat_to_transformation()
                });
                let t_inertial_struct =
                    astrodyn::compute_t_inertial_struct(&body.t_struct_body, &t_inertial_body);
                let force_inertial = t_inertial_struct.transpose() * body.external_force_struct;
                let torque_body = body.t_struct_body * body.external_torque_struct;
                body.total_force.force += force_inertial;
                body.total_force.torque += torque_body;
                if let Some(mass) = &body.mass {
                    if force_inertial != DVec3::ZERO {
                        body.frame_derivs.trans_accel += force_inertial * mass.inverse_mass;
                    }
                    if torque_body != DVec3::ZERO {
                        body.frame_derivs.rot_accel += mass.inverse_inertia * torque_body;
                    }
                }
            }
        }

        // ── 7b. Wrench aggregation ──
        // When the mass tree contains any non-root nodes, walk every
        // chain leaves → root and sum each child's `(force, torque)`
        // into the chain's integration root. After this pass, only the
        // root node's `total_force` / `frame_derivs` carries the
        // composite-rigid-body wrench; non-root nodes are zeroed so
        // their (skipped) integration step does not double-count their
        // own contribution. Mirrors JEOD's
        // `DynBody::collect_forces_and_torques` walk in
        // `models/dynamics/dyn_body/src/dyn_body_collect.cc:128-202`.
        // JEOD_INV: DB.16 — child forces propagated to parent recursively
        // JEOD_INV: DB.17 — only the root's TotalForce/FrameDerivatives carry the whole-composite total
        self.aggregate_chain_wrenches();

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
            Vec<astrodyn_gravity::relativistic::RelativisticSource>,
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
                                Some(astrodyn_gravity::relativistic::RelativisticSource {
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
            let integ_origin = body_integ_origins[body_idx].position.raw_si();
            let integ_vel = body_integ_origins[body_idx].velocity.raw_si();
            let stage_dt = time_frac * integ_dt;
            let origin = integ_origin + integ_vel * stage_dt;
            let sub_dt = if integ_vel != DVec3::ZERO {
                stage_dt
            } else {
                0.0
            };
            // `pos` arrives in integration-frame coordinates from the
            // integrator's stage state; shift to root inertial here per
            // RF.10 before passing to gravity. Equivalent to today's
            // `pos + origin` arithmetic, just routed through `IntegOrigin`.
            let mut accel =
                accumulate_gravity(pos + origin, controls, origin, |source_id: usize| {
                    let grav = gravity_data.get(source_id)?;
                    let sfids = &source_frame_ids[source_id];
                    let position = base_positions[source_id] + base_velocities[source_id] * sub_dt;
                    let rotation = sfids
                        .pfix
                        .map(|pfix_id| &frame_tree.get(pfix_id).state.rot.t_parent_this);
                    Some(astrodyn::ResolvedSource {
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
                accel += astrodyn_gravity::relativistic::compute_relativistic_correction(
                    mu, src_pos, pos_eci, vel_eci, src_vel, other,
                );
            }
            accel
        };

        if self.contact_pairs.is_empty() && self.ground_contact_pairs.is_empty() {
            // ── Standard path: per-body RK4 / GJ integration ──
            // No clone of gravity_controls: the outer iter_mut gives us
            // a &mut SimBody, and Rust's disjoint-field split borrow lets
            // the closure capture &body.gravity_controls while other
            // fields of `body` are borrowed mutably for the integrator.
            let time_scale_factor = self.time.time_scale_factor;
            for (body_idx, body) in self.bodies.iter_mut().enumerate() {
                // JEOD_INV: DB.17 — only the root's state is integrated;
                // kinematic children's `trans`/`rot` were already
                // overwritten by `propagate_kinematic_state` earlier in
                // this step from the parent's state composed with the
                // link geometry. Skip integration here so we don't
                // stomp the kinematic value with a force-driven update.
                //
                // Integrator-coupled sub-states that are *not* part of
                // the orbital integration (flat-plate plate
                // temperatures via the derivative-class arm) are
                // routed through the Scheduled path inside
                // `compute_interactions` for kinematic_only bodies,
                // so a kinematic child's thermal state still advances
                // and `radiation_force` is still published. Mirrors
                // the analogous Bevy fix for `flat_plate_srp_system`
                // (PR #287). The skip below only gates orbital
                // trans/rot integration.
                if body.kinematic_only {
                    continue;
                }
                // JEOD_INV: DB.21 — frame-attached bodies do not
                // integrate. JEOD's `DynBody::integrate` at
                // `models/dynamics/dyn_body/src/dyn_body_integration.cc:309-333`
                // skips both the translational and rotational
                // integrators when `frame_attach.isAttached()` and
                // instead resets the body's state to the parent
                // frame's state + offset. The pre- and
                // post-integration `propagate_frame_attached_state`
                // sweeps in `step::mod` do the equivalent state reset
                // for us; here we simply skip the integrator so it
                // doesn't stomp the kinematic value with a
                // force-driven update.
                if body.frame_attach.is_some() {
                    continue;
                }
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
                        matches!(body.integrator, astrodyn_dynamics::IntegratorType::Rk4),
                        "ThermalIntegrationOrder::{thermal_order:?} requires \
                         astrodyn_dynamics::IntegratorType::Rk4 for body {body_idx}; \
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

                    // body.trans flows end-to-end as typed through the
                    // coupled integrator boundary. The kernel's `stage_fn`
                    // closure still receives untyped `&TranslationalState`
                    // because RK4 stage state is integrator-internal.
                    integrate_body_coupled_typed::<SelfRef, IntegrationFrame>(
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
                            let t_inertial_struct = astrodyn::compute_t_inertial_struct(
                                &t_struct_body,
                                &t_inertial_body,
                            );
                            // Per-stage flux recompute from intermediate vehicle
                            // position — matches JEOD's derivative-class
                            // `RadiationSource::calculate_flux`. Sun position is
                            // step-constant (ephemeris is scheduled-class).
                            //
                            // JEOD_INV: RF.10 — `stage_trans.position` is the
                            // integrator's intermediate `DVec3` in the body's
                            // integration frame; `srp_inputs.sun_position` is
                            // typed `Position<RootInertial>`. The structural
                            // guard refuses `stage_trans.position
                            // - srp_inputs.sun_position` (DVec3 -
                            // Position<RootInertial>) at compile time — the
                            // bug shape that slipped past review on PR #258.
                            // Lift the integration-frame `DVec3` to typed,
                            // then call `IntegOrigin::shift_position_at_stage`
                            // for the stage-time interpolation matching
                            // `eval_body_gravity`.
                            let p_integ =
                                Position::<IntegrationFrame>::from_raw_si(stage_trans.position); // allowed: legitimate kernel-internal SRP stage interpolation boundary
                            let stage_pos_root: Position<astrodyn::RootInertial> =
                                body_integ_origins[body_idx]
                                    .shift_position_at_stage(p_integ, time_frac * integ_dt);
                            let sun_to_vehicle: Position<astrodyn::RootInertial> =
                                stage_pos_root - srp_inputs.sun_position;
                            let sun_to_vehicle = sun_to_vehicle.raw_si();
                            let distance = sun_to_vehicle.length().max(1.0);
                            let stage_flux_inertial_hat = sun_to_vehicle / distance;
                            let stage_flux_mag = astrodyn::solar_flux_at_distance(distance);
                            let flux_struct_hat = t_inertial_struct * stage_flux_inertial_hat;
                            let srp_result = astrodyn::compute_flat_plate_srp_thermal(
                                &stage_thermal.plates,
                                &stage_thermal.t_pow4_cached,
                                flux_struct_hat,
                                stage_flux_mag,
                                // The kernel works in raw `DVec3` for
                                // structural-frame arithmetic; the
                                // typed field on `FlatPlateStageInputs`
                                // is the storage-time guard.
                                srp_inputs.center_grav.raw_si(),
                                srp_inputs.illum_factor,
                            );
                            let srp_force_inertial =
                                t_inertial_struct.transpose() * srp_result.force;
                            // Stage 4 (time_frac == 1.0) is the representative
                            // final-state SRP — cached for writeback below.
                            final_srp_inertial_force = srp_force_inertial;
                            final_srp_torque = srp_result.torque;
                            let temp_dots = match thermal_order {
                                astrodyn::ThermalIntegrationOrder::DerivativeRk4 => {
                                    srp_result.temp_dots
                                }
                                astrodyn::ThermalIntegrationOrder::DerivativeFirstOrder => {
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
                                astrodyn::ThermalIntegrationOrder::Scheduled => {
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
                    // Typed integrator boundary: `body.trans` flows
                    // end-to-end as `TranslationalStateTyped<IntegrationFrame>`.
                    // Force/torque/dt are lifted once at the call site;
                    // the per-stage gravity-fn raw lift is a sanctioned
                    // kernel-internal boundary inside `integrate_body_typed`.
                    integrate_body_typed::<SelfRef, IntegrationFrame>(
                        &body.config,
                        &mut body.trans,
                        body.rot.as_mut(),
                        body.mass.as_ref(),
                        |pos, vel, time_frac| {
                            // allowed: typed-sibling call boundary — wraps gravity_fn closure output for the typed gravity contract
                            Acceleration::<IntegrationFrame>::from_raw_si(eval_body_gravity(
                                controls,
                                body_idx,
                                pos.raw_si(),
                                vel.raw_si(),
                                time_frac,
                            ))
                        },
                        Force::<IntegrationFrame>::from_raw_si(body.total_force.force), // allowed: typed-sibling call boundary — step-constant force lift
                        Torque::<BodyFrame<SelfRef>>::from_raw_si(body.total_force.torque), // allowed: typed-sibling call boundary — step-constant torque lift
                        uom::si::f64::Time::new::<uom::si::time::second>(dt),
                        time_scale_factor,
                        // Runner stores the raw astrodyn_dynamics enum; the
                        // astrodyn kernel takes the wrapped flavor —
                        // convert at the boundary (cheap copy).
                        body.integrator.into(),
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
                    .all(|b| matches!(b.integrator, astrodyn_dynamics::IntegratorType::Rk4)),
                "contact-coupled path (inter-body or ground-contact pairs) requires \
                 RK4 integrator on all bodies"
            );
            assert!(
                self.bodies.iter().all(|b| !b.kinematic_only),
                "contact-coupled path is incompatible with kinematic-only bodies: \
                 contact forces require an integrated state for every participant. \
                 Either clear `kinematic_only` on every contact-coupled body, or \
                 detach the kinematic child before registering contact pairs."
            );
            // Frame-attached bodies have no integrator to feed contact
            // forces back through; they are kinematically driven by
            // the parent frame. Surface a fail-loud diagnostic
            // matching the kinematic_only assertion above.
            assert!(
                self.bodies.iter().all(|b| b.frame_attach.is_none()),
                "contact-coupled path is incompatible with frame-attached bodies: \
                 contact forces require an integrated state for every participant, \
                 but the contact-coupled kernel integrates ALL bodies in the sim — \
                 so any frame-attached body present is a misconfiguration. Call \
                 `Simulation::detach_from_frame(body_idx)` on every frame-attached \
                 body before stepping (or unregister the contact pairs)."
            );
            assert!(
                self.bodies
                    .iter()
                    .all(|b| b.rot.is_some() && b.mass.is_some()),
                "contact-coupled path (inter-body or ground-contact pairs) requires \
                 6-DOF (rotational state + mass) on all bodies"
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
                "Derivative-class thermal integration is not yet supported when \
                 inter-body or ground-contact pairs are registered; use \
                 ThermalIntegrationOrder::Scheduled on flat-plate SRP bodies \
                 when contact pairs are active",
            );
            // Contact pair states must share the root inertial frame, since
            // the coupled contact evaluator uses each body's stage state
            // directly without any per-step frame transform. `validate()`
            // catches this at config time (both for inter-body
            // `contact_pairs` and for `ground_contact_pairs` —
            // `ValidationError::ContactPairNonRootFrame` /
            // `GroundContactPairNonRootFrame`); the asserts here are
            // defense-in-depth for callers that skip validation.
            assert!(
                self.contact_pairs.iter().all(|p| {
                    let fa = self.bodies[p.body_a].integ_frame_id;
                    let fb = self.bodies[p.body_b].integ_frame_id;
                    fa == fb && fa == self.root_frame_id
                }),
                "inter-body contact pair bodies must share the root inertial integration frame"
            );
            assert!(
                self.ground_contact_pairs
                    .iter()
                    .all(|p| self.bodies[p.body_a].integ_frame_id == self.root_frame_id),
                "ground-contact pair bodies must integrate in the root inertial frame"
            );

            // `integ_dt` (dynamic timestep) is defined above the gravity
            // closure; reuse it here for the coupled integrator call.

            // Snapshot the ground-contact planet's pfix rotation BEFORE
            // taking the mutable borrow of bodies. For SphericalTerrain
            // the pfix rotation cancels in the ground-point computation
            // and we may pass identity, but other Terrain implementations
            // would need this matrix.
            //
            // Defense-in-depth: ground contact's terrain query assumes
            // the planet center is at the inertial origin
            // (`compute_ground_contact_geometry` projects
            // `vehicle_pos_inertial` directly into pfix without any
            // planet-translation subtraction). `validate()` catches
            // non-central planets via
            // `ValidationError::GroundContactNonCentralPlanet`, but
            // assert here too in case a caller skips validation.
            let ground_t_inertial_pfix: DMat3 =
                if let Some(planet_idx) = self.ground_contact_planet_source {
                    let sfids = &self.source_frame_ids[planet_idx];
                    assert_eq!(
                        sfids.inertial, self.root_frame_id,
                        "ground contact requires the planet source's inertial frame to be \
                         the root frame (`compute_ground_contact_geometry` projects \
                         vehicle inertial position into pfix as if the planet center \
                         were at the inertial origin); planet_source={planet_idx} has \
                         inertial frame {} but root is {}. Use a central planet for \
                         ground contact, or call `Simulation::validate()` to surface \
                         this as a configuration error before stepping.",
                        sfids.inertial, self.root_frame_id
                    );
                    if let Some(pfix_id) = sfids.pfix {
                        self.frame_tree.get(pfix_id).state.rot.t_parent_this
                    } else {
                        DMat3::IDENTITY
                    }
                } else {
                    DMat3::IDENTITY
                };

            // Split disjoint `self` fields up front so we can keep mutable
            // access to bodies (and to coupled_integ_scratch below) while
            // borrowing contact pairs immutably — no per-step clone of the
            // facet/material data.
            let contact_pairs: &Vec<ContactPairConfig> = &self.contact_pairs;
            let ground_contact_pairs: &Vec<GroundContactPairConfig> = &self.ground_contact_pairs;
            // Stage-1 gate for JEOD's pre-propagation init impulse:
            // mirrors `ContactSurface::collect_forces_torques` zeroing
            // `facet.force` after stage 1's collection. The closure
            // applies impulses on the first call and the gate flips to
            // `true`, so stages 2-4 see no init force. Cell<bool> instead
            // of Vec<Cell<Option<…>>> to keep the per-step path
            // allocation-free (the impulses themselves are stored on
            // each `GroundContactPairConfig` and cleared after the
            // integrator returns).
            let init_impulses_drained = std::cell::Cell::new(false);
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
            // body.trans flows end-to-end as typed through the typed
            // sibling; the kernel allocates the transient untyped
            // buffer and writes back through the same `&mut`s.
            let inputs: Vec<CoupledBodyInputTyped<'_, IntegrationFrame>> = bodies_mut
                .iter_mut()
                .enumerate()
                .map(|(i, b)| {
                    let rot = b
                        .rot
                        .as_mut()
                        .expect("validated: 6-DOF required for contact-coupled path");
                    CoupledBodyInputTyped {
                        trans: &mut b.trans,
                        rot,
                        mass: &mass_vec[i],
                        non_grav_non_contact_force: non_grav_non_contact_vec[i],
                        non_contact_torque_body: non_contact_torque_vec[i],
                    }
                })
                .collect();

            integrate_bodies_contact_coupled_typed(
                inputs,
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
                    // integrator (`eval_stage` in astrodyn::integration)
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
                    // Ground contact: drain JEOD's pre-propagation init
                    // impulses on the first stage call only (mirrors
                    // ContactSurface::collect_forces_torques zeroing
                    // facet.force after stage 1). Subsequent stages
                    // evaluate the SteadyState path which produces no
                    // contact for above-surface vehicles — matching
                    // JEOD's runtime check_contact_ground behaviour.
                    let drain_now = !init_impulses_drained.replace(true);
                    for pair in ground_contact_pairs {
                        if drain_now {
                            if let Some(impulse) = pair.pending_initial_impulse {
                                out[pair.body_a].0 += impulse.force_inertial;
                                out[pair.body_a].1 += impulse.torque_body;
                            }
                        }
                        if let Some(eval) = evaluate_ground_contact_pair(
                            &pair.vehicle_facet,
                            &pair.ground_facet,
                            &stage_trans[pair.body_a],
                            &stage_rot[pair.body_a],
                            t_struct_body_vec[pair.body_a],
                            &mass_vec[pair.body_a],
                            ground_t_inertial_pfix,
                            Phase::SteadyState,
                        ) {
                            out[pair.body_a].0 += eval.force_on_a;
                            out[pair.body_a].1 += eval.torque_a_body;
                        }
                    }
                },
                integ_dt,
            );

            // `integrate_bodies_contact_coupled_typed` consumed the
            // `&mut body.trans` borrows; the integrated state is
            // already written back through them.

            // Mark pending init impulses as consumed so subsequent steps
            // don't reapply them. (Matches JEOD: the impulsive force is
            // applied only at stage 1 of step 1 and is then zeroed by
            // collect_forces_torques.)
            for pair in self.ground_contact_pairs.iter_mut() {
                pair.pending_initial_impulse = None;
            }
        }

        // Sync body positions back to frame tree after integration.
        for body in &self.bodies {
            let node = self.frame_tree.get_mut(body.body_frame_id);
            node.state.trans.position = body.trans.position.raw_si();
            node.state.trans.velocity = body.trans.velocity.raw_si();
        }

        // ── 8b. Frame switch (body actions) ──
        // Applied AFTER integration, matching JEOD's pipeline where
        // DynBodyFrameSwitch is a body action evaluated post-integration.
        // The body has already been integrated in its current frame for this
        // step; the switch transforms to the new frame for the NEXT step.
        // The lifted helper in `astrodyn::frame_orchestration` performs the
        // distance check, reparent, state copy-out, and gravity-controls
        // flip — same logic that previously lived inline here, now shared
        // with ECS adapters (issue #71). Phase C made the helper generic
        // over the source-id type via a closure-based source lookup; the
        // runner uses the default `usize` instantiation.
        let inertial_fids: Vec<astrodyn_frames::FrameId> =
            self.source_frame_ids.iter().map(|sf| sf.inertial).collect();
        let num_sources = inertial_fids.len();
        let root_frame_id = self.root_frame_id;
        for body_idx in 0..self.bodies.len() {
            let body = &mut self.bodies[body_idx];
            // The lifted `evaluate_and_apply_frame_switch` helper takes
            // an untyped `TranslationalState`. After #255, `body.trans`
            // is `TranslationalStateTyped<IntegrationFrame>` — bridge
            // by extracting raw values, running the helper, then
            // re-wrapping with the `IntegrationFrame` phantom on success.
            let mut raw_trans = TranslationalState {
                position: body.trans.position.raw_si(),
                velocity: body.trans.velocity.raw_si(),
            };
            let switched = evaluate_and_apply_frame_switch(
                &mut self.frame_tree,
                root_frame_id,
                body.body_frame_id,
                &mut body.integ_frame_id,
                &mut raw_trans,
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
            if switched {
                // allowed: frame-switch writeback — `evaluate_and_apply_frame_switch` operates on raw `TranslationalState`; lift the post-switch result back into typed storage
                body.trans.position = Position::<IntegrationFrame>::from_raw_si(raw_trans.position);
                // allowed: frame-switch writeback (velocity sibling of the line above)
                body.trans.velocity = Velocity::<IntegrationFrame>::from_raw_si(raw_trans.velocity);
            }
        }

        Ok(())
    }

    /// Walk the live mass tree from leaves to roots and accumulate each
    /// child node's `(force, torque)` into the chain's integration root.
    ///
    /// After this pass:
    ///
    /// - The integrated **root** of every chain carries the
    ///   composite-rigid-body wrench of the whole subtree on its
    ///   `total_force` / `frame_derivs` accumulators (force in inertial,
    ///   torque in body frame — same shape `collect_and_resolve_forces`
    ///   produces).
    /// - Every **non-root** chain member has its `total_force` and
    ///   `frame_derivs` zeroed so the per-body integration loop (which
    ///   already skips kinematic children) does not see stale values
    ///   for downstream accessors.
    ///
    /// No-ops when there is no mass tree, no chains (every node is a
    /// root), or no non-root nodes — the per-body force collection
    /// output is already correct for those configurations.
    ///
    /// JEOD precedent: `DynBody::collect_forces_and_torques`
    /// (`models/dynamics/dyn_body/src/dyn_body_collect.cc:128-202`).
    /// The structural-frame parallel-axis math lives in
    /// [`astrodyn::aggregate_wrenches_via_storage`]; this method is the
    /// runner-side glue that builds the per-edge geometry from the live
    /// `MassTree` and flips between inertial / structural / body frames
    /// at the entry / exit boundaries.
    fn aggregate_chain_wrenches(&mut self) {
        let Some(tree) = self.mass_tree.as_ref() else {
            return;
        };
        if tree.node_count() == 0 {
            return;
        }
        // Fast path: no chains in the tree (every node is a root) — the
        // per-body force collection is already final.
        let any_non_root = (0..tree.node_count()).any(|raw| {
            let id: MassBodyId = raw;
            tree.parent(id).is_some()
        });
        if !any_non_root {
            return;
        }

        // Map each tree node → SimBody index (only nodes registered as
        // bodies contribute / consume wrenches; tree-only nodes
        // contribute zero and absorb zero, matching JEOD's behaviour
        // for child mass-only sub-bodies).
        let mut sim_body_for_id: HashMap<MassBodyId, usize> = HashMap::new();
        for (idx, body) in self.bodies.iter().enumerate() {
            if let Some(id) = body.mass_body_id {
                sim_body_for_id.insert(id, idx);
            }
        }

        // Build per-node wrenches in the node's **own structural frame**.
        // JEOD walks in structural so the per-link `t_parent_child^T`
        // rotation correctly composes child→parent transformations and
        // the parallel-axis arm `pcm_to_ccm` (in parent struct) lands a
        // parent-struct torque from a parent-struct force.
        //
        // Conversion at this boundary is the same composition
        // `force_collection_system` already uses for the root:
        //   T_inertial_struct = T_struct_body^T · T_inertial_body
        //   force_struct  = T_inertial_struct · force_inertial
        //   torque_struct = T_struct_body^T · torque_body
        // The defaults (`T_struct_body = I`, `T_inertial_body = I` from
        // an absent `RotationalState`) collapse every transform to
        // identity — bit-exact with the previous inertial-frame walk for
        // single-body / identity-attitude chains.
        let mut wrenches: HashMap<MassBodyId, Wrench> = HashMap::new();
        for (id, &idx) in &sim_body_for_id {
            let body = &self.bodies[idx];
            let t_inertial_body = body.rot.as_ref().map_or(DMat3::IDENTITY, |r| {
                r.quaternion.left_quat_to_transformation()
            });
            let t_inertial_struct =
                astrodyn::compute_t_inertial_struct(&body.t_struct_body, &t_inertial_body);
            let force_struct = t_inertial_struct * body.total_force.force;
            let torque_struct = body.t_struct_body.transpose() * body.total_force.torque;
            wrenches.insert(*id, Wrench::new(force_struct, torque_struct));
        }

        // Build per-edge geometry from each non-root node's
        // `composite_wrt_pstr` and the parent's `composite_properties`
        // — the same arithmetic
        // [`astrodyn::edge_geometry_from_composites`] does, but read
        // directly from the live `MassTree` (which already keeps these
        // up to date through every attach / detach via
        // `recompute_composites`). JEOD `dyn_body_collect.cc:181`.
        let mut edges: HashMap<MassBodyId, EdgeGeometry> = HashMap::new();
        for raw in 0..tree.node_count() {
            let id: MassBodyId = raw;
            let Some(parent_id) = tree.parent(id) else {
                continue;
            };
            let child = tree.get(id);
            let parent = tree.get(parent_id);
            let pcm_to_ccm =
                child.composite_wrt_pstr.position - parent.composite_properties.position;
            edges.insert(
                id,
                EdgeGeometry {
                    pcm_to_ccm,
                    t_parent_child: child.composite_wrt_pstr.t_parent_this,
                },
            );
        }

        // Aggregate. Returns one entry per tree root with the chain's
        // wrench in the root's structural frame.
        let aggregated = aggregate_wrenches_via_storage(tree, &wrenches, &edges);

        // Identify roots (cheap scan; matches `tree.roots()` semantics
        // without re-borrowing the tree mutably here).
        let mut roots: HashMap<MassBodyId, ()> = HashMap::new();
        for raw in 0..tree.node_count() {
            let id: MassBodyId = raw;
            if tree.parent(id).is_none() {
                roots.insert(id, ());
            }
        }

        // Writeback. Roots: aggregated struct-frame wrench → inertial
        // force, body torque, plus refreshed `frame_derivs`. Non-roots:
        // zero, so any downstream consumer reading them sees the
        // post-aggregation invariant.
        for (id, idx) in sim_body_for_id {
            let body = &mut self.bodies[idx];
            if roots.contains_key(&id) {
                let agg = aggregated.get(&id).copied().unwrap_or_else(Wrench::zero);
                let t_inertial_body = body.rot.as_ref().map_or(DMat3::IDENTITY, |r| {
                    r.quaternion.left_quat_to_transformation()
                });
                let t_inertial_struct =
                    astrodyn::compute_t_inertial_struct(&body.t_struct_body, &t_inertial_body);
                let force_inertial = t_inertial_struct.transpose() * agg.force;
                let torque_body = body.t_struct_body * agg.torque;
                body.total_force.force = force_inertial;
                body.total_force.torque = torque_body;
                // allowed: typed→raw boundary — `frame_derivs` is the
                // untyped derivative struct used by the integrator
                // contract; the runner's gravity storage is typed
                // against `RootInertial`.
                let grav_accel_raw = body.gravity_accel.grav_accel.raw_si();
                if let Some(mass) = &body.mass {
                    body.frame_derivs.trans_accel =
                        grav_accel_raw + force_inertial * mass.inverse_mass;
                    if let Some(rot) = body.rot.as_ref() {
                        // Refresh rotational dynamics with the
                        // aggregated body-frame torque. Matches
                        // `compute_frame_derivatives` for the
                        // rotational arm only.
                        let h_body = mass.inertia * rot.ang_vel_body;
                        let euler = rot.ang_vel_body.cross(h_body);
                        body.frame_derivs.rot_accel = mass.inverse_inertia * (torque_body - euler);
                    } else {
                        body.frame_derivs.rot_accel = DVec3::ZERO;
                    }
                } else {
                    body.frame_derivs.trans_accel = grav_accel_raw;
                    body.frame_derivs.rot_accel = DVec3::ZERO;
                }
            } else {
                body.total_force.force = DVec3::ZERO;
                body.total_force.torque = DVec3::ZERO;
                body.frame_derivs.trans_accel = DVec3::ZERO;
                body.frame_derivs.rot_accel = DVec3::ZERO;
            }
        }
    }
}
