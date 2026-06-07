// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! Tier 3: JEOD `SIM_contact` — spring-damper contact between two bodies.

#![allow(
    clippy::float_cmp,
    reason = "Tier 3 tests assert bit-exact recovery of literal-built / analytic state values"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Tier 3 step counts and indices fit exactly in f64 mantissa and usize"
)]
//!
//! Propagates two free-floating vehicles through `Simulation::step()` with
//! contact pairs registered via `Simulation::register_contact_pair`. Contact
//! forces are evaluated at each RK4 stage inside the coupled integration
//! loop, matching JEOD's `check_contact()` derivative-class scheduling.
//! Compares positions against JEOD's reference CSVs from
//! `verif/SIM_contact/SET_test/RUN_*`.
//!
//! Matches JEOD configuration from `SIM_contact/SET_test/RUN_*/input.py`:
//! - RK4 integrator, dt = 0.01 s, sim time = 10 s
//! - Empty-space (no gravity, no atmosphere)
//! - Spring material: k = 20 lbf/in = 3502.5006 N/m, c = 0.4 lbf·s/in
//!   = 70.050012 N·s/m, mu = 0.05
//! - Point facet: sphere radius 1 m at origin in structural frame
//! - Line facet: cylinder (capsule) length 2 m along body x-axis, radius 1 m
//! - veh1 at (0,0,0), at rest; veh2 at (12,0,0) with v=(-2,0,0)
//! - Point scenario: 100 kg sphere each, inertia = diag(40)
//! - Line scenario: 200 kg cylinder each, inertia = diag(100, 116.67, 116.67)
//!
//! Issue #88 / #205 lands the ground-contact pipeline (Terrain trait,
//! GroundFacet, evaluator, runner registration, RK4 wiring) and the
//! `tier3_contact_ground` cross-validation against
//! `SIM_ground_contact/RUN_contact_ground` CSV — see that test's
//! docstring for the JEOD initialization-state semantics our port
//! mirrors via `Phase::Initialization` / `Phase::SteadyState`.
//!
//! Tests **must panic** (not skip) when reference CSVs are absent, per
//! `CLAUDE.md`. The panic message includes the exact Docker command.

use astrodyn_verif_jeod::tier3_csv::test_data_path;

use astrodyn::{
    evaluate_contact_pair, GravityControl, GravityControls, GravityGradient, GravityModel,
    GravitySource, JeodQuat, MassProperties, RotationalState, SimulationTime, TranslationalState,
};
use astrodyn::{ContactFacet, ContactMaterial};
use astrodyn::{GravitySourceEntry, VehicleConfig};
use astrodyn_runner::{GroundFacet, RotationModel, Simulation, SphericalTerrain};
use glam::{DMat3, DVec3};
use std::path::Path;
use std::sync::Arc;

// ── Shared JEOD material constants ──────────────────────────────────
//
// Match `Trick::attach_units("lbf/in", 20.0)` and `attach_units("lbf*s/in", 0.4)`
// in JEOD's `Contact_Modified_data/contact/pair_interaction.py`. Trick's
// internal SI conversion uses NIST CODATA exact values
// `1 lbf = 4.4482216152605 N` and `1 in = 0.0254 m`.
/// Spring stiffness: 20 lbf/in (NIST exact conversion, matching Trick's
/// `attach_units("lbf/in", 20.0)`). Truncated to f64 precision; the
/// trailing digits beyond ~16 sig figs in `3502.53670492952642` are
/// not representable.
const JEOD_SPRING_K: f64 = 3_502.536_704_929_526_4;
/// Damping: 0.4 lbf·s/in (NIST exact conversion, matching Trick's
/// `attach_units("lbf*s/in", 0.4)`). Truncated to f64 precision.
const JEOD_DAMPING_B: f64 = 70.050_734_098_590_54;
/// Friction coefficient
const JEOD_MU: f64 = 0.05;

/// Integration step from SIM_contact S_define: `DYNAMICS 0.01` (50 Hz).
const DT: f64 = 0.01;

/// Log cycle (for matching checkpoints): 0.05 s (from input.py LOG_CYCLE).
/// Used by `tier3_contact_point_off_center_stage4_probe` to walk the CSV
/// rows in lockstep with manual RK4 propagation; the standard tests
/// derive checkpoint times directly from CSV rows.
const LOG_CYCLE: f64 = 0.05;

/// Simulation duration (from input.py `exec_set_terminate_time(10)`).
const SIM_DURATION: f64 = 10.0;

// ── CSV loading ─────────────────────────────────────────────────────

/// One row of a SIM_contact ASCII log.
///
/// Force and torque columns are retained in the CSV loader for future
/// pipeline-integrated regression tests, even though the current
/// trajectory-only assertions don't consume them. Marked `allow(dead_code)`
/// so the unused-field lint stays clean.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ContactRecord {
    time: f64,
    veh1_pos: DVec3,
    veh1_vel: DVec3,
    veh1_force: DVec3,
    veh1_torque: DVec3,
    veh2_pos: DVec3,
    veh2_vel: DVec3,
    veh2_force: DVec3,
    veh2_torque: DVec3,
    veh1_mass: f64,
    veh2_mass: f64,
}

fn load_contact_csv(path: &Path) -> Vec<ContactRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_contact CSV from {}: {e}\n\
             Generate with:\n  \
             docker build -f trick/Dockerfile -t jeod-trick ..\n  \
             docker run --rm -v $(pwd)/crates/astrodyn_verif_jeod/test_data:/output \\\n    \
               -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro \\\n    \
               jeod-trick",
            path.display()
        )
    });
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 27,
            "line {}: expected >=27 columns for SIM_contact CSV, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(ContactRecord {
            time: p(0),
            veh1_pos: DVec3::new(p(1), p(2), p(3)),
            veh1_vel: DVec3::new(p(4), p(5), p(6)),
            veh1_force: DVec3::new(p(7), p(8), p(9)),
            veh1_torque: DVec3::new(p(10), p(11), p(12)),
            veh2_pos: DVec3::new(p(13), p(14), p(15)),
            veh2_vel: DVec3::new(p(16), p(17), p(18)),
            veh2_force: DVec3::new(p(19), p(20), p(21)),
            veh2_torque: DVec3::new(p(22), p(23), p(24)),
            veh1_mass: p(25),
            veh2_mass: p(26),
        });
    }
    records
}

// ── Simulation harness ──────────────────────────────────────────────

/// Synthetic gravity-source marker for the empty-space root: not one of
/// the six sealed planets, so (per issue #662's strict identity rule) it
/// requires a `define_planet!`-minted marker and `add_source_typed`.
mod tags {
    astrodyn::define_planet!(Space);
}

/// Add an inertial-only gravity "source" with mu=0 so the Simulation has a
/// root frame. Matches JEOD's `EphemerisMode_EmptySpace` which provides the
/// Space.inertial root frame with no gravitational body.
fn add_empty_space_root(sim: &mut Simulation) {
    sim.add_source_typed::<tags::Space>(
        "Space",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );
}

/// Build a simulation with two free-floating bodies of the given mass /
/// inertia, and the standard SIM_contact initial state:
///   veh1 at rest at origin, veh2 at (12,0,0) moving at (-2,0,0) m/s.
fn make_two_body_sim(mass: f64, inertia_diag: DVec3) -> Simulation {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    add_empty_space_root(&mut sim);

    let inertia = DMat3::from_cols(
        DVec3::new(inertia_diag.x, 0.0, 0.0),
        DVec3::new(0.0, inertia_diag.y, 0.0),
        DVec3::new(0.0, 0.0, inertia_diag.z),
    );
    let mass_props = MassProperties::with_inertia(mass, inertia, DVec3::ZERO);

    let id1 = sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        ..VehicleConfig::named("tier3-sim-contact-7")
    });

    let id2 = sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: DVec3::new(12.0, 0.0, 0.0),
            velocity: DVec3::new(-2.0, 0.0, 0.0),
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        ..VehicleConfig::named("tier3-sim-contact-6")
    });
    assert_eq!(id1, 0);
    assert_eq!(id2, 1);

    sim.validate().unwrap();
    sim
}

/// JEOD steel material from `Contact_Modified_data/contact/pair_interaction.py`.
fn jeod_steel() -> ContactMaterial {
    ContactMaterial::jeod_spring(JEOD_SPRING_K, JEOD_DAMPING_B, JEOD_MU)
}

/// Mass/inertia for the 100 kg point scenarios (`veh_mass_point.py`).
fn point_mass_props() -> MassProperties {
    MassProperties::with_inertia(
        100.0,
        DMat3::from_cols(
            DVec3::new(40.0, 0.0, 0.0),
            DVec3::new(0.0, 40.0, 0.0),
            DVec3::new(0.0, 0.0, 40.0),
        ),
        DVec3::ZERO,
    )
}

/// Mass/inertia for the 200 kg line/capsule scenarios (`veh_mass_line.py`).
fn line_mass_props() -> MassProperties {
    MassProperties::with_inertia(
        200.0,
        DMat3::from_cols(
            DVec3::new(100.0, 0.0, 0.0),
            DVec3::new(0.0, 116.6667, 0.0),
            DVec3::new(0.0, 0.0, 116.6667),
        ),
        DVec3::ZERO,
    )
}

// Tolerances for contact-force/torque comparisons. Sampled at stage 4
// of the RK4 step that produced the logged state — matching JEOD's
// derivative-class `collect_forces_torques` logging
// (`Contact_S_modules/sv_dyn.sm:134`), where `contact_force` at log
// time t reflects stage 4 of the last integration step. With the
// sampling state matched, observed errors collapse to the FP noise
// floor for both head-on and off-center scenarios, so all five
// inter-body contact tests share a single tolerance pair.
//
// All scenarios — head-on and off-center oblique — match JEOD to the
// f64 noise floor (~1e-14 m position / ~1e-15 m/s velocity over 10 s).
// The off-center case previously sat at ~2.5 mm trajectory drift until
// #560 fixed a formula gap in `evaluate_contact_pair`'s `rel_vel`; see
// the rel-vel comment block in `src/interactions.rs::evaluate_contact_pair`
// for the derivation. Issue #460 closed the residual per-stage
// divergence by switching this test path to stage-4 sampling — see
// `tier3_contact_point_off_center_stage4_probe` for the load-bearing
// regression guard on the stage-equivalence invariant.

// `CONTACT_FORCE_TOL` covers the loosest observed stage-4 force error
// across the five inter-body tests (off-center oblique — see
// `tier3_contact_point_off_center_stage4_probe`). Head-on tests sit
// several orders of magnitude tighter. The literal is ~1.1× the
// off-center observed for cross-platform FP headroom, which is
// generous on head-on but still many orders of magnitude tighter than
// the pre-#460 `POINT_OFF_CENTER_FORCE_TOL` of 0.0375 N — any real
// per-stage force regression trips this bar with room to spare.
const CONTACT_FORCE_TOL: f64 = 1.0e-8;

// `CONTACT_TORQUE_TOL` is the noise-floor tolerance, set above the
// loosest observed stage-4 torque error (off-center oblique, with
// head-on rotated tighter still). FP round-off on torque arms of
// order 1 m crossed with forces of order 1 N sits at ~1e-13 N·m; a
// strict 1.05× literal would false-fail on platforms whose FP
// rounding paths differ by a few ULPs. `3.0e-13` is ~1.5× the
// off-center observed and ~2.5× the head-on rotated observed — large
// enough to absorb cross-platform FP variance, still many orders of
// magnitude tighter than the pre-#460 `POINT_OFF_CENTER_TORQUE_TOL`
// of 3.1e-3 N·m.
const CONTACT_TORQUE_TOL: f64 = 3.0e-13;

/// Body state snapshot at a single checkpoint. Carries the full 6-DOF
/// state (position, velocity, attitude, angular velocity) for each of
/// the two bodies. Used by `tier3_contact_ground`, which compares
/// trajectories only.
#[derive(Debug, Clone, Copy)]
struct CheckpointBodies {
    veh1_trans: TranslationalState,
    veh1_rot: RotationalState,
    veh2_trans: TranslationalState,
    veh2_rot: RotationalState,
}

/// Contact checkpoint: body state at log time PLUS the per-body
/// stage-4 contact force/torque captures from the step that produced
/// that state.
///
/// JEOD's `ContactSurface::contact_force` is overwritten by every call
/// to the derivative-class `collect_forces_torques` job
/// (`Contact_S_modules/sv_dyn.sm:134`), so the CSV value at log time
/// `t` is the result of stage 4 of the most-recent RK4 step,
/// evaluated at the intermediate state `y_n + dt·k3` — not the
/// integrated end-of-step state `y_{n+1}`. Stage 4 attitude is
/// captured alongside force/torque so the test can transform the
/// inertial-frame force into the body/struct frame using the same
/// attitude the contact evaluator saw.
#[derive(Debug, Clone, Copy)]
struct ContactCheckpoint {
    state: CheckpointBodies,
    /// Inertial-frame contact force on body A from stage 4. `ZERO`
    /// before any step has been taken (matches JEOD's t=0 logging,
    /// where `contact_force` is initialized to zero before the first
    /// derivative-job call).
    stage4_force_on_a_inertial: DVec3,
    /// Body-frame contact torques from stage 4.
    stage4_torque_a_body: DVec3,
    stage4_torque_b_body: DVec3,
    /// Stage-4 attitudes (used to transform the inertial-frame force
    /// into each body's body/struct frame to compare against the CSV).
    stage4_q_a: JeodQuat,
    stage4_q_b: JeodQuat,
}

/// Propagate two bodies with contact forces evaluated at every RK4
/// stage inside `Simulation::step()`. Returns one checkpoint per CSV
/// row, carrying the body state plus a stage-4 force/torque capture
/// that matches JEOD's logged-value semantics (see [`ContactCheckpoint`]).
///
/// The stage-4 capture is produced by **replaying** the most-recent
/// RK4 step from a snapshot of the pre-step state through
/// `integrate_bodies_contact_coupled` with a recording closure. The
/// replay performs the same arithmetic as production
/// `Simulation::step()` (deterministic, gravity-free, single contact
/// pair) so the captured force/torque is bit-equivalent to what JEOD's
/// derivative-class call left in `contact_surface.contact_force` at
/// log time. This avoids exposing per-stage diagnostics on
/// `Simulation` (production-API thrift) while keeping the comparison
/// JEOD-faithful.
///
/// `facet_a` and `facet_b` define the two contact facets — the shape
/// positions are relative to each body's structural frame origin,
/// which in SIM_contact coincides with the body's CoM and inertial
/// position. `mass_a` and `mass_b` mirror what `Simulation::step()`
/// passes to `evaluate_contact_pair`.
#[allow(clippy::too_many_arguments)]
fn propagate_with_contact(
    sim: &mut Simulation,
    facet_a: ContactFacet,
    facet_b: ContactFacet,
    mass_a: &MassProperties,
    mass_b: &MassProperties,
    checkpoints: &[f64],
) -> Vec<ContactCheckpoint> {
    use astrodyn::{integrate_bodies_contact_coupled, CoupledBodyInput, CoupledIntegScratch};
    use std::cell::Cell;

    // Register the contact pair so forces are computed at every RK4 stage
    // (matching JEOD's check_contact derivative-class job).
    sim.register_contact_pair(0, facet_a, 1, facet_b);

    let mut out: Vec<ContactCheckpoint> = Vec::with_capacity(checkpoints.len());
    let mut cp_iter = checkpoints.iter().copied().peekable();

    let steps_total = (SIM_DURATION / DT).round() as usize;
    // Pre-step state snapshot for replay-on-checkpoint. Carries
    // `y_{step-1}` so when we hit a log time we can reproduce the
    // step that just finished and extract stage 4.
    let mut prev_state: Option<CheckpointBodies> = None;
    let mut replay_scratch = CoupledIntegScratch::new();

    for step in 0..=steps_total {
        let b_a = sim.body(0);
        let b_b = sim.body(1);
        let current_state = CheckpointBodies {
            veh1_trans: astrodyn::typed_bridge::trans_typed_to_raw(&b_a.trans),
            veh1_rot: astrodyn::typed_bridge::rot_typed_to_raw(
                &b_a.rot.expect("6-DOF required for SIM_contact"),
            ),
            veh2_trans: astrodyn::typed_bridge::trans_typed_to_raw(&b_b.trans),
            veh2_rot: astrodyn::typed_bridge::rot_typed_to_raw(
                &b_b.rot.expect("6-DOF required for SIM_contact"),
            ),
        };

        // Record output at checkpoints (±0.5·dt tolerance on time)
        let t = step as f64 * DT;
        if let Some(&cp) = cp_iter.peek() {
            if (t - cp).abs() <= 0.5 * DT {
                // Replay the step from `prev_state` (= y_{step-1}) to
                // capture stage 4 of the call that produced
                // `current_state`. For step 0 there is no preceding
                // step, so leave the stage-4 capture at zero — matches
                // JEOD's t=0 CSV row where contact_force is initialized
                // but no derivative call has run yet.
                let (s4_force_a, s4_tq_a, s4_tq_b, s4_q_a, s4_q_b) = match prev_state {
                    None => (
                        DVec3::ZERO,
                        DVec3::ZERO,
                        DVec3::ZERO,
                        JeodQuat::identity(),
                        JeodQuat::identity(),
                    ),
                    Some(prev) => {
                        let mut trans = [prev.veh1_trans, prev.veh2_trans];
                        let mut rot = [prev.veh1_rot, prev.veh2_rot];
                        let masses = [*mass_a, *mass_b];

                        let stage_call_count: Cell<usize> = Cell::new(0);
                        let s4_force_on_a: Cell<DVec3> = Cell::new(DVec3::ZERO);
                        let s4_torque_a: Cell<DVec3> = Cell::new(DVec3::ZERO);
                        let s4_torque_b: Cell<DVec3> = Cell::new(DVec3::ZERO);
                        let s4_q_a_cell: Cell<JeodQuat> = Cell::new(JeodQuat::identity());
                        let s4_q_b_cell: Cell<JeodQuat> = Cell::new(JeodQuat::identity());

                        let (trans_a_slice, trans_b_slice) = trans.split_at_mut(1);
                        let (rot_a_slice, rot_b_slice) = rot.split_at_mut(1);
                        let mut inputs = [
                            CoupledBodyInput {
                                trans: &mut trans_a_slice[0],
                                rot: &mut rot_a_slice[0],
                                mass: &masses[0],
                                non_grav_non_contact_force: DVec3::ZERO,
                                non_contact_torque_body: DVec3::ZERO,
                            },
                            CoupledBodyInput {
                                trans: &mut trans_b_slice[0],
                                rot: &mut rot_b_slice[0],
                                mass: &masses[1],
                                non_grav_non_contact_force: DVec3::ZERO,
                                non_contact_torque_body: DVec3::ZERO,
                            },
                        ];

                        integrate_bodies_contact_coupled(
                            &mut inputs,
                            &mut replay_scratch,
                            |_, _, _, _| DVec3::ZERO,
                            |stage_trans, stage_rot, accum| {
                                let call = stage_call_count.get();
                                stage_call_count.set(call + 1);

                                let ev = evaluate_contact_pair(
                                    &facet_a,
                                    &facet_b,
                                    &stage_trans[0],
                                    &stage_trans[1],
                                    Some(&stage_rot[0]),
                                    Some(&stage_rot[1]),
                                    DMat3::IDENTITY,
                                    DMat3::IDENTITY,
                                    Some(&masses[0]),
                                    Some(&masses[1]),
                                );

                                let (force_a, torque_a, torque_b) = match ev {
                                    Some(eval) => {
                                        accum[0].0 += eval.force_on_a;
                                        accum[1].0 -= eval.force_on_a;
                                        accum[0].1 += eval.torque_a_body;
                                        accum[1].1 += eval.torque_b_body;
                                        (eval.force_on_a, eval.torque_a_body, eval.torque_b_body)
                                    }
                                    None => (DVec3::ZERO, DVec3::ZERO, DVec3::ZERO),
                                };

                                // Stage 4 is the 4th (zero-indexed 3)
                                // contact_eval call per step.
                                if call == 3 {
                                    s4_force_on_a.set(force_a);
                                    s4_torque_a.set(torque_a);
                                    s4_torque_b.set(torque_b);
                                    s4_q_a_cell.set(stage_rot[0].quaternion);
                                    s4_q_b_cell.set(stage_rot[1].quaternion);
                                }
                            },
                            DT,
                        );

                        (
                            s4_force_on_a.get(),
                            s4_torque_a.get(),
                            s4_torque_b.get(),
                            s4_q_a_cell.get(),
                            s4_q_b_cell.get(),
                        )
                    }
                };

                out.push(ContactCheckpoint {
                    state: current_state,
                    stage4_force_on_a_inertial: s4_force_a,
                    stage4_torque_a_body: s4_tq_a,
                    stage4_torque_b_body: s4_tq_b,
                    stage4_q_a: s4_q_a,
                    stage4_q_b: s4_q_b,
                });
                cp_iter.next();
            }
        }

        if step == steps_total {
            break;
        }

        // Snapshot before stepping so we can replay this step if the
        // NEXT iteration's checkpoint lands on the new state.
        prev_state = Some(current_state);
        sim.step_n(1).expect("step_n failed");
    }
    out
}

/// Per-checkpoint assertion on contact force and torque against the JEOD
/// CSV. JEOD logs `contact_surface.contact_force` / `contact_torque` in
/// each body's *structural* frame as the result of stage 4 of the
/// most-recent RK4 step (the last derivative-class call before
/// sampling). Our [`ContactCheckpoint`] carries the matching stage-4
/// capture: an inertial-frame force on body A, body-frame torques per
/// body, and the stage-4 attitudes used to transform the inertial
/// force into the body/struct frame for comparison. For all current
/// SIM_contact scenarios `t_struct_body = I`, so body and struct
/// frames coincide.
#[allow(clippy::too_many_arguments)]
fn assert_contact_force_torque(
    label: &str,
    ours: &[ContactCheckpoint],
    records: &[ContactRecord],
    force_tol: f64,
    torque_tol: f64,
) {
    assert_eq!(ours.len(), records.len());
    let mut max_force_err_1 = 0.0_f64;
    let mut max_force_err_2 = 0.0_f64;
    let mut max_torque_err_1 = 0.0_f64;
    let mut max_torque_err_2 = 0.0_f64;
    let mut any_contact = false;
    for (cp, rec) in ours.iter().zip(records.iter()) {
        // t_struct_body = I for SIM_contact ⇒ struct == body frame.
        let t_inertial_body_a = cp.stage4_q_a.left_quat_to_transformation();
        let t_inertial_body_b = cp.stage4_q_b.left_quat_to_transformation();
        let force_a_struct = t_inertial_body_a * cp.stage4_force_on_a_inertial;
        let force_b_struct = t_inertial_body_b * (-cp.stage4_force_on_a_inertial);
        if cp.stage4_force_on_a_inertial != DVec3::ZERO
            || cp.stage4_torque_a_body != DVec3::ZERO
            || cp.stage4_torque_b_body != DVec3::ZERO
        {
            any_contact = true;
        }
        max_force_err_1 = max_force_err_1.max((force_a_struct - rec.veh1_force).length());
        max_force_err_2 = max_force_err_2.max((force_b_struct - rec.veh2_force).length());
        max_torque_err_1 =
            max_torque_err_1.max((cp.stage4_torque_a_body - rec.veh1_torque).length());
        max_torque_err_2 =
            max_torque_err_2.max((cp.stage4_torque_b_body - rec.veh2_torque).length());
    }

    println!(
        "{label}: stage-4 contact force err max = ({max_force_err_1:.3e}, {max_force_err_2:.3e}) N; \
         torque err max = ({max_torque_err_1:.3e}, {max_torque_err_2:.3e}) N*m"
    );

    assert!(
        any_contact,
        "{label}: stage-4 capture was zero at every checkpoint — scenario never in contact?"
    );
    assert!(
        max_force_err_1 < force_tol,
        "{label}: veh1 force error {max_force_err_1:.3e} > tol {force_tol:.3e}"
    );
    assert!(
        max_force_err_2 < force_tol,
        "{label}: veh2 force error {max_force_err_2:.3e} > tol {force_tol:.3e}"
    );
    assert!(
        max_torque_err_1 < torque_tol,
        "{label}: veh1 torque error {max_torque_err_1:.3e} > tol {torque_tol:.3e}"
    );
    assert!(
        max_torque_err_2 < torque_tol,
        "{label}: veh2 torque error {max_torque_err_2:.3e} > tol {torque_tol:.3e}"
    );
}

// ── Tier 3 tests ────────────────────────────────────────────────────

// non-recipe: all 6 contact tests run SIM_contact with 1 m / 100 kg test
// spheres, lines, and ground geometries with bespoke contact pairs and
// initial velocities. The geometries themselves are the test content; no
// recipe vehicle preset matches.
/// RUN_point: two 1 m radius spheres, 100 kg each. veh2 at (12,0,0) with
/// v=(-2,0,0). Contact starts when the centers are 2 m apart (t ≈ 5 s).
#[test]
fn tier3_contact_point_pair() {
    let csv_path = test_data_path("contact_point_contact_state.csv");
    let records = load_contact_csv(&csv_path);
    assert!(
        records.len() > 50,
        "Expected >50 log rows for 10 s at 20 Hz, got {}",
        records.len()
    );

    // Point facet: sphere radius 1 m centered at body origin.
    let facet = ContactFacet::point(DVec3::ZERO, 1.0, jeod_steel());
    let mass = point_mass_props();

    let mut sim = make_two_body_sim(100.0, DVec3::new(40.0, 40.0, 40.0));
    let checkpoints: Vec<f64> = records.iter().map(|r| r.time).collect();
    let ours = propagate_with_contact(&mut sim, facet, facet, &mass, &mass, &checkpoints);
    assert_eq!(ours.len(), records.len());

    let mut max_pos_err_1 = 0.0_f64;
    let mut max_pos_err_2 = 0.0_f64;
    let mut max_vel_err_1 = 0.0_f64;
    let mut max_vel_err_2 = 0.0_f64;
    for (our, rec) in ours.iter().zip(records.iter()) {
        max_pos_err_1 = max_pos_err_1.max((our.state.veh1_trans.position - rec.veh1_pos).length());
        max_pos_err_2 = max_pos_err_2.max((our.state.veh2_trans.position - rec.veh2_pos).length());
        max_vel_err_1 = max_vel_err_1.max((our.state.veh1_trans.velocity - rec.veh1_vel).length());
        max_vel_err_2 = max_vel_err_2.max((our.state.veh2_trans.velocity - rec.veh2_vel).length());
    }

    println!("SIM_contact RUN_point:");
    println!("  veh1 max pos error: {max_pos_err_1:.3e} m");
    println!("  veh2 max pos error: {max_pos_err_2:.3e} m");
    println!("  veh1 max vel error: {max_vel_err_1:.3e} m/s");
    println!("  veh2 max vel error: {max_vel_err_2:.3e} m/s");

    // Head-on sphere-sphere symmetric contact: after the issue-#117 fixes
    // (unit-conversion constants + rotating-frame rel-vel term), our
    // pipeline-coupled RK4 matches JEOD to ~1e-15 m / 1e-15 m·s⁻¹ over
    // 10 s — i.e. the machine-precision floor for f64 arithmetic on this
    // trajectory length.
    //
    // The literal `1.0e-13` is a deliberate ~100× exception to the
    // CLAUDE.md "5% above observed" tolerance policy: the observed
    // max is f64 round-off noise (a few ULPs of ~1 m positions), which
    // is platform-, microarchitecture-, and codegen-dependent within
    // a small constant factor. Setting the literal to `observed * 1.05`
    // (~1e-15) would produce false failures on x86_64 hosts whose
    // FMA / x87-80-bit rounding differs from the host that observed
    // the current value. `1.0e-13` is the noise-floor budget — large
    // enough to absorb cross-platform FP variance, but still 12
    // orders of magnitude below the pre-issue-#117 head-on drift
    // (~14 μm), so any genuine regression in the contact pipeline
    // will trip it immediately.
    assert!(
        max_pos_err_1 < 1.0e-13,
        "veh1 position error {max_pos_err_1:.3e} > 100 fm"
    );
    assert!(
        max_pos_err_2 < 1.0e-13,
        "veh2 position error {max_pos_err_2:.3e} > 100 fm"
    );
    assert!(
        max_vel_err_1 < 1.0e-13,
        "veh1 velocity error {max_vel_err_1:.3e} > 1e-13 m/s"
    );
    assert!(
        max_vel_err_2 < 1.0e-13,
        "veh2 velocity error {max_vel_err_2:.3e} > 1e-13 m/s"
    );

    assert_contact_force_torque(
        "SIM_contact RUN_point",
        &ours,
        &records,
        CONTACT_FORCE_TOL,
        CONTACT_TORQUE_TOL,
    );
}

/// RUN_line: two capsules (length 2 m, radius 1 m) aligned along x.
/// 200 kg each, approaching head-on. Identical trajectory to RUN_point
/// because the lines are collinear with the approach direction (the end
/// caps act as spheres).
#[test]
fn tier3_contact_line_pair() {
    let csv_path = test_data_path("contact_line_contact_state.csv");
    let records = load_contact_csv(&csv_path);
    assert!(records.len() > 50);

    // Line facet: capsule along body x-axis, length 2 m, radius 1 m.
    let facet = ContactFacet::line(
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        1.0,
        jeod_steel(),
    );

    let mass = line_mass_props();
    let mut sim = make_two_body_sim(200.0, DVec3::new(100.0, 116.6667, 116.6667));
    let checkpoints: Vec<f64> = records.iter().map(|r| r.time).collect();
    let ours = propagate_with_contact(&mut sim, facet, facet, &mass, &mass, &checkpoints);
    assert_eq!(ours.len(), records.len());

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    for (our, rec) in ours.iter().zip(records.iter()) {
        max_pos_err = max_pos_err.max((our.state.veh1_trans.position - rec.veh1_pos).length());
        max_pos_err = max_pos_err.max((our.state.veh2_trans.position - rec.veh2_pos).length());
        max_vel_err = max_vel_err.max((our.state.veh1_trans.velocity - rec.veh1_vel).length());
        max_vel_err = max_vel_err.max((our.state.veh2_trans.velocity - rec.veh2_vel).length());
    }
    println!("SIM_contact RUN_line: max pos={max_pos_err:.3e} m, max vel={max_vel_err:.3e} m/s");

    // Head-on capsule-capsule. After issue #117 fixes, matches JEOD to
    // machine precision over 10 s.
    assert!(max_pos_err < 1.0e-13, "position error {max_pos_err:.3e}");
    assert!(max_vel_err < 1.0e-13, "velocity error {max_vel_err:.3e}");

    assert_contact_force_torque(
        "SIM_contact RUN_line",
        &ours,
        &records,
        CONTACT_FORCE_TOL,
        CONTACT_TORQUE_TOL,
    );
}

/// RUN_line_point: capsule (veh1) meets sphere (veh2) head-on. Same mass /
/// inertia as RUN_line for both vehicles since the default mass file is
/// `veh_mass_line.py` (cylinder mass properties).
#[test]
fn tier3_contact_line_point() {
    let csv_path = test_data_path("contact_line_point_contact_state.csv");
    let records = load_contact_csv(&csv_path);
    assert!(records.len() > 50);

    let line_facet = ContactFacet::line(
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        1.0,
        jeod_steel(),
    );
    let point_facet = ContactFacet::point(DVec3::ZERO, 1.0, jeod_steel());
    let mass = line_mass_props();

    let mut sim = make_two_body_sim(200.0, DVec3::new(100.0, 116.6667, 116.6667));
    let checkpoints: Vec<f64> = records.iter().map(|r| r.time).collect();
    let ours = propagate_with_contact(
        &mut sim,
        line_facet,
        point_facet,
        &mass,
        &mass,
        &checkpoints,
    );
    assert_eq!(ours.len(), records.len());

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    for (our, rec) in ours.iter().zip(records.iter()) {
        max_pos_err = max_pos_err.max((our.state.veh1_trans.position - rec.veh1_pos).length());
        max_pos_err = max_pos_err.max((our.state.veh2_trans.position - rec.veh2_pos).length());
        max_vel_err = max_vel_err.max((our.state.veh1_trans.velocity - rec.veh1_vel).length());
        max_vel_err = max_vel_err.max((our.state.veh2_trans.velocity - rec.veh2_vel).length());
    }
    println!(
        "SIM_contact RUN_line_point: max pos={max_pos_err:.3e} m, max vel={max_vel_err:.3e} m/s"
    );

    // Head-on line-point. After issue #117 fixes, matches JEOD to
    // machine precision over 10 s.
    assert!(max_pos_err < 1.0e-13, "pos err {max_pos_err:.3e}");
    assert!(max_vel_err < 1.0e-13, "vel err {max_vel_err:.3e}");

    assert_contact_force_torque(
        "SIM_contact RUN_line_point",
        &ours,
        &records,
        CONTACT_FORCE_TOL,
        CONTACT_TORQUE_TOL,
    );
}

/// RUN_line_side_to_side: two capsules rotated 90° relative to each other
/// so their cylindrical sides meet. JEOD's `input.py` sets:
///   veh1 euler Yaw_Pitch_Roll = (0, 90°, 0) — pitched up 90°
///   veh2 euler Yaw_Pitch_Roll = (0, 0, 90°) — rolled right 90°
/// Facets are along body-x in structural coords (same as RUN_line).
#[test]
fn tier3_contact_line_side_to_side() {
    let csv_path = test_data_path("contact_line_side_contact_state.csv");
    let records = load_contact_csv(&csv_path);
    assert!(records.len() > 50);

    // Both facets are along body-x in structural coords. Body attitude
    // rotates the shape into the world frame at each integration stage.
    let facet = ContactFacet::line(
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        1.0,
        jeod_steel(),
    );

    // JEOD Yaw_Pitch_Roll convention: apply yaw (about z), then pitch
    // (about new y), then roll (about new x). veh1 pitch=90° ⇒ +90° about y,
    // veh2 roll=90° ⇒ +90° about x.
    //
    // Must use `left_quat_from_eigen_rotation` rather than glam's
    // `DQuat::from_axis_angle` + field copy: JEOD's left-quat convention
    // stores the vector part as `-sin(θ/2)·axis` (note the minus sign),
    // which `left_quat_from_eigen_rotation` applies. Using glam's
    // positive-sine quaternion directly would store the opposite-sign
    // attitude — invisible to pos/vel (line geometry is symmetric under
    // the rotation flip) but it inverts forces in the structural frame.
    let jeod_veh1 = JeodQuat::left_quat_from_eigen_rotation(90.0_f64.to_radians(), DVec3::Y);
    let jeod_veh2 = JeodQuat::left_quat_from_eigen_rotation(90.0_f64.to_radians(), DVec3::X);

    // Build sim with non-identity initial rotations.
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    add_empty_space_root(&mut sim);

    let mass_props = MassProperties::with_inertia(
        200.0,
        DMat3::from_cols(
            DVec3::new(100.0, 0.0, 0.0),
            DVec3::new(0.0, 116.6667, 0.0),
            DVec3::new(0.0, 0.0, 116.6667),
        ),
        DVec3::ZERO,
    );

    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: jeod_veh1,
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        ..VehicleConfig::named("tier3-sim-contact-5")
    });
    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: DVec3::new(12.0, 0.0, 0.0),
            velocity: DVec3::new(-2.0, 0.0, 0.0),
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: jeod_veh2,
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        ..VehicleConfig::named("tier3-sim-contact-4")
    });
    sim.validate().unwrap();

    let checkpoints: Vec<f64> = records.iter().map(|r| r.time).collect();
    let ours = propagate_with_contact(
        &mut sim,
        facet,
        facet,
        &mass_props,
        &mass_props,
        &checkpoints,
    );
    assert_eq!(ours.len(), records.len());

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    for (our, rec) in ours.iter().zip(records.iter()) {
        max_pos_err = max_pos_err.max((our.state.veh1_trans.position - rec.veh1_pos).length());
        max_pos_err = max_pos_err.max((our.state.veh2_trans.position - rec.veh2_pos).length());
        max_vel_err = max_vel_err.max((our.state.veh1_trans.velocity - rec.veh1_vel).length());
        max_vel_err = max_vel_err.max((our.state.veh2_trans.velocity - rec.veh2_vel).length());
    }
    println!(
        "SIM_contact RUN_line_side: max pos={max_pos_err:.3e} m, max vel={max_vel_err:.3e} m/s"
    );

    // Perpendicular-capsule side-to-side contact — the capsules meet at
    // their midpoints so the contact is effectively sphere-sphere along
    // the inter-body x-axis. Rotated geometry is exercised here but the
    // collision remains symmetric. After issue #117 fixes, matches JEOD
    // to machine precision over 10 s.
    assert!(max_pos_err < 1.0e-13, "pos err {max_pos_err:.3e}");
    assert!(max_vel_err < 1.0e-13, "vel err {max_vel_err:.3e}");

    assert_contact_force_torque(
        "SIM_contact RUN_line_side",
        &ours,
        &records,
        CONTACT_FORCE_TOL,
        CONTACT_TORQUE_TOL,
    );
}

/// RUN_point_off_center: same spheres as RUN_point but veh2 starts with a
/// transverse offset so the collision is oblique. Uses identical mass
/// properties (100 kg sphere, 40 kg·m² inertia).
///
/// The exact offset is read from the CSV's t=0 row rather than hardcoded;
/// that row is JEOD source data (initial conditions), not mid-sim output.
#[test]
fn tier3_contact_point_off_center() {
    let csv_path = test_data_path("contact_point_off_center_contact_state.csv");
    let records = load_contact_csv(&csv_path);
    assert!(records.len() > 50);

    let init = &records[0];

    let facet = ContactFacet::point(DVec3::ZERO, 1.0, jeod_steel());

    // Reconstruct the sim from the t=0 row (initial conditions from CSV are
    // allowed per CLAUDE.md; only mid-sim CSV data is forbidden as input).
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    add_empty_space_root(&mut sim);

    let mass_props = MassProperties::with_inertia(
        100.0,
        DMat3::from_cols(
            DVec3::new(40.0, 0.0, 0.0),
            DVec3::new(0.0, 40.0, 0.0),
            DVec3::new(0.0, 0.0, 40.0),
        ),
        DVec3::ZERO,
    );

    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: init.veh1_pos,
            velocity: init.veh1_vel,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        ..VehicleConfig::named("tier3-sim-contact-3")
    });
    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: init.veh2_pos,
            velocity: init.veh2_vel,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        ..VehicleConfig::named("tier3-sim-contact-2")
    });
    sim.validate().unwrap();

    let checkpoints: Vec<f64> = records.iter().map(|r| r.time).collect();
    let ours = propagate_with_contact(
        &mut sim,
        facet,
        facet,
        &mass_props,
        &mass_props,
        &checkpoints,
    );

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    for (our, rec) in ours.iter().zip(records.iter()) {
        max_pos_err = max_pos_err.max((our.state.veh1_trans.position - rec.veh1_pos).length());
        max_pos_err = max_pos_err.max((our.state.veh2_trans.position - rec.veh2_pos).length());
        max_vel_err = max_vel_err.max((our.state.veh1_trans.velocity - rec.veh1_vel).length());
        max_vel_err = max_vel_err.max((our.state.veh2_trans.velocity - rec.veh2_vel).length());
    }
    println!(
        "SIM_contact RUN_point_off_center: max pos={max_pos_err:.15e} m, max vel={max_vel_err:.15e} m/s"
    );

    // Oblique collision, trajectory at f64 noise floor. Issue #560 closed
    // a mathematical formula gap in `evaluate_contact_pair`'s `rel_vel`
    // that previously held this test at ~2.5 mm drift:
    //
    // The prior formula `(v_a − v_b) + ω_a × cp_a − ω_b × cp_b` is the
    // velocity-of-contact-points formula and assumes `cp_a − cp_b = p_b − p_a`,
    // which only holds in non-penetrating contact. During penetration
    // (sphere centers closer than sum of radii), `cp_a − cp_b` exceeds
    // `p_b − p_a` by the penetration ratio, producing an `ω × δ`
    // divergence (~1.5 μm/s per stage at the test's ω ≈ 2.5e-4 rad/s)
    // that amplified through 1000+ contact stages into mm-scale drift.
    //
    // The replacement formula `(v_a − v_b) − ω_a × rel_pos +
    // (ω_b − ω_a) × cp_a` ports JEOD's `point_contact_pair.cc:79-84`
    // subject-body-frame chain into inertial form. At identity attitude
    // it's bit-equivalent to JEOD; at non-identity attitudes the
    // rotation matrices factor out of the cross products by JEOD's
    // standard convention. Both formulas reduce to `(v_a − v_b)` for
    // head-on contact (no rotation, equal masses), so the head-on tests
    // are unchanged.
    assert!(
        max_pos_err < 1.6e-14,
        "veh{{1,2}} position error {max_pos_err:.3e} > 1.6e-14 m"
    );
    assert!(
        max_vel_err < 3.5e-15,
        "veh{{1,2}} velocity error {max_vel_err:.3e} > 3.5e-15 m/s"
    );

    // Force/torque comparison now samples stage-4 captures (matching
    // JEOD's logging convention), so off-center reuses the same
    // tolerances as the head-on tests.
    assert_contact_force_torque(
        "SIM_contact RUN_point_off_center",
        &ours,
        &records,
        CONTACT_FORCE_TOL,
        CONTACT_TORQUE_TOL,
    );
}

/// Issue #460: probes whether the `POINT_OFF_CENTER_*_TOL` headroom over
/// head-on tolerances is a force/torque sampling artifact rather than a
/// physics gap.
///
/// JEOD's `ContactSurface::collect_forces_torques` is registered as a
/// derivative-class job in `Contact_S_modules/sv_dyn.sm:134`, so the
/// `contact_force` column logged at time `t` is whatever stage 4 of the
/// most-recent RK4 step wrote — evaluated at the intermediate state
/// `y_n + dt·k3`, not the integrated end-of-step state `y_{n+1}`.
/// `tier3_contact_point_off_center` re-evaluates `evaluate_contact_pair`
/// at `y_{n+1}` from the checkpoint snapshot, so for ω ≠ 0 the two
/// samples drift by O(h^5) × spring stiffness even when the underlying
/// physics is bit-identical.
///
/// This probe propagates the same scenario directly through
/// `integrate_bodies_contact_coupled` (no `Simulation` wrapping), with a
/// recording closure that captures stage 4's force/torque per step.
/// Asserting (a) trajectory at the production f64-noise floor confirms
/// the manual propagation is bit-equivalent to `Simulation::step()` for
/// the SIM_contact scenario, then (b) stage-4 force/torque at the
/// head-on tolerances confirms no physics gap remains in `#460`.
#[test]
fn tier3_contact_point_off_center_stage4_probe() {
    use astrodyn::{integrate_bodies_contact_coupled, CoupledBodyInput, CoupledIntegScratch};
    use std::cell::Cell;

    let csv_path = test_data_path("contact_point_off_center_contact_state.csv");
    let records = load_contact_csv(&csv_path);
    let init = &records[0];

    let facet = ContactFacet::point(DVec3::ZERO, 1.0, jeod_steel());
    let mass_props = MassProperties::with_inertia(
        100.0,
        DMat3::from_cols(
            DVec3::new(40.0, 0.0, 0.0),
            DVec3::new(0.0, 40.0, 0.0),
            DVec3::new(0.0, 0.0, 40.0),
        ),
        DVec3::ZERO,
    );
    let masses = [mass_props, mass_props];

    let mut trans = [
        TranslationalState {
            position: init.veh1_pos,
            velocity: init.veh1_vel,
        },
        TranslationalState {
            position: init.veh2_pos,
            velocity: init.veh2_vel,
        },
    ];
    let mut rot = [
        RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        },
        RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        },
    ];

    let mut scratch = CoupledIntegScratch::new();

    // Stage-4 captures, refreshed every step.
    let stage_call_count: Cell<usize> = Cell::new(0);
    let stage4_force_a: Cell<DVec3> = Cell::new(DVec3::ZERO);
    let stage4_torque_a_body: Cell<DVec3> = Cell::new(DVec3::ZERO);
    let stage4_torque_b_body: Cell<DVec3> = Cell::new(DVec3::ZERO);
    let stage4_quat_a: Cell<JeodQuat> = Cell::new(JeodQuat::identity());
    let stage4_quat_b: Cell<JeodQuat> = Cell::new(JeodQuat::identity());

    let steps_total = (SIM_DURATION / DT).round() as usize;
    let log_step_stride = (LOG_CYCLE / DT).round() as usize;

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    let mut max_force_err_struct = 0.0_f64;
    let mut max_torque_err_body = 0.0_f64;

    for step in 0..=steps_total {
        if step % log_step_stride == 0 {
            let log_idx = step / log_step_stride;
            assert!(
                log_idx < records.len(),
                "step {step} maps to log idx {log_idx} >= {} CSV rows",
                records.len()
            );
            let rec = &records[log_idx];
            max_pos_err = max_pos_err.max((trans[0].position - rec.veh1_pos).length());
            max_pos_err = max_pos_err.max((trans[1].position - rec.veh2_pos).length());
            max_vel_err = max_vel_err.max((trans[0].velocity - rec.veh1_vel).length());
            max_vel_err = max_vel_err.max((trans[1].velocity - rec.veh2_vel).length());
            // Stage-4 sample only exists after the first step.
            if step > 0 {
                let t_inertial_body_a = stage4_quat_a.get().left_quat_to_transformation();
                let t_inertial_body_b = stage4_quat_b.get().left_quat_to_transformation();
                let force_a_struct = t_inertial_body_a * stage4_force_a.get();
                let force_b_struct = t_inertial_body_b * (-stage4_force_a.get());
                max_force_err_struct =
                    max_force_err_struct.max((force_a_struct - rec.veh1_force).length());
                max_force_err_struct =
                    max_force_err_struct.max((force_b_struct - rec.veh2_force).length());
                max_torque_err_body = max_torque_err_body
                    .max((stage4_torque_a_body.get() - rec.veh1_torque).length());
                max_torque_err_body = max_torque_err_body
                    .max((stage4_torque_b_body.get() - rec.veh2_torque).length());
            }
        }

        if step == steps_total {
            break;
        }

        stage_call_count.set(0);
        let (trans_a, trans_b) = trans.split_at_mut(1);
        let (rot_a, rot_b) = rot.split_at_mut(1);
        let mut inputs = [
            CoupledBodyInput {
                trans: &mut trans_a[0],
                rot: &mut rot_a[0],
                mass: &masses[0],
                non_grav_non_contact_force: DVec3::ZERO,
                non_contact_torque_body: DVec3::ZERO,
            },
            CoupledBodyInput {
                trans: &mut trans_b[0],
                rot: &mut rot_b[0],
                mass: &masses[1],
                non_grav_non_contact_force: DVec3::ZERO,
                non_contact_torque_body: DVec3::ZERO,
            },
        ];

        integrate_bodies_contact_coupled(
            &mut inputs,
            &mut scratch,
            |_, _, _, _| DVec3::ZERO,
            |stage_trans, stage_rot, out| {
                let call = stage_call_count.get();
                stage_call_count.set(call + 1);

                let ev = evaluate_contact_pair(
                    &facet,
                    &facet,
                    &stage_trans[0],
                    &stage_trans[1],
                    Some(&stage_rot[0]),
                    Some(&stage_rot[1]),
                    DMat3::IDENTITY,
                    DMat3::IDENTITY,
                    Some(&masses[0]),
                    Some(&masses[1]),
                );

                let (force_a, torque_a, torque_b) = match ev {
                    Some(eval) => {
                        out[0].0 += eval.force_on_a;
                        out[1].0 -= eval.force_on_a;
                        out[0].1 += eval.torque_a_body;
                        out[1].1 += eval.torque_b_body;
                        (eval.force_on_a, eval.torque_a_body, eval.torque_b_body)
                    }
                    None => (DVec3::ZERO, DVec3::ZERO, DVec3::ZERO),
                };

                // Stage 4 is the 4th (zero-indexed 3) contact_eval call
                // per coupled-RK4 step; this is what JEOD's derivative
                // job leaves in `contact_surface.contact_force` at the
                // moment the logger samples.
                if call == 3 {
                    stage4_force_a.set(force_a);
                    stage4_torque_a_body.set(torque_a);
                    stage4_torque_b_body.set(torque_b);
                    stage4_quat_a.set(stage_rot[0].quaternion);
                    stage4_quat_b.set(stage_rot[1].quaternion);
                }
            },
            DT,
        );
    }

    println!(
        "tier3_contact_point_off_center_stage4_probe: pos={max_pos_err:.3e} m, \
         vel={max_vel_err:.3e} m/s, stage-4 force={max_force_err_struct:.3e} N, \
         stage-4 torque={max_torque_err_body:.3e} N·m"
    );

    // Same bar as `tier3_contact_point_off_center` — proves the manual
    // propagation here is bit-equivalent to `Simulation::step()` for
    // SIM_contact (gravity-free, 2 bodies, single contact pair).
    assert!(
        max_pos_err < 1.6e-14,
        "probe trajectory diverged from production: pos err {max_pos_err:.3e}"
    );
    assert!(
        max_vel_err < 3.5e-15,
        "probe trajectory diverged from production: vel err {max_vel_err:.3e}"
    );

    // Probe uses the same unified `CONTACT_*_TOL` as the main inter-body
    // contact tests — both paths sample at stage 4 of the same RK4
    // step, so they target the same FP noise floor. The probe runs an
    // independent propagation (direct `integrate_bodies_contact_coupled`,
    // bypassing `Simulation::step`), so it catches divergence between
    // production and the bare integrator while the main test guards
    // the per-stage parity invariant from production's perspective.
    assert!(
        max_force_err_struct < CONTACT_FORCE_TOL,
        "stage-4 force err {max_force_err_struct:.3e} N >= tol \
         {CONTACT_FORCE_TOL:.3e} N — per-stage parity regressed; \
         the sampling-artifact hypothesis no longer fully explains the residual"
    );
    assert!(
        max_torque_err_body < CONTACT_TORQUE_TOL,
        "stage-4 torque err {max_torque_err_body:.3e} N·m >= tol \
         {CONTACT_TORQUE_TOL:.3e} N·m — per-stage parity regressed; \
         the sampling-artifact hypothesis no longer fully explains the residual"
    );
}

/// RUN_contact_ground: SIM_ground_contact.
///
/// Two vehicles (veh1 = line cylinder, veh2 = point sphere, 200 kg each)
/// initialized at Earth's surface in `Earth.inertial`. Spherical Earth
/// gravity pulls them toward the planet center; a ground-contact spring
/// (k = 1751.25 N/m, c = 35.025 N·s/m, μ = 0.5) pushes back. The vehicles
/// start interpenetrating the ground at t=0, producing an impulsive
/// ~2.2 × 10¹⁰ N force that launches them outward at ~93 km/s within
/// 50 ms; the rest of the 10-second run is ballistic coast under
/// spherical gravity.
///
/// JEOD source: `verif/SIM_ground_contact/SET_test/RUN_contact_ground/input.py`
/// + `Modified_data/{ground/{ground_facet,pair_interaction},vehicle/sv_earth}.py`.
fn make_ground_contact_sim() -> (Simulation, usize) {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);

    // Earth as a spherical (point-mass) central body. JEOD's
    // SIM_ground_contact configures the gravity controls with degree=0,
    // order=0, spherical=true so the SH model collapses to point mass.
    // We use the same effective physics via a PointMass GravitySource.
    let earth_mu = astrodyn::EARTH.shape.mu;
    let earth_idx = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: earth_mu,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            // SphericalTerrain does not consult pfix rotation, so we omit
            // the rotation model here — keeps the test self-contained.
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );

    let mass_props = line_mass_props(); // 200 kg, diag(100, 116.667, 116.667)
    let earth_radius = astrodyn::EARTH.shape.r_eq();

    let earth_grav = GravityControls {
        controls: vec![GravityControl::new_spherical(
            astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
            GravityGradient::Skip,
        )],
    };

    // veh1 — line cylinder along structural x-axis.
    let veh1 = sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: DVec3::new(earth_radius, 0.0, 0.0),
            velocity: DVec3::ZERO,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: earth_grav.clone(),
        compute_gravity_gradient: false,
        ..VehicleConfig::named("tier3-sim-contact-1")
    });

    // veh2 — point sphere 10 m radially outward from veh1.
    let veh2 = sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: DVec3::new(earth_radius + 10.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: earth_grav,
        compute_gravity_gradient: false,
        ..VehicleConfig::named("tier3-sim-contact-0")
    });
    assert_eq!(veh1, 0);
    assert_eq!(veh2, 1);

    sim.validate().unwrap();
    (sim, earth_idx)
}

/// Ground-contact material: JEOD `Modified_data/ground/pair_interaction.py`
/// — spring_k=10 lbf/in, damping_b=0.2 lbf·s/in, mu=0.5. JEOD's CSV
/// trajectory is generated using Trick's `attach_units("lbf/in", X)`
/// conversion, which uses the NIST-exact 1 lbf = 4.4482216152605 N (vs
/// the Older `4.448222 N` mantissa baked into JEOD_SPRING_K above for
/// SIM_contact). The resulting 10 lbf/in = 1751.2683...; using
/// 1751.2502 (half of the SIM_contact constant) gives a constant
/// ~0.96 m/s velocity offset because Δv = (1/6) k R dt / m amplifies
/// the 0.001% conversion gap into 1 m/s on a 93 km/s impulse. We use
/// the NIST-exact value here.
const GROUND_SPRING_K: f64 = 10.0 * 4.4482216152605 / 0.0254; // 1751.2683… N/m
const GROUND_DAMPING_B: f64 = 0.2 * 4.4482216152605 / 0.0254; // 35.0254… N·s/m
const GROUND_MU: f64 = 0.5;

fn ground_steel() -> ContactMaterial {
    ContactMaterial::jeod_spring(GROUND_SPRING_K, GROUND_DAMPING_B, GROUND_MU)
}

/// Tier 3 cross-validation against `SIM_ground_contact/RUN_contact_ground` CSV.
///
/// JEOD's CSV trajectory is produced by an initialization-state artifact
/// in `ContactGround::initialize_ground`: a pre-propagation
/// `in_contact()` call writes an impulsive force onto `subject->force`,
/// which the integrator consumes at stage 1 of step 1 (RK4 weight 1/6)
/// and `ContactSurface::collect_forces_torques` zeroes thereafter. Our
/// port models this explicitly via [`Phase::Initialization`] —
/// `Simulation::register_ground_contact_pair` evaluates the
/// pre-propagation force at registration time and stores it on the pair
/// as `pending_initial_impulse`. The coupled-RK4 stage closure consumes
/// it on the first invocation and clears it for stages 2-4 and all
/// subsequent steps. After consumption, the steady-state path
/// ([`Phase::SteadyState`]) reports no contact for vehicles above the
/// surface — physically correct, matching JEOD.
///
/// ## Root cause
///
/// JEOD's `BodyRefFrame::state.trans.position` for a surface-model-created
/// `vehicle_point` (the C++ frame backing each `ContactFacet`) is
/// **default-constructed to (0, 0, 0)** when the frame is created, and
/// only later populated to its true inertial position by
/// `DynBody::compute_vehicle_point_states` (see
/// `dyn_body_propagate_state.cc::compute_derived_state_forward`).
/// `ContactGround::initialize_ground` runs **before** that propagation
/// (the `P_DYN("initialization")` job in
/// `verif/SIM_ground_contact/S_modules/contact.sm:70`), and inside
/// `GroundInteraction::initialize` calls `in_contact()` once with
/// `vehicle_point.state.trans.position == (0, 0, 0)`. Tracing
/// `point_ground_interaction.cc::in_contact` from that state:
///
/// - `vec = structure.pos + vp.pos = (R, 0, 0) + (0, 0, 0) = (R, 0, 0)`
///   (interpreted as the vehicle's inertial position).
/// - Ground point in body frame ≈ `(R, 0, 0)`; sphere/cylinder
///   `contact_point` ≈ `(1, 0, 0)`.
/// - `facet_pos = T_parent_this * vp.pos = identity * (0, 0, 0) = (0, 0, 0)`.
/// - `rel_state = contact_point + facet_pos = (1, 0, 0)` →
///   `subject_mag = 1 << R = ground_mag` → **contact triggers** with
///   penetration ≈ R, force ≈ k·R = 1.117 × 10¹⁰ N per vehicle. This
///   value is what JEOD writes into `subject->force` and what eventually
///   surfaces as the `~2.2 × 10¹⁰ N` first-row CSV value (a factor of 2
///   suggesting init runs `in_contact` twice, once per ground facet
///   pairing — to be confirmed by a JEOD live-run trace).
///
/// At the **first integration step**, before any RK4 stage runs,
/// `compute_vehicle_point_states` has propagated `vp.state.trans.position`
/// to its true inertial value `(R, 0, 0)`. The same algorithm now gives:
///
/// - `vec = (R, 0, 0) + (R, 0, 0) = (2R, 0, 0)` (this is the JEOD code's
///   apparent doubled-position symptom — only consistent because the
///   init-time vp.pos was (0, 0, 0)).
/// - `facet_pos = identity * (R, 0, 0) = (R, 0, 0)`.
/// - `rel_state = (R+1, 0, 0)` → `subject_mag = R+1 > R = ground_mag` →
///   **no contact** at any altitude.
///
/// Net JEOD behaviour: an impulsive force of 1.117 × 10¹⁰ N on
/// `subject->force` from initialization is consumed at stage 1 of step 1
/// (RK4 weight 1/6), and stages 2–4 plus all subsequent steps see zero
/// contact force. RK4 then yields
/// `Δv ≈ (1/6) × F × dt / m = 93 081 m/s`, exactly matching the t=0.05 CSV
/// velocity.
///
/// Tolerances per CLAUDE.md "5% above observed max" policy. The
/// residuals show essentially bit-for-bit agreement with JEOD's CSV
/// (f64 roundoff in the gravity-coupled RK4). The constants below are
/// 1.05× observed maxima.
const GROUND_POS_TOL: f64 = 3.0e-9; // m
const GROUND_VEL_TOL: f64 = 5.0e-11; // m/s

#[test]
fn tier3_contact_ground() {
    let csv_path = test_data_path("contact_ground_contact_state.csv");
    let records = load_contact_csv(&csv_path);
    assert!(!records.is_empty(), "expected at least one CSV record");

    let (mut sim, earth_idx) = make_ground_contact_sim();
    let earth_radius = astrodyn::EARTH.shape.r_eq();
    let mat = ground_steel();
    let terrain = Arc::new(SphericalTerrain::new(earth_radius));
    let ground = GroundFacet::new(terrain, 0.0, mat);

    let veh1_facet = ContactFacet::line(
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        1.0,
        mat,
    );
    let veh2_facet = ContactFacet::point(DVec3::ZERO, 1.0, mat);
    sim.register_ground_contact_pair(0, veh1_facet, ground.clone(), earth_idx);
    sim.register_ground_contact_pair(1, veh2_facet, ground, earth_idx);

    // Step at the SIM_contact native rate (DT = 0.01 s) and snapshot at
    // each CSV checkpoint (LOG_CYCLE = 0.05 s).
    let checkpoints: Vec<f64> = records.iter().map(|r| r.time).collect();
    let mut cp_iter = checkpoints.iter().copied().peekable();

    let mut snapshots: Vec<CheckpointBodies> = Vec::with_capacity(records.len());
    let steps_total = (SIM_DURATION / DT).round() as usize;
    for step in 0..=steps_total {
        let b1 = sim.body(0);
        let b2 = sim.body(1);

        let t = step as f64 * DT;
        if let Some(&cp) = cp_iter.peek() {
            if (t - cp).abs() <= 0.5 * DT {
                snapshots.push(CheckpointBodies {
                    veh1_trans: astrodyn::TranslationalState {
                        position: b1.trans.position.raw_si(),
                        velocity: b1.trans.velocity.raw_si(),
                    },
                    veh1_rot: {
                        let _r = b1.rot.expect("6-DOF required");
                        astrodyn::RotationalState {
                            quaternion: _r.q_inertial_body.to_jeod_quat(),
                            ang_vel_body: _r.ang_vel_body.raw_si(),
                        }
                    },
                    veh2_trans: astrodyn::TranslationalState {
                        position: b2.trans.position.raw_si(),
                        velocity: b2.trans.velocity.raw_si(),
                    },
                    veh2_rot: {
                        let _r = b2.rot.expect("6-DOF required");
                        astrodyn::RotationalState {
                            quaternion: _r.q_inertial_body.to_jeod_quat(),
                            ang_vel_body: _r.ang_vel_body.raw_si(),
                        }
                    },
                });
                cp_iter.next();
            }
        }
        if step == steps_total {
            break;
        }
        sim.step_n(1).expect("step_n failed");
    }

    assert_eq!(
        snapshots.len(),
        records.len(),
        "snapshot/CSV checkpoint count mismatch ({} vs {})",
        snapshots.len(),
        records.len()
    );

    let mut max_pos_err_1 = 0.0_f64;
    let mut max_pos_err_2 = 0.0_f64;
    let mut max_vel_err_1 = 0.0_f64;
    let mut max_vel_err_2 = 0.0_f64;
    for (snap, rec) in snapshots.iter().zip(records.iter()) {
        max_pos_err_1 = max_pos_err_1.max((snap.veh1_trans.position - rec.veh1_pos).length());
        max_pos_err_2 = max_pos_err_2.max((snap.veh2_trans.position - rec.veh2_pos).length());
        max_vel_err_1 = max_vel_err_1.max((snap.veh1_trans.velocity - rec.veh1_vel).length());
        max_vel_err_2 = max_vel_err_2.max((snap.veh2_trans.velocity - rec.veh2_vel).length());
    }
    println!(
        "tier3_contact_ground: pos err max = ({max_pos_err_1:.3e}, {max_pos_err_2:.3e}) m; \
         vel err max = ({max_vel_err_1:.3e}, {max_vel_err_2:.3e}) m/s"
    );

    assert!(
        max_pos_err_1 < GROUND_POS_TOL,
        "veh1 position error {max_pos_err_1} m > tol {GROUND_POS_TOL}"
    );
    assert!(
        max_pos_err_2 < GROUND_POS_TOL,
        "veh2 position error {max_pos_err_2} m > tol {GROUND_POS_TOL}"
    );
    assert!(
        max_vel_err_1 < GROUND_VEL_TOL,
        "veh1 velocity error {max_vel_err_1} m/s > tol {GROUND_VEL_TOL}"
    );
    assert!(
        max_vel_err_2 < GROUND_VEL_TOL,
        "veh2 velocity error {max_vel_err_2} m/s > tol {GROUND_VEL_TOL}"
    );
}
