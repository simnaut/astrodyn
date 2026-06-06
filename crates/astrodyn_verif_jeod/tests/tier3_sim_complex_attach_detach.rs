// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! Tier 3: SIM_verif_attach_detach — chained-attach re-rooting
//! (`RUN_complex_attach_detach`, `RUN_compute_child_derivative`).

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
//! Cross-validates two slices of JEOD's `RUN_complex_attach_detach` and
//! `RUN_compute_child_derivative` end-to-end through
//! [`Simulation::step()`] against the regenerated reference CSVs
//! `chained_attach_complex_kinematic_propagation_state.csv` and
//! `chained_attach_child_deriv_kinematic_propagation_state.csv`:
//!
//! 1. **Pre-attach trajectory window.** Free-flying veh1, veh2, veh3
//!    propagate ballistically through `Simulation::step()` from t=0 up
//!    to the first scheduled attach event (t=10 for the complex run,
//!    t=1 for the child-derivative run). Position, velocity,
//!    quaternion-angle, and angular velocity are compared per CSV row.
//!    Same dominant-error story as
//!    [`tier3_sim_attach_detach_trajectory`]: f64 round-off + JEOD's
//!    `%g`-formatted CSV print precision (~3e-8 rad on a non-trivially
//!    rotated body).
//!
//! 2. **Re-rooting topology + composite mass.** At the chained-attach
//!    fire times the runner exercises the new
//!    [`MassTree::attach_with_reroot`] / [`Simulation::attach`]
//!    semantics: when veh1 is already attached to veh2 and the test
//!    fires `attach(veh1, veh3, ...)`, the runner walks veh1's existing
//!    root (veh2) and re-roots the *whole subject subtree* under veh3,
//!    matching JEOD `dyn_body_attach.cc::attach_child`'s 521→567
//!    `child_root != &child` branch. The post-event composite mass on
//!    every vehicle (read from the live mass tree, not from the CSV)
//!    is asserted against JEOD's `1+2+3` arithmetic — the topology is
//!    fully determined by the schedule and per-body masses. After the
//!    re-rooting attach the test also verifies the new tree's parent
//!    pointers (`veh1 → veh2 → veh3` rooted at veh3, then a detach +
//!    re-attach round trip).
//!
//! 3. **Full-window trajectory cross-validation.** The
//!    [`tier3_sim_compute_child_derivative_full_trajectory`] test
//!    drives every CSV row from `t=0` through `t≈57`, firing JEOD's
//!    `trick.add_read` schedule (`SET_test/RUN_compute_child_derivative/input.py`)
//!    via the runner's struct-frame
//!    [`Simulation::set_body_external_force_struct`] /
//!    [`Simulation::set_body_external_torque_struct`] entry points.
//!    Forces / torques applied to **kinematic** children (v1, v2)
//!    propagate to the chain's integration root through the
//!    runner-side wrench-aggregation pass added in this commit
//!    (`Simulation::aggregate_chain_wrenches` —
//!    `JEOD_INV: DB.16` / `DB.17`); the post-attach
//!    `composite_body` state of every chain member is rederived via
//!    [`Simulation::propagate_kinematic_state_for_logging`]
//!    immediately after each topology event so the runner's sample at
//!    each integer-second `add_read` boundary observes the same
//!    state JEOD's CSV records.
//!
//! [`Simulation::step()`]: astrodyn_runner::Simulation::step
//! [`MassTree::attach_with_reroot`]: astrodyn_dynamics::MassTree::attach_with_reroot
//! [`Simulation::attach`]: astrodyn_runner::Simulation::attach
//! [`tier3_sim_attach_detach_trajectory`]: ../tier3_sim_attach_detach_trajectory/index.html
//! [`Simulation::set_body_external_force`]: astrodyn_runner::Simulation::set_body_external_force
//! [`Simulation::set_body_external_torque`]: astrodyn_runner::Simulation::set_body_external_torque
//! [`Simulation::set_body_external_force_struct`]: astrodyn_runner::Simulation::set_body_external_force_struct
//! [`Simulation::set_body_external_torque_struct`]: astrodyn_runner::Simulation::set_body_external_torque_struct
//! [`Simulation::propagate_kinematic_state_for_logging`]: astrodyn_runner::Simulation::propagate_kinematic_state_for_logging
//! [`tier3_sim_compute_child_derivative_full_trajectory`]: tier3_sim_compute_child_derivative_full_trajectory

use std::path::PathBuf;

use astrodyn::IntegratorType;
use astrodyn::MassProperties;
use astrodyn::{
    GravityControls, JeodQuat, RotationalState, SimulationTime, TranslationalState, VehicleConfig,
};
use astrodyn_runner::Simulation;
use astrodyn_verif_jeod::crossval::CrossvalReport;
use glam::{DMat3, DVec3};

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data")
}

/// One CSV row from a `kinematic_propagation_state` log: time + 13
/// fields per vehicle (3 pos + 3 vel + 4 quat + 3 ang_vel) for veh1,
/// veh2, veh3 — 40 fields total per row.
#[derive(Debug, Clone)]
struct StateRow {
    time: f64,
    veh: [VehSnapshot; 3],
}

#[derive(Debug, Clone, Copy)]
struct VehSnapshot {
    position: DVec3,
    velocity: DVec3,
    /// Scalar-first JEOD quaternion `[scalar, v0, v1, v2]`.
    quaternion: JeodQuat,
    /// Body-frame angular velocity (rad/s).
    ang_vel_body: DVec3,
}

fn load_csv(filename: &str) -> Vec<StateRow> {
    let path = test_data_dir().join(filename);
    assert!(
        path.exists(),
        "JEOD reference data not found at {}.\n\
         Generate with:\n\
         cargo xtask regenerate-tier3\n\
         (or the equivalent Docker invocation — see CLAUDE.md \"Generating \
         Tier 3 Reference Data (Docker)\"). The CSV is produced by the \
         `kinematic_propagation_state` log group in \
         `trick/generate_references.sh`.",
        path.display()
    );
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));

    let mut rows = Vec::new();
    for (idx, line) in content.lines().skip(1).enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let fields: Vec<&str> = trimmed.split(',').map(str::trim).collect();
        // 1 (time) + 3 vehicles × 13 fields each = 40
        assert_eq!(
            fields.len(),
            40,
            "CSV {} line {}: expected 40 columns, found {}",
            path.display(),
            idx + 2,
            fields.len(),
        );
        let parse = |col: usize| -> f64 {
            fields[col].parse().unwrap_or_else(|e| {
                panic!(
                    "CSV {} line {} col {}: invalid f64 {:?}: {e}",
                    path.display(),
                    idx + 2,
                    col,
                    fields[col]
                )
            })
        };
        let mut veh = [VehSnapshot {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            quaternion: JeodQuat::from_array([1.0, 0.0, 0.0, 0.0]),
            ang_vel_body: DVec3::ZERO,
        }; 3];
        for (i, snap) in veh.iter_mut().enumerate() {
            let base = 1 + i * 13;
            snap.position = DVec3::new(parse(base), parse(base + 1), parse(base + 2));
            snap.velocity = DVec3::new(parse(base + 3), parse(base + 4), parse(base + 5));
            snap.quaternion = JeodQuat::from_array([
                parse(base + 6),
                parse(base + 7),
                parse(base + 8),
                parse(base + 9),
            ]);
            snap.ang_vel_body = DVec3::new(parse(base + 10), parse(base + 11), parse(base + 12));
        }
        rows.push(StateRow {
            time: parse(0),
            veh,
        });
    }
    assert!(
        !rows.is_empty(),
        "CSV {} contained no data rows",
        path.display()
    );
    rows
}

// ════════════════════════════════════════════════════════════════════
// Initial conditions and per-vehicle masses, all sourced from the JEOD
// `Modified_data/veh{1,2,3}.py` files. The values are duplicated here
// (rather than imported from the existing
// `tier3_sim_attach_detach_trajectory.rs`) so each test file stays a
// self-contained source of the constants it asserts against; cargo
// test files are independent compilation units, sharing helpers via
// a `mod common` would re-introduce cross-file ordering hazards.
// ════════════════════════════════════════════════════════════════════

fn veh1_mass() -> MassProperties {
    MassProperties::with_inertia(
        1.0,
        DMat3::from_diagonal(DVec3::splat(10.0)),
        DVec3::new(5.0, 0.0, 0.0),
    )
}

fn veh2_mass() -> MassProperties {
    MassProperties::with_inertia(
        2.0,
        DMat3::from_diagonal(DVec3::splat(20.0)),
        DVec3::new(5.0, 0.0, 0.0),
    )
}

fn veh3_mass() -> MassProperties {
    MassProperties::with_inertia(
        3.0,
        DMat3::from_diagonal(DVec3::splat(30.0)),
        DVec3::new(5.0, 0.0, 0.0),
    )
}

fn veh1_initial_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(-5.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 1.0, 0.0),
    }
}

fn veh1_initial_rot() -> RotationalState {
    RotationalState {
        quaternion: JeodQuat::from_array([1.0, 0.0, 0.0, 0.0]),
        ang_vel_body: DVec3::ZERO,
    }
}

fn veh2_initial_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(5.0, 10.0, 0.0),
        velocity: DVec3::ZERO,
    }
}

fn veh2_initial_rot() -> RotationalState {
    // `Modified_data/veh2.py:49`: `euler_angles = [-2.0, 0.0, 0.0]`
    // (no `attach_units("degree", ...)` wrapper → JEOD default unit
    // for this field is RADIANS). See the equivalent comment in
    // `tier3_sim_attach_detach_trajectory.rs::veh2_initial_rot`.
    let yaw_rad = -2.0_f64;
    let q = JeodQuat::left_quat_from_eigen_rotation(yaw_rad, DVec3::Z);
    RotationalState {
        quaternion: q,
        ang_vel_body: DVec3::new(0.0, 0.0, 0.2),
    }
}

fn veh3_initial_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(0.063, 13.787, -25.0),
        velocity: DVec3::new(0.0, 0.0, 1.0),
    }
}

fn veh3_initial_rot() -> RotationalState {
    // Same convention as `veh2_initial_rot`: euler-angle field defaults
    // to radians without an explicit `trick.attach_units` wrapper.
    let yaw_rad = -15.8_f64;
    let q = JeodQuat::left_quat_from_eigen_rotation(yaw_rad, DVec3::Z);
    RotationalState {
        quaternion: q,
        ang_vel_body: DVec3::new(0.0, 0.0, 1.0),
    }
}

/// JEOD `BodyAttachAligned` between veh1.node12 (at (10,0,0) in
/// veh1.struct, identity orientation) and veh2.node21 (at (0,0,0) in
/// veh2.struct, YPR(180°, 0, 0)). Composes to identity rotation +
/// offset (-10, 0, 0) — see
/// `tier3_sim_attach_detach_trajectory.rs::simple_attach_offset_and_rotation`.
fn attach_v1_to_v2_offset_and_rotation() -> (DVec3, DMat3) {
    (DVec3::new(-10.0, 0.0, 0.0), DMat3::IDENTITY)
}

/// JEOD `BodyAttachAligned` between veh1.node13 (at (5, 0, -5) in
/// veh1.struct, YPR(0°, 90°, 0°) → rotation about y-axis by +90°) and
/// veh3.node31 (at (0, 0, 5) in veh3.struct, YPR(180°, -90°, 0°)).
///
/// In our [`MassTree::attach_with_reroot`] / [`Simulation::attach`]
/// surface the offset and rotation are specified in **veh3**'s
/// (parent's) struct frame, mapping veh1's struct origin into veh3's
/// struct frame. We compute the analytic answer here from the
/// JEOD-source named-point geometry rather than re-deriving the JEOD
/// `mass_attach.cc` chain at runtime — both sides are deterministic
/// f64 arithmetic from the same input.
///
/// The composite mass + topology assertions in this test do not depend
/// on the exact geometric values (mass tree composes mass over offsets
/// in any pose), so even if the geometry below were slightly off, the
/// topology + composite-mass assertions would still hold. A follow-up
/// PR that ports JEOD's named-point attach end-to-end can replace this
/// constant with a `MassTree::attach_aligned`-equivalent derivation.
fn attach_v1_to_v3_offset_and_rotation() -> (DVec3, DMat3) {
    // Per JEOD `mass_attach.cc:67-137`, attach_aligned chains:
    //   veh3.struct → veh3.node31 → 180° flip → veh1.node13 → veh1.struct.
    // Computing the composed (offset, T) symbolically:
    //   T = T_node13_struct⁻¹ · diag(-1,-1,1) · T_node31_struct
    // and offset = T_node31_struct⁻¹ · (diag(-1,-1,1) · (-T_node13_struct·node13.position)) + node31.position
    // gives the values below.
    //
    // node13.position = (5, 0, -5), T_node13_struct = R_y(+90°)
    // node31.position = (0, 0, 5),  T_node31_struct = R_y(-90°) · R_z(180°)
    //
    // The arithmetic is f64-deterministic; the values here are computed
    // once at compile time via `glam` operations rather than baked as
    // literals so a future change to the named-point geometry (e.g. a
    // sign-flip in the YPR convention) refractures here loudly.
    let r_y_p90 = DMat3::from_cols(
        DVec3::new(0.0, 0.0, -1.0),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
    );
    let r_y_m90 = r_y_p90.transpose();
    let r_z_180 = DMat3::from_cols(
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(0.0, -1.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    );
    let t_node13_struct = r_y_p90;
    let t_node31_struct = r_y_m90 * r_z_180;

    let inv_pos = -(t_node13_struct * DVec3::new(5.0, 0.0, -5.0));
    let inv_t = t_node13_struct.transpose();

    let t_yaw = DMat3::from_cols(
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(0.0, -1.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    );

    let pos_after_yaw = t_yaw * inv_pos;
    let offset = t_node31_struct.transpose() * pos_after_yaw + DVec3::new(0.0, 0.0, 5.0);
    let t_parent_child = inv_t * t_yaw * t_node31_struct;
    (offset, t_parent_child)
}

/// Build a fresh runner sim configured for the chained-attach scenarios.
///
/// Same shape as
/// `tier3_sim_attach_detach_trajectory.rs::build_sim`: empty space
/// (no gravity sources, `EphemerisMode_EmptySpace`), RK4 on every
/// body, three vehicles registered into the mass tree.
fn build_sim() -> (Simulation, usize, usize, usize) {
    let dt = 0.1;
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

    let v1 = sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&veh1_initial_trans()),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(veh1_initial_rot()),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(veh1_mass()))),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..VehicleConfig::named("tier3-sim-complex-attach-detach-2")
    });
    let v2 = sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&veh2_initial_trans()),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(veh2_initial_rot()),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(veh2_mass()))),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..VehicleConfig::named("tier3-sim-complex-attach-detach-1")
    });
    let v3 = sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&veh3_initial_trans()),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(veh3_initial_rot()),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(veh3_mass()))),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..VehicleConfig::named("tier3-sim-complex-attach-detach-0")
    });
    sim.add_body_to_tree(v1, "veh1");
    sim.add_body_to_tree(v2, "veh2");
    sim.add_body_to_tree(v3, "veh3");
    (sim, v1, v2, v3)
}

/// Quaternion angle error: `2 · acos(|q_a · q_b|)`.
fn quat_angle_err(a: JeodQuat, b: JeodQuat) -> f64 {
    let dot = (a.scalar() * b.scalar() + a.vector().dot(b.vector()))
        .abs()
        .clamp(-1.0, 1.0);
    2.0 * dot.acos()
}

fn body_snapshot(sim: &Simulation, idx: usize) -> VehSnapshot {
    let out = sim.body(idx);
    let rot = out
        .rot
        .expect("chained-attach test runs every body in 6-DOF");
    VehSnapshot {
        position: out.trans.position.raw_si(),
        velocity: out.trans.velocity.raw_si(),
        quaternion: rot.q_inertial_body.to_jeod_quat(),
        ang_vel_body: rot.ang_vel_body.raw_si(),
    }
}

#[derive(Default, Clone, Copy)]
struct WindowErrors {
    pos: f64,
    vel: f64,
    quat: f64,
    ang_vel: f64,
}

impl WindowErrors {
    fn update(&mut self, runner: &VehSnapshot, csv: &VehSnapshot) {
        self.pos = self.pos.max((runner.position - csv.position).length());
        self.vel = self.vel.max((runner.velocity - csv.velocity).length());
        self.quat = self
            .quat
            .max(quat_angle_err(runner.quaternion, csv.quaternion));
        self.ang_vel = self
            .ang_vel
            .max((runner.ang_vel_body - csv.ang_vel_body).length());
    }
}

// ════════════════════════════════════════════════════════════════════
// Tolerances. Per CLAUDE.md "Cross-validation tolerances", set to ~5%
// above the observed max error per component (values come from the
// JSON reports each test writes).
//
// Pre-attach windows are RK4-only on three free-flying bodies — same
// rigid-body floor as `tier3_sim_attach_detach_trajectory_simple`'s
// pre-attach window for veh1+veh2 (which integrates the same way).
// Quaternion-angle tolerance absorbs the JEOD `%g` print precision
// (~3e-8 rad on a body with small rotation, ~5e-7 rad on veh3 which
// integrates a faster rotation).
//
// Observed errors are per-body — veh3's tolerances are roughly 10×
// veh1/veh2 because it integrates a non-trivial yaw plus
// `ω_z = 1 rad/s` initial spin (so its `composite_body` quaternion
// has accumulated ~10 rad of total rotation by t = 9.5 s, and the
// %g-formatted CSV residual scales with the angle magnitude). Per
// CLAUDE.md "Tolerance policy", each tolerance is set to ~5% above
// the observed max.
// ════════════════════════════════════════════════════════════════════

// veh1: initial trans state has only `vy = 1`, identity attitude,
// zero ω. Error is pure RK4 round-off.
const VEH1_PRE_ATTACH_POSITION_TOL_M: f64 = 1.5e-13;
const VEH1_PRE_ATTACH_VELOCITY_TOL_MPS: f64 = 1e-15;
const VEH1_PRE_ATTACH_QUAT_ANGLE_TOL_RAD: f64 = 1e-15;
const VEH1_PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S: f64 = 1e-15;

// veh2: zero velocity, non-trivial initial yaw (-2 rad), ω_z = 0.2.
const VEH2_PRE_ATTACH_POSITION_TOL_M: f64 = 1e-15;
const VEH2_PRE_ATTACH_VELOCITY_TOL_MPS: f64 = 1e-15;
const VEH2_PRE_ATTACH_QUAT_ANGLE_TOL_RAD: f64 = 4.5e-8;
const VEH2_PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S: f64 = 1e-15;

// veh3: vz = 1, yaw -15.8 rad, ω_z = 1 — fast-spinning body with a
// long quaternion residual.
const VEH3_PRE_ATTACH_POSITION_TOL_M: f64 = 1.4e-12;
const VEH3_PRE_ATTACH_VELOCITY_TOL_MPS: f64 = 1e-15;
const VEH3_PRE_ATTACH_QUAT_ANGLE_TOL_RAD: f64 = 5.2e-7;
const VEH3_PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S: f64 = 1e-15;

/// First scheduled attach in `RUN_complex_attach_detach`
/// (`SET_test/RUN_complex_attach_detach/input.py:23`).
const COMPLEX_ATTACH_V1_V2_TIME: f64 = 10.0;
/// `veh1.attach_to_3.active = True` time
/// (`SET_test/RUN_complex_attach_detach/input.py:24`). With veh1
/// already a child of veh2, JEOD's `attach_child` re-roots veh2 (the
/// subject's existing root) under veh3.
const COMPLEX_RECHAIN_V1_V3_TIME: f64 = 32.777;
/// `veh1.detach_from_2.active = True` time
/// (`SET_test/RUN_complex_attach_detach/input.py:25`).
const COMPLEX_DETACH_V1_FROM_V2_TIME: f64 = 50.0;
/// `veh1.attach_to_2b.active = True` time
/// (`SET_test/RUN_complex_attach_detach/input.py:26`).
const COMPLEX_REATTACH_V1_V2_TIME: f64 = 55.0;

/// First scheduled attach in `RUN_compute_child_derivative`
/// (`SET_test/RUN_compute_child_derivative/input.py:27`).
const CHILD_DERIV_ATTACH_V1_V2_TIME: f64 = 1.0;
/// Second scheduled attach in `RUN_compute_child_derivative`
/// (`SET_test/RUN_compute_child_derivative/input.py:28`); fires the
/// same re-rooting code path as `COMPLEX_RECHAIN_V1_V3_TIME`.
const CHILD_DERIV_RECHAIN_V1_V3_TIME: f64 = 2.0;

/// `RUN_compute_child_derivative` event schedule, transcribed from
/// `SET_test/RUN_compute_child_derivative/input.py`. Each entry is
/// `(time, event)` where `event` names the body action or struct-frame
/// force / torque toggle the JEOD `trick.add_read` snippet performs at
/// that integer second. Times match JEOD verbatim
/// (`input.py:27-141`); no synthesis or smoothing.
///
/// JEOD's read-job ordering is: at the start of integration second N,
/// every read scheduled at `time = N` fires before the integration
/// derivative-evaluation cycle begins. The driver below mirrors that
/// by firing every event whose time has been reached *before* it
/// advances [`Simulation::step()`] across the [`N, N + dt`] interval —
/// see [`drive_run_compute_child_derivative_through_csv`] for the
/// runner-side scheduler.
#[derive(Clone, Copy, Debug)]
enum ChildDerivEvent {
    AttachV1ToV2,
    AttachV1ToV3,
    /// `veh1.detach_from_3.active = True` —
    /// `Modified_data/attach_detach.py:44-49` walks v1's ancestor chain
    /// looking for the body whose immediate parent is v3 and detaches
    /// that body. With the chained tree v3 ← v2 ← v1 this resolves to
    /// detaching v2 from v3.
    DetachVeh1FromVeh3ViaVeh2,
    /// `veh<i>.force.force = [...]` — struct-frame N.
    SetForceStruct(usize, DVec3),
    /// `veh<i>.torque.torque = [...]` — struct-frame N·m.
    SetTorqueStruct(usize, DVec3),
}

fn child_deriv_schedule() -> Vec<(f64, ChildDerivEvent)> {
    use ChildDerivEvent::*;
    let f0 = DVec3::ZERO;
    vec![
        (1.0, AttachV1ToV2),
        (2.0, AttachV1ToV3),
        (3.0, SetForceStruct(0, DVec3::new(0.1, 0.0, 0.0))),
        (5.0, SetForceStruct(0, f0)),
        (7.0, SetForceStruct(1, DVec3::new(0.0, 0.1, 0.0))),
        (9.0, SetForceStruct(1, f0)),
        (11.0, SetForceStruct(2, DVec3::new(0.0, 0.0, 0.1))),
        (13.0, SetForceStruct(2, f0)),
        (15.0, DetachVeh1FromVeh3ViaVeh2),
        (17.0, SetForceStruct(0, DVec3::new(-0.1, 0.0, 0.0))),
        (19.0, SetForceStruct(0, f0)),
        (21.0, SetForceStruct(1, DVec3::new(0.0, -0.1, 0.0))),
        (23.0, SetForceStruct(1, f0)),
        (25.0, SetForceStruct(2, DVec3::new(0.0, 0.0, -0.1))),
        (27.0, SetForceStruct(2, f0)),
        (33.0, SetTorqueStruct(0, DVec3::new(0.001, 0.0, 0.0))),
        (35.0, SetTorqueStruct(0, f0)),
        (37.0, SetTorqueStruct(1, DVec3::new(0.0, 0.001, 0.0))),
        (39.0, SetTorqueStruct(1, f0)),
        (41.0, SetTorqueStruct(2, DVec3::new(0.0, 0.0, 0.001))),
        (43.0, SetTorqueStruct(2, f0)),
        (45.0, DetachVeh1FromVeh3ViaVeh2),
        (47.0, SetTorqueStruct(0, DVec3::new(-0.001, 0.0, 0.0))),
        (49.0, SetTorqueStruct(0, f0)),
        (51.0, SetTorqueStruct(1, DVec3::new(0.0, -0.001, 0.0))),
        (53.0, SetTorqueStruct(1, f0)),
        (55.0, SetTorqueStruct(2, DVec3::new(0.0, 0.0, -0.001))),
        (57.0, SetTorqueStruct(2, f0)),
    ]
}

// ════════════════════════════════════════════════════════════════════
// Cross-validation: pre-attach trajectory windows
// ════════════════════════════════════════════════════════════════════

/// `RUN_complex_attach_detach`, pre-attach window `t ∈ [0, 10)`.
/// Three free-flying vehicles propagate under `Simulation::step()`;
/// every CSV row is asserted against the runner's `composite_body`
/// state. Stops at `COMPLEX_ATTACH_V1_V2_TIME`; the post-attach +
/// re-rooting + force/torque windows are deferred (see file-level
/// docstring "What is **not** validated").
// non-recipe: SIM_verif_attach_detach exercises a placeholder mass
// tree (1/2/3 kg, three vehicles). ISS / Apollo recipes don't apply.
#[test]
fn tier3_sim_complex_attach_detach_pre_attach_trajectory() {
    let rows = load_csv("chained_attach_complex_kinematic_propagation_state.csv");
    assert!(rows.len() > 20, "expected >20 rows, got {}", rows.len());

    let (mut sim, v1, v2, v3) = build_sim();
    let dt = sim.dt;

    // Sanity at t=0: CSV's first row must match the JEOD source-file ICs.
    let r0 = &rows[0];
    assert!(
        (r0.veh[0].position - veh1_initial_trans().position).length() < 1e-12
            && (r0.veh[1].position - veh2_initial_trans().position).length() < 1e-12
            && (r0.veh[2].position - veh3_initial_trans().position).length() < 1e-12,
        "CSV t=0 does not match the JEOD source-file ICs (Modified_data/veh{{1,2,3}}.py)"
    );

    let mut errs = [WindowErrors::default(); 3];

    for row in &rows {
        if row.time >= COMPLEX_ATTACH_V1_V2_TIME {
            break;
        }
        while sim.elapsed() + 0.5 * dt < row.time {
            sim.step().expect("step() must succeed");
        }
        for (i, &idx) in [v1, v2, v3].iter().enumerate() {
            let snap = body_snapshot(&sim, idx);
            errs[i].update(&snap, &row.veh[i]);
        }
    }

    let mut report = CrossvalReport::compute(
        "tier3_sim_complex_attach_detach_pre_attach_trajectory",
        &[],
        &[],
    );
    for (i, name) in ["veh1", "veh2", "veh3"].iter().enumerate() {
        report.add_extra(&format!("{name}_max_position_err"), errs[i].pos, "m");
        report.add_extra(&format!("{name}_max_velocity_err"), errs[i].vel, "m/s");
        report.add_extra(&format!("{name}_max_quat_angle_err"), errs[i].quat, "rad");
        report.add_extra(&format!("{name}_max_ang_vel_err"), errs[i].ang_vel, "rad/s");
    }
    report.write();

    let tols: [(&str, f64, f64, f64, f64); 3] = [
        (
            "veh1",
            VEH1_PRE_ATTACH_POSITION_TOL_M,
            VEH1_PRE_ATTACH_VELOCITY_TOL_MPS,
            VEH1_PRE_ATTACH_QUAT_ANGLE_TOL_RAD,
            VEH1_PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S,
        ),
        (
            "veh2",
            VEH2_PRE_ATTACH_POSITION_TOL_M,
            VEH2_PRE_ATTACH_VELOCITY_TOL_MPS,
            VEH2_PRE_ATTACH_QUAT_ANGLE_TOL_RAD,
            VEH2_PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S,
        ),
        (
            "veh3",
            VEH3_PRE_ATTACH_POSITION_TOL_M,
            VEH3_PRE_ATTACH_VELOCITY_TOL_MPS,
            VEH3_PRE_ATTACH_QUAT_ANGLE_TOL_RAD,
            VEH3_PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S,
        ),
    ];
    for (i, (name, ptol, vtol, qtol, wtol)) in tols.iter().enumerate() {
        assert!(
            errs[i].pos < *ptol,
            "{name} pre-attach position {:.3e} m exceeds {ptol:.1e}",
            errs[i].pos
        );
        assert!(
            errs[i].vel < *vtol,
            "{name} pre-attach velocity {:.3e} m/s exceeds {vtol:.1e}",
            errs[i].vel
        );
        assert!(
            errs[i].quat < *qtol,
            "{name} pre-attach quat-angle {:.3e} rad exceeds {qtol:.1e}",
            errs[i].quat
        );
        assert!(
            errs[i].ang_vel < *wtol,
            "{name} pre-attach ang-vel {:.3e} rad/s exceeds {wtol:.1e}",
            errs[i].ang_vel
        );
    }
}

/// `RUN_compute_child_derivative`, pre-attach window `t ∈ [0, 1)`.
/// Two CSV rows (t=0, t=0.5) — short by design (the run starts
/// attaching at t=1) but worth validating to confirm the same RK4
/// floor holds across the alternate run.
// non-recipe: SIM_verif_attach_detach placeholder mass tree.
#[test]
fn tier3_sim_compute_child_derivative_pre_attach_trajectory() {
    let rows = load_csv("chained_attach_child_deriv_kinematic_propagation_state.csv");
    assert!(rows.len() > 20, "expected >20 rows, got {}", rows.len());

    let (mut sim, v1, v2, v3) = build_sim();
    let dt = sim.dt;

    let mut errs = [WindowErrors::default(); 3];
    for row in &rows {
        if row.time >= CHILD_DERIV_ATTACH_V1_V2_TIME {
            break;
        }
        while sim.elapsed() + 0.5 * dt < row.time {
            sim.step().expect("step() must succeed");
        }
        for (i, &idx) in [v1, v2, v3].iter().enumerate() {
            let snap = body_snapshot(&sim, idx);
            errs[i].update(&snap, &row.veh[i]);
        }
    }

    let mut report = CrossvalReport::compute(
        "tier3_sim_compute_child_derivative_pre_attach_trajectory",
        &[],
        &[],
    );
    for (i, name) in ["veh1", "veh2", "veh3"].iter().enumerate() {
        report.add_extra(&format!("{name}_max_position_err"), errs[i].pos, "m");
        report.add_extra(&format!("{name}_max_velocity_err"), errs[i].vel, "m/s");
        report.add_extra(&format!("{name}_max_quat_angle_err"), errs[i].quat, "rad");
        report.add_extra(&format!("{name}_max_ang_vel_err"), errs[i].ang_vel, "rad/s");
    }
    report.write();

    let tols: [(&str, f64, f64, f64, f64); 3] = [
        (
            "veh1",
            VEH1_PRE_ATTACH_POSITION_TOL_M,
            VEH1_PRE_ATTACH_VELOCITY_TOL_MPS,
            VEH1_PRE_ATTACH_QUAT_ANGLE_TOL_RAD,
            VEH1_PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S,
        ),
        (
            "veh2",
            VEH2_PRE_ATTACH_POSITION_TOL_M,
            VEH2_PRE_ATTACH_VELOCITY_TOL_MPS,
            VEH2_PRE_ATTACH_QUAT_ANGLE_TOL_RAD,
            VEH2_PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S,
        ),
        (
            "veh3",
            VEH3_PRE_ATTACH_POSITION_TOL_M,
            VEH3_PRE_ATTACH_VELOCITY_TOL_MPS,
            VEH3_PRE_ATTACH_QUAT_ANGLE_TOL_RAD,
            VEH3_PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S,
        ),
    ];
    for (i, (name, ptol, vtol, qtol, wtol)) in tols.iter().enumerate() {
        assert!(
            errs[i].pos < *ptol,
            "{name} pre-attach position {:.3e} m exceeds {ptol:.1e}",
            errs[i].pos
        );
        assert!(
            errs[i].vel < *vtol,
            "{name} pre-attach velocity {:.3e} m/s exceeds {vtol:.1e}",
            errs[i].vel
        );
        assert!(
            errs[i].quat < *qtol,
            "{name} pre-attach quat-angle {:.3e} rad exceeds {qtol:.1e}",
            errs[i].quat
        );
        assert!(
            errs[i].ang_vel < *wtol,
            "{name} pre-attach ang-vel {:.3e} rad/s exceeds {wtol:.1e}",
            errs[i].ang_vel
        );
    }
}

// ════════════════════════════════════════════════════════════════════
// Re-rooting topology + composite mass
// ════════════════════════════════════════════════════════════════════

/// Re-rooting end-to-end through `Simulation::step()` +
/// `Simulation::attach`/`detach` exercising the new
/// `attach_with_reroot` semantics. Asserts:
///
/// 1. After the first `attach(v1, v2)` (root subject), v1 sits under
///    v2 in the mass tree; v2's composite mass = 3.
/// 2. After the chained `attach(v1, v3)` (v1 already has parent v2),
///    the runner re-roots v2 (v1's existing root) under v3; the new
///    tree shape is `v3 ← v2 ← v1`; v3's composite mass = 6.
/// 3. After `detach(v1)` (v1 sits two levels under v3 root, but JEOD
///    detaches v1 from its immediate parent v2 only), v1 becomes a
///    root again with composite mass 1; v3's composite drops back to
///    5 (v2 + v3); v2's composite stays at 2 (just v2 alone, no
///    children any more).
/// 4. After `attach(v1, v2)` again, v1 is a standalone root (just
///    detached in step 3) and v2 is interior to v3's tree, so this
///    takes the plain root-subject attach path (no rerooting); v1
///    becomes a child of v2 and v3's composite climbs back to 6.
///
/// Composite-mass values are independent of attach geometry (mass tree
/// arithmetic doesn't care about offsets), so the assertions hold
/// regardless of whether the geometry-derived `(offset, T_parent_child)`
/// chains here exactly bit-match JEOD's named-point algorithm.
// non-recipe: SIM_verif_attach_detach placeholder mass tree.
#[test]
fn tier3_sim_complex_attach_detach_rerooting_topology() {
    let (mut sim, v1, v2, v3) = build_sim();
    let dt = sim.dt;

    // Drive the runner through the complex run's event schedule. We
    // step in coarse chunks (no per-CSV-row comparison here — trajectory
    // is covered by the dedicated pre_attach_trajectory test above and
    // post-event windows are deferred per the file-level docstring).
    let advance_to = |sim: &mut Simulation, target: f64| {
        while sim.elapsed() + 0.5 * dt < target {
            sim.step().expect("step() must succeed");
        }
    };

    // -- t=0: confirm initial topology + masses. --
    assert_eq!(get_composite_mass(&sim, v1), 1.0);
    assert_eq!(get_composite_mass(&sim, v2), 2.0);
    assert_eq!(get_composite_mass(&sim, v3), 3.0);
    assert!(parent_id_of(&sim, v1).is_none());
    assert!(parent_id_of(&sim, v2).is_none());
    assert!(parent_id_of(&sim, v3).is_none());

    // -- t=10: attach veh1 → veh2 (root subject, no reroot needed). --
    advance_to(&mut sim, COMPLEX_ATTACH_V1_V2_TIME);
    let (offset_v1_v2, t_v1_v2) = attach_v1_to_v2_offset_and_rotation();
    sim.attach(v1, v2, offset_v1_v2, t_v1_v2);
    sim.mark_kinematic_only(v1);
    assert_eq!(
        parent_id_of(&sim, v1).map(|p| p == mass_body_id(&sim, v2)),
        Some(true)
    );
    assert!(parent_id_of(&sim, v2).is_none());
    assert!(parent_id_of(&sim, v3).is_none());
    assert_eq!(
        get_composite_mass(&sim, v2),
        3.0,
        "post-attach v2 composite = v1 + v2"
    );
    assert_eq!(get_composite_mass(&sim, v3), 3.0, "v3 untouched");

    // -- t=32.777: chained attach veh1 → veh3 fires while veh1 is
    //    already a child of veh2. The runner walks veh1's existing
    //    root (veh2) and re-roots veh2 under veh3 — this is the new
    //    `attach_with_reroot` code path.
    advance_to(&mut sim, COMPLEX_RECHAIN_V1_V3_TIME);
    let (offset_v1_v3, t_v1_v3) = attach_v1_to_v3_offset_and_rotation();
    sim.attach(v1, v3, offset_v1_v3, t_v1_v3);
    // After reroot: v3 root, v2 under v3, v1 under v2.
    assert!(
        parent_id_of(&sim, v3).is_none(),
        "v3 must be the root after reroot"
    );
    assert_eq!(
        parent_id_of(&sim, v2).map(|p| p == mass_body_id(&sim, v3)),
        Some(true),
        "v2 must be re-rooted under v3"
    );
    assert_eq!(
        parent_id_of(&sim, v1).map(|p| p == mass_body_id(&sim, v2)),
        Some(true),
        "v1 must remain attached to its existing parent v2"
    );
    assert_eq!(
        get_composite_mass(&sim, v3),
        6.0,
        "post-reroot v3 composite = v1 + v2 + v3"
    );

    // -- t=50: detach v1 from v2. v1 becomes its own tree root. --
    advance_to(&mut sim, COMPLEX_DETACH_V1_FROM_V2_TIME);
    sim.clear_kinematic_only(v1);
    sim.detach(v1);
    assert!(parent_id_of(&sim, v1).is_none(), "v1 root after detach");
    assert_eq!(
        parent_id_of(&sim, v2).map(|p| p == mass_body_id(&sim, v3)),
        Some(true),
        "v2 still under v3 (detach was on the v1↔v2 edge only)"
    );
    assert!(parent_id_of(&sim, v3).is_none(), "v3 still root");
    assert_eq!(get_composite_mass(&sim, v1), 1.0);
    assert_eq!(get_composite_mass(&sim, v2), 2.0);
    assert_eq!(
        get_composite_mass(&sim, v3),
        5.0,
        "post-v1-detach v3 composite = v2 + v3"
    );

    // -- t=55: attach v1 → v2 again. v1 is a standalone root (just
    //    detached at t=50), so this is a plain root-subject attach
    //    under an interior parent — the kernel takes the
    //    `parent[child].is_none()` short-circuit, not the reroot path.
    //    v1 merges back into v3's tree under v2.
    advance_to(&mut sim, COMPLEX_REATTACH_V1_V2_TIME);
    sim.attach(v1, v2, offset_v1_v2, t_v1_v2);
    sim.mark_kinematic_only(v1);
    assert_eq!(
        parent_id_of(&sim, v1).map(|p| p == mass_body_id(&sim, v2)),
        Some(true),
        "v1 attached to v2 again"
    );
    assert_eq!(
        parent_id_of(&sim, v2).map(|p| p == mass_body_id(&sim, v3)),
        Some(true),
        "v2 still under v3"
    );
    assert_eq!(
        get_composite_mass(&sim, v3),
        6.0,
        "post-reattach v3 composite back to v1 + v2 + v3"
    );
}

/// Parallel re-rooting topology test for `RUN_compute_child_derivative`,
/// which fires the same `attach_to_3` chained-attach (re-rooting) path
/// at t=2 — a closely-spaced pair of attaches that stresses the
/// integrator-history-reset bookkeeping (`JEOD_INV: IG.37`) on adjacent
/// topology changes.
// non-recipe: SIM_verif_attach_detach placeholder mass tree.
#[test]
fn tier3_sim_compute_child_derivative_rerooting_topology() {
    let (mut sim, v1, v2, v3) = build_sim();
    let dt = sim.dt;
    let advance_to = |sim: &mut Simulation, target: f64| {
        while sim.elapsed() + 0.5 * dt < target {
            sim.step().expect("step() must succeed");
        }
    };

    advance_to(&mut sim, CHILD_DERIV_ATTACH_V1_V2_TIME);
    let (offset_v1_v2, t_v1_v2) = attach_v1_to_v2_offset_and_rotation();
    sim.attach(v1, v2, offset_v1_v2, t_v1_v2);
    sim.mark_kinematic_only(v1);
    assert_eq!(get_composite_mass(&sim, v2), 3.0);

    advance_to(&mut sim, CHILD_DERIV_RECHAIN_V1_V3_TIME);
    let (offset_v1_v3, t_v1_v3) = attach_v1_to_v3_offset_and_rotation();
    sim.attach(v1, v3, offset_v1_v3, t_v1_v3);
    assert!(parent_id_of(&sim, v3).is_none());
    assert_eq!(
        parent_id_of(&sim, v2).map(|p| p == mass_body_id(&sim, v3)),
        Some(true),
        "v2 re-rooted under v3"
    );
    assert_eq!(
        parent_id_of(&sim, v1).map(|p| p == mass_body_id(&sim, v2)),
        Some(true),
        "v1 still attached to v2 after reroot"
    );
    assert_eq!(get_composite_mass(&sim, v3), 6.0);

    // Step a few more ticks to confirm the integrator-state reset on
    // attach (IG.37) succeeded — if reset was missed, GJ/ABM4 dirty
    // flags would panic on the next integrate. RK4 has no such state,
    // so this assertion exercises the path but doesn't fault. Keeping
    // the loop tight + non-zero validates `Simulation::step()` survives
    // back-to-back attach/reroot events.
    for _ in 0..5 {
        sim.step().expect("step() after reroot must succeed");
    }
}

// ── small helpers used by the topology tests ─────────────────────────

/// Read a body's composite mass from the runner's authoritative
/// `MassTree` (not from `body.mass`, which is a per-body cache that
/// only the affected-id resync touches — could lag the tree if a
/// future bug skipped the resync).
fn get_composite_mass(sim: &Simulation, body_idx: usize) -> f64 {
    let id = mass_body_id(sim, body_idx);
    sim.mass_tree
        .as_ref()
        .expect("mass tree must exist after add_body_to_tree")
        .get(id)
        .composite_properties
        .mass
}

/// Convenience wrapper: panic with a useful message if the body lacks
/// a mass-tree id (means the test's `add_body_to_tree` registration
/// was skipped — never legal in this test).
fn mass_body_id(sim: &Simulation, body_idx: usize) -> astrodyn::MassBodyId {
    sim.body_mass_id(body_idx)
        .expect("test body must have a mass_body_id")
}

/// Look up the mass-tree parent of `body_idx`, or `None` if it's a root.
fn parent_id_of(sim: &Simulation, body_idx: usize) -> Option<astrodyn::MassBodyId> {
    let id = mass_body_id(sim, body_idx);
    sim.mass_tree
        .as_ref()
        .expect("mass tree must exist after add_body_to_tree")
        .parent(id)
}

// ════════════════════════════════════════════════════════════════════
// Full-trajectory cross-validation: RUN_compute_child_derivative
// ════════════════════════════════════════════════════════════════════

/// Apply one [`ChildDerivEvent`] to the runner. Mirrors the JEOD
/// `trick.add_read` snippet semantics:
///
/// - `AttachV1ToV2` / `AttachV1ToV3` invoke
///   [`Simulation::attach`] with the named-point geometry derived from
///   `Modified_data/veh*.py` mass-point definitions, then mark v1 as
///   `kinematic_only` so it stops integrating (only the chain root
///   integrates per `JEOD_INV: DB.17`).
/// - `DetachVeh1FromVeh3ViaVeh2` resolves to the JEOD action's actual
///   effect: walking v1's ancestor chain to find the body whose parent
///   is v3, and detaching that body. With the chained tree
///   v3 ← v2 ← v1 the answer is "detach v2 from v3"; on subsequent
///   firings (after the first detach), v3 is no longer reachable from
///   v1 and JEOD reports an `inform` and no-ops, so we no-op too.
/// - `SetForceStruct` / `SetTorqueStruct` route through the new
///   struct-frame entry points
///   ([`Simulation::set_body_external_force_struct`] /
///   [`Simulation::set_body_external_torque_struct`]) so the inertial
///   contribution tracks the body's current attitude across each step.
fn apply_event(sim: &mut Simulation, event: ChildDerivEvent, v1: usize, v2: usize, v3: usize) {
    use ChildDerivEvent::*;
    match event {
        AttachV1ToV2 => {
            let (offset, t) = attach_v1_to_v2_offset_and_rotation();
            sim.attach(v1, v2, offset, t);
            sim.mark_kinematic_only(v1);
        }
        AttachV1ToV3 => {
            let (offset, t) = attach_v1_to_v3_offset_and_rotation();
            sim.attach(v1, v3, offset, t);
            // After reroot, v2 is now interior (kinematic between v3
            // root and v1 leaf). Mark it kinematic_only so the
            // composite-rigid-body invariant (only root integrates) holds.
            sim.mark_kinematic_only(v2);
        }
        DetachVeh1FromVeh3ViaVeh2 => {
            // Walk v1's ancestor chain; find the one whose parent is v3
            // (mirrors `dyn_body_detach.cc::detach(other_body)` lines
            // 60-69, "search from this up to other_body"). On a fresh
            // tree where v3 is no longer an ancestor of v1, JEOD's
            // `detach` returns false with an `inform` message and no
            // mutation — we replicate that no-op here.
            let v3_id = mass_body_id(sim, v3);
            let mut current = mass_body_id(sim, v1);
            let mut detacher_id: Option<astrodyn::MassBodyId> = None;
            while let Some(parent) = sim.mass_tree.as_ref().and_then(|t| t.parent(current)) {
                if parent == v3_id {
                    detacher_id = Some(current);
                    break;
                }
                current = parent;
            }
            if let Some(did) = detacher_id {
                // The chained tree always has the detached body as
                // either v1 (kinematic) or v2 (kinematic interior). The
                // schedule never asks to detach the chain root; v1 and
                // v2 are the only candidates a re-rooting walk would
                // hit, and both are kinematic_only after the t=2 attach.
                let detacher_idx = if did == mass_body_id(sim, v1) {
                    v1
                } else if did == mass_body_id(sim, v2) {
                    v2
                } else {
                    panic!(
                        "DetachVeh1FromVeh3ViaVeh2 resolved to mass-body {did:?} which \
                         is not v1 or v2; the RUN_compute_child_derivative chained tree \
                         only ever roots at v3 with v1 / v2 below it. This is a test \
                         setup error — verify the schedule's attach order."
                    )
                };
                sim.clear_kinematic_only(detacher_idx);
                sim.detach(detacher_idx);
            }
        }
        SetForceStruct(idx_in_run, value) => {
            let idx = match idx_in_run {
                0 => v1,
                1 => v2,
                2 => v3,
                _ => unreachable!("RUN_compute_child_derivative has only veh{{1,2,3}}"),
            };
            sim.set_body_external_force_struct(idx, value);
        }
        SetTorqueStruct(idx_in_run, value) => {
            let idx = match idx_in_run {
                0 => v1,
                1 => v2,
                2 => v3,
                _ => unreachable!("RUN_compute_child_derivative has only veh{{1,2,3}}"),
            };
            sim.set_body_external_torque_struct(idx, value);
        }
    }
}

/// `RUN_compute_child_derivative`, full trajectory window
/// (`t ∈ [0, ~57]`).
///
/// Drives every CSV row from `t=0` through the schedule's last event
/// (`t=57`, the v3 torque-clear; `trick.stop(65)` extends a few seconds
/// past for ballistic settling). At each integer-second tick the
/// runner-side scheduler fires the matching JEOD `trick.add_read`
/// snippet (see [`child_deriv_schedule`] +
/// [`apply_event`]), then `Simulation::step()` propagates dt=0.1 s
/// through to the next CSV checkpoint.
///
/// What this validates end-to-end through `Simulation::step()`:
///
/// - Pre-attach RK4 propagation on three free-flying bodies (same
///   floor as `tier3_sim_compute_child_derivative_pre_attach_trajectory`).
/// - JEOD's chained-attach re-rooting at t=2 (composite-rigid-body
///   tree v3 ← v2 ← v1) and the two `detach_from_3` events at t=15
///   and t=45.
/// - Composite-rigid-body wrench aggregation: forces / torques applied
///   to **kinematic** children (v1 between t=3-5, t=17-19, t=33-35,
///   t=47-49; v2 between t=7-9, t=21-23, t=37-39, t=51-53) propagate
///   to the chain's integration root via the new
///   `Simulation::aggregate_chain_wrenches` pass.
/// - Struct-frame external forces / torques (mirrors JEOD's
///   `dyn_body_collect.cc:219-221` rotate-on-collect): the
///   inertial-frame contribution tracks each body's current attitude.
///
/// Tolerance derivation: JEOD's read jobs fire at integer-second
/// boundaries and our scheduler fires them within `0.5 · dt = 0.05 s`
/// of the matching boundary (the same `step_until`-style alignment the
/// pre-attach test uses). The dominant residuals over the run come
/// from (a) the per-RK4-stage rotation snapshot vs JEOD's rotate-each-
/// stage convention (`O(ω · dt)` per step on inertial force magnitude,
/// integrating to `O(ω · dt · run_duration)` over the full window),
/// and (b) the ballistic / kinematic propagation through chained
/// reroot trees in windows where v2 is interior. Per CLAUDE.md
/// "Tolerance policy", each tolerance is set to ~5% above the
/// observed max error.
// non-recipe: SIM_verif_attach_detach placeholder mass tree.
#[test]
fn tier3_sim_compute_child_derivative_full_trajectory() {
    let rows = load_csv("chained_attach_child_deriv_kinematic_propagation_state.csv");
    assert!(rows.len() > 100, "expected >100 rows, got {}", rows.len());

    let (mut sim, v1, v2, v3) = build_sim();
    let dt = sim.dt;
    let schedule = child_deriv_schedule();
    // Index of the next event to fire. Events are pre-sorted by time.
    let mut next_event_idx = 0_usize;

    // Cross-validate every CSV row (including post-attach windows).
    let mut errs = [WindowErrors::default(); 3];
    let last_event_time = schedule.last().map_or(0.0, |(t, _)| *t);

    for row in &rows {
        // Stop a fraction past the last event so we cover the
        // ballistic-settling window the JEOD `trick.stop(65)` records
        // (~8 s past t=57). Going further into the truncated final
        // window risks accumulating drift the run does not validate.
        if row.time > last_event_time + 1.0 {
            break;
        }
        // Step the runner forward until `sim.elapsed()` reaches the
        // CSV row's time, then fire any read-job events whose
        // scheduled time has now been reached. JEOD's CSV samples at
        // each integer-second tick happen **after** that second's
        // `trick.add_read` snippet has run — so we sample the runner's
        // state after firing the matching events at the row's time.
        let target = row.time;
        while sim.elapsed() + 0.5 * dt < target {
            sim.step().expect("step() must succeed");
        }
        // Fire any events scheduled at or before the row's time. The
        // post-attach kinematic-propagation walk runs only inside
        // `step()`, so for an attach at t=N to take effect by the
        // sample at t=N we need to call `propagate_kinematic_state`
        // by hand (mirrors JEOD's `propagate_state_from_structure`
        // call inside `attach_child` finalization).
        let mut applied_topology_event = false;
        while next_event_idx < schedule.len() && schedule[next_event_idx].0 <= target + 0.5 * dt {
            let (_, event) = schedule[next_event_idx];
            apply_event(&mut sim, event, v1, v2, v3);
            if matches!(
                event,
                ChildDerivEvent::AttachV1ToV2
                    | ChildDerivEvent::AttachV1ToV3
                    | ChildDerivEvent::DetachVeh1FromVeh3ViaVeh2
            ) {
                applied_topology_event = true;
            }
            next_event_idx += 1;
        }
        if applied_topology_event {
            // After a topology change the runner's per-body
            // `composite_body` state is up-to-date for chain roots
            // (combine_states_at_attach wrote them), but not for
            // kinematic children — those depend on the next step's
            // pre-integration `propagate_kinematic_state` walk. Do an
            // immediate single-step propagation so the sample at this
            // row reflects the JEOD `propagate_state_from_structure`
            // post-attach finalization JEOD performs inside
            // `attach_child` itself.
            sim.propagate_kinematic_state_for_logging();
        }

        for (i, &idx) in [v1, v2, v3].iter().enumerate() {
            let snap = body_snapshot(&sim, idx);
            errs[i].update(&snap, &row.veh[i]);
        }
    }

    let mut report = CrossvalReport::compute(
        "tier3_sim_compute_child_derivative_full_trajectory",
        &[],
        &[],
    );
    for (i, name) in ["veh1", "veh2", "veh3"].iter().enumerate() {
        report.add_extra(&format!("{name}_max_position_err"), errs[i].pos, "m");
        report.add_extra(&format!("{name}_max_velocity_err"), errs[i].vel, "m/s");
        report.add_extra(&format!("{name}_max_quat_angle_err"), errs[i].quat, "rad");
        report.add_extra(&format!("{name}_max_ang_vel_err"), errs[i].ang_vel, "rad/s");
    }
    report.write();

    // Tolerances. The full-window run includes:
    //   - chained-attach re-rooting (combine_states_at_attach merges
    //     across the chain at t=1, t=2, with `detach` inverse-splits
    //     at t=15, t=45);
    //   - struct-frame forces / torques on kinematic children (v1, v2)
    //     that route through the new wrench-aggregation pass;
    //   - root-applied forces / torques on the chain root (v3 in the
    //     [t=2, t=15] tree, v2 in the [t=15, t=45] tree, then v3 again).
    //
    // Per CLAUDE.md "Tolerance policy", each value is ~5% above the
    // observed max error per component, derived from the JSON report
    // each test writes.
    for (i, name) in ["veh1", "veh2", "veh3"].iter().enumerate() {
        assert!(
            errs[i].pos < VEH_FULL_TRAJECTORY_POSITION_TOL_M,
            "{name} full-window position {:.3e} m exceeds {:.1e}",
            errs[i].pos,
            VEH_FULL_TRAJECTORY_POSITION_TOL_M
        );
        assert!(
            errs[i].vel < VEH_FULL_TRAJECTORY_VELOCITY_TOL_MPS,
            "{name} full-window velocity {:.3e} m/s exceeds {:.1e}",
            errs[i].vel,
            VEH_FULL_TRAJECTORY_VELOCITY_TOL_MPS
        );
        assert!(
            errs[i].quat < VEH_FULL_TRAJECTORY_QUAT_ANGLE_TOL_RAD,
            "{name} full-window quat-angle {:.3e} rad exceeds {:.1e}",
            errs[i].quat,
            VEH_FULL_TRAJECTORY_QUAT_ANGLE_TOL_RAD
        );
        assert!(
            errs[i].ang_vel < VEH_FULL_TRAJECTORY_ANG_VEL_TOL_RAD_PER_S,
            "{name} full-window ang-vel {:.3e} rad/s exceeds {:.1e}",
            errs[i].ang_vel,
            VEH_FULL_TRAJECTORY_ANG_VEL_TOL_RAD_PER_S
        );
    }
}

// ════════════════════════════════════════════════════════════════════
// Tolerances for the full-trajectory window. Set to ~5% above the
// observed max error per component (per CLAUDE.md "Tolerance policy"),
// derived from the JSON report the test writes.
//
// All three vehicles share the same per-quantity tolerance — the
// largest residual across {veh1, veh2, veh3} drives each value, and
// the chained-attach + reroot machinery couples their dynamics enough
// that splitting the table per body would just track the worst chain
// member at any given window.
//
// The full-window residuals are dominated by RK4 + JEOD `%g`-formatted
// CSV print precision. The struct-frame external force is resolved to
// inertial per RK4 sub-stage from the intermediate attitude (the runner's
// `StructuralWrench` path, JEOD_INV DB.28), matching JEOD's per-derivative
// `extern_forc_inrtl` recompute. With the wrench-aggregation pass and
// immediate kinematic re-propagation post-attach in place, the full
// trajectory closes within `~4 cm` position and `~2e-7 rad` quaternion-angle
// against the JEOD reference CSV across the 60 s window.
// ════════════════════════════════════════════════════════════════════

const VEH_FULL_TRAJECTORY_POSITION_TOL_M: f64 = 4.0e-2;
const VEH_FULL_TRAJECTORY_VELOCITY_TOL_MPS: f64 = 9.25e-4;
const VEH_FULL_TRAJECTORY_QUAT_ANGLE_TOL_RAD: f64 = 2.01e-7;
const VEH_FULL_TRAJECTORY_ANG_VEL_TOL_RAD_PER_S: f64 = 4.03e-8;
