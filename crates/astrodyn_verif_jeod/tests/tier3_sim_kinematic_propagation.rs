// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! Tier 3: SIM_verif_attach_detach RUN_simple_attach_detach with
//! kinematic-propagation cross-validation.
//!
//! Wires the runner's `propagate_state_via_storage` pipeline end-to-end
//! (design-doc § 15.3 — kinematic propagation) and validates the
//! kinematic-child state derivation against the JEOD verification sim.
//! Drives [`Simulation::attach`] / [`Simulation::detach`] end-to-end —
//! both methods run JEOD's `combine_states_at_attach` momentum-
//! conservation kernel, so the runner-side composite-body trajectories
//! match JEOD's reference verbatim across the attach and detach events.
//!
//! # What is validated
//!
//! 1. **Veh3 free trajectory**: the third vehicle never participates in
//!    any attach/detach in `RUN_simple_attach_detach`, so its composite-
//!    body state is just rigid-body integration under no force / no
//!    torque. This pins the runner's per-step pipeline (time advance →
//!    integration → composite-state writeback) bit-identically against
//!    JEOD's reference CSV across the entire run.
//!
//! 2. **Kinematic-propagation invariant during the attached window**:
//!    for `t ∈ [10, 20)` JEOD has veh1 attached to veh2 as a kinematic
//!    child. JEOD's recorded composite-body states must satisfy
//!    `kernel(veh2_state, link_geometry) ≈ veh1_state`, where `kernel`
//!    is `compute_kinematic_child_state`. Cross-validating that
//!    relationship across the JEOD CSV's logged states pins the
//!    pure-math kernel to JEOD's runtime semantics — independent of
//!    whether we drive the kernel from `Simulation::step()` or from
//!    a Bevy `App::update()`.
//!
//! 3. **End-to-end runner pipeline against JEOD's CSV**: the runner
//!    reproduces the veh1/veh2 ballistic trajectories in the
//!    **pre-attach** window `t ∈ [0, 10)`, the **attached** window
//!    `t ∈ [10, 20)` (after `Simulation::attach` runs JEOD's
//!    momentum-conservation combine), and the **post-detach** window
//!    `t ∈ [20, 30)` (after `Simulation::detach` runs the inverse split)
//!    against JEOD's CSV. Veh1 in the attached window is derived each
//!    tick by [`propagate_state_via_storage`](astrodyn::propagate_state_via_storage)
//!    from veh2's integrated composite-body state plus the link
//!    geometry — the absolute trajectory match against JEOD's recorded
//!    `composite_body` states pins the full
//!    integration → combine → propagation pipeline.
//!
//! 4. **Runner self-consistency in the attached window**: veh1's
//!    runner-derived state equals the kernel applied to veh2's
//!    runner-integrated state at the same tick. This is a structural
//!    check that `propagate_kinematic_state` actually runs each tick
//!    and writes through to `body.trans` / `body.rot`, independent of
//!    JEOD's CSV.
//!
//! # What is **not** validated
//!
//! - **Re-rooting attaches**: `RUN_complex_attach_detach` and
//!   `RUN_compute_child_derivative` exercise chained `attach_to(name1,
//!   name2)` invocations whose parent-of-the-attaching-subtree's-root
//!   is repointed; the mass-tree algebra check at
//!   `crates/astrodyn_dynamics/tests/mass_tree_placeholder.rs` covers the
//!   composite-mass output of those re-rooting paths but does not drive
//!   the full pipeline. This test does not address the trajectory gap.
//! - **Reference-frame attaches and dynamic-body-action rate changes**:
//!   past `FRAME_ATTACH_PHASE_START = 30 s` the JEOD run fires
//!   `set_attitude_rate` and a sequence of reference-frame attach/detach
//!   events that are not modelled here. Cross-validation stops at
//!   `t = 30`.
//!
//! # JEOD source data ingested
//!
//! Initial conditions only — never intermediate-step CSV values. The
//! attach geometry (offset + rotation) and per-vehicle initial state
//! both come from `Modified_data/veh{1,2,3}.py` and the named
//! `BodyAttachAligned` invocation in `attach_detach.py`. The CSV
//! drives only the comparison.

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

/// One row of the `kinematic_propagation_state` CSV: time + per-vehicle
/// composite-body state for veh1, veh2, veh3 (39 + 1 columns).
#[derive(Debug, Clone)]
struct StateRow {
    time: f64,
    veh: [VehSnapshot; 3],
}

#[derive(Debug, Clone, Copy)]
struct VehSnapshot {
    position: DVec3,
    velocity: DVec3,
    /// JEOD scalar-first quaternion `[scalar, v0, v1, v2]`.
    quaternion: JeodQuat,
    /// `composite_body.state.rot.ang_vel_this`: angular velocity in
    /// the composite-body frame (rad/s).
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
         Tier 3 Reference Data (Docker)\"). The new \
         `kinematic_propagation_state.csv` is produced by the run group \
         added to `trick/generate_references.sh` for the kinematic-\
         propagation Tier 3 test.",
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
// Initial conditions, all from JEOD source files
// (Modified_data/veh{1,2,3}.py and attach_detach.py).
// ════════════════════════════════════════════════════════════════════

/// Time of `BodyAttachAligned veh1.attach_to_2`, from
/// `SET_test/RUN_simple_attach_detach/input.py:24`.
const ATTACH_TIME: f64 = 10.0;
/// Time of `BodyDetach veh1.detach_from_2`,
/// `SET_test/RUN_simple_attach_detach/input.py:25`.
const DETACH_TIME: f64 = 20.0;
/// Past this time the JEOD run starts firing reference-frame attaches
/// and rate-changes (`set_attitude_rate`, `attach_to_frame`, etc. —
/// `RUN_simple_attach_detach/input.py:27-39`). Reference-frame
/// attaches and dynamic-body-action rate changes are deferred
/// dynamics scope and are not validated in this test.
const FRAME_ATTACH_PHASE_START: f64 = 30.0;

/// Veh1: mass=1.0, properties.position=(5, 0, 0), inertia=10·I.
/// For an atomic body the mass tree's `composite_properties.position`
/// equals the core's, so the CoM in struct frame is (5, 0, 0).
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
    // JEOD `Modified_data/veh2.py:48`:
    //   `veh2.rot_init.orientation.euler_angles = [ -2.0, 0.0, 0.0 ]`
    // No `trick.attach_units("degree", ...)` wrapper, so the value is
    // in JEOD's default units for euler-angle fields: RADIANS, not
    // degrees. The CSV's `Q_parent_this.scalar=0.5403` at t=0 confirms
    // (cos(1.0)=0.5403 ⇒ half-angle 1.0 rad ⇒ θ=2.0 rad — matches
    // |yaw|=2.0 rad, with the sign flipped by JEOD's quaternion
    // convention `qv = -sin(θ/2)·axis`).
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
    // Same convention as `veh2_initial_rot`: euler angles in radians.
    let yaw_rad = -15.8_f64;
    let q = JeodQuat::left_quat_from_eigen_rotation(yaw_rad, DVec3::Z);
    // `Modified_data/veh3.py:50` sets `ang_velocity = [0, 0, 1.0]`,
    // but `RUN_simple_attach_detach/input.py:19` overrides it back to
    // zero before the run starts. Matching the override here keeps
    // initial conditions JEOD-source-faithful.
    RotationalState {
        quaternion: q,
        ang_vel_body: DVec3::ZERO,
    }
}

/// Build a fresh runner sim configured for the simple attach-detach
/// scenario. Returns the simulation plus the integer body indices for
/// (veh1, veh2, veh3).
///
/// Empty space — no gravity sources, no central body — matching JEOD's
/// `EphemerisMode_EmptySpace` config. RK4 integrator on every body.
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
        ..VehicleConfig::named("tier3-sim-kinematic-propagation-2")
    });
    let v2 = sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&veh2_initial_trans()),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(veh2_initial_rot()),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(veh2_mass()))),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..VehicleConfig::named("tier3-sim-kinematic-propagation-1")
    });
    let v3 = sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&veh3_initial_trans()),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(veh3_initial_rot()),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(veh3_mass()))),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..VehicleConfig::named("tier3-sim-kinematic-propagation-0")
    });
    sim.add_body_to_tree(v1, "veh1");
    sim.add_body_to_tree(v2, "veh2");
    sim.add_body_to_tree(v3, "veh3");
    (sim, v1, v2, v3)
}

/// JEOD's `BodyAttachAligned veh1.attach_to_2` resolves to a
/// structural-frame `(offset, T_parent_child)` pair via JEOD's
/// named-point attach algorithm
/// (`models/dynamics/mass/src/mass_attach.cc:67-137`). Both arguments
/// below are derived from JEOD source files only:
///
/// - `veh1.node12` lives at `(10, 0, 0)` in `veh1.struct` with identity
///   orientation (`Modified_data/veh1.py:21-23`).
/// - `veh2.node21` lives at `(0, 0, 0)` in `veh2.struct` with
///   YPR(180°, 0, 0) — i.e. `T_pstr_node21 = R(180°, Z)`
///   (`Modified_data/veh2.py:18-26`).
/// - JEOD's `attach_to` inserts a fixed 180° yaw between attach
///   points (`mass_attach.cc:111-117`).
///
/// Composing the chain
/// (veh2.struct → veh2.node21 → 180° flip → veh1.node12 → veh1.struct)
/// collapses to identity rotation and offset `(-10, 0, 0)`:
/// veh1's `node12` (at +10x in veh1) is the attach point, the
/// 180°-yaw-pair flip cancels, and veh2's `node21` is at the origin.
fn simple_attach_offset_and_rotation() -> (DVec3, DMat3) {
    (DVec3::new(-10.0, 0.0, 0.0), DMat3::IDENTITY)
}

// ════════════════════════════════════════════════════════════════════
// Tolerances. Per project policy, set to ~5% above observed max error.
// ════════════════════════════════════════════════════════════════════

// Tolerances are set to ~5% above observed max error per project
// policy (CLAUDE.md "Cross-validation tolerances"). Observed values
// come from `target/tier3_crossval/tier3_sim_kinematic_propagation_
// simple.json`.

/// Free-flying veh3 across the full run: no force, no torque, RK4 on
/// rigid-body kinematics. Error floor is rounding from JEOD's CSV
/// (~17 sig figures) plus integrator round-off.
const VEH3_POSITION_TOL_M: f64 = 1.5e-12;
const VEH3_VELOCITY_TOL_MPS: f64 = 1e-15;
const VEH3_QUAT_ANGLE_TOL_RAD: f64 = 1e-15;
const VEH3_ANG_VEL_TOL_RAD_PER_S: f64 = 1e-15;

/// Pre-attach veh1/veh2 in the window `t ∈ [0, 10)`: same
/// rigid-body-only conditions as veh3. Quaternion-angle tolerance
/// absorbs JEOD's scalar-first quaternion CSV print precision —
/// the recorder writes `Q_parent_this.{scalar,vector}` as separate
/// %g-formatted f64 fields and the round-trip drifts ~4e-8 rad on a
/// non-trivially-rotated body.
const PRE_ATTACH_POSITION_TOL_M: f64 = 1.5e-13;
const PRE_ATTACH_VELOCITY_TOL_MPS: f64 = 1e-15;
const PRE_ATTACH_QUAT_ANGLE_TOL_RAD: f64 = 4.5e-8;
const PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S: f64 = 1e-15;

/// During-attach kinematic-propagation invariant in the JEOD CSV:
/// `kernel(veh2_csv, link) ≈ veh1_csv`. The CSV samples are JEOD's
/// own `composite_body` recorder; the residual reflects rounding
/// from JEOD's printf format plus the kernel's f64 round-off.
const ATTACH_INVARIANT_POSITION_TOL_M: f64 = 1e-14;
const ATTACH_INVARIANT_VELOCITY_TOL_MPS: f64 = 1.5e-16;
const ATTACH_INVARIANT_QUAT_ANGLE_TOL_RAD: f64 = 3.5e-8;

/// During-attach runner self-consistency: veh1's
/// runner-propagation-derived state must equal the kernel applied to
/// veh2's runner-integrated state. Rounding-only on position /
/// velocity / angular velocity; the quaternion-angle tolerance
/// absorbs the same `2 · acos(|q · q'|)` ULP residual that pins the
/// kernel-vs-CSV invariant — when both quaternions are unit-norm but
/// differ by a few ULPs, the angle test reports ~1e-7–1e-8 rad even
/// though the underlying components agree to 1 ULP.
const RUNNER_PROP_POSITION_TOL_M: f64 = 1e-15;
const RUNNER_PROP_VELOCITY_TOL_MPS: f64 = 1e-15;
const RUNNER_PROP_QUAT_ANGLE_TOL_RAD: f64 = 4.5e-8;
const RUNNER_PROP_ANG_VEL_TOL_RAD_PER_S: f64 = 1e-15;

/// Attached-window absolute trajectory match against JEOD's CSV in
/// `t ∈ [10, 20)`: veh2 is the integrated tree root carrying the
/// momentum-conservation merge of the pre-attach pair, veh1 is the
/// kinematic child derived from veh2 by `propagate_state_via_storage`
/// each tick. Position / velocity residual is dominated by the
/// momentum-conservation kernel's f64 round-off propagating through
/// the 10-second integrated window (the f64 angular-momentum solve on
/// veh1, half that on veh2). Quaternion tolerance absorbs the same
/// JEOD-recorder %g print rounding that pins the pre-attach window.
const ATTACHED_POSITION_TOL_M: f64 = 1.75e-9;
const ATTACHED_VELOCITY_TOL_MPS: f64 = 3.25e-11;
const ATTACHED_QUAT_ANGLE_TOL_RAD: f64 = 3.5e-8;
const ATTACHED_ANG_VEL_TOL_RAD_PER_S: f64 = 1e-15;

/// Post-detach absolute trajectory match against JEOD's CSV in
/// `t ∈ [20, 30)`: veh1 and veh2 each integrate independently from
/// the inverse-split inertial states `Simulation::detach` derives.
/// The error floor mirrors the attached-window residual since both
/// bodies have just resumed independent RK4 integration from JEOD-
/// equivalent initial conditions.
const POST_DETACH_POSITION_TOL_M: f64 = 1.75e-9;
const POST_DETACH_VELOCITY_TOL_MPS: f64 = 3.25e-11;
const POST_DETACH_QUAT_ANGLE_TOL_RAD: f64 = 3.5e-8;
const POST_DETACH_ANG_VEL_TOL_RAD_PER_S: f64 = 1e-15;

/// Compute the kinematic kernel from veh2's state and the link
/// geometry, returning the predicted veh1 composite-body state.
fn kernel_from_veh2(veh2: &VehSnapshot, veh1_mass: f64, veh2_mass: f64) -> VehSnapshot {
    use astrodyn::{compute_kinematic_child_state, KinematicChildInputs};
    let parent_t_inertial_body = veh2.quaternion.left_quat_to_transformation();
    let (offset, t_pc) = simple_attach_offset_and_rotation();
    // Combined composite CoM in veh2's structural frame:
    //   parent_composite_in_pstr =
    //     (m_v2 · cm_v2 + m_v1 · (offset + cm_v1_in_v2struct))
    //     / (m_v2 + m_v1)
    // In our test, both vehicles' core CoM is at (5, 0, 0) in their own
    // struct frame and `T_parent_child` is identity, so veh1's CoM in
    // veh2's struct frame is `offset + (5, 0, 0) = (-5, 0, 0)`.
    let veh1_cm_in_v1_struct = DVec3::new(5.0, 0.0, 0.0);
    let veh2_cm_in_v2_struct = DVec3::new(5.0, 0.0, 0.0);
    let veh1_cm_in_v2_struct = offset + t_pc.transpose() * veh1_cm_in_v1_struct;
    let combined_cm_in_v2_struct = (veh2_cm_in_v2_struct * veh2_mass
        + veh1_cm_in_v2_struct * veh1_mass)
        / (veh1_mass + veh2_mass);
    let inputs = KinematicChildInputs {
        parent_t_inertial_body,
        parent_ang_vel_body: veh2.ang_vel_body,
        parent_position_inertial: veh2.position,
        parent_velocity_inertial: veh2.velocity,
        parent_t_struct_body: DMat3::IDENTITY,
        parent_composite_in_pstr: combined_cm_in_v2_struct,
        t_parent_child: t_pc,
        link_offset_in_pstr: offset,
        child_t_struct_body: DMat3::IDENTITY,
        child_composite_in_cstr: veh1_cm_in_v1_struct,
    };
    let out = compute_kinematic_child_state(inputs);
    VehSnapshot {
        position: out.child_position_inertial,
        velocity: out.child_velocity_inertial,
        quaternion: out.child_q_inertial_body,
        ang_vel_body: out.child_ang_vel_body,
    }
}

/// Quaternion angle error: `2 · acos(|q_a · q_b|)`.
fn quat_angle_err(a: JeodQuat, b: JeodQuat) -> f64 {
    let dot = (a.scalar() * b.scalar() + a.vector().dot(b.vector()))
        .abs()
        .clamp(-1.0, 1.0);
    2.0 * dot.acos()
}

/// Run the simulation through the recorded timeline, applying attach at
/// t=10. Validates all three exit-criteria items from the file-level
/// docstring.
// non-recipe: SIM_verif_attach_detach exercises a placeholder mass
// tree (1/2/3 kg, three free vehicles in empty space) directly through
// the runner's `attach`/`detach`. ISS / Apollo recipes don't apply.
#[test]
fn tier3_sim_kinematic_propagation_simple() {
    let rows = load_csv("kinematic_propagation_simple_kinematic_propagation_state.csv");
    assert!(rows.len() > 60, "expected >60 rows, got {}", rows.len());

    // ── Validation 1: kinematic-propagation invariant in the JEOD
    //    CSV during the attached window. Pin the kernel against
    //    JEOD's runtime semantics independently of `Simulation`.
    let mut max_invariant_pos = 0.0f64;
    let mut max_invariant_vel = 0.0f64;
    let mut max_invariant_quat = 0.0f64;
    for row in &rows {
        // After detach the relationship no longer holds (veh1 has its
        // own integrated state again). Restrict to the strict-inside
        // window — the row at t=10 is the first sample after the
        // attach event fires, the row at t=20 is the first after
        // detach.
        if row.time < ATTACH_TIME || row.time >= DETACH_TIME {
            continue;
        }
        let predicted = kernel_from_veh2(&row.veh[1], 1.0, 2.0);
        max_invariant_pos =
            max_invariant_pos.max((predicted.position - row.veh[0].position).length());
        max_invariant_vel =
            max_invariant_vel.max((predicted.velocity - row.veh[0].velocity).length());
        max_invariant_quat =
            max_invariant_quat.max(quat_angle_err(predicted.quaternion, row.veh[0].quaternion));
    }

    // ── Validation 2 + 3: drive the runner end-to-end.
    let (mut sim, v1, v2, v3) = build_sim();

    // Sanity at t=0.
    let r0 = &rows[0];
    assert!(
        (r0.veh[0].position - veh1_initial_trans().position).length() < 1e-12
            && (r0.veh[1].position - veh2_initial_trans().position).length() < 1e-12
            && (r0.veh[2].position - veh3_initial_trans().position).length() < 1e-12,
        "CSV t=0 does not match the JEOD source-file initial conditions \
         (Modified_data/veh{{1,2,3}}.py); refresh the fixture or fix the IC \
         constants in this test."
    );

    let mut max_pre_pos = [0.0f64; 2];
    let mut max_pre_vel = [0.0f64; 2];
    let mut max_pre_quat = [0.0f64; 2];
    let mut max_pre_avel = [0.0f64; 2];

    let mut max_v3_pos = 0.0f64;
    let mut max_v3_vel = 0.0f64;
    let mut max_v3_quat = 0.0f64;
    let mut max_v3_avel = 0.0f64;

    let mut max_runner_pos = 0.0f64;
    let mut max_runner_vel = 0.0f64;
    let mut max_runner_quat = 0.0f64;
    let mut max_runner_avel = 0.0f64;

    // Attached-window absolute trajectory: veh1 + veh2 vs JEOD's CSV
    // (post-`Simulation::attach` momentum-conservation combine, with
    // veh1 derived as a kinematic child each tick).
    let mut max_attached_pos = [0.0f64; 2];
    let mut max_attached_vel = [0.0f64; 2];
    let mut max_attached_quat = [0.0f64; 2];
    let mut max_attached_avel = [0.0f64; 2];

    // Post-detach absolute trajectory: veh1 + veh2 vs JEOD's CSV
    // (after `Simulation::detach` runs the inverse-split, both bodies
    // resume independent integration).
    let mut max_post_detach_pos = [0.0f64; 2];
    let mut max_post_detach_vel = [0.0f64; 2];
    let mut max_post_detach_quat = [0.0f64; 2];
    let mut max_post_detach_avel = [0.0f64; 2];

    let mut attach_fired = false;
    let mut detach_fired = false;
    let dt = sim.dt;

    for row in &rows {
        // Stop validation past `FRAME_ATTACH_PHASE_START` — the JEOD
        // run fires `set_attitude_rate` at t=30, then a sequence of
        // reference-frame attach/detach events that aren't modelled
        // here. See the file-level docstring's "What is **not**
        // validated" section for the deferred-dynamics scope.
        if row.time >= FRAME_ATTACH_PHASE_START {
            break;
        }
        // Step until sim.elapsed() ≈ row.time.
        while sim.elapsed() + 0.5 * dt < row.time {
            sim.step().expect("step() must succeed");
        }
        // JEOD's `trick.add_read(t, ...)` job runs at the *start* of
        // sim-second `t`, before that second's integration cycle, and
        // before the recorder logs the row for `t`. Apply our attach /
        // detach at the matching moment — once `sim.elapsed() ≈ t`
        // (post-step from `t-dt` to `t`) but before the row at `t` is
        // sampled and before the `t → t+dt` integration step begins.
        // `Simulation::attach` runs JEOD's `combine_states_at_attach`
        // momentum-conservation kernel and writes the merged composite-
        // body state back into veh2 (the integrated tree root);
        // `Simulation::detach` runs the inverse-split, derives veh1's
        // instantaneous composite-body state via the body-aware tree
        // walk, and shifts veh2's state by the inertial CoM-delta.
        if !attach_fired && row.time + 0.5 * dt > ATTACH_TIME {
            let (offset, t_pc) = simple_attach_offset_and_rotation();
            sim.attach(v1, v2, offset, t_pc);
            sim.mark_kinematic_only(v1);
            // Mirror JEOD's `DynBody::attach_child` which calls
            // `propagate_state_from_structure` inside the attach so
            // the chain's child states are coherent immediately on
            // return — without waiting for the next derivative cycle.
            // JEOD's recorder logs the post-attach state at the
            // `add_read` boundary before the next integration step.
            sim.propagate_kinematic_state_for_logging();
            attach_fired = true;
        }
        if attach_fired && !detach_fired && row.time + 0.5 * dt > DETACH_TIME {
            // `Simulation::detach` auto-clears the `kinematic_only`
            // flag on the freshly-detached child, so no paired
            // `clear_kinematic_only` call is needed.
            sim.detach(v1);
            detach_fired = true;
        }

        // Veh3: free-flying across entire run.
        let v3_out = sim.body(v3);
        let v3_csv = &row.veh[2];
        max_v3_pos = max_v3_pos.max((v3_out.trans.position.raw_si() - v3_csv.position).length());
        max_v3_vel = max_v3_vel.max((v3_out.trans.velocity.raw_si() - v3_csv.velocity).length());
        if let Some(rot) = v3_out.rot {
            max_v3_quat = max_v3_quat.max(quat_angle_err(
                rot.q_inertial_body.to_jeod_quat(),
                v3_csv.quaternion,
            ));
            max_v3_avel =
                max_v3_avel.max((rot.ang_vel_body.raw_si() - v3_csv.ang_vel_body).length());
        }

        // Pre-attach veh1 + veh2: free-flying.
        if row.time < ATTACH_TIME {
            for i in 0..2 {
                let body_idx = if i == 0 { v1 } else { v2 };
                let out = sim.body(body_idx);
                let csv = &row.veh[i];
                max_pre_pos[i] =
                    max_pre_pos[i].max((out.trans.position.raw_si() - csv.position).length());
                max_pre_vel[i] =
                    max_pre_vel[i].max((out.trans.velocity.raw_si() - csv.velocity).length());
                if let Some(rot) = out.rot {
                    max_pre_quat[i] = max_pre_quat[i].max(quat_angle_err(
                        rot.q_inertial_body.to_jeod_quat(),
                        csv.quaternion,
                    ));
                    max_pre_avel[i] = max_pre_avel[i]
                        .max((rot.ang_vel_body.raw_si() - csv.ang_vel_body).length());
                }
            }
        } else if row.time < DETACH_TIME {
            // Attached window: two complementary checks.
            //
            //   (a) Absolute trajectory match against JEOD's CSV. The
            //       runner ran `combine_states_at_attach` at
            //       `Simulation::attach` time and now derives veh1
            //       from veh2 each tick via `propagate_state_via_storage`.
            //       Veh2 is the integrated tree root; veh1 is the
            //       kinematic child. Compare both against JEOD's
            //       recorded `composite_body` states.
            //
            //   (b) Runner self-consistency: veh1's runner-derived
            //       state equals the kernel applied to veh2's
            //       runner-integrated state at the same tick. Pins
            //       that `propagate_kinematic_state` actually runs
            //       each tick and writes through to
            //       `body.trans` / `body.rot`, independent of JEOD.
            for i in 0..2 {
                let body_idx = if i == 0 { v1 } else { v2 };
                let out = sim.body(body_idx);
                let csv = &row.veh[i];
                max_attached_pos[i] =
                    max_attached_pos[i].max((out.trans.position.raw_si() - csv.position).length());
                max_attached_vel[i] =
                    max_attached_vel[i].max((out.trans.velocity.raw_si() - csv.velocity).length());
                if let Some(rot) = out.rot {
                    max_attached_quat[i] = max_attached_quat[i].max(quat_angle_err(
                        rot.q_inertial_body.to_jeod_quat(),
                        csv.quaternion,
                    ));
                    max_attached_avel[i] = max_attached_avel[i]
                        .max((rot.ang_vel_body.raw_si() - csv.ang_vel_body).length());
                }
            }

            let v1_runner = sim.body(v1);
            let v2_runner = sim.body(v2);
            let v2_rot = v2_runner.rot.expect("veh2 6-DOF");
            let v2_snap = VehSnapshot {
                position: v2_runner.trans.position.raw_si(),
                velocity: v2_runner.trans.velocity.raw_si(),
                quaternion: v2_rot.q_inertial_body.to_jeod_quat(),
                ang_vel_body: v2_rot.ang_vel_body.raw_si(),
            };
            let predicted = kernel_from_veh2(&v2_snap, 1.0, 2.0);
            max_runner_pos = max_runner_pos
                .max((v1_runner.trans.position.raw_si() - predicted.position).length());
            max_runner_vel = max_runner_vel
                .max((v1_runner.trans.velocity.raw_si() - predicted.velocity).length());
            if let Some(rot) = v1_runner.rot {
                max_runner_quat = max_runner_quat.max(quat_angle_err(
                    rot.q_inertial_body.to_jeod_quat(),
                    predicted.quaternion,
                ));
                max_runner_avel = max_runner_avel
                    .max((rot.ang_vel_body.raw_si() - predicted.ang_vel_body).length());
            }
        } else {
            // Post-detach absolute trajectory match against JEOD's
            // CSV (`t ∈ [20, 30)`): veh1 and veh2 each integrate
            // independently from the inverse-split inertial states
            // `Simulation::detach` derived.
            for i in 0..2 {
                let body_idx = if i == 0 { v1 } else { v2 };
                let out = sim.body(body_idx);
                let csv = &row.veh[i];
                max_post_detach_pos[i] = max_post_detach_pos[i]
                    .max((out.trans.position.raw_si() - csv.position).length());
                max_post_detach_vel[i] = max_post_detach_vel[i]
                    .max((out.trans.velocity.raw_si() - csv.velocity).length());
                if let Some(rot) = out.rot {
                    max_post_detach_quat[i] = max_post_detach_quat[i].max(quat_angle_err(
                        rot.q_inertial_body.to_jeod_quat(),
                        csv.quaternion,
                    ));
                    max_post_detach_avel[i] = max_post_detach_avel[i]
                        .max((rot.ang_vel_body.raw_si() - csv.ang_vel_body).length());
                }
            }
        }
    }

    // Emit the cross-validation report.
    let mut report = CrossvalReport::compute("tier3_sim_kinematic_propagation_simple", &[], &[]);
    report.add_extra("invariant_max_position_err", max_invariant_pos, "m");
    report.add_extra("invariant_max_velocity_err", max_invariant_vel, "m/s");
    report.add_extra("invariant_max_quat_angle_err", max_invariant_quat, "rad");
    for i in 0..2 {
        let label = format!("veh{}_pre_attach", i + 1);
        report.add_extra(&format!("{label}_max_position_err"), max_pre_pos[i], "m");
        report.add_extra(&format!("{label}_max_velocity_err"), max_pre_vel[i], "m/s");
        report.add_extra(
            &format!("{label}_max_quat_angle_err"),
            max_pre_quat[i],
            "rad",
        );
        report.add_extra(
            &format!("{label}_max_ang_vel_err"),
            max_pre_avel[i],
            "rad/s",
        );
    }
    for i in 0..2 {
        let label = format!("veh{}_attached", i + 1);
        report.add_extra(
            &format!("{label}_max_position_err"),
            max_attached_pos[i],
            "m",
        );
        report.add_extra(
            &format!("{label}_max_velocity_err"),
            max_attached_vel[i],
            "m/s",
        );
        report.add_extra(
            &format!("{label}_max_quat_angle_err"),
            max_attached_quat[i],
            "rad",
        );
        report.add_extra(
            &format!("{label}_max_ang_vel_err"),
            max_attached_avel[i],
            "rad/s",
        );
    }
    for i in 0..2 {
        let label = format!("veh{}_post_detach", i + 1);
        report.add_extra(
            &format!("{label}_max_position_err"),
            max_post_detach_pos[i],
            "m",
        );
        report.add_extra(
            &format!("{label}_max_velocity_err"),
            max_post_detach_vel[i],
            "m/s",
        );
        report.add_extra(
            &format!("{label}_max_quat_angle_err"),
            max_post_detach_quat[i],
            "rad",
        );
        report.add_extra(
            &format!("{label}_max_ang_vel_err"),
            max_post_detach_avel[i],
            "rad/s",
        );
    }
    report.add_extra("veh3_max_position_err", max_v3_pos, "m");
    report.add_extra("veh3_max_velocity_err", max_v3_vel, "m/s");
    report.add_extra("veh3_max_quat_angle_err", max_v3_quat, "rad");
    report.add_extra("veh3_max_ang_vel_err", max_v3_avel, "rad/s");
    report.add_extra("runner_prop_max_position_err", max_runner_pos, "m");
    report.add_extra("runner_prop_max_velocity_err", max_runner_vel, "m/s");
    report.add_extra("runner_prop_max_quat_angle_err", max_runner_quat, "rad");
    report.add_extra("runner_prop_max_ang_vel_err", max_runner_avel, "rad/s");
    report.write();

    // ── Assertions ──
    assert!(
        max_invariant_pos < ATTACH_INVARIANT_POSITION_TOL_M,
        "kernel-vs-CSV invariant: position {max_invariant_pos:.3e} m \
         exceeds {ATTACH_INVARIANT_POSITION_TOL_M:.1e} m"
    );
    assert!(
        max_invariant_vel < ATTACH_INVARIANT_VELOCITY_TOL_MPS,
        "kernel-vs-CSV invariant: velocity {max_invariant_vel:.3e} m/s \
         exceeds {ATTACH_INVARIANT_VELOCITY_TOL_MPS:.1e}"
    );
    assert!(
        max_invariant_quat < ATTACH_INVARIANT_QUAT_ANGLE_TOL_RAD,
        "kernel-vs-CSV invariant: quaternion {max_invariant_quat:.3e} rad \
         exceeds {ATTACH_INVARIANT_QUAT_ANGLE_TOL_RAD:.1e}"
    );

    for i in 0..2 {
        let label = format!("veh{}", i + 1);
        assert!(
            max_pre_pos[i] < PRE_ATTACH_POSITION_TOL_M,
            "{label} pre-attach position {:.3e} m exceeds {PRE_ATTACH_POSITION_TOL_M:.1e}",
            max_pre_pos[i]
        );
        assert!(
            max_pre_vel[i] < PRE_ATTACH_VELOCITY_TOL_MPS,
            "{label} pre-attach velocity {:.3e} m/s exceeds {PRE_ATTACH_VELOCITY_TOL_MPS:.1e}",
            max_pre_vel[i]
        );
        assert!(
            max_pre_quat[i] < PRE_ATTACH_QUAT_ANGLE_TOL_RAD,
            "{label} pre-attach quat-angle {:.3e} rad exceeds {PRE_ATTACH_QUAT_ANGLE_TOL_RAD:.1e}",
            max_pre_quat[i]
        );
        assert!(
            max_pre_avel[i] < PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S,
            "{label} pre-attach ang-vel {:.3e} rad/s exceeds {PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S:.1e}",
            max_pre_avel[i]
        );
    }

    assert!(
        max_v3_pos < VEH3_POSITION_TOL_M,
        "veh3 position {max_v3_pos:.3e} m exceeds {VEH3_POSITION_TOL_M:.1e}"
    );
    assert!(
        max_v3_vel < VEH3_VELOCITY_TOL_MPS,
        "veh3 velocity {max_v3_vel:.3e} m/s exceeds {VEH3_VELOCITY_TOL_MPS:.1e}"
    );
    assert!(
        max_v3_quat < VEH3_QUAT_ANGLE_TOL_RAD,
        "veh3 quaternion {max_v3_quat:.3e} rad exceeds {VEH3_QUAT_ANGLE_TOL_RAD:.1e}"
    );
    assert!(
        max_v3_avel < VEH3_ANG_VEL_TOL_RAD_PER_S,
        "veh3 ang-vel {max_v3_avel:.3e} rad/s exceeds {VEH3_ANG_VEL_TOL_RAD_PER_S:.1e}"
    );

    assert!(
        max_runner_pos < RUNNER_PROP_POSITION_TOL_M,
        "runner kinematic propagation: position {max_runner_pos:.3e} m \
         exceeds {RUNNER_PROP_POSITION_TOL_M:.1e}"
    );
    assert!(
        max_runner_vel < RUNNER_PROP_VELOCITY_TOL_MPS,
        "runner kinematic propagation: velocity {max_runner_vel:.3e} m/s \
         exceeds {RUNNER_PROP_VELOCITY_TOL_MPS:.1e}"
    );
    assert!(
        max_runner_quat < RUNNER_PROP_QUAT_ANGLE_TOL_RAD,
        "runner kinematic propagation: quaternion {max_runner_quat:.3e} rad \
         exceeds {RUNNER_PROP_QUAT_ANGLE_TOL_RAD:.1e}"
    );
    assert!(
        max_runner_avel < RUNNER_PROP_ANG_VEL_TOL_RAD_PER_S,
        "runner kinematic propagation: ang-vel {max_runner_avel:.3e} rad/s \
         exceeds {RUNNER_PROP_ANG_VEL_TOL_RAD_PER_S:.1e}"
    );

    for i in 0..2 {
        let label = format!("veh{}", i + 1);
        assert!(
            max_attached_pos[i] < ATTACHED_POSITION_TOL_M,
            "{label} attached-window position {:.3e} m exceeds {ATTACHED_POSITION_TOL_M:.1e}",
            max_attached_pos[i]
        );
        assert!(
            max_attached_vel[i] < ATTACHED_VELOCITY_TOL_MPS,
            "{label} attached-window velocity {:.3e} m/s exceeds {ATTACHED_VELOCITY_TOL_MPS:.1e}",
            max_attached_vel[i]
        );
        assert!(
            max_attached_quat[i] < ATTACHED_QUAT_ANGLE_TOL_RAD,
            "{label} attached-window quat-angle {:.3e} rad exceeds {ATTACHED_QUAT_ANGLE_TOL_RAD:.1e}",
            max_attached_quat[i]
        );
        assert!(
            max_attached_avel[i] < ATTACHED_ANG_VEL_TOL_RAD_PER_S,
            "{label} attached-window ang-vel {:.3e} rad/s exceeds {ATTACHED_ANG_VEL_TOL_RAD_PER_S:.1e}",
            max_attached_avel[i]
        );
    }

    for i in 0..2 {
        let label = format!("veh{}", i + 1);
        assert!(
            max_post_detach_pos[i] < POST_DETACH_POSITION_TOL_M,
            "{label} post-detach position {:.3e} m exceeds {POST_DETACH_POSITION_TOL_M:.1e}",
            max_post_detach_pos[i]
        );
        assert!(
            max_post_detach_vel[i] < POST_DETACH_VELOCITY_TOL_MPS,
            "{label} post-detach velocity {:.3e} m/s exceeds {POST_DETACH_VELOCITY_TOL_MPS:.1e}",
            max_post_detach_vel[i]
        );
        assert!(
            max_post_detach_quat[i] < POST_DETACH_QUAT_ANGLE_TOL_RAD,
            "{label} post-detach quat-angle {:.3e} rad exceeds {POST_DETACH_QUAT_ANGLE_TOL_RAD:.1e}",
            max_post_detach_quat[i]
        );
        assert!(
            max_post_detach_avel[i] < POST_DETACH_ANG_VEL_TOL_RAD_PER_S,
            "{label} post-detach ang-vel {:.3e} rad/s exceeds {POST_DETACH_ANG_VEL_TOL_RAD_PER_S:.1e}",
            max_post_detach_avel[i]
        );
    }
}
