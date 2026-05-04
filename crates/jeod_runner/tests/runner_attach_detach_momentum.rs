//! Runner integration tests for [`jeod_runner::Simulation::attach`] /
//! [`jeod_runner::Simulation::detach`] momentum conservation
//! (sub-issue #297).
//!
//! These tests pin three properties of the runner's single-body
//! attach/detach API:
//!
//! 1. **Attach matches the kernel byte-for-byte.** Spawning two free
//!    bodies with non-trivial state, snapshotting their pre-attach
//!    composite-body state, calling `Simulation::attach`, and comparing
//!    the parent's `body.trans` / `body.rot` against the standalone
//!    [`jeod_dynamics::combine_states_at_attach`] output for the same
//!    inputs must agree to f64 rounding. The runner adapter is a thin
//!    wrapper around the kernel — any drift indicates a bug in the
//!    snapshot/writeback plumbing.
//!
//! 2. **Linear momentum about the integration-frame origin is
//!    conserved across attach.** `m_p · v_p + m_c · v_c == m_t · v_t`
//!    is the kernel's central promise; this test guards it at the
//!    runner integration level (against a regression where, say, the
//!    orchestration accidentally writes the parent's mass-weighted
//!    velocity into the child's storage instead of the parent's).
//!
//! 3. **Detach is the inverse split.** After `Simulation::attach` then
//!    `Simulation::detach`, the parent's `body.trans` recovers its
//!    pre-attach inertial position to f64 rounding. The child's state
//!    is the rigid-body composition of the merged-body state with the
//!    child-composite-wrt-combined-CoM offset (matching the body-aware
//!    tree walk in `detach_subtree` and the Bevy adapter's detach
//!    handler).
//!
//! The fourth scope item — Bevy ↔ runner bit-identity — is guarded in
//! `tests/bevy_parity_attach_detach_momentum.rs` (Bevy crate-level
//! integration tests), not here, because it has to stand up a Bevy
//! `App`.

use glam::{DMat3, DVec3};
use jeod_dynamics::{combine_states_at_attach, AttachCombineInputs};
use jeod_frames::{RefFrameRot, RefFrameState, RefFrameTrans};
use jeod_math::JeodQuat;
use jeod_runner::{Simulation, SimulationBuilderExt};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, GravitySourceEntry,
    IntegratorType, MassProperties as SimMassProperties, RotationalState, SimulationTime,
    TranslationalState, VehicleConfig,
};

/// Build a one-source (pure-inertial — `mu = 0`) simulation containing
/// a parent + child pair with explicit pre-attach state. We disable
/// gravity (`mu = 0`) so the kernel's combine output is unaffected by
/// the integrator's force evaluation between snapshot and writeback —
/// these tests guard the *attach orchestration*, not the integrator.
fn build_pair(
    dt: f64,
    parent_mass: SimMassProperties,
    parent_trans: TranslationalState,
    parent_rot: Option<RotationalState>,
    child_mass: SimMassProperties,
    child_trans: TranslationalState,
    child_rot: Option<RotationalState>,
) -> (
    Simulation,
    /* parent_idx */ usize,
    /* child_idx */ usize,
    /* parent_id */ jeod_dynamics::MassBodyId,
    /* child_id */ jeod_dynamics::MassBodyId,
) {
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

    // Inertial-only environment: no gravity sources contributing
    // acceleration. We still need at least one source frame for the
    // pipeline to be valid.
    let inertial = sim.add_source(
        "InertialAnchor",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
            velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: jeod_runner::RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    // Empty gravity controls: the kernel's accumulate path returns
    // zero acceleration, which is what we want.
    let parent_idx = sim.add_body(VehicleConfig {
        trans: parent_trans,
        rot: parent_rot,
        mass: Some(parent_mass),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(inertial, false)],
        },
        ..Default::default()
    });
    let child_idx = sim.add_body(VehicleConfig {
        trans: child_trans,
        rot: child_rot,
        mass: Some(child_mass),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(inertial, false)],
        },
        ..Default::default()
    });

    let parent_id = sim.add_body_to_tree(parent_idx, "Parent");
    let child_id = sim.add_body_to_tree(child_idx, "Child");
    sim.validate().expect("pre-attach validation");

    (sim, parent_idx, child_idx, parent_id, child_id)
}

/// Build a `RefFrameState` that mirrors the runner's
/// `body_composite_state_or_default`: read pos/vel directly, fall back
/// to identity attitude + zero ang_vel when `rot` is `None`.
fn body_state_or_default(trans: TranslationalState, rot: Option<RotationalState>) -> RefFrameState {
    let (q, w) = match rot {
        Some(r) => (r.quaternion, r.ang_vel_body),
        None => (JeodQuat::identity(), DVec3::ZERO),
    };
    RefFrameState {
        trans: RefFrameTrans {
            position: trans.position,
            velocity: trans.velocity,
        },
        rot: RefFrameRot {
            q_parent_this: q,
            t_parent_this: q.left_quat_to_transformation(),
            ang_vel_this: w,
        },
    }
}

/// `Simulation::attach` writes the kernel's output into the parent
/// entity's `body.trans` / `body.rot` byte-for-byte.
#[test]
fn runner_attach_matches_kernel_byte_for_byte() {
    let parent_mass = SimMassProperties::with_inertia(
        420.0,
        DMat3::from_diagonal(DVec3::new(150.0, 200.0, 250.0)),
        DVec3::ZERO,
    );
    let child_mass = SimMassProperties::with_inertia(
        80.0,
        DMat3::from_diagonal(DVec3::new(40.0, 50.0, 60.0)),
        DVec3::ZERO,
    );

    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let parent_rot = Some(RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::new(0.0, -1e-3, 0.0),
    });
    // Child structurally offset 3 m along +x; with identity attitude
    // the child's structural offset matches its inertial offset on the
    // first tick.
    let child_trans = TranslationalState {
        position: DVec3::new(7e6 + 3.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7610.0, 5.0),
    };
    let child_rot = Some(RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::new(0.0, 5e-4, 0.0),
    });

    let (mut sim, parent_idx, child_idx, _parent_id, _child_id) = build_pair(
        1.0,
        parent_mass,
        parent_trans,
        parent_rot,
        child_mass,
        child_trans,
        child_rot,
    );

    // Snapshot pre-attach state for the kernel comparison.
    let parent_pre_state = body_state_or_default(parent_trans, parent_rot);
    let child_pre_state = body_state_or_default(child_trans, child_rot);

    // Run the runner attach (our wiring under test).
    let offset = DVec3::new(3.0, 0.0, 0.0);
    let t_parent_child = DMat3::IDENTITY;
    sim.attach(child_idx, parent_idx, offset, t_parent_child);

    // Read the post-attach combined mass props from the runner's mass
    // tree — the kernel needs the same combined mass that
    // `Simulation::attach` fed itself.
    let combined_mass = sim
        .mass_tree
        .as_ref()
        .expect("mass tree present after attach")
        .get(_parent_id)
        .composite_properties;

    // Run the standalone kernel with the same inputs.
    let parent_t_struct_to_body = parent_mass.t_parent_this;
    let parent_t_inertial_struct = jeod_dynamics::compute_t_inertial_struct(
        &parent_t_struct_to_body,
        &parent_pre_state.rot.t_parent_this,
    );
    let expected = combine_states_at_attach(AttachCombineInputs {
        parent_composite: parent_pre_state,
        parent_mass,
        parent_t_inertial_struct,
        child_composite: child_pre_state,
        child_mass,
        combined_mass,
        orig_parent_cm_struct: parent_mass.position,
    });

    // Pull the runner's post-attach parent state.
    let parent_out = sim.body(parent_idx);
    let runner_pos = parent_out.trans.position;
    let runner_vel = parent_out.trans.velocity;
    let parent_rot_out = parent_out
        .rot
        .expect("6-DOF parent must keep rotational state");
    let runner_q = parent_rot_out.quaternion;
    let runner_w = parent_rot_out.ang_vel_body;

    // Bit-identical: every f64 must match its kernel counterpart to
    // `to_bits()`.
    assert_eq!(
        runner_pos.to_array().map(f64::to_bits),
        expected
            .composite_state
            .trans
            .position
            .to_array()
            .map(f64::to_bits),
        "post-attach parent position must be bit-identical to the kernel output"
    );
    assert_eq!(
        runner_vel.to_array().map(f64::to_bits),
        expected
            .composite_state
            .trans
            .velocity
            .to_array()
            .map(f64::to_bits),
        "post-attach parent velocity must be bit-identical to the kernel output"
    );
    assert_eq!(
        [
            runner_q.scalar().to_bits(),
            runner_q.vector().x.to_bits(),
            runner_q.vector().y.to_bits(),
            runner_q.vector().z.to_bits(),
        ],
        [
            expected
                .composite_state
                .rot
                .q_parent_this
                .scalar()
                .to_bits(),
            expected
                .composite_state
                .rot
                .q_parent_this
                .vector()
                .x
                .to_bits(),
            expected
                .composite_state
                .rot
                .q_parent_this
                .vector()
                .y
                .to_bits(),
            expected
                .composite_state
                .rot
                .q_parent_this
                .vector()
                .z
                .to_bits(),
        ],
        "post-attach parent quaternion must be bit-identical to the kernel output"
    );
    assert_eq!(
        runner_w.to_array().map(f64::to_bits),
        expected
            .composite_state
            .rot
            .ang_vel_this
            .to_array()
            .map(f64::to_bits),
        "post-attach parent ang_vel must be bit-identical to the kernel output"
    );
}

/// Linear momentum about the integration-frame origin must be
/// conserved across `Simulation::attach`. This is the runner-side
/// integration check that the kernel's promise is honored end-to-end
/// by the orchestration code (not just at the kernel boundary).
#[test]
fn runner_attach_conserves_linear_momentum() {
    let parent_mass = SimMassProperties::with_inertia(
        1000.0,
        DMat3::from_diagonal(DVec3::new(500.0, 500.0, 500.0)),
        DVec3::ZERO,
    );
    let child_mass = SimMassProperties::with_inertia(
        1000.0,
        DMat3::from_diagonal(DVec3::new(500.0, 500.0, 500.0)),
        DVec3::ZERO,
    );
    let parent_v = DVec3::new(7300.0, 0.0, 0.0);
    let child_v = DVec3::new(7400.0, 50.0, -10.0);
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: parent_v,
    };
    let parent_rot = Some(RotationalState::default());
    let child_trans = TranslationalState {
        position: DVec3::new(7e6 + 2.0, 0.0, 0.0),
        velocity: child_v,
    };
    let child_rot = Some(RotationalState::default());

    let (mut sim, parent_idx, child_idx, parent_id, _) = build_pair(
        1.0,
        parent_mass,
        parent_trans,
        parent_rot,
        child_mass,
        child_trans,
        child_rot,
    );

    let p_pre = parent_mass.mass * parent_v + child_mass.mass * child_v;

    sim.attach(
        child_idx,
        parent_idx,
        DVec3::new(2.0, 0.0, 0.0),
        DMat3::IDENTITY,
    );

    let combined_mass = sim
        .mass_tree
        .as_ref()
        .unwrap()
        .get(parent_id)
        .composite_properties
        .mass;
    let parent_post = sim.body(parent_idx);
    let p_post = combined_mass * parent_post.trans.velocity;

    // Bit-identical f64 conservation is unrealistic (the kernel does
    // a divide by `inverse_mass`), but the round-off envelope is
    // bounded by a few ULP at the per-component level.
    let err = (p_post - p_pre).length();
    assert!(
        err < 1e-9,
        "linear momentum violation across runner attach: pre={p_pre:?} post={p_post:?} err={err}"
    );
}

/// `Simulation::detach` is the inverse split: applied right after
/// `Simulation::attach`, the parent's inertial position must recover
/// its pre-attach value to rounding, and the child's state must equal
/// the rigid-body composition of the merged composite + the
/// child-composite-wrt-combined-CoM offset.
#[test]
fn runner_attach_then_detach_recovers_parent_position() {
    let parent_mass = SimMassProperties::with_inertia(
        420.0,
        DMat3::from_diagonal(DVec3::new(150.0, 200.0, 250.0)),
        DVec3::ZERO,
    );
    let child_mass = SimMassProperties::with_inertia(
        80.0,
        DMat3::from_diagonal(DVec3::new(40.0, 50.0, 60.0)),
        DVec3::ZERO,
    );
    let parent_trans = TranslationalState {
        position: DVec3::new(6.7e6, 1.0e5, -3.0e4),
        velocity: DVec3::new(7300.0, -50.0, 13.0),
    };
    let parent_rot = Some(RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    });
    let child_trans = TranslationalState {
        position: parent_trans.position + DVec3::new(3.0, 0.0, 0.0),
        velocity: parent_trans.velocity, // co-moving so the merge is "soft"
    };
    let child_rot = Some(RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    });

    let (mut sim, parent_idx, child_idx, _, _) = build_pair(
        1.0,
        parent_mass,
        parent_trans,
        parent_rot,
        child_mass,
        child_trans,
        child_rot,
    );

    sim.attach(
        child_idx,
        parent_idx,
        DVec3::new(3.0, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    sim.detach(child_idx);

    let parent_post = sim.body(parent_idx);
    let err = (parent_post.trans.position - parent_trans.position).length();
    assert!(
        err < 1e-9,
        "parent position drift across attach + detach: pre={:?} post={:?} err={err}",
        parent_trans.position,
        parent_post.trans.position
    );
    // For a co-moving merge, velocity is unchanged across both
    // operations (linear momentum trivially preserved at v_p == v_c).
    let v_err = (parent_post.trans.velocity - parent_trans.velocity).length();
    assert!(
        v_err < 1e-9,
        "parent velocity drift across attach + detach: pre={:?} post={:?} err={v_err}",
        parent_trans.velocity,
        parent_post.trans.velocity
    );

    // The detached child's state is the body-aware tree-walk
    // composition of the merged composite + the offset between the
    // combined CoM and the child's composite. For axis-aligned bodies
    // with identity attitude and offset = (3,0,0), the combined CoM in
    // parent-struct is at (80·3 / 500) = 0.48; the child's composite
    // sits at (3,0,0); so the offset from combined-CoM to child-CoM in
    // parent-struct is (2.52, 0, 0). With identity attitude the inertial
    // shift matches the struct shift, so the post-detach child position
    // is parent_pre_position + (2.52 + 0.48, 0, 0) = parent_pre_position
    // + (3, 0, 0) = child's pre-attach position.
    let child_post = sim.body(child_idx);
    let child_err = (child_post.trans.position - child_trans.position).length();
    assert!(
        child_err < 1e-9,
        "child position drift across attach + detach (rigid co-mover): pre={:?} post={:?} err={child_err}",
        child_trans.position,
        child_post.trans.position
    );
}

/// 3-DOF bodies must still attach + detach without panicking — the
/// runner orchestration treats missing rotational state as identity
/// attitude + zero ang_vel and skips the rotational writeback. This
/// pins the degenerate-case path against a regression where the
/// kernel call's I⁻¹·L solve panics on a zero inertia tensor or where
/// the writeback unconditionally calls `body.rot = Some(...)`.
#[test]
fn runner_attach_detach_handles_3dof_bodies() {
    let parent_mass = SimMassProperties::new(1000.0);
    let child_mass = SimMassProperties::new(1000.0);
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let child_trans = TranslationalState {
        position: DVec3::new(7e6, 1.0, 0.0),
        velocity: DVec3::new(0.0, 7700.0, 0.0),
    };
    let (mut sim, parent_idx, child_idx, _, _) = build_pair(
        1.0,
        parent_mass,
        parent_trans,
        None, // 3-DOF
        child_mass,
        child_trans,
        None, // 3-DOF
    );

    sim.attach(child_idx, parent_idx, DVec3::ZERO, DMat3::IDENTITY);
    // Linear momentum still conserved.
    let parent_post = sim.body(parent_idx);
    assert!(
        parent_post.rot.is_none(),
        "3-DOF parent must remain 3-DOF after attach"
    );
    let expected_v = (1000.0 * 7600.0 + 1000.0 * 7700.0) / 2000.0;
    assert!(
        (parent_post.trans.velocity.y - expected_v).abs() < 1e-9,
        "linear momentum conservation failed for 3-DOF attach: y_vel={}",
        parent_post.trans.velocity.y
    );

    // Detach must also work without rotational state.
    sim.detach(child_idx);
    let parent_post_detach = sim.body(parent_idx);
    assert!(parent_post_detach.rot.is_none());
    let child_post_detach = sim.body(child_idx);
    assert!(child_post_detach.rot.is_none());
}

/// Multi-level attach: the named `parent_idx` passed to
/// `Simulation::attach` is itself an interior, kinematic-only SimBody
/// in an existing two-level tree (root A → kinematic-only B). A free
/// body C is attached underneath B. Per JEOD_INV: DB.13 / DB.17, only
/// the integrated tree root carries authoritative composite-body
/// state; an interior SimBody's `body.trans` / `body.rot` are derived
/// from the root each tick by `propagate_kinematic_state`. The combine
/// must therefore route through A's state, not B's, even though the
/// caller named B as the parent.
///
/// Without the multi-level fix in `Simulation::attach`, the kernel
/// reads B's storage directly (potentially stale, or — when the
/// kinematic walk hasn't run — frozen at the configured-but-not-
/// integrated value), and writes the merged state back to B instead
/// of A. The integrated root A's `body.trans` is then unchanged by
/// the attach, so the next integrate step propagates the pre-attach
/// momentum and silently violates linear-momentum conservation
/// across the topology change.
///
/// This test pins the post-attach root state to the mass-weighted
/// combined velocity of (A+B) and C — the value the kernel produces
/// when fed the root's authoritative pre-attach state.
#[test]
fn runner_attach_handles_interior_kinematic_parent() {
    // Two-level pre-attach tree: root A (integrated) and interior B
    // (kinematic-only child of A, no integrator state of its own).
    // Free body C joins underneath B in this test.
    let a_mass = SimMassProperties::with_inertia(
        500.0,
        DMat3::from_diagonal(DVec3::new(200.0, 200.0, 200.0)),
        DVec3::ZERO,
    );
    let b_mass = SimMassProperties::with_inertia(
        500.0,
        DMat3::from_diagonal(DVec3::new(200.0, 200.0, 200.0)),
        DVec3::ZERO,
    );
    let c_mass = SimMassProperties::with_inertia(
        1000.0,
        DMat3::from_diagonal(DVec3::new(500.0, 500.0, 500.0)),
        DVec3::ZERO,
    );

    // A is the only body with integrator-written state. With identity
    // attitude on both A and B, the rigid-body invariant gives B's
    // inertial position = A's + offset_b, and B's inertial velocity =
    // A's velocity (zero ang_vel).
    let a_v = DVec3::new(7000.0, 0.0, 0.0);
    let c_v = DVec3::new(7700.0, 0.0, 0.0);
    let a_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: a_v,
    };
    let a_rot = Some(RotationalState::default());
    let offset_b = DVec3::new(2.0, 0.0, 0.0);
    // B's storage is set consistent with the rigid-body invariant —
    // exactly what `propagate_kinematic_state` would produce on the
    // first step. A correct combine reading from A and walking down
    // through the tree must yield the same value.
    let b_trans = TranslationalState {
        position: a_trans.position + offset_b,
        velocity: a_v,
    };
    let b_rot = Some(RotationalState::default());
    let c_trans = TranslationalState {
        position: a_trans.position + DVec3::new(5.0, 0.0, 0.0),
        velocity: c_v,
    };
    let c_rot = Some(RotationalState::default());

    // Build the sim manually since `build_pair` only creates two
    // bodies; we need three.
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, 1.0);

    let inertial = sim.add_source(
        "InertialAnchor",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
            velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: jeod_runner::RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    let a_idx = sim.add_body(VehicleConfig {
        trans: a_trans,
        rot: a_rot,
        mass: Some(a_mass),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(inertial, false)],
        },
        ..Default::default()
    });
    let b_idx = sim.add_body(VehicleConfig {
        trans: b_trans,
        rot: b_rot,
        mass: Some(b_mass),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(inertial, false)],
        },
        ..Default::default()
    });
    let c_idx = sim.add_body(VehicleConfig {
        trans: c_trans,
        rot: c_rot,
        mass: Some(c_mass),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(inertial, false)],
        },
        ..Default::default()
    });

    let _a_id = sim.add_body_to_tree(a_idx, "A");
    let _b_id = sim.add_body_to_tree(b_idx, "B");
    let _c_id = sim.add_body_to_tree(c_idx, "C");

    // Form the existing A → B link first; then mark B kinematic-only
    // so its storage is officially derived from A every tick.
    sim.attach(b_idx, a_idx, offset_b, DMat3::IDENTITY);
    sim.mark_kinematic_only(b_idx);

    // Snapshot A's authoritative pre-attach composite-body state for
    // the second attach. The runner's mass tree already merged B into
    // A's composite when we ran `sim.attach(b_idx, a_idx, ...)`, so
    // reading sim.body(a_idx) here gives the (A+B) composite state +
    // (A+B) composite mass — what the kernel needs as the "parent
    // composite" when adding C.
    let a_pre_attach_v = sim.body(a_idx).trans.velocity;
    let combined_ab_mass = sim
        .mass_tree
        .as_ref()
        .expect("mass tree present")
        .get(_a_id)
        .composite_properties
        .mass;
    // Sanity: m_A + m_B = 1000 (parallel-axis with offset adds zero
    // mass; mass is additive). The combined-AB velocity equals A's
    // pre-second-attach velocity (the first attach was a co-mover so
    // momentum conservation is the identity).
    assert!(
        (combined_ab_mass - 1000.0).abs() < 1e-9,
        "AB combined mass: expected 1000, got {combined_ab_mass}"
    );

    // Now the failure mode under test. With the multi-level fix the
    // attach walks up from B → A and runs the combine using A's
    // authoritative composite-body state; the resulting whole-tree
    // composite is written back to A. Without the fix the combine
    // reads B's storage and writes to B, leaving A's velocity
    // unchanged at its pre-attach value (assertion below).
    sim.attach(c_idx, b_idx, DVec3::new(3.0, 0.0, 0.0), DMat3::IDENTITY);

    // Expected: linear-momentum conservation about the integration
    // origin gives v_post = (m_AB · v_AB + m_C · v_C) / (m_AB + m_C).
    let expected_v =
        (combined_ab_mass * a_pre_attach_v + c_mass.mass * c_v) / (combined_ab_mass + c_mass.mass);

    let a_post = sim.body(a_idx);
    let v_err = (a_post.trans.velocity - expected_v).length();
    assert!(
        v_err < 1e-9,
        "interior-parent attach must update the integrated tree root's velocity \
         to the mass-weighted combine (linear momentum conservation across the \
         whole tree). Without the multi-level fix the combine writes to the \
         interior parent and the root's velocity stays at {a_pre_attach_v:?}. \
         Expected={expected_v:?}, got={:?}, err={v_err}",
        a_post.trans.velocity
    );

    // The integrated root must still carry rotational state (the
    // root-level writeback path mirrors the parent-as-root case).
    assert!(
        a_post.rot.is_some(),
        "6-DOF root must keep rotational state after multi-level attach"
    );

    // Total linear momentum about the integration origin is conserved
    // to f64 rounding (defense-in-depth on the same property).
    let combined_total_mass = sim
        .mass_tree
        .as_ref()
        .expect("mass tree present")
        .get(_a_id)
        .composite_properties
        .mass;
    let p_pre = combined_ab_mass * a_pre_attach_v + c_mass.mass * c_v;
    let p_post = combined_total_mass * a_post.trans.velocity;
    let p_err = (p_post - p_pre).length();
    assert!(
        p_err < 1e-9,
        "total linear momentum must be conserved across the multi-level attach: \
         pre={p_pre:?} post={p_post:?} err={p_err}"
    );
}

/// `Simulation::detach` on a tree whose bodies integrate in a non-root
/// `PlanetInertial<P>` source frame must round-trip parent and child
/// state through the planet origin. The detach handler runs the
/// body-aware tree walk in root-inertial coordinates (the only inertial
/// frame in which `propagate_forward` arithmetic is valid across a
/// chain of bodies that may live in different integ frames). Both the
/// seed and the writeback have to cross the `IntegOrigin` boundary.
///
/// Without the fix:
/// - the seed `root_pre_state` is read directly from
///   `TranslationalStateTyped<IntegrationFrame>` storage but treated as
///   root-inertial, so `derive_subtree_composite_state` walks the chain
///   from a planet-relative seed and produces a planet-relative child;
/// - the post-detach writeback then casts root-inertial values
///   (`new_root_position`/`new_root_velocity`) back into
///   `TranslationalStateTyped<IntegrationFrame>` storage without the
///   inverse `IntegOrigin` shift, mis-labelling the typed state.
///
/// For a non-root-integrated body this corrupts post-detach storage by
/// the planet's offset from root (`SSB_TO_EARTH_OFFSET ~ 1.5e11 m` in
/// the realistic frame-tree layout exercised here).
///
/// With the fix the round-trip recovers the pre-attach storage values
/// to f64 rounding for both parent and child — same property the
/// `runner_attach_then_detach_recovers_parent_position` test pins for
/// the root-integrated case, generalised across `IntegOrigin`.
#[test]
fn runner_detach_lifts_through_integ_origin() {
    use jeod_sim::{RotationModel, SimulationBuilder};

    // Earth as a non-central source 1.5e11 m from the SSB-rooted frame
    // — the same realistic geometry the frame-translation invariance
    // test (`integ_frame_translation_invariance.rs`) uses to make
    // `IntegOrigin` non-zero. The shift through the planet origin is
    // what distinguishes this test from
    // `runner_attach_then_detach_recovers_parent_position` (which has
    // an implicitly zero `IntegOrigin`).
    const SSB_TO_EARTH: DVec3 = DVec3::new(1.5e11, 0.0, 0.0);

    let parent_mass = SimMassProperties::with_inertia(
        420.0,
        DMat3::from_diagonal(DVec3::new(150.0, 200.0, 250.0)),
        DVec3::ZERO,
    );
    let child_mass = SimMassProperties::with_inertia(
        80.0,
        DMat3::from_diagonal(DVec3::new(40.0, 50.0, 60.0)),
        DVec3::ZERO,
    );

    // Earth-relative ECI coords. Both bodies integrate in
    // `Earth.inertial` (a non-root frame in this setup), so these are
    // the values stored in `body.trans` — and the values the round-trip
    // must recover post-detach.
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 1.0e5, -3.0e4),
        velocity: DVec3::new(7300.0, -50.0, 13.0),
    };
    let parent_rot = Some(RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    });
    let child_trans = TranslationalState {
        position: parent_trans.position + DVec3::new(3.0, 0.0, 0.0),
        velocity: parent_trans.velocity, // co-mover so the merge is "soft"
    };
    let child_rot = Some(RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    });

    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, 1.0);

    // Root = SSB barycenter (mu=0 placeholder); Earth is a non-central
    // child at SSB_TO_EARTH so `IntegOrigin{Earth} != 0`.
    let _ssb = sb.add_source(
        "SSB",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
            velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
            t_inertial_pfix: None,
            rotation_model: RotationModel::None,
            delta_c20: 0.0,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );
    let earth = sb.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0, // disable gravity so the test guards orchestration only
                model: GravityModel::PointMass,
            },
            position: jeod_sim::Position::<jeod_sim::RootInertial>::from_raw_si(SSB_TO_EARTH),
            velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
            t_inertial_pfix: None,
            rotation_model: RotationModel::None,
            delta_c20: 0.0,
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
        },
    );

    let parent_idx = sb.add_body(VehicleConfig {
        trans: parent_trans,
        rot: parent_rot,
        mass: Some(parent_mass),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        integ_source: Some(earth),
        ..Default::default()
    });
    let child_idx = sb.add_body(VehicleConfig {
        trans: child_trans,
        rot: child_rot,
        mass: Some(child_mass),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        integ_source: Some(earth),
        ..Default::default()
    });
    let mut sim = sb.build().expect("non-root-integ pair builds");

    let _parent_id = sim.add_body_to_tree(parent_idx, "Parent");
    let _child_id = sim.add_body_to_tree(child_idx, "Child");

    sim.attach(
        child_idx,
        parent_idx,
        DVec3::new(3.0, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    sim.detach(child_idx);

    // Parent's `body.trans` is typed `<IntegrationFrame>` and the
    // integ frame is `Earth.inertial`; the round-trip must recover the
    // pre-attach Earth-relative values. Without the writeback shift the
    // post-detach value would be off by exactly `SSB_TO_EARTH` (the
    // root-inertial seed lift never gets reversed).
    let parent_post = sim.body(parent_idx);
    let pos_err = (parent_post.trans.position - parent_trans.position).length();
    assert!(
        pos_err < 1e-6,
        "parent position must round-trip across attach + detach when the \
         integ frame has a non-zero `IntegOrigin`: pre={:?} post={:?} \
         err={pos_err}. A failure of order {} m indicates the seed lift / \
         writeback shift is missing.",
        parent_trans.position,
        parent_post.trans.position,
        SSB_TO_EARTH.length()
    );
    let v_err = (parent_post.trans.velocity - parent_trans.velocity).length();
    assert!(
        v_err < 1e-9,
        "parent velocity must round-trip across attach + detach (co-mover): \
         pre={:?} post={:?} err={v_err}",
        parent_trans.velocity,
        parent_post.trans.velocity,
    );

    // Same property for the child — its storage is also typed
    // `<IntegrationFrame>` and the rederived state must come out in
    // Earth-relative coords, not root-inertial coords.
    let child_post = sim.body(child_idx);
    let child_err = (child_post.trans.position - child_trans.position).length();
    assert!(
        child_err < 1e-6,
        "child position must round-trip across attach + detach when the \
         integ frame has a non-zero `IntegOrigin`: pre={:?} post={:?} \
         err={child_err}. A failure of order {} m indicates the writeback \
         shift is missing on the child side.",
        child_trans.position,
        child_post.trans.position,
        SSB_TO_EARTH.length()
    );
    let child_v_err = (child_post.trans.velocity - child_trans.velocity).length();
    assert!(
        child_v_err < 1e-9,
        "child velocity must round-trip across attach + detach (co-mover): \
         pre={:?} post={:?} err={child_v_err}",
        child_trans.velocity,
        child_post.trans.velocity,
    );
}
