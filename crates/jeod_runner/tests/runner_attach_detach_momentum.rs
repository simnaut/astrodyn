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
use jeod_runner::Simulation;
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
