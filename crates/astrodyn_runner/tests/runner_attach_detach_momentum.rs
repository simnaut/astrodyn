// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! Runner integration tests for [`astrodyn_runner::Simulation::attach`] /
//! [`astrodyn_runner::Simulation::detach`] momentum conservation
//! (sub-issue #297).
//!
//! These tests pin three properties of the runner's single-body
//! attach/detach API:
//!
//! 1. **Attach matches the kernel byte-for-byte.** Spawning two free
//!    bodies with non-trivial state, snapshotting their pre-attach
//!    composite-body state, calling `Simulation::attach`, and comparing
//!    the parent's `body.trans` / `body.rot` against the standalone
//!    [`astrodyn::combine_states_at_attach`] output for the same
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

use astrodyn::JeodQuat;
use astrodyn::{combine_states_at_attach, AttachCombineInputs};
use astrodyn::{
    AngularVelocity, BodyAttitude, BodyFrame, GravityControl, GravityControls, GravityGradient,
    GravityModel, GravitySource, GravitySourceEntry, InertiaTensor, IntegratorType,
    MassProperties as SimMassProperties, MassPropertiesTyped, Position, RootInertial,
    RotationalState, RotationalStateTyped, SelfRef, SimulationTime, StructuralFrame,
    TranslationalState, TranslationalStateTyped, Vec3Ext, VehicleConfig, Velocity,
};
use astrodyn::{RefFrameRot, RefFrameState, RefFrameTrans};
use astrodyn_runner::{Simulation, SimulationBuilderExt};
use glam::{DMat3, DVec3};
use uom::si::f64::Mass;
use uom::si::mass::kilogram;

/// Synthetic gravity-source markers: these tests anchor bodies to
/// non-planet sources, which (per issue #662's strict identity rule)
/// require `define_planet!`-minted markers and `add_source_typed`.
mod tags {
    astrodyn::define_planet!(InertialAnchor);
    astrodyn::define_planet!(Ssb);
}

// allowed: typed↔raw kernel-boundary helpers used in test scaffolding
// (issue #397).
fn trans_typed(t: &TranslationalState) -> TranslationalStateTyped<RootInertial> {
    TranslationalStateTyped::<RootInertial> {
        position: Position::<RootInertial>::from_raw_si(t.position), // allowed: typed↔raw kernel boundary
        velocity: Velocity::<RootInertial>::from_raw_si(t.velocity), // allowed: typed↔raw kernel boundary
    }
}

fn rot_typed(r: &RotationalState) -> RotationalStateTyped<SelfRef> {
    RotationalStateTyped::<SelfRef>::new(
        BodyAttitude::from_jeod_quat(r.quaternion),
        AngularVelocity::<BodyFrame<SelfRef>>::from_raw_si(r.ang_vel_body), // allowed: typed↔raw kernel boundary
    )
}

fn mass_typed(mp: &SimMassProperties) -> MassPropertiesTyped<SelfRef> {
    MassPropertiesTyped::<SelfRef>::with_inertia(
        Mass::new::<kilogram>(mp.mass),
        InertiaTensor::<BodyFrame<SelfRef>>::from_dmat3_unchecked(mp.inertia), // allowed: typed↔raw kernel boundary
        Position::<StructuralFrame<SelfRef>>::from_raw_si(mp.position), // allowed: typed↔raw kernel boundary
    )
    .with_t_parent_this(mp.t_parent_this)
}

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
    /* parent_id */ astrodyn::MassBodyId,
    /* child_id */ astrodyn::MassBodyId,
) {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

    // Inertial-only environment: no gravity sources contributing
    // acceleration. We still need at least one source frame for the
    // pipeline to be valid.
    let _inertial = sim.add_source_typed::<tags::InertialAnchor>(
        "InertialAnchor",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: astrodyn_runner::RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );

    // Empty gravity controls: the kernel's accumulate path returns
    // zero acceleration, which is what we want.
    let parent_idx = sim.add_body(VehicleConfig {
        trans: trans_typed(&parent_trans),
        rot: parent_rot.as_ref().map(rot_typed),
        mass: Some(mass_typed(&(parent_mass))),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<tags::InertialAnchor>>(),
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("runner-attach-detach-momentum-8")
    });
    let child_idx = sim.add_body(VehicleConfig {
        trans: trans_typed(&child_trans),
        rot: child_rot.as_ref().map(rot_typed),
        mass: Some(mass_typed(&(child_mass))),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<tags::InertialAnchor>>(),
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("runner-attach-detach-momentum-7")
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
    let parent_t_inertial_struct = astrodyn::compute_t_inertial_struct(
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
    let runner_pos = parent_out.trans.position.raw_si();
    let runner_vel = parent_out.trans.velocity.raw_si();
    let parent_rot_out = parent_out
        .rot
        .expect("6-DOF parent must keep rotational state");
    let runner_q = parent_rot_out.q_inertial_body.to_jeod_quat();
    let runner_w = parent_rot_out.ang_vel_body.raw_si();

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
    let p_post = combined_mass * parent_post.trans.velocity.raw_si();

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
    let parent_post_pos = parent_post.trans.position.raw_si();
    let parent_post_vel = parent_post.trans.velocity.raw_si();
    let err = (parent_post_pos - parent_trans.position).length();
    assert!(
        err < 1e-9,
        "parent position drift across attach + detach: pre={:?} post={:?} err={err}",
        parent_trans.position,
        parent_post_pos
    );
    // For a co-moving merge, velocity is unchanged across both
    // operations (linear momentum trivially preserved at v_p == v_c).
    let v_err = (parent_post_vel - parent_trans.velocity).length();
    assert!(
        v_err < 1e-9,
        "parent velocity drift across attach + detach: pre={:?} post={:?} err={v_err}",
        parent_trans.velocity,
        parent_post_vel
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
    let child_post_pos = child_post.trans.position.raw_si();
    let child_err = (child_post_pos - child_trans.position).length();
    assert!(
        child_err < 1e-9,
        "child position drift across attach + detach (rigid co-mover): pre={:?} post={:?} err={child_err}",
        child_trans.position,
        child_post_pos
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
    let parent_vel = parent_post.trans.velocity.raw_si();
    assert!(
        (parent_vel.y - expected_v).abs() < 1e-9,
        "linear momentum conservation failed for 3-DOF attach: y_vel={}",
        parent_vel.y
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
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, 1.0);

    let _inertial = sim.add_source_typed::<tags::InertialAnchor>(
        "InertialAnchor",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: astrodyn_runner::RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );

    let a_idx = sim.add_body(VehicleConfig {
        trans: trans_typed(&a_trans),
        rot: a_rot.as_ref().map(rot_typed),
        mass: Some(mass_typed(&(a_mass))),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<tags::InertialAnchor>>(),
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("runner-attach-detach-momentum-6")
    });
    let b_idx = sim.add_body(VehicleConfig {
        trans: trans_typed(&b_trans),
        rot: b_rot.as_ref().map(rot_typed),
        mass: Some(mass_typed(&(b_mass))),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<tags::InertialAnchor>>(),
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("runner-attach-detach-momentum-5")
    });
    let c_idx = sim.add_body(VehicleConfig {
        trans: trans_typed(&c_trans),
        rot: c_rot.as_ref().map(rot_typed),
        mass: Some(mass_typed(&(c_mass))),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<tags::InertialAnchor>>(),
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("runner-attach-detach-momentum-4")
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
    let a_pre_attach_v = sim.body(a_idx).trans.velocity.raw_si();
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
    let a_post_vel = a_post.trans.velocity.raw_si();
    let v_err = (a_post_vel - expected_v).length();
    assert!(
        v_err < 1e-9,
        "interior-parent attach must update the integrated tree root's velocity \
         to the mass-weighted combine (linear momentum conservation across the \
         whole tree). Without the multi-level fix the combine writes to the \
         interior parent and the root's velocity stays at {a_pre_attach_v:?}. \
         Expected={expected_v:?}, got={a_post_vel:?}, err={v_err}"
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
    let p_post = combined_total_mass * a_post_vel;
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
    use astrodyn::{RotationModel, SimulationBuilder};

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

    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, 1.0);

    // Root = SSB barycenter (mu=0 placeholder); Earth is a non-central
    // child at SSB_TO_EARTH so `IntegOrigin{Earth} != 0`.
    let _ssb = sb.add_source_typed::<tags::Ssb>(
        "SSB",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            rotation_model: RotationModel::None,
            delta_c20: 0.0,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );
    let _earth = sb.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0, // disable gravity so the test guards orchestration only
                model: GravityModel::PointMass,
            },
            position: SSB_TO_EARTH.m_at::<RootInertial>(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            rotation_model: RotationModel::None,
            delta_c20: 0.0,
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
            marker_only: false,
        },
    );

    let parent_idx = sb.add_body(VehicleConfig {
        trans: trans_typed(&parent_trans),
        rot: parent_rot.as_ref().map(rot_typed),
        mass: Some(mass_typed(&(parent_mass))),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                GravityGradient::Skip,
            )],
        },
        integ_source: Some(astrodyn::FrameUid::of::<
            astrodyn::PlanetInertial<astrodyn::Earth>,
        >()),
        ..VehicleConfig::named("runner-attach-detach-momentum-3")
    });
    let child_idx = sb.add_body(VehicleConfig {
        trans: trans_typed(&child_trans),
        rot: child_rot.as_ref().map(rot_typed),
        mass: Some(mass_typed(&(child_mass))),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                GravityGradient::Skip,
            )],
        },
        integ_source: Some(astrodyn::FrameUid::of::<
            astrodyn::PlanetInertial<astrodyn::Earth>,
        >()),
        ..VehicleConfig::named("runner-attach-detach-momentum-2")
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
    let parent_post_pos = parent_post.trans.position.raw_si();
    let parent_post_vel = parent_post.trans.velocity.raw_si();
    let pos_err = (parent_post_pos - parent_trans.position).length();
    assert!(
        pos_err < 1e-6,
        "parent position must round-trip across attach + detach when the \
         integ frame has a non-zero `IntegOrigin`: pre={:?} post={parent_post_pos:?} \
         err={pos_err}. A failure of order {} m indicates the seed lift / \
         writeback shift is missing.",
        parent_trans.position,
        SSB_TO_EARTH.length()
    );
    let v_err = (parent_post_vel - parent_trans.velocity).length();
    assert!(
        v_err < 1e-9,
        "parent velocity must round-trip across attach + detach (co-mover): \
         pre={:?} post={parent_post_vel:?} err={v_err}",
        parent_trans.velocity,
    );

    // Same property for the child — its storage is also typed
    // `<IntegrationFrame>` and the rederived state must come out in
    // Earth-relative coords, not root-inertial coords.
    let child_post = sim.body(child_idx);
    let child_post_pos = child_post.trans.position.raw_si();
    let child_post_vel = child_post.trans.velocity.raw_si();
    let child_err = (child_post_pos - child_trans.position).length();
    assert!(
        child_err < 1e-6,
        "child position must round-trip across attach + detach when the \
         integ frame has a non-zero `IntegOrigin`: pre={:?} post={child_post_pos:?} \
         err={child_err}. A failure of order {} m indicates the writeback \
         shift is missing on the child side.",
        child_trans.position,
        SSB_TO_EARTH.length()
    );
    let child_v_err = (child_post_vel - child_trans.velocity).length();
    assert!(
        child_v_err < 1e-9,
        "child velocity must round-trip across attach + detach (co-mover): \
         pre={:?} post={child_post_vel:?} err={child_v_err}",
        child_trans.velocity,
    );
}

/// `Simulation::from_builder` materialising a
/// [`SimulationBuilder::attach_bodies`] declaration must preserve each
/// body's caller-supplied `VehicleConfig::trans` / `rot` verbatim — the
/// build-time topology declaration is a configuration step, not an
/// in-flight impulse merge. PR #307 review thread `PRRT_kwDORtae6c5_Q3me`.
///
/// Pre-#307-thread-fix the builder routed `attach_bodies` through the
/// public `Simulation::attach`, which had been newly wired to
/// JEOD's `combine_states_at_attach` momentum-conservation kernel.
/// That turned every build-time attached pair into a kinematic merge:
/// the parent's spec'd inertial position drifted by the
/// composite-CoM-shift towards the child, and its velocity became the
/// mass-weighted average of the parent's and child's spec'd
/// velocities. For any caller that registered a parent on its orbital
/// initial state and a child on a deliberately offset / co-moving
/// initial state, that was a silent corruption of `VehicleConfig`.
///
/// The fix routes builder-time attaches through
/// `attach_preserving_initial_state`, which performs the tree
/// mutation, composite-mass resync, and integrator-history reset
/// while leaving `body.trans` / `body.rot` untouched. This test pins
/// that contract: the post-build state on both bodies is bit-identical
/// to the spec'd `VehicleConfig` state.
#[test]
fn from_builder_preserves_attached_bodies_initial_state() {
    use astrodyn::SimulationBuilder;

    // Asymmetric pair so any combine writeback would *change* both
    // bodies' state by orders of magnitude — the spec'd values are
    // structurally distinct from any mass-weighted merge.
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
    let parent_rot = RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    };
    // Child spec'd at a different inertial position with a different
    // velocity (not the parent's) — this is the case the public
    // runtime `attach` would treat as a pre-attach pair to merge.
    let child_trans = TranslationalState {
        position: DVec3::new(7e6 + 100.0, 50.0, -25.0),
        velocity: DVec3::new(10.0, 7700.0, 5.0),
    };
    let child_rot = RotationalState {
        quaternion: JeodQuat::left_quat_from_eigen_rotation(0.3, DVec3::Y),
        ang_vel_body: DVec3::new(0.1, -0.05, 0.02),
    };

    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, 1.0);
    let _inertial = sb.add_source_typed::<tags::InertialAnchor>(
        "InertialAnchor",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: astrodyn_runner::RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );
    let parent_idx = sb.add_body(VehicleConfig {
        trans: trans_typed(&parent_trans),
        rot: Some(rot_typed(&(parent_rot))),
        mass: Some(mass_typed(&(parent_mass))),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<tags::InertialAnchor>>(),
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("runner-attach-detach-momentum-1")
    });
    let child_idx = sb.add_body(VehicleConfig {
        trans: trans_typed(&child_trans),
        rot: Some(rot_typed(&(child_rot))),
        mass: Some(mass_typed(&(child_mass))),
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<tags::InertialAnchor>>(),
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("runner-attach-detach-momentum-0")
    });
    sb.register_in_mass_tree(parent_idx, "Parent");
    sb.register_in_mass_tree(child_idx, "Child");
    sb.attach_bodies(
        child_idx,
        parent_idx,
        DVec3::new(3.0, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    let sim = sb
        .build()
        .expect("two-body builder with attach_bodies must validate");

    // Bit-identical preservation: post-build state equals spec'd state.
    let parent_post = sim.body(parent_idx);
    let parent_post_pos = parent_post.trans.position.raw_si();
    let parent_post_vel = parent_post.trans.velocity.raw_si();
    assert_eq!(
        parent_post_pos.to_array().map(f64::to_bits),
        parent_trans.position.to_array().map(f64::to_bits),
        "from_builder must preserve parent position verbatim: spec={:?} got={parent_post_pos:?}",
        parent_trans.position,
    );
    assert_eq!(
        parent_post_vel.to_array().map(f64::to_bits),
        parent_trans.velocity.to_array().map(f64::to_bits),
        "from_builder must preserve parent velocity verbatim: spec={:?} got={parent_post_vel:?}",
        parent_trans.velocity,
    );
    let parent_rot_post = parent_post
        .rot
        .expect("6-DOF parent must keep rot through builder materialization");
    let parent_quat_post = parent_rot_post.q_inertial_body.to_jeod_quat();
    let parent_omega_post = parent_rot_post.ang_vel_body.raw_si();
    assert_eq!(
        [
            parent_quat_post.scalar().to_bits(),
            parent_quat_post.vector().x.to_bits(),
            parent_quat_post.vector().y.to_bits(),
            parent_quat_post.vector().z.to_bits(),
        ],
        [
            parent_rot.quaternion.scalar().to_bits(),
            parent_rot.quaternion.vector().x.to_bits(),
            parent_rot.quaternion.vector().y.to_bits(),
            parent_rot.quaternion.vector().z.to_bits(),
        ],
        "from_builder must preserve parent quaternion verbatim",
    );
    assert_eq!(
        parent_omega_post.to_array().map(f64::to_bits),
        parent_rot.ang_vel_body.to_array().map(f64::to_bits),
        "from_builder must preserve parent ang_vel verbatim",
    );

    let child_post = sim.body(child_idx);
    let child_post_pos = child_post.trans.position.raw_si();
    let child_post_vel = child_post.trans.velocity.raw_si();
    assert_eq!(
        child_post_pos.to_array().map(f64::to_bits),
        child_trans.position.to_array().map(f64::to_bits),
        "from_builder must preserve child position verbatim: spec={:?} got={child_post_pos:?}",
        child_trans.position,
    );
    assert_eq!(
        child_post_vel.to_array().map(f64::to_bits),
        child_trans.velocity.to_array().map(f64::to_bits),
        "from_builder must preserve child velocity verbatim",
    );
    let child_rot_post = child_post
        .rot
        .expect("6-DOF child must keep rot through builder materialization");
    let child_quat_post = child_rot_post.q_inertial_body.to_jeod_quat();
    let child_omega_post = child_rot_post.ang_vel_body.raw_si();
    assert_eq!(
        [
            child_quat_post.scalar().to_bits(),
            child_quat_post.vector().x.to_bits(),
            child_quat_post.vector().y.to_bits(),
            child_quat_post.vector().z.to_bits(),
        ],
        [
            child_rot.quaternion.scalar().to_bits(),
            child_rot.quaternion.vector().x.to_bits(),
            child_rot.quaternion.vector().y.to_bits(),
            child_rot.quaternion.vector().z.to_bits(),
        ],
        "from_builder must preserve child quaternion verbatim",
    );
    assert_eq!(
        child_omega_post.to_array().map(f64::to_bits),
        child_rot.ang_vel_body.to_array().map(f64::to_bits),
        "from_builder must preserve child ang_vel verbatim",
    );

    // Topology: the tree mutation still happened. The bypass path
    // skips only the `combine_states_at_attach` writeback — tree
    // mutation, composite-mass resync, and integrator-history reset
    // all still run. `MassBodyId` is `pub type MassBodyId = usize`
    // and `MassTree::add_body` returns sequential ids in registration
    // order; `from_builder` calls `add_body_to_tree` in the order the
    // builder registered them, so parent gets id 0 and child gets id 1.
    let tree = sim
        .mass_tree
        .as_ref()
        .expect("from_builder must wire the mass tree when attach_bodies is called");
    let parent_mass_body_id: astrodyn::MassBodyId = 0;
    let child_mass_body_id: astrodyn::MassBodyId = 1;
    assert_eq!(
        tree.parent(child_mass_body_id),
        Some(parent_mass_body_id),
        "child must be parented under parent in the mass tree post-builder"
    );
    let parent_composite_mass = tree.get(parent_mass_body_id).composite_properties.mass;
    let total_core_mass = parent_mass.mass + child_mass.mass;
    assert!(
        (parent_composite_mass - total_core_mass).abs() < 1e-12,
        "parent's post-attach composite mass must equal parent + child: \
         got {parent_composite_mass}, expected {total_core_mass}"
    );
}

/// Inertia of a point mass `m` at offset `r` (parallel-axis term),
/// computed independently of the kernel's `point_mass_inertia`.
fn point_mass(m: f64, r: DVec3) -> DMat3 {
    let outer = DMat3::from_cols(r * r.x, r * r.y, r * r.z);
    DMat3::from_diagonal(DVec3::splat(r.length_squared())) * m - outer * m
}

/// Max per-column L2 distance between two matrices.
fn mat3_max_col_diff(a: DMat3, b: DMat3) -> f64 {
    let d = a - b;
    [d.x_axis, d.y_axis, d.z_axis]
        .into_iter()
        .map(|c| c.length())
        .fold(0.0_f64, f64::max)
}

/// A composite whose **parent has a non-identity, non-180° struct→body
/// orientation** must carry its inertia in the body frame end-to-end:
/// `recompute_composites` builds the composite in the parent body frame and
/// `sync_body_mass_from_tree` hands it to the integrated body unchanged, so the
/// rotational integrator (Euler's equation, body-frame `inertia · ω`) consumes
/// a body-frame tensor.
///
/// `tier3_sim_attach_mass::RUN_09` cross-validates the body-frame composite
/// *value* against JEOD's `mass.out`. This test guards the full Simulation
/// `attach → sync_body_mass_from_tree → body.mass → integrate` pipeline for a
/// **general** orientation — the case every existing trajectory scenario is
/// blind to, because the only non-identity orientations in the suite are
/// Apollo's `yaw_180`, which is inertia-invariant on its diagonal tensors.
///
/// The reference is derived independently via the frame-invariance identity:
/// the same composite built in the **struct** frame (core rotated body→struct
/// by `Sᵀ·I·S`, struct-frame parallel-axis offsets) and conjugated by `S`
/// equals the body-frame composite. A sensitivity guard asserts the pipeline
/// result is *not* the unrotated struct composite, so a regression that dropped
/// the body-frame rotation fails loudly rather than silently.
#[test]
fn runner_attach_composite_inertia_is_body_frame() {
    // General struct→body rotation (0.5 rad about a tilted axis): a yaw_180 or
    // identity would hide the struct/body distinction this test exists to pin.
    let s = DMat3::from_axis_angle(DVec3::new(1.0, 2.0, 3.0).normalize(), 0.5);

    // Parent: asymmetric inertia in the BODY frame (a StructCG init would have
    // already rotated it to body), with the non-identity struct→body transform.
    let parent_body_inertia = DMat3::from_diagonal(DVec3::new(150.0, 200.0, 250.0));
    let parent_mass = SimMassProperties::with_inertia(2.0, parent_body_inertia, DVec3::ZERO)
        .with_t_parent_this(s);
    // Child: identity orientation, body-frame inertia, attached at an off-axis
    // struct offset so the composite is genuinely asymmetric (off-diagonal).
    let child_body_inertia = DMat3::from_diagonal(DVec3::new(40.0, 50.0, 60.0));
    let child_mass = SimMassProperties::with_inertia(1.0, child_body_inertia, DVec3::ZERO);
    let offset = DVec3::new(2.0, 1.0, -0.5);

    let omega0 = DVec3::new(0.05, 0.02, -0.01);
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let parent_rot = Some(RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: omega0,
    });
    let child_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0) + offset,
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let child_rot = Some(RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: omega0,
    });

    let (mut sim, parent_idx, child_idx, _pid, _cid) = build_pair(
        0.5,
        parent_mass,
        parent_trans,
        parent_rot,
        child_mass,
        child_trans,
        child_rot,
    );
    sim.attach(child_idx, parent_idx, offset, DMat3::IDENTITY);

    // Pipeline result: the integrated parent's composite inertia, body-frame.
    let pipeline_body = sim
        .body_mass(parent_idx)
        .expect("parent mass after attach")
        .inertia
        .as_dmat3();

    // Independent reference: build the composite in the struct frame and
    // conjugate by S. Composite CoM (struct) = m_c/(m_p+m_c) along the offset.
    let cm_struct = offset * (child_mass.mass / (parent_mass.mass + child_mass.mass));
    let parent_core_struct = s.transpose() * parent_body_inertia * s;
    let struct_composite = parent_core_struct
        + point_mass(parent_mass.mass, -cm_struct)
        + child_body_inertia
        + point_mass(child_mass.mass, offset - cm_struct);
    let expected_body = s * struct_composite * s.transpose();

    assert!(
        mat3_max_col_diff(pipeline_body, expected_body) < 1e-9,
        "pipeline composite inertia must be the body-frame composite \
         (S·I_struct·Sᵀ); got {pipeline_body:?}, expected {expected_body:?}"
    );
    // Sensitivity: for this general S the body-frame composite differs
    // substantially from the unrotated struct composite — proves the
    // struct→body rotation actually happened along the full pipeline.
    assert!(
        mat3_max_col_diff(pipeline_body, struct_composite) > 1.0,
        "body-frame composite must differ from the struct-frame composite for a \
         general orientation (else the body-frame rotation was dropped)"
    );

    // Smoke: the integrator propagates torque-free (mu = 0, no external torque)
    // with the body-frame inertia. Body-frame angular momentum magnitude
    // |I·ω| is conserved for a torque-free rigid body.
    let omega_after_attach = sim
        .body(parent_idx)
        .rot
        .expect("parent rot after attach")
        .ang_vel_body
        .raw_si();
    let h0 = (pipeline_body * omega_after_attach).length();
    for _ in 0..200 {
        sim.step().expect("torque-free step");
    }
    let omega_final = sim
        .body(parent_idx)
        .rot
        .expect("parent rot after stepping")
        .ang_vel_body
        .raw_si();
    let hf = (pipeline_body * omega_final).length();
    assert!(
        omega_final.is_finite() && (hf - h0).abs() <= 1e-6 * h0.max(1.0),
        "torque-free body-frame angular momentum |I·ω| must be conserved: \
         |H0|={h0}, |Hf|={hf}"
    );
}
