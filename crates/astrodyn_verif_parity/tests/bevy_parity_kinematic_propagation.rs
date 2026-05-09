// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Bevy ECS vs `astrodyn_runner::Simulation` parity for kinematic state
//! propagation.
//!
//! Builds the same parent + kinematic-child topology in both runtimes
//! with identical initial conditions (no force, no torque, RK4 on
//! 6-DOF rigid bodies). Drives the production attach surfaces in both
//! runtimes — `AttachEvent` on the Bevy side, `Simulation::attach` on
//! the runner side — so each adapter runs JEOD's
//! `combine_states_at_attach` momentum-conservation kernel on the
//! pre-attach pair before installing the kinematic chain. Steps both
//! forward and asserts the parent's and child's `composite_body`
//! inertial states match between runtimes at every checkpoint.
//!
//! # What is pinned
//!
//! 1. **Combine parity**: both adapters feed
//!    `combine_states_at_attach` bit-identical pre-merge inputs (each
//!    side reads `parent_mass` from its own pre-attach storage —
//!    `body.mass` on the runner, `MassPropertiesC` on Bevy — so the
//!    `composite_mass_system` revert race
//!    cannot reach either side's kernel input on the attach tick).
//!    The merged composite-body state lands in the parent's
//!    `body.trans` / `body.rot` (runner) and `TranslationalStateC` /
//!    `RotationalStateC` (Bevy) bit-identically.
//!
//! 2. **Kinematic propagation parity**: with the chain established
//!    (`MassChildOf` ECS edge on Bevy, `mark_kinematic_only` flag on
//!    the runner) the child is derived each tick by
//!    [`astrodyn::propagate_state_via_storage`]. Both runtimes
//!    delegate to the same storage-agnostic kernel, so the per-tick
//!    derivation is bit-identical. Any drift between runtimes would
//!    mean one of the adapters mis-routes the kernel inputs, not that
//!    the physics kernel itself diverged.
//!
//! # Tick-1 / steady-state separation
//!
//! The Bevy adapter's simple-attach contract leaves the `MassChildOf`
//! ECS edge insertion to mission code (only the chained-attach reroot
//! path inserts it inside `staging_system`). To keep the kernel input
//! parity from being broken by a pre-installed `MassChildOf` —
//! `composite_mass_system` would write the combined mass into the
//! parent's `MassPropertiesC` *before* `staging_system` reads it,
//! feeding the kernel a doubled-mass `parent_mass` — the test
//! installs `MassChildOf` on Bevy *and* calls `mark_kinematic_only`
//! on the runner *after* the first post-attach tick. The runner
//! mirrors Bevy's tick-1 schedule by leaving the child as a regular
//! integrator-driven body during tick 1 and transitioning it to
//! kinematic-only only from tick 2 onward.

#![allow(deprecated)]

use astrodyn::IntegratorType;
use astrodyn::MassProperties;
use astrodyn::{
    DynamicsConfig, GravityControls, JeodQuat, MassTree, RotationalState, SimulationTime,
    SixDofState, TranslationalState, VehicleConfig,
};
use astrodyn_bevy::{
    AstrodynPlugin, AttachEvent, DynamicsConfigC, ExternalForceC, ExternalTorqueC,
    FrameDerivativesC, GravityControlsC, MassBodyIdC, MassChildOf, MassPropertiesC, MassTreeR,
    RotationalStateC, TotalForceC, TranslationalStateC,
};
use bevy::prelude::*;
use glam::{DMat3, DVec3};

mod common;
use common::assert_sixdof_eq;

const DT: f64 = 0.1;
/// Total `FixedUpdate` ticks (Bevy) and `Simulation::step` calls
/// (runner) after the production attach is fired. The first tick
/// processes the attach, with the chain transitioning to kinematic-
/// only state at the start of tick 2 — see the file-level docstring's
/// "Tick-1 / steady-state separation" section.
const NUM_STEPS: usize = 30;

fn parent_mass() -> MassProperties {
    let inertia = DMat3::from_diagonal(DVec3::splat(20.0));
    MassProperties::with_inertia(2.0, inertia, DVec3::new(5.0, 0.0, 0.0))
}

fn child_mass() -> MassProperties {
    let inertia = DMat3::from_diagonal(DVec3::splat(10.0));
    MassProperties::with_inertia(1.0, inertia, DVec3::new(5.0, 0.0, 0.0))
}

fn parent_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(5.0, 10.0, 0.0),
        velocity: DVec3::new(0.0, 0.0, 0.5),
    }
}

fn parent_rot() -> RotationalState {
    let q = JeodQuat::left_quat_from_eigen_rotation(-0.5, DVec3::Z);
    RotationalState {
        quaternion: q,
        ang_vel_body: DVec3::new(0.0, 0.0, 0.2),
    }
}

/// Initial child translational state. Set equal to the parent's at
/// `t = 0` so the [`combine_states_at_attach`] merge is a soft merge
/// (no relative motion). The merge still runs the full algorithm on
/// both adapters — the parent's composite-body position shifts by
/// the mass-weighted CoM-delta in struct frame, the merged inertia
/// is recomputed via parallel-axis, and the angular-momentum solve
/// runs against the new combined inertia — but the two sides feed
/// the kernel bit-identical pre-merge inputs so the post-merge
/// parent state is deterministic across runtimes without the test
/// having to predict the kernel output ahead of time.
fn child_initial_trans() -> TranslationalState {
    parent_trans()
}

fn child_initial_rot() -> RotationalState {
    parent_rot()
}

/// Link geometry: child structural origin at (-10, 0, 0) in parent's
/// struct frame, identity link rotation. Mirrors the simple-attach
/// case from `tier3_sim_kinematic_propagation`.
fn link_offset() -> DVec3 {
    DVec3::new(-10.0, 0.0, 0.0)
}

fn link_t_parent_child() -> DMat3 {
    DMat3::IDENTITY
}

/// Build the runner-side simulation with parent + child registered as
/// independent free-flying bodies, then drive
/// [`astrodyn_runner::Simulation::attach`] to install the kinematic chain.
/// `Simulation::attach` runs `combine_states_at_attach` (post-#297)
/// and writes the merged composite-body state back into the parent's
/// `body.trans` / `body.rot`. The child is *not* marked kinematic-
/// only here — see the file-level "Tick-1 / steady-state separation"
/// section: the Bevy adapter's simple-attach contract does not
/// install `MassChildOf` from `staging_system`, so the chain becomes
/// kinematic-only only from tick 2 on the Bevy side, and the runner
/// mirrors that by leaving the child as a regular integrator-driven
/// body during tick 1.
fn build_runner_sim() -> (astrodyn_runner::Simulation, usize, usize) {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = astrodyn_runner::Simulation::new(time, DT);
    let parent_idx = sim.add_body(VehicleConfig {
        trans: astrodyn_bevy::typed_bridge::trans_raw_to_root(&parent_trans()),
        rot: Some(astrodyn_bevy::typed_bridge::rot_raw_to_self_ref(
            &(parent_rot()),
        )),
        mass: Some(astrodyn_bevy::typed_bridge::mass_raw_to_self_ref(
            &(parent_mass()),
        )),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    let child_idx = sim.add_body(VehicleConfig {
        trans: astrodyn_bevy::typed_bridge::trans_raw_to_root(&child_initial_trans()),
        rot: Some(astrodyn_bevy::typed_bridge::rot_raw_to_self_ref(
            &(child_initial_rot()),
        )),
        mass: Some(astrodyn_bevy::typed_bridge::mass_raw_to_self_ref(
            &(child_mass()),
        )),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    sim.add_body_to_tree(parent_idx, "parent");
    sim.add_body_to_tree(child_idx, "child");
    // Production runtime attach: runs `combine_states_at_attach` on
    // the pre-attach (parent, child) pair and writes the merged
    // composite-body state back into the parent's `body.trans` /
    // `body.rot`. Mirrors what the Bevy adapter's `staging_system`
    // does for `AttachEvent`.
    sim.attach(child_idx, parent_idx, link_offset(), link_t_parent_child());
    (sim, parent_idx, child_idx)
}

/// Build the Bevy app with parent + child as free-flying bodies and
/// fire an [`AttachEvent`] through the production message bus so
/// `staging_system` runs `combine_states_at_attach` (JEOD's momentum-
/// conservation algorithm — `models/dynamics/dyn_body/src/dyn_body_attach.cc`).
/// The first `FixedUpdate` tick processes the event:
/// `staging_system` writes the merged composite-body state into the
/// parent's `TranslationalStateC` / `RotationalStateC`. Mirrors the
/// runner's `Simulation::attach` orchestration so both adapters feed
/// the kernel bit-identical inputs.
///
/// The kinematic-chain `MassChildOf` ECS edge is *not* installed
/// here — see the file-level "Tick-1 / steady-state separation"
/// section. Pre-installing the edge would race
/// `composite_mass_system` against `staging_system` on the attach
/// tick, so the test installs it after the first tick instead.
fn build_bevy_app() -> (App, Entity, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);

    let mut tree = MassTree::new();
    let parent_id = tree.add_body("Parent".into(), parent_mass());
    let child_id = tree.add_body("Child".into(), child_mass());
    app.insert_resource(MassTreeR(tree));

    let parent = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            MassPropertiesC::from(astrodyn_bevy::typed_bridge::mass_raw_to_self_ref(
                &(parent_mass()),
            )),
            MassBodyIdC(parent_id),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(parent_trans()),
            RotationalStateC::from(astrodyn_bevy::typed_bridge::rot_raw_to_self_ref(
                &(parent_rot()),
            )),
            TotalForceC::default(),
            FrameDerivativesC::default(),
            ExternalForceC::default(),
            ExternalTorqueC::default(),
            GravityControlsC(GravityControls { controls: vec![] }),
        ))
        .id();
    let child = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            MassPropertiesC::from(astrodyn_bevy::typed_bridge::mass_raw_to_self_ref(
                &(child_mass()),
            )),
            MassBodyIdC(child_id),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(child_initial_trans()),
            RotationalStateC::from(astrodyn_bevy::typed_bridge::rot_raw_to_self_ref(
                &(child_initial_rot()),
            )),
            TotalForceC::default(),
            FrameDerivativesC::default(),
            ExternalForceC::default(),
            ExternalTorqueC::default(),
            GravityControlsC(GravityControls { controls: vec![] }),
        ))
        .id();

    // Fire the production attach event. `staging_system` consumes it
    // on the next `FixedUpdate` tick: it runs `combine_states_at_attach`
    // and writes the merged composite-body state back into the
    // parent's components.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child,
            parent,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                link_offset(),
            ),
            t_parent_child: astrodyn::FrameTransform::from_matrix(link_t_parent_child()),
        });

    (app, parent, child)
}

/// Step the Bevy app `n` `FixedUpdate` ticks. `run_schedule(FixedUpdate)`
/// is the canonical way to drive the JEOD pipeline forward in tests
/// (see existing parity tests in `tests/bevy_parity_*.rs`); `app.update()`
/// runs only the `Update` schedule which the JEOD plugin doesn't use.
fn step_bevy(app: &mut App, n: usize) {
    for _ in 0..n {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(std::time::Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
    }
}

/// Read child + parent state from Bevy at the latest tick.
fn read_bevy_state(
    app: &App,
    parent: Entity,
    child: Entity,
) -> (
    TranslationalState,
    RotationalState,
    TranslationalState,
    RotationalState,
) {
    let p_trans = astrodyn_bevy::typed_bridge::trans_typed_to_raw(
        &app.world()
            .get::<TranslationalStateC<astrodyn::Earth>>(parent)
            .unwrap()
            .0,
    );
    let p_rot = astrodyn_bevy::typed_bridge::rot_typed_to_raw(
        &app.world().get::<RotationalStateC>(parent).unwrap().0,
    );
    let c_trans = astrodyn_bevy::typed_bridge::trans_typed_to_raw(
        &app.world()
            .get::<TranslationalStateC<astrodyn::Earth>>(child)
            .unwrap()
            .0,
    );
    let c_rot = astrodyn_bevy::typed_bridge::rot_typed_to_raw(
        &app.world().get::<RotationalStateC>(child).unwrap().0,
    );
    (p_trans, p_rot, c_trans, c_rot)
}

/// Inputs to `kernel_from_parent` plus the helper itself. Used to
/// run the kinematic kernel against the runner's parent state and
/// assert the runner's own child state matches the kernel output.
fn kernel_from_parent(
    parent: &TranslationalState,
    parent_rot: &RotationalState,
) -> (TranslationalState, RotationalState) {
    use astrodyn::{compute_kinematic_child_state, KinematicChildInputs};
    let parent_t_inertial_body = parent_rot.quaternion.left_quat_to_transformation();
    // Combined composite CoM in parent struct frame (for atomic
    // bodies the core position equals the composite position).
    let parent_cm = DVec3::new(5.0, 0.0, 0.0);
    let child_cm = DVec3::new(5.0, 0.0, 0.0);
    let child_in_parent_struct = link_offset() + link_t_parent_child().transpose() * child_cm;
    // Match `propagate_kinematic_state` in the runner and
    // `propagate_state_from_root_system` in Bevy: both read
    // `composite_properties.position` post-attach which reflects the
    // mass-weighted combined CoM in the parent's struct frame.
    let combined_in_pstr = (parent_cm * parent_mass().mass
        + child_in_parent_struct * child_mass().mass)
        / (parent_mass().mass + child_mass().mass);
    let inputs = KinematicChildInputs {
        parent_t_inertial_body,
        parent_ang_vel_body: parent_rot.ang_vel_body,
        parent_position_inertial: parent.position,
        parent_velocity_inertial: parent.velocity,
        parent_t_struct_body: DMat3::IDENTITY,
        parent_composite_in_pstr: combined_in_pstr,
        t_parent_child: link_t_parent_child(),
        link_offset_in_pstr: link_offset(),
        child_t_struct_body: DMat3::IDENTITY,
        child_composite_in_cstr: child_cm,
    };
    let out = compute_kinematic_child_state(inputs);
    (
        TranslationalState {
            position: out.child_position_inertial,
            velocity: out.child_velocity_inertial,
        },
        RotationalState {
            quaternion: out.child_q_inertial_body,
            ang_vel_body: out.child_ang_vel_body,
        },
    )
}

/// Cross-runtime parity helper: pack the runner-side and Bevy-side 6-DOF
/// state pairs into [`SixDofState`] and delegate to the shared
/// [`assert_sixdof_eq`] (which compares each of the 13 components via
/// `to_bits()`).
///
/// Bit-identity is the right contract here because both runtimes drive
/// the same `combine_states_at_attach` /
/// `propagate_state_via_storage` kernels and the same RK4 integrator
/// with no scheduling non-determinism — any drift would indicate one
/// of the adapters mis-routes the kernel inputs, not a physics
/// divergence. Loose `< 1e-12` tolerances would silently mask
/// exactly that class of bug, so this helper is wired through the same
/// `to_bits()` checker used by every other `bevy_parity_*.rs` test.
fn assert_states_bit_identical(
    bevy: &TranslationalState,
    runner: &TranslationalState,
    bevy_rot: &RotationalState,
    runner_rot: &RotationalState,
    label: &str,
) {
    let bevy_state = SixDofState {
        trans: *bevy,
        rot: *bevy_rot,
    };
    let runner_state = SixDofState {
        trans: *runner,
        rot: *runner_rot,
    };
    assert_sixdof_eq(label, &bevy_state, &runner_state);
}

/// Bevy adapter and runner produce bit-identical parent **and** child
/// state when both runtimes drive their respective production attach
/// surfaces (`AttachEvent` for Bevy, `Simulation::attach` for the
/// runner) and run the kinematic chain through `MassChildOf` /
/// `mark_kinematic_only` from tick 2 onward.
///
/// Per-tick structure (matching the file-level "Tick-1 / steady-state
/// separation" docstring):
///
/// 1. `build_runner_sim` calls `Simulation::attach`; `build_bevy_app`
///    queues an `AttachEvent`. Neither side has the chain established
///    yet (no `MassChildOf` on Bevy, no `kinematic_only` flag on the
///    runner).
/// 2. Tick 1: Bevy's `staging_system` consumes the `AttachEvent` and
///    writes the merged composite-body state into the parent. The
///    runner has already merged synchronously. The child integrates
///    one free-flight tick on both sides because the chain is not
///    yet established. Both runtimes are bit-identical at end of
///    tick 1.
/// 3. Between ticks 1 and 2, mission code installs `MassChildOf`
///    (Bevy) and `mark_kinematic_only` (runner). `composite_mass_system`
///    on tick 2's start sees the new edge and writes the combined
///    mass into the parent's `MassPropertiesC` via
///    `bypass_change_detection`.
/// 4. Ticks 2..NUM_STEPS: the chain is steady-state. The integrator
///    advances the parent only; `propagate_state_from_root_post_integration_system`
///    (Bevy) and `propagate_kinematic_state` (runner) derive the
///    child from the just-integrated parent.
///
/// Asserts at end of `NUM_STEPS`:
/// 1. parent translational + rotational state is bit-identical
///    between Bevy and runner (`to_bits()` per component);
/// 2. child translational + rotational state is bit-identical
///    between Bevy and runner;
/// 3. runner's child state == `kernel(runner.parent, link)`,
///    bit-identical (kernel-self-consistency sanity check that the
///    parity comes from the correct kernel inputs, not coincidence).
#[test]
fn bevy_parity_kinematic_propagation_simple_chain() {
    let (mut sim, parent_idx, child_idx) = build_runner_sim();
    let (mut app, parent_entity, child_entity) = build_bevy_app();

    // Tick 1: process the production attach. Both runtimes merge
    // the parent state via `combine_states_at_attach`; the child
    // integrates one free-flight tick because neither side has
    // installed the kinematic-chain marker yet.
    sim.step().expect("runner step must succeed");
    step_bevy(&mut app, 1);

    // Install the kinematic-chain handles on both sides for the
    // remaining ticks. On Bevy this is the `MassChildOf` edge that
    // mission code owns under JEOD's simple-attach contract; on the
    // runner the equivalent handle is the `kinematic_only` flag set
    // via `mark_kinematic_only`.
    app.world_mut()
        .entity_mut(child_entity)
        .insert(MassChildOf::with_rotation(
            parent_entity,
            link_offset(),
            link_t_parent_child(),
        ));
    sim.mark_kinematic_only(child_idx);

    // Ticks 2..NUM_STEPS: steady-state kinematic propagation.
    sim.step_n(NUM_STEPS - 1)
        .expect("runner step_n must succeed");
    step_bevy(&mut app, NUM_STEPS - 1);

    let runner_p = sim.body(parent_idx);
    let runner_c = sim.body(child_idx);
    let (bevy_p_trans, bevy_p_rot, bevy_c_trans, bevy_c_rot) =
        read_bevy_state(&app, parent_entity, child_entity);

    let runner_p_trans = astrodyn_bevy::typed_bridge::trans_typed_to_raw(&runner_p.trans);
    let runner_p_rot = astrodyn_bevy::typed_bridge::rot_typed_to_raw(&runner_p.rot.unwrap());
    let runner_c_trans = astrodyn_bevy::typed_bridge::trans_typed_to_raw(&runner_c.trans);
    let runner_c_rot = astrodyn_bevy::typed_bridge::rot_typed_to_raw(&runner_c.rot.unwrap());

    // ── Invariant 1: parent state is bit-identical across runtimes.
    assert_states_bit_identical(
        &bevy_p_trans,
        &runner_p_trans,
        &bevy_p_rot,
        &runner_p_rot,
        "parent",
    );

    // ── Invariant 2: child state is bit-identical across runtimes.
    //    Both Bevy and runner now run kinematic propagation pre+post
    //    integration, so the child reflects the same-tick parent in
    //    both runtimes.
    assert_states_bit_identical(
        &bevy_c_trans,
        &runner_c_trans,
        &bevy_c_rot,
        &runner_c_rot,
        "child",
    );

    // ── Invariant 3: runner's child state matches its kernel on
    //    the runner's parent state. Both `Simulation::step()` and
    //    `kernel_from_parent` here drive the *same*
    //    `compute_kinematic_child_state` with bit-identical inputs
    //    (the hand-rolled mass-weighted CoM matches what
    //    `MassTree::attach` stores in `composite_properties.position`),
    //    so this is a true parity check — assert via `to_bits()` per
    //    component, mirroring `assert_sixdof_eq`.
    let (predicted_child_trans, predicted_child_rot) =
        kernel_from_parent(&runner_p_trans, &runner_p_rot);
    let runner_predicted = SixDofState {
        trans: predicted_child_trans,
        rot: predicted_child_rot,
    };
    let runner_actual = SixDofState {
        trans: astrodyn_bevy::typed_bridge::trans_typed_to_raw(&runner_c.trans),
        rot: astrodyn_bevy::typed_bridge::rot_typed_to_raw(&runner_c.rot.unwrap()),
    };
    assert_sixdof_eq(
        "runner kernel-consistency (child vs kernel(runner.parent))",
        &runner_actual,
        &runner_predicted,
    );
}
