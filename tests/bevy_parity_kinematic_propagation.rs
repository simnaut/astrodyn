//! Bevy ECS vs `jeod_runner::Simulation` parity for kinematic state
//! propagation.
//!
//! Builds the same parent + kinematic-child topology in both runtimes
//! with identical initial conditions (no force, no torque, RK4 on
//! 6-DOF rigid bodies). Steps both forward and asserts the child's
//! `composite_body` inertial state matches between runtimes at every
//! checkpoint.
//!
//! This pins that the Bevy adapter's
//! `propagate_state_from_root_system` and the runner's
//! `propagate_kinematic_state` produce bit-equivalent state for the
//! same topology. Both delegate to the storage-agnostic kernel
//! [`jeod_sim::propagate_state_via_storage`], so the parity is
//! structural — any drift between runtimes would mean one of the
//! adapters mis-routes the kernel inputs, not that the physics
//! kernel itself diverged.

#![allow(deprecated)]

use bevy::prelude::*;
use bevy_jeod::{
    DynamicsConfigC, ExternalForceC, ExternalTorqueC, FrameDerivativesC, GravityControlsC,
    JeodPlugin, KinematicChildC, MassBodyIdC, MassChildOf, MassPropertiesC, MassTreeR,
    RotationalStateC, TotalForceC, TranslationalStateC,
};
use glam::{DMat3, DVec3};
use jeod_dynamics::{IntegratorType, MassProperties};
use jeod_sim::{
    DynamicsConfig, GravityControls, JeodQuat, MassTree, RotationalState, SimulationTime,
    SixDofState, TranslationalState, VehicleConfig,
};

mod common;
use common::assert_sixdof_eq;

const DT: f64 = 0.1;
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

/// Link geometry: child structural origin at (-10, 0, 0) in parent's
/// struct frame, identity link rotation. Mirrors the simple-attach
/// case from `tier3_sim_kinematic_propagation`.
fn link_offset() -> DVec3 {
    DVec3::new(-10.0, 0.0, 0.0)
}

fn link_t_parent_child() -> DMat3 {
    DMat3::IDENTITY
}

/// Build the runner-side simulation with parent + kinematic child.
fn build_runner_sim() -> (jeod_runner::Simulation, usize, usize) {
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = jeod_runner::Simulation::new(time, DT);
    let parent_idx = sim.add_body(VehicleConfig {
        trans: parent_trans(),
        rot: Some(parent_rot()),
        mass: Some(parent_mass()),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    // Child starts with junk state; propagation must overwrite it from
    // the parent every tick. The link is wired via direct mass-tree
    // mutation (rather than `Simulation::attach`) to bypass the JEOD
    // momentum-conservation combine that #297 added: `staging_system`
    // on the Bevy side likewise installs `MassChildOf` directly here,
    // so both adapters carry the parent's pre-attach integrated state
    // and the kinematic propagation derives the child the same way in
    // both. The combine path is exercised separately in
    // `bevy_parity_attach_detach_momentum`.
    let child_idx = sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: DVec3::splat(1e9),
            velocity: DVec3::splat(1e9),
        },
        rot: Some(RotationalState::default()),
        mass: Some(child_mass()),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    let parent_id = sim.add_body_to_tree(parent_idx, "parent");
    let child_id = sim.add_body_to_tree(child_idx, "child");
    // Tree-only attach: skip `Simulation::attach`'s combine to mirror
    // the Bevy app builder's direct `MassChildOf` insertion. The
    // low-level contract documented on `sync_body_mass_from_tree`
    // (`crates/jeod_runner/src/simulation/bodies.rs:586-602`) requires
    // syncing **every** SimBody whose tree node was touched by the
    // mutation — the directly-attached child plus the parent's full
    // ancestor chain. The child's composite_properties still equals
    // its core mass at a leaf attach (no grandchildren contribute), so
    // the mass-write is numerically a no-op there; the load-bearing
    // part of the call is the integrator-history book-keeping
    // (`gj_state` / `abm4_state` topology-dirty flag) that the same
    // contract requires on every topology change. With RK4 (this
    // fixture's integrator) those fields are `None` and the call is
    // a no-op end-to-end, but invoking it here keeps the shared parity
    // setup correct for any future multistep variant that reuses
    // `build_runner_sim`.
    sim.mass_tree
        .as_mut()
        .expect("mass tree present after add_body_to_tree")
        .attach(child_id, parent_id, link_offset(), link_t_parent_child());
    sim.sync_body_mass_from_tree(parent_idx);
    sim.sync_body_mass_from_tree(child_idx);
    sim.mark_kinematic_only(child_idx);
    (sim, parent_idx, child_idx)
}

/// Build the Bevy app with parent + kinematic child, installing the
/// `MassChildOf` link directly (bypassing `AttachEvent`).
///
/// Why bypass `AttachEvent`: `staging_system` calls
/// `combine_states_at_attach` (JEOD's momentum-conservation algorithm
/// — `models/dynamics/dyn_body/src/dyn_body_attach.cc`) when processing
/// the event, which shifts the parent's `composite_body` inertial
/// state by the inertial CoM-delta. The runner-side counterpart
/// (`Simulation::attach`, post-#297) runs the same kernel. Both are
/// covered end-to-end by `bevy_parity_attach_detach_momentum`. To keep
/// the kinematic-propagation parity check focused on the per-tick
/// child-derivation walk (the one structural invariant this test
/// pins), the runner side wires the link via direct
/// [`MassTree::attach`] mutation and the Bevy side inserts
/// `MassChildOf` directly — both adapters skip the combine and carry
/// the parent's pre-attach integrated state, so the kinematic
/// propagation derives the child the same way in both.
fn build_bevy_app() -> (App, Entity, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    let mut tree = MassTree::new();
    let parent_id = tree.add_body("Parent".into(), parent_mass());
    let child_id = tree.add_body("Child".into(), child_mass());
    // Wire the topology in the arena too so composite-mass agrees
    // with the Bevy ECS view.
    tree.attach(child_id, parent_id, link_offset(), link_t_parent_child());
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
            MassPropertiesC::from(parent_mass()),
            MassBodyIdC(parent_id),
            TranslationalStateC::<jeod_sim::Earth>::from(parent_trans()),
            RotationalStateC::from(parent_rot()),
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
            MassPropertiesC::from(child_mass()),
            MassBodyIdC(child_id),
            // The link itself — pre-installed so the first
            // FixedUpdate tick already sees the chain. Mirrors the
            // runner's direct `mass_tree.attach` setup.
            MassChildOf::with_rotation(parent, link_offset(), link_t_parent_child()),
            // Pin the child as kinematic-only up front rather than
            // letting `wrench_aggregation_system` infer the marker
            // from topology on the first tick. The integration system
            // gates on `Without<KinematicChildC>`, so without an
            // explicit insertion the test would depend on the inferral
            // running before integration on tick 0 — order-fragile
            // behaviour the parity check shouldn't pivot on.
            KinematicChildC,
            // Stale state — propagation must overwrite both.
            TranslationalStateC::<jeod_sim::Earth>::default(),
            RotationalStateC::default(),
            TotalForceC::default(),
            FrameDerivativesC::default(),
            ExternalForceC::default(),
            ExternalTorqueC::default(),
            GravityControlsC(GravityControls { controls: vec![] }),
        ))
        .id();

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
    let p_trans = app
        .world()
        .get::<TranslationalStateC<jeod_sim::Earth>>(parent)
        .unwrap()
        .0
        .to_untyped();
    let p_rot = app
        .world()
        .get::<RotationalStateC>(parent)
        .unwrap()
        .0
        .to_untyped();
    let c_trans = app
        .world()
        .get::<TranslationalStateC<jeod_sim::Earth>>(child)
        .unwrap()
        .0
        .to_untyped();
    let c_rot = app
        .world()
        .get::<RotationalStateC>(child)
        .unwrap()
        .0
        .to_untyped();
    (p_trans, p_rot, c_trans, c_rot)
}

/// Inputs to `kernel_from_parent` plus the helper itself. Used to
/// run the kinematic kernel against each runtime's parent state and
/// assert the runtime's own child state matches the kernel output.
fn kernel_from_parent(
    parent: &TranslationalState,
    parent_rot: &RotationalState,
) -> (TranslationalState, RotationalState) {
    use jeod_dynamics::kinematic_propagation::{
        compute_kinematic_child_state, KinematicChildInputs,
    };
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
/// the same `propagate_state_via_storage` kernel and the same RK4
/// integrator with no scheduling non-determinism — any drift would
/// indicate one of the adapters mis-routes the kernel inputs, not a
/// physics divergence. Loose `< 1e-12` tolerances would silently mask
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
/// state for the same kinematic-chain topology.
///
/// The Bevy adapter runs `propagate_state_from_root_system` *both*
/// before and after integration each tick (mirroring the runner's
/// stage 3b / 8d pre+post sweeps in
/// `crates/jeod_runner/src/simulation/step/mod.rs`), so a kinematic
/// child's `RotationalStateC` / `TranslationalStateC` reflects the
/// just-integrated parent state — the same value the runner's
/// post-integration pass installs. The previous one-tick lag (Bevy
/// pre-only vs runner pre+post) was a schedule-placement gap, not a
/// physics divergence.
///
/// Runs `NUM_STEPS` ticks at `DT=0.1 s`. Asserts:
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

    sim.step_n(NUM_STEPS).expect("runner step_n must succeed");
    step_bevy(&mut app, NUM_STEPS);

    let runner_p = sim.body(parent_idx);
    let runner_c = sim.body(child_idx);
    let (bevy_p_trans, bevy_p_rot, bevy_c_trans, bevy_c_rot) =
        read_bevy_state(&app, parent_entity, child_entity);

    // ── Invariant 1: parent state is bit-identical across runtimes.
    assert_states_bit_identical(
        &bevy_p_trans,
        &runner_p.trans,
        &bevy_p_rot,
        &runner_p.rot.unwrap(),
        "parent",
    );

    // ── Invariant 2: child state is bit-identical across runtimes.
    //    Both Bevy and runner now run kinematic propagation pre+post
    //    integration, so the child reflects the same-tick parent in
    //    both runtimes.
    assert_states_bit_identical(
        &bevy_c_trans,
        &runner_c.trans,
        &bevy_c_rot,
        &runner_c.rot.unwrap(),
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
        kernel_from_parent(&runner_p.trans, &runner_p.rot.unwrap());
    let runner_predicted = SixDofState {
        trans: predicted_child_trans,
        rot: predicted_child_rot,
    };
    let runner_actual = SixDofState {
        trans: runner_c.trans,
        rot: runner_c.rot.unwrap(),
    };
    assert_sixdof_eq(
        "runner kernel-consistency (child vs kernel(runner.parent))",
        &runner_actual,
        &runner_predicted,
    );
}
