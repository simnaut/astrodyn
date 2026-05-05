//! Bevy ECS vs `jeod_runner::Simulation` parity for the SIM_verif_
//! attach_detach RUN_simple_attach_detach scenario, driven through the
//! production `AttachEvent` surface end-to-end.
//!
//! Companion to `crates/jeod_runner/tests/tier3_sim_attach_detach_
//! trajectory.rs`: the Tier 3 test cross-validates the runner against
//! JEOD's CSV across the *full* attach+detach scenario; this parity
//! test cross-validates the Bevy adapter against the runner across
//! the *attach* portion only. Together they pin that the same JEOD-
//! faithful trajectory is produced regardless of which consumer of
//! `jeod_sim` drives the pipeline up to the detach event.
//!
//! # Attach scheduling — canonical timing
//!
//! Both runtimes fire their respective attach surfaces *after* the
//! integration step that lands at `t = ATTACH_TIME`, mirroring JEOD's
//! `trick.add_read(10, "veh1.attach_to_2.active = True")` semantics:
//! Trick's `input_processor_run` queue at simtime `t` fires the read
//! job at the start of the dispatch cycle for time `t`, *after* the
//! integrator has produced the `t = T` state from the previous tick's
//! `t-dt → t` integration of the still-separate bodies. The CSV row
//! at `t = ATTACH_TIME` therefore captures the post-read-job
//! (combined) state. This matches the timing in
//! `tier3_sim_attach_detach_trajectory_simple` and is the JEOD-
//! faithful schedule.
//!
//! The older "fire before the step that crosses `ATTACH_TIME`"
//! pattern integrates the `t-dt → t` step with the bodies *already*
//! attached, which produces a different post-attach trajectory:
//! `combine_states_at_attach` preserves the parent's quaternion and
//! averages momenta, so the combine input being the integrated-
//! once-attached state versus the separate state at `t` shifts the
//! merged angular and translational state. Keeping both new tests
//! on the same post-step timing guards against re-introducing the
//! inconsistency.
//!
//! Schedule asymmetry between the two runtimes on the attach-event
//! tick (see "What is pinned" below): `Simulation::attach` is
//! synchronous and writes the post-combine state to the runner's
//! integrated tree root in the same call; the Bevy adapter consumes
//! `AttachEvent` only at the start of the *next* `FixedUpdate`'s
//! `staging_system`. Both runtimes therefore feed the same separate
//! `t = ATTACH_TIME` state to the same `combine_states_at_attach`
//! kernel, but at different schedule positions; bit-identity holds
//! from the tick *after* the attach event onward.
//!
//! # What is pinned
//!
//! Both runtimes:
//!
//! - register the same three bodies (veh1 / veh2 / veh3) with
//!   identical initial conditions from `Modified_data/veh{1,2,3}.py`
//!   and `RUN_simple_attach_detach/input.py`,
//! - integrate forward through `Simulation::step()` (runner) and
//!   `App::run_schedule(FixedUpdate)` (Bevy) in lock-step,
//! - fire the in-flight attach via `Simulation::attach` (runner) and
//!   `AttachEvent` (Bevy) just after the step that lands at
//!   `t = ATTACH_TIME` (see "Attach scheduling" above).
//!
//! Validation runs from `t = 0` through `t < DETACH_TIME` and asserts
//! bit-identical `composite_body` state on the integrated tree root
//! (veh2 in the attached window, plus the always-free-flying veh3) at
//! every tick *except* the attach-event tick itself, where the
//! synchronous-vs-deferred schedule asymmetry above leaves
//! `runner.veh2` post-combine while Bevy still holds the pre-combine
//! state (the `AttachEvent` is in the queue but not yet consumed).
//! The kinematic-only veh1 in the attached window has a known one-
//! tick schedule asymmetry (Bevy runs propagation only before
//! integration; the runner runs it both before and after) and is
//! structurally covered by `bevy_parity_kinematic_propagation_
//! simple_chain` — this trajectory parity therefore asserts veh1
//! only while it is itself integrated.
//!
//! # What is **not** pinned (and why)
//!
//! Detach-time parent-side trajectory parity is excluded: the Bevy
//! adapter's `composite_mass_system` reverts the parent's
//! `MassPropertiesC` to its `CoreMassPropertiesC` each tick when the
//! ECS tree (`MassChildOf`) is empty, so `staging_system`'s detach
//! handler reads the parent's `parent_pre_composite_props.position`
//! as the core CoM rather than the post-attach combined CoM and the
//! parent-side inverse-combine writeback collapses to a no-op. This
//! is the dual-write coordination bug tracked under sub-issue #308
//! (`composite_mass_system` reverts `MassPropertiesC` for detached
//! entities before the detach handler reads it) — not in scope for
//! sub-issue #305. The same exclusion is documented at the bottom of
//! `bevy_parity_attach_detach_momentum::bevy_runner_parity_attach_
//! detach_momentum`. Once #308 is resolved, this test will gain a
//! detach-window parity assertion as a follow-up.
//!
//! # Why a separate parity test instead of extending
//! `bevy_parity_kinematic_propagation`
//!
//! `bevy_parity_kinematic_propagation` deliberately bypasses
//! `AttachEvent` (it inserts `MassChildOf` directly) so it can pin
//! kinematic-propagation parity in isolation from the
//! momentum-conservation combine. This test does the opposite — it
//! goes through `AttachEvent` end-to-end, exercising the full event
//! handler + staging-system path. The two tests are orthogonal: a
//! regression in `staging_system`'s combine math would pass the
//! kinematic-propagation parity (it bypasses the combine) and fail
//! this one; a regression in `propagate_state_from_root_system` would
//! fail both.

use bevy::prelude::*;
use bevy_jeod::{
    AttachEvent, DynamicsConfigC, ExternalForceC, ExternalTorqueC, FrameDerivativesC,
    GravityControlsC, JeodPlugin, MassBodyIdC, MassPropertiesC, MassTreeR, RotationalStateC,
    TotalForceC, TranslationalStateC,
};
use glam::{DMat3, DVec3};
use jeod_dynamics::{IntegratorType, MassProperties};
use jeod_runner::Simulation;
use jeod_sim::{
    DynamicsConfig, GravityControls, JeodQuat, MassTree, RotationalState, SimulationTime,
    SixDofState, TranslationalState, VehicleConfig,
};
use std::time::Duration;

mod common;
use common::assert_sixdof_eq;

const DT: f64 = 0.1;

/// `BodyAttachAligned veh1.attach_to_2` time
/// (`SET_test/RUN_simple_attach_detach/input.py:24`).
const ATTACH_TIME: f64 = 10.0;
/// `BodyDetach veh1.detach_from_2` time
/// (`SET_test/RUN_simple_attach_detach/input.py:25`). We stop the
/// parity comparison strictly before this — see the file-level
/// docstring's "What is **not** pinned" section for the #308 dual-
/// write reason.
const DETACH_TIME: f64 = 20.0;

/// Number of `DT`-sized steps to run. Equals `(DETACH_TIME / DT) - 1`
/// so the loop's last `step()`/`step_bevy()` call lands at
/// `t = DETACH_TIME - DT` (= 19.9 s) and never reaches
/// `t == DETACH_TIME`. The `-1` is the "stop strictly before
/// detach" fence: the bare `DETACH_TIME / DT` would advance the sim
/// to exactly `t = DETACH_TIME`, where `staging_system` would
/// observe the detach time the file-level "What is **not** pinned"
/// section explicitly excludes.
const NUM_STEPS: usize = (DETACH_TIME / DT) as usize - 1;

// ── Initial conditions, all from JEOD Modified_data files. ──

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

fn veh1_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(-5.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 1.0, 0.0),
    }
}

fn veh1_rot() -> RotationalState {
    RotationalState {
        quaternion: JeodQuat::from_array([1.0, 0.0, 0.0, 0.0]),
        ang_vel_body: DVec3::ZERO,
    }
}

fn veh2_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(5.0, 10.0, 0.0),
        velocity: DVec3::ZERO,
    }
}

fn veh2_rot() -> RotationalState {
    let q = JeodQuat::left_quat_from_eigen_rotation(-2.0, DVec3::Z);
    RotationalState {
        quaternion: q,
        ang_vel_body: DVec3::new(0.0, 0.0, 0.2),
    }
}

fn veh3_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(0.063, 13.787, -25.0),
        velocity: DVec3::new(0.0, 0.0, 1.0),
    }
}

fn veh3_rot() -> RotationalState {
    let q = JeodQuat::left_quat_from_eigen_rotation(-15.8, DVec3::Z);
    RotationalState {
        quaternion: q,
        ang_vel_body: DVec3::ZERO,
    }
}

/// JEOD's `BodyAttachAligned veh1.attach_to_2` — see the matching
/// docstring on `simple_attach_offset_and_rotation` in the Tier 3 test.
fn link_offset_and_rotation() -> (DVec3, DMat3) {
    (DVec3::new(-10.0, 0.0, 0.0), DMat3::IDENTITY)
}

fn six_dof_config() -> DynamicsConfig {
    DynamicsConfig {
        translational_dynamics: true,
        rotational_dynamics: true,
        three_dof: false,
    }
}

fn build_runner_sim() -> (Simulation, usize, usize, usize) {
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let v1 = sim.add_body(VehicleConfig {
        trans: veh1_trans(),
        rot: Some(veh1_rot()),
        mass: Some(veh1_mass()),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    let v2 = sim.add_body(VehicleConfig {
        trans: veh2_trans(),
        rot: Some(veh2_rot()),
        mass: Some(veh2_mass()),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    let v3 = sim.add_body(VehicleConfig {
        trans: veh3_trans(),
        rot: Some(veh3_rot()),
        mass: Some(veh3_mass()),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    sim.add_body_to_tree(v1, "veh1");
    sim.add_body_to_tree(v2, "veh2");
    sim.add_body_to_tree(v3, "veh3");
    (sim, v1, v2, v3)
}

fn build_bevy_app() -> (App, Entity, Entity, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    let mut tree = MassTree::new();
    let id_v1 = tree.add_body("veh1".into(), veh1_mass());
    let id_v2 = tree.add_body("veh2".into(), veh2_mass());
    let id_v3 = tree.add_body("veh3".into(), veh3_mass());
    app.insert_resource(MassTreeR(tree));

    let v1 = spawn_body(
        &mut app,
        "veh1",
        id_v1,
        veh1_mass(),
        veh1_trans(),
        veh1_rot(),
    );
    let v2 = spawn_body(
        &mut app,
        "veh2",
        id_v2,
        veh2_mass(),
        veh2_trans(),
        veh2_rot(),
    );
    let v3 = spawn_body(
        &mut app,
        "veh3",
        id_v3,
        veh3_mass(),
        veh3_trans(),
        veh3_rot(),
    );

    (app, v1, v2, v3)
}

fn spawn_body(
    app: &mut App,
    name: &str,
    id: jeod_sim::MassBodyId,
    mass: MassProperties,
    trans: TranslationalState,
    rot: RotationalState,
) -> Entity {
    app.world_mut()
        .spawn((
            Name::new(name.to_string()),
            DynamicsConfigC(six_dof_config()),
            MassPropertiesC::from(mass),
            MassBodyIdC(id),
            TranslationalStateC::from(trans),
            RotationalStateC::from(rot),
            TotalForceC::default(),
            FrameDerivativesC::default(),
            ExternalForceC::default(),
            ExternalTorqueC::default(),
            GravityControlsC(GravityControls { controls: vec![] }),
        ))
        .id()
}

fn step_bevy(app: &mut App) {
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);
}

fn read_bevy_state(app: &App, entity: Entity) -> SixDofState {
    let trans = app
        .world()
        .get::<TranslationalStateC>(entity)
        .expect("entity has TranslationalStateC")
        .0
        .to_untyped();
    let rot = app
        .world()
        .get::<RotationalStateC>(entity)
        .expect("entity has RotationalStateC")
        .0
        .to_untyped();
    SixDofState { trans, rot }
}

fn read_runner_state(sim: &Simulation, idx: usize) -> SixDofState {
    let out = sim.body(idx);
    SixDofState {
        trans: out.trans,
        rot: out
            .rot
            .expect("attach/detach trajectory parity runs every body in 6-DOF"),
    }
}

/// Bevy adapter and runner produce bit-identical `composite_body`
/// state for the integrator-written bodies (veh2 in the attached
/// window, plus the always-free-flying veh3) across the *attach
/// portion* of the SIM_verif_attach_detach RUN_simple_attach_detach
/// scenario, when both runtimes drive the in-flight attach through
/// their respective production event surfaces (`AttachEvent` in
/// Bevy, `Simulation::attach` in the runner). veh1 is asserted only
/// in the pre-attach window — the kinematic-only schedule asymmetry
/// during and after the attach is structurally covered by
/// `bevy_parity_kinematic_propagation_simple_chain`.
///
/// Detach-side parity is *not* asserted here. The loop terminates
/// strictly before `t == DETACH_TIME` (see `NUM_STEPS`) and no
/// `DetachEvent` is queued in either runtime; full attach+detach
/// runner-vs-CSV trajectory cross-validation is the job of the
/// companion `tier3_sim_attach_detach_trajectory_simple` test, and
/// the Bevy-vs-runner detach parity will be added here as a follow-
/// up once the #308 dual-write coordination is resolved (see the
/// file-level "What is **not** pinned" docstring section).
///
/// Per-step lock-step structure:
///
/// 1. step both runtimes by one tick of `DT = 0.1 s`,
/// 2. *immediately after* the step that lands at `t = ATTACH_TIME`,
///    fire the attach in both runtimes (synchronous
///    `Simulation::attach` on the runner, queued `AttachEvent` on
///    Bevy),
/// 3. read each runtime's state at the new simtime and assert
///    bit-identical agreement (`to_bits()` per component) on every
///    integrator-written body's full 6-DOF state — see the inline
///    contract comments below for the per-body skip rules on the
///    attach-event tick.
///
/// Both runtimes are deterministic and drive the same JEOD physics
/// kernels (`combine_states_at_attach`, `propagate_state_via_storage`,
/// RK4) with the same ordering, so bit-identity is the right contract:
/// any drift would mean one of the adapters mis-routes inputs at the
/// kernel boundary, not a physics divergence.
#[test]
fn bevy_parity_attach_detach_trajectory_simple() {
    let (mut sim, r_v1, r_v2, r_v3) = build_runner_sim();
    let (mut app, b_v1, b_v2, b_v3) = build_bevy_app();

    let mut attach_fired = false;

    for tick in 0..NUM_STEPS {
        // Time the simulations will reach after this tick's
        // `step()` / `step_bevy()` call (`tick + 1` ticks of `DT`
        // from t=0). Used to gate the attach event and to label
        // the attach-event tick for the bit-identity skip below.
        let next_time = (tick as f64 + 1.0) * DT;

        sim.step().expect("runner step must succeed");
        step_bevy(&mut app);

        // Fire the attach *after* the step that lands at simtime =
        // ATTACH_TIME so the (`t-DT → t`) integration runs with the
        // bodies still separate, matching JEOD's `trick.add_read(t,
        // ...)` semantics — see the file-level "Attach scheduling —
        // canonical timing" docstring and the matching
        // `tier3_sim_attach_detach_trajectory_simple`. The runner
        // applies `Simulation::attach` synchronously: it runs
        // `combine_states_at_attach` and writes the post-combine
        // state to `r_v2` immediately. The Bevy adapter's
        // `AttachEvent` is *queued* — `staging_system` consumes it
        // at the top of the *next* `FixedUpdate`, before that step's
        // integration runs. Both runtimes therefore feed the same
        // separate `t = ATTACH_TIME` state to the same kernel, just
        // at different schedule positions.
        let is_attach_event_tick = !attach_fired && (next_time - ATTACH_TIME).abs() < 0.5 * DT;
        if is_attach_event_tick {
            let (offset, t_pc) = link_offset_and_rotation();
            sim.attach(r_v1, r_v2, offset, t_pc);
            sim.mark_kinematic_only(r_v1);
            app.world_mut()
                .resource_mut::<bevy::ecs::message::Messages<AttachEvent>>()
                .write(AttachEvent {
                    child: b_v1,
                    parent: b_v2,
                    offset: jeod_sim::Vec3Ext::m_at::<jeod_sim::StructuralFrame<jeod_sim::SelfRef>>(
                        offset,
                    ),
                    t_parent_child: t_pc,
                });
            attach_fired = true;
        }

        // Compare each body's 6-DOF state. The contract differs by
        // body:
        //
        // - **veh3** (free-flying root, never attached) is
        //   integrator-written every tick under no force/torque.
        //   Both runtimes apply the same RK4 kernel to the same
        //   inputs; bit-identity holds on every tick.
        //
        // - **veh2** (integrated tree root in the attached window)
        //   is integrator-written every tick. Bit-identity holds
        //   *except* on the attach-event tick: the runner's
        //   synchronous `Simulation::attach` has just written the
        //   post-combine state, while Bevy's `AttachEvent` is
        //   queued but won't be consumed until the next
        //   `FixedUpdate`'s `staging_system`. The next iteration's
        //   `step_bevy` runs the same combine kernel on the same
        //   separate-veh2 state that the runner consumed, so parity
        //   re-aligns on the tick after the attach event.
        //
        // - **veh1** in the attached window is *kinematic-only*: its
        //   `composite_body` state is derived from veh2 by
        //   `propagate_state_from_root_system` (Bevy) and
        //   `propagate_kinematic_state` (runner). The two runtimes
        //   differ in *when* that walk fires within a tick: the
        //   runner runs propagation both before *and* after
        //   integration so `Simulation::body(idx)` returns
        //   same-tick-derived state; Bevy runs propagation only
        //   before integration, so `TranslationalStateC` reflects
        //   the *previous* tick's parent. Combined with
        //   `KinematicChildC` being installed by
        //   `wrench_aggregation_system` via Commands (so it's not
        //   visible to `propagate_state_from_root_system` until
        //   after the next sync point), there is a transient two-
        //   tick lag at the attach event. This is a documented
        //   schedule asymmetry, structurally covered by the kernel-
        //   self-consistency invariants in
        //   `bevy_parity_kinematic_propagation_simple_chain`. This
        //   trajectory parity therefore asserts bit-identity on
        //   veh1 only when veh1 is itself integrating — i.e. in
        //   the pre-attach window. The attached window's veh1
        //   parity is delegated to the kinematic-propagation parity
        //   test.
        let r_v3_state = read_runner_state(&sim, r_v3);
        let b_v3_state = read_bevy_state(&app, b_v3);

        let label = format!("t={:.3}s", sim.elapsed());
        assert_sixdof_eq(&format!("veh3 {label}"), &r_v3_state, &b_v3_state);

        if !is_attach_event_tick {
            let r_v2_state = read_runner_state(&sim, r_v2);
            let b_v2_state = read_bevy_state(&app, b_v2);
            assert_sixdof_eq(&format!("veh2 {label}"), &r_v2_state, &b_v2_state);
        }

        if !attach_fired {
            let r_v1_state = read_runner_state(&sim, r_v1);
            let b_v1_state = read_bevy_state(&app, b_v1);
            assert_sixdof_eq(&format!("veh1 {label}"), &r_v1_state, &b_v1_state);
        }
    }
}
