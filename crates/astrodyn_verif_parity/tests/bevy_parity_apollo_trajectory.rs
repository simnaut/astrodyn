//! Bevy ↔ runner parity for the SIM_Apollo `apollo_trajectory` 12-second
//! launch-stack sequence (11 mass-tree events: 9 detaches + 2 attaches).
//!
//! Drives the same builder + arena setup the runner-side tier3 test
//! consumes
//! (`crates/astrodyn_verif_jeod/tests/tier3_sim_apollo_trajectory.rs`)
//! through both runtimes — `astrodyn_runner::Simulation` and the Bevy
//! `populate_app::<Earth>` bridge — and asserts bit-identical
//! translational state at every reference-CSV checkpoint over the
//! full 12 s window. Each of the eleven mass-tree events fires on both
//! sides at the same tick via
//! [`SimContext::detach_subtree`] /
//! [`SimContext::attach_subtree_aligned`], routed through the runner's
//! mass-tree mutation surface and the Bevy adapter's
//! `AttachEvent`/`DetachEvent` bus respectively (the Bevy adapter
//! resolves the `MassBodyId` to a mass-only entity and computes the
//! named-mass-point reduction internally — see
//! [`BevySimContext::attach_subtree_aligned`]).
//!
//! ## Topology mirroring
//!
//! Apollo's mass tree pairs one integrated body (`cm`) with seven
//! tree-only mass bodies (`sm`, `lm`, `dm`, `s3`, `s2`, `s1`, `les`)
//! plus fourteen named attachment points and seven launch-stack
//! attaches — a shape the declarative `SimulationBuilder` doesn't
//! model. Both runtimes augment their post-build arena via the shared
//! [`setup_apollo_arena`] helper, which allocates the seven tree-only
//! bodies in deterministic order so the two arenas end up with
//! identical `MassBodyId` layouts. The Bevy side additionally spawns
//! a mass-only entity (`MassBodyIdC` + `MassPropertiesC` only) for
//! each tree-only body so the existing `DetachEvent`/`AttachEvent`
//! staging path — which already supports mass-only attach participants
//! — can address the subtree roots without any new event types.
//!
//! ## Comparison cadence
//!
//! The CSV cadence is 0.1 s (5 dt-ticks at `dt = 0.02 s`); the parity
//! assertion runs at every checkpoint in `apollo_trajectory.csv`.
//! Bit-identity at the CSV cadence implies bit-identity at every
//! intermediate integration tick by the monotonic-divergence argument
//! `VerificationCaseParityExt::run_and_assert_parity` uses (once two
//! runtimes drift, they stay drifted), so a coarser checkpoint set
//! is equivalent in detection strength to a per-tick scan.

#![allow(
    clippy::float_cmp,
    reason = "bevy-parity tests assert bit-exact identity between runner and Bevy state fields"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Apollo recipe step counts and indices fit exactly in f64 mantissa and usize"
)]

use std::path::PathBuf;
use std::time::Duration;

use astrodyn::{typed_bridge, JeodQuat, MassBodyId, TranslationalState};
use astrodyn_bevy::{
    DynamicsConfigC, KinematicChildC, MassBodyIdC, MassChildOf, MassPropertiesC, MassTreeR,
    RotationalStateC, SimulationBuilderBevyExt, TranslationalStateC,
};
use astrodyn_runner::{Simulation, SimulationBuilderExt};
use astrodyn_verif_jeod::run_verification::sim_apollo_trajectory::{
    apollo_trajectory_builder, apply_event, setup_apollo_arena, ApolloTopology, Event, DT, EVENTS,
    SIM_DURATION_S,
};
use astrodyn_verif_jeod::verification::SimContext;
use astrodyn_verif_parity::BevySimContext;
use bevy::prelude::*;
use glam::DVec3;

/// Load every reference-CSV timestamp from `apollo_trajectory.csv`.
/// Only column 0 (sim time) is consumed — the parity assertion never
/// reads JEOD-logged state, so the per-row position / velocity / quat
/// payload is irrelevant. Skipping the t=0 row matches the trajectory
/// loop's "skip CSV row 0 (initial state — no integration yet)"
/// convention.
fn load_apollo_times() -> Vec<f64> {
    let csv_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace crate dir")
        .join("astrodyn_verif_jeod")
        .join("test_data")
        .join("apollo_trajectory.csv");
    assert!(
        csv_path.exists(),
        "apollo_trajectory.csv missing at {}. Generate with: cargo xtask regenerate-tier3",
        csv_path.display(),
    );
    let content = std::fs::read_to_string(&csv_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", csv_path.display()));
    let mut times = Vec::new();
    for line in content.lines().skip(1) {
        let first = line.split(',').next().unwrap_or("");
        let t: f64 = first
            .trim()
            .parse()
            .unwrap_or_else(|e| panic!("apollo_trajectory.csv: bad time column {first:?}: {e}"));
        times.push(t);
    }
    times
}

/// Build the runner-side `Simulation` for the apollo trajectory: shared
/// builder + arena setup + sync_body_mass_from_tree + core-to-composite
/// conversion. Returns the simulation plus the topology so the lockstep
/// driver can address subtree roots by `MassBodyId`.
fn build_runner() -> (Simulation, ApolloTopology) {
    let handles = apollo_trajectory_builder();
    let mut sim = handles
        .builder
        .build()
        .expect("apollo simulation must validate");
    let cm_id = sim
        .body_mass_id(0)
        .expect("cm body must be registered in mass tree by the builder");
    let tree = sim.mass_tree.as_mut().expect("mass tree was just created");
    let topology = setup_apollo_arena(tree, cm_id);
    sim.sync_body_mass_from_tree(0);
    sim.convert_body_trans_core_to_composite(0);
    (sim, topology)
}

/// Build the Bevy app for the apollo trajectory: same builder fed to
/// `populate_app::<Earth>`, then mirror the runner's post-build
/// arena-setup steps on the Bevy world (augment `MassTreeR`, spawn
/// mass-only entities for the seven tree-only bodies, sync the cm
/// entity's `MassPropertiesC` from the composite, shift its
/// `TranslationalStateC<Earth>` from core_body to composite_body).
///
/// Returns the app, the cm body entity, and the topology — bit-identical
/// to the runner's `MassBodyId` layout because both adapters allocate
/// `MassBodyId`s in the same order through the same
/// [`setup_apollo_arena`] helper.
fn build_bevy_app() -> (App, Entity, ApolloTopology) {
    let handles_runner = apollo_trajectory_builder();
    // Set up Bevy app from the same `SimulationBuilder`.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let scenario_handles = handles_runner
        .builder
        .populate_app::<astrodyn::Earth>(&mut app)
        .expect("populate_app under <Earth>");
    // `MinimalPlugins` does not auto-run `Startup`; the parity loop
    // drives `FixedUpdate` directly. Trigger Startup so per-source
    // frame trees are wired before the first integration tick.
    app.world_mut().run_schedule(Startup);

    let cm_entity = scenario_handles.body_entities[0];
    let cm_id = app
        .world()
        .get::<MassBodyIdC>(cm_entity)
        .expect("cm entity carries MassBodyIdC after populate_app")
        .0;

    // Augment the live arena with the seven tree-only bodies + named
    // mass points + launch-stack attaches.
    let topology = {
        let mut tree_r = app.world_mut().resource_mut::<MassTreeR>();
        setup_apollo_arena(&mut tree_r.0, cm_id)
    };

    // Spawn mass-only entities for the seven tree-only mass bodies.
    // The staging-system `bodies` query filters on `MassBodyIdC` +
    // `MassPropertiesC`; both components are sufficient for an entity
    // to participate in attach / detach events as a mass-only body
    // (no `DynamicsConfigC` / no `TranslationalStateC` — the
    // staging-system carve-out at integration.rs:1700 documents the
    // legitimate `MassBody`-without-`DynBody` shape this matches).
    // Read the core mass for each tree-only body from the arena so the
    // ECS `MassPropertiesC` mirrors the arena's `core_properties` — the
    // staging system reads from the arena (not `MassPropertiesC`) for
    // the composite-state walk, so this is for the kernel-input read on
    // the child side of `attach_subtree_aligned` (the algorithm reads
    // `subtree_composite_props = tree.get(subtree_root_id).composite_properties`
    // before the topology mutation — that's an arena read; the bevy
    // adapter's staging system mirrors the same arena reads).
    // Spawn an entity for each of the seven tree-only mass bodies
    // carrying the **full** state component set: `MassBodyIdC`,
    // `MassPropertiesC`, `TranslationalStateC<Earth>`,
    // `RotationalStateC`, plus the `KinematicChildC` integrator
    // filter. The state components start at zero / identity; per-tick
    // `propagate_state_from_root_system` overwrites them from the
    // integrated cm root every step, so the seed values are
    // immaterial — the same is true on the runner side, where the
    // tree-only bodies have no `SimBody.trans` / `body.rot` (they
    // live in `MassTree` alone and ride the integrated root's state
    // through the arena composition kernel).
    //
    // `KinematicChildC` is mandatory: without it,
    // `integration_system` would also visit these entities each
    // tick, doubling the gravity reads against the cm root. The
    // marker mirrors `wrench_aggregation_system`'s auto-insert on
    // non-root nodes of a `MassChildOf` chain, just installed
    // explicitly here because the apollo recipe never produces a
    // tick where the wrench walk would see all eight bodies
    // simultaneously (the very first staging tick already detaches
    // s1).
    let tree_only = [
        topology.sm,
        topology.lm,
        topology.dm,
        topology.s3,
        topology.s2,
        topology.s1,
        topology.les,
    ];
    let mut mass_id_to_entity: std::collections::HashMap<MassBodyId, Entity> =
        std::collections::HashMap::new();
    mass_id_to_entity.insert(topology.cm, cm_entity);
    for &mass_id in &tree_only {
        let core = {
            let tree_r = app.world().resource::<MassTreeR>();
            tree_r.0.get(mass_id).core_properties
        };
        let entity = app
            .world_mut()
            .spawn((
                astrodyn_bevy::FrameUidC(astrodyn::named_body_frame_uid(
                    "bevy-parity-apollo-trajectory-b1",
                )),
                MassBodyIdC(mass_id),
                MassPropertiesC(typed_bridge::mass_raw_to_self_ref(&core)),
                TranslationalStateC::<astrodyn::Earth>::default(),
                RotationalStateC::default(),
                DynamicsConfigC::default(),
                KinematicChildC,
            ))
            .id();
        mass_id_to_entity.insert(mass_id, entity);
    }

    // Install `MassChildOf` Relations to mirror the arena's
    // parent↔child topology, taking the offset and rotation from
    // each body's `structure_point`. The Bevy adapter's
    // `composite_mass_system` slow path activates when at least one
    // `MassChildOf` edge is present; without these edges the fast
    // path activates and (post-staging detach) reverts cm's
    // `MassPropertiesC` from the composite-after-detach back to the
    // stale core seeded at first sight (the cache mismatch the
    // staging system's `bypass_change_detection` write deliberately
    // leaves behind). Installing the edges so composite_mass_system
    // stays on the slow path keeps the Bevy-side composite in
    // lock-step with the runner's `Simulation::sync_body_mass_from_tree`
    // after every event.
    let all_ids = [
        topology.cm,
        topology.sm,
        topology.lm,
        topology.dm,
        topology.s3,
        topology.s2,
        topology.s1,
        topology.les,
    ];
    for &mass_id in &all_ids {
        let edge = {
            let tree_r = app.world().resource::<MassTreeR>();
            tree_r.0.parent(mass_id).map(|parent_id| {
                let sp = tree_r.0.get(mass_id).structure_point;
                (parent_id, sp.position, sp.t_parent_this)
            })
        };
        if let Some((parent_id, offset, t_parent_child)) = edge {
            let parent_entity = mass_id_to_entity[&parent_id];
            let child_entity = mass_id_to_entity[&mass_id];
            app.world_mut()
                .entity_mut(child_entity)
                .insert(MassChildOf::with_rotation(
                    parent_entity,
                    offset,
                    t_parent_child,
                ));
        }
    }

    // Cm's `MassPropertiesC` is NOT manually overwritten with the
    // composite here: the Bevy adapter's `composite_mass_system`
    // walks `MassChildOf` chains from per-entity `CoreMassPropertiesC`
    // each tick and writes the recomputed composite into
    // `MassPropertiesC` automatically. The cm entity was spawned
    // with its CORE mass (cm-only) via `populate_app`, and the
    // tree-only entities just spawned above carry their own core
    // masses — feeding the slow-path Steiner accumulation exactly
    // the shape the arena tree holds. Manually writing the composite
    // here would double-count (the slow-path walk would Steiner-shift
    // the children's mass on top of an already-composite cm core).

    // Mirror `Simulation::convert_body_trans_core_to_composite(0)`: the
    // builder seeded `TranslationalStateC<Earth>` with the CSV row-0
    // core_body inertial state; convert in place to composite_body.
    {
        let (cw_inertial, dvel_inertial) = {
            let tree_r = app.world().resource::<MassTreeR>();
            let node = tree_r.0.get(cm_id);
            let cw_struct = node.core_wrt_composite.position;
            let t_struct_to_body = node.composite_properties.t_parent_this;
            let cw_body = t_struct_to_body * cw_struct;
            let rot = app
                .world()
                .get::<RotationalStateC>(cm_entity)
                .expect("cm carries RotationalStateC");
            let body_quat = rot.0.q_inertial_body.to_jeod_quat();
            let t_inertial_to_body = body_quat.left_quat_to_transformation();
            let t_body_to_inertial = t_inertial_to_body.transpose();
            let cw_inertial = t_body_to_inertial * cw_body;
            let omega_body = rot.0.ang_vel_body.raw_si();
            let dvel_inertial = t_body_to_inertial * omega_body.cross(cw_body);
            (cw_inertial, dvel_inertial)
        };
        let mut entity = app.world_mut().entity_mut(cm_entity);
        let mut trans = entity
            .get_mut::<TranslationalStateC<astrodyn::Earth>>()
            .expect("cm carries TranslationalStateC<Earth>");
        // composite = core − cw_inertial; subtract the rigid-body
        // ω × r contribution on velocity. All values stay in the body's
        // integration frame, so the relabel-only `from_raw_si` lift
        // matches the runner's `Position::<IntegrationFrame>::from_raw_si`.
        let new_pos = trans.0.position.raw_si() - cw_inertial;
        let new_vel = trans.0.velocity.raw_si() - dvel_inertial;
        *trans = TranslationalStateC::from(typed_bridge::trans_raw_to_root(&TranslationalState {
            position: new_pos,
            velocity: new_vel,
        }));
    }

    (app, cm_entity, topology)
}

/// Step the Bevy app by exactly one `dt` and assert the runner stayed
/// in lockstep (advance the same `dt` on the runner separately).
fn step_bevy(app: &mut App) {
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);
}

/// Apply one Apollo event to both runtimes through the SimContext
/// surface. The runner forwards to `Simulation::detach_subtree` /
/// `attach_subtree_aligned`; the Bevy adapter writes an
/// `AttachEvent`/`DetachEvent` onto the message bus that the next
/// `FixedUpdate`'s `staging_system` drains.
fn apply_event_both<P: astrodyn::Planet>(
    runner: &mut Simulation,
    app: &mut App,
    body_entities: &[Entity],
    topology: &ApolloTopology,
    event: Event,
) {
    {
        let ctx: &mut dyn SimContext = runner;
        apply_event(ctx, topology, event);
    }
    {
        let world = app.world_mut();
        // `BevySimContext::new` takes parallel slices of source +
        // body entities. The apollo recipe doesn't drive any source
        // operations from the SimContext, so the source slice is an
        // empty placeholder.
        let mut ctx = BevySimContext::<P>::new(world, &[], body_entities);
        apply_event(&mut ctx, topology, event);
    }
}

/// Read the runner-side cm composite translational state (raw glam).
fn runner_cm_trans_raw(sim: &Simulation) -> (DVec3, DVec3) {
    let trans = sim.body(0).trans;
    (trans.position.raw_si(), trans.velocity.raw_si())
}

/// Read the Bevy-side cm composite translational state (raw glam).
fn bevy_cm_trans_raw(app: &App, cm: Entity) -> (DVec3, DVec3) {
    let trans = app
        .world()
        .get::<TranslationalStateC<astrodyn::Earth>>(cm)
        .expect("cm entity carries TranslationalStateC<Earth>");
    (trans.0.position.raw_si(), trans.0.velocity.raw_si())
}

/// Read the runner-side cm composite rotational state (raw glam).
fn runner_cm_rot_raw(sim: &Simulation) -> (JeodQuat, DVec3) {
    let body = sim.body(0);
    let rot = body.rot.as_ref().expect("cm carries RotationalState");
    let untyped = typed_bridge::rot_typed_to_raw(rot);
    (untyped.quaternion, untyped.ang_vel_body)
}

/// Read the Bevy-side cm composite rotational state (raw glam).
fn bevy_cm_rot_raw(app: &App, cm: Entity) -> (JeodQuat, DVec3) {
    let rot = app
        .world()
        .get::<RotationalStateC>(cm)
        .expect("cm entity carries RotationalStateC");
    let untyped = typed_bridge::rot_typed_to_raw(&rot.0);
    (untyped.quaternion, untyped.ang_vel_body)
}

fn assert_bit_eq_vec(case: &str, t: f64, label: &str, runner: DVec3, bevy: DVec3) {
    for i in 0..3 {
        assert!(
            runner[i].to_bits() == bevy[i].to_bits(),
            "{case} bit-parity broke at t = {t:.6}s on {label}[{i}]:\n  \
             runner = {runner_v} (bits = {runner_bits:#018x})\n  \
             bevy   = {bevy_v} (bits = {bevy_bits:#018x})",
            runner_v = runner[i],
            bevy_v = bevy[i],
            runner_bits = runner[i].to_bits(),
            bevy_bits = bevy[i].to_bits(),
        );
    }
}

fn assert_bit_eq_quat(case: &str, t: f64, runner: JeodQuat, bevy: JeodQuat) {
    for i in 0..4 {
        assert!(
            runner.data[i].to_bits() == bevy.data[i].to_bits(),
            "{case} bit-parity broke at t = {t:.6}s on quat[{i}]:\n  \
             runner = {runner_v} (bits = {runner_bits:#018x})\n  \
             bevy   = {bevy_v} (bits = {bevy_bits:#018x})",
            runner_v = runner.data[i],
            bevy_v = bevy.data[i],
            runner_bits = runner.data[i].to_bits(),
            bevy_bits = bevy.data[i].to_bits(),
        );
    }
}

#[test]
fn bevy_parity_apollo_trajectory() {
    let csv_times = load_apollo_times();
    assert!(
        (csv_times.last().copied().unwrap_or(0.0) - SIM_DURATION_S).abs() < 0.05,
        "apollo_trajectory.csv last time {} disagrees with SIM_Apollo terminate_time={SIM_DURATION_S}",
        csv_times.last().copied().unwrap_or(0.0),
    );

    let (mut runner, runner_topology) = build_runner();
    let (mut app, cm_entity, bevy_topology) = build_bevy_app();
    // Both arenas allocate ids in the same order through the shared
    // `setup_apollo_arena`; check explicitly so a future refactor that
    // diverges the two adapters' build orders fails here rather than
    // at a confusing per-event arena lookup.
    assert_eq!(
        runner_topology.cm, bevy_topology.cm,
        "cm MassBodyId mismatch between runtimes"
    );
    assert_eq!(
        runner_topology.sm, bevy_topology.sm,
        "sm MassBodyId mismatch between runtimes"
    );
    assert_eq!(
        runner_topology.lm, bevy_topology.lm,
        "lm MassBodyId mismatch between runtimes"
    );
    assert_eq!(
        runner_topology.s3, bevy_topology.s3,
        "s3 MassBodyId mismatch between runtimes"
    );
    let topology = runner_topology;

    // Sanity check: at t = 0, both runtimes hold the same cm composite
    // state (the bevy-side core-to-composite conversion above mirrors
    // the runner's `convert_body_trans_core_to_composite`).
    let (r_pos0, r_vel0) = runner_cm_trans_raw(&runner);
    let (b_pos0, b_vel0) = bevy_cm_trans_raw(&app, cm_entity);
    assert_bit_eq_vec("apollo_trajectory (init)", 0.0, "position", r_pos0, b_pos0);
    assert_bit_eq_vec("apollo_trajectory (init)", 0.0, "velocity", r_vel0, b_vel0);

    let body_entities = vec![cm_entity];
    let mut event_iter = EVENTS.iter().peekable();
    let mut current_t = 0.0_f64;

    // Iterate the reference timestamps so the lockstep checkpoint
    // cadence is identical to the runner-vs-JEOD tier3 test. Per-step
    // structure of one iteration, mirroring JEOD's `add_read(t)`
    // semantics (event fires at END of cycle ending at t):
    //
    //   1. Step both runtimes up to `reference_t` under the
    //      currently-active mass-tree topology.
    //   2. Compare bit-identity *before* firing any event at
    //      `reference_t`. This is the integrator-output instant at
    //      end of cycle `[reference_t - DT, reference_t]`; the
    //      pre-event state is what the runner-vs-JEOD CSV row at the
    //      previous reference time integrated forward to, and is the
    //      naturally synchronised point between the two runtimes
    //      (runner has not yet detached; bevy has nothing queued).
    //   3. Fire events scheduled at `reference_t` through both
    //      runtimes: runner detaches / attaches synchronously, bevy
    //      queues `DetachEvent` / `AttachEvent` onto the message bus.
    //
    // The bevy queue is drained at the TOP of the next iteration's
    // first `FixedUpdate` (staging_system runs before integration),
    // so by the next comparison point both runtimes have applied
    // the event AND integrated one tick under the post-event
    // topology — matching the runner's "synchronous detach +
    // integrate-next-cycle" flow tick-for-tick.
    for &reference_t in csv_times.iter().skip(1) {
        // Step both runtimes up to the reference timestamp.
        while current_t + DT * 0.5 < reference_t {
            runner.step().expect("runner step failed");
            step_bevy(&mut app);
            current_t += DT;
        }

        // Compare BEFORE firing any event scheduled at `reference_t`.
        let (r_pos, r_vel) = runner_cm_trans_raw(&runner);
        let (b_pos, b_vel) = bevy_cm_trans_raw(&app, cm_entity);
        assert_bit_eq_vec("apollo_trajectory", reference_t, "position", r_pos, b_pos);
        assert_bit_eq_vec("apollo_trajectory", reference_t, "velocity", r_vel, b_vel);
        let (r_q, r_w) = runner_cm_rot_raw(&runner);
        let (b_q, b_w) = bevy_cm_rot_raw(&app, cm_entity);
        assert_bit_eq_quat("apollo_trajectory", reference_t, r_q, b_q);
        assert_bit_eq_vec("apollo_trajectory", reference_t, "ang_vel", r_w, b_w);

        // Fire events scheduled at exactly this reference time.
        // `current_t` lands on `reference_t` within the half-DT loop-
        // exit slack; the event-firing condition is the same one the
        // runner-side tier3 test uses (event_t <= reference_t and
        // strictly above the previous current_t).
        while let Some(&&(event_t, event)) = event_iter.peek() {
            if event_t <= current_t + 1e-9 {
                apply_event_both::<astrodyn::Earth>(
                    &mut runner,
                    &mut app,
                    &body_entities,
                    &topology,
                    event,
                );
                event_iter.next();
            } else {
                break;
            }
        }
    }
}
