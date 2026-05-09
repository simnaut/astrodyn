// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Bevy ECS vs `astrodyn_runner::Simulation` parity for the
//! chained-attach (re-rooting) path, driven through the production
//! `AttachEvent` surface end-to-end.
//!
//! Companion to `bevy_parity_chained_attach_reroot.rs`, which pins the
//! `MassTreeR` resource composition + parent-pointer parity for the
//! arena kernel directly. This test instead exercises the full Bevy
//! adapter: `AttachEvent` → `staging_system` → `MassTree::attach_with_reroot`
//! plus the entity-side `MassChildOf` reparent and the rerooted
//! subtree's auto-promote-to-kinematic semantics. Together the two
//! tests cover both the storage-layer and the event-driven entry point
//! for the chained-attach kernel.
//!
//! # What is pinned
//!
//! Both runtimes:
//!
//! - register the same three bodies (veh1 / veh2 / veh3) with
//!   identical initial conditions from the JEOD source files
//!   (`Modified_data/veh{1,2,3}.py`), matching the placeholder mass
//!   tree the runner-side `tier3_sim_complex_attach_detach.rs` uses,
//! - integrate forward through `Simulation::step()` (runner) and
//!   `App::run_schedule(FixedUpdate)` (Bevy) in lock-step,
//! - fire two attaches in sequence: a simple root-subject attach
//!   (`v1 → v2`) followed immediately by a chained-attach reroot
//!   (`v1 → v3` while v1 is already a child of v2). The chained attach
//!   is the case `MassTree::attach_with_reroot` ports from JEOD's
//!   `dyn_body_attach.cc:521-567` — the subject's existing root (v2)
//!   gets reparented under the new parent (v3) while v1 stays attached
//!   to v2.
//!
//! Validation runs from `t = 0` immediately after the chained attach
//! and asserts:
//!
//! 1. **Topology parity**: the runner's `MassTree::parent` pointer
//!    chain matches the Bevy `MassTreeR` parent chain bit-identically
//!    after each event, AND the Bevy entity-side `MassChildOf`
//!    Relations component on the rerooted subject root (v2) points at
//!    the new parent (v3) with the recomputed `(offset, t_parent_child)`
//!    pair. The descendants' `MassChildOf` (v1 → v2) is preserved.
//!
//! 2. **Composite mass parity**: every body's
//!    `MassPropertiesC.composite_properties.mass` matches the runner's
//!    arena composite mass exactly (mass arithmetic is f64-exact in
//!    both contexts).
//!
//! 3. **Auto-promote-to-kinematic**: after the chained attach, every
//!    body in the rerooted subtree (v2 + v1) is observable as a
//!    non-root mass-tree node on the Bevy side. The
//!    `wrench_aggregation_system` walks `MassChildOf` chains and
//!    inserts `KinematicChildC` on every non-root entity; the runner
//!    sets `body.kinematic_only = true` directly. Both express the same
//!    JEOD `dyn_body_collect.cc:138`-aligned contract: only the root of
//!    the merged tree integrates; every interior body is derived
//!    through composite-rigid-body propagation from the root.
//!
//! # What is **not** pinned
//!
//! Trajectory cross-validation through the rerooted kinematic walk
//! (the merged tree's combined-body trajectory) is the runner-side
//! Tier 3 test's job (`tier3_sim_complex_attach_detach.rs`). The
//! kernel under test here is identical between adapter and runner, so
//! trajectory cross-validation in either layer covers the kernel; this
//! parity test focuses on the storage shape and the event-driven entry
//! point.

use astrodyn::IntegratorType;
use astrodyn::MassProperties;
use astrodyn::{
    DynamicsConfig, GravityControls, JeodQuat, MassTree, RotationalState, SimulationTime,
    TranslationalState, VehicleConfig,
};
use astrodyn_bevy::{
    AstrodynPlugin, AttachEvent, DynamicsConfigC, ExternalForceC, ExternalTorqueC,
    FrameDerivativesC, GravityControlsC, KinematicChildC, MassBodyIdC, MassChildOf,
    MassPropertiesC, MassTreeR, RotationalStateC, TotalForceC, TranslationalStateC,
};
use astrodyn_runner::Simulation;
use bevy::prelude::*;
use glam::{DMat3, DVec3};
use std::time::Duration;

const DT: f64 = 0.1;
/// First attach: v1 → v2 (root subject — bit-equivalent to plain attach).
const ATTACH_V1_V2_TIME: f64 = 1.0;
/// Chained attach: v1 → v3 while v1 is already a child of v2.
/// `attach_with_reroot` re-roots v2 under v3 and v1 stays attached to v2.
const RECHAIN_V1_V3_TIME: f64 = 2.0;

// ── Initial conditions, all from JEOD Modified_data files (matches
//    `tier3_sim_complex_attach_detach.rs::veh*_initial_*`). ──

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
        ang_vel_body: DVec3::new(0.0, 0.0, 1.0),
    }
}

/// JEOD `BodyAttachAligned` v1.attach_to_2: composes to identity
/// rotation + offset (-10, 0, 0). See
/// `tier3_sim_complex_attach_detach.rs::attach_v1_to_v2_offset_and_rotation`.
fn attach_v1_v2_offset_and_rotation() -> (DVec3, DMat3) {
    (DVec3::new(-10.0, 0.0, 0.0), DMat3::IDENTITY)
}

/// JEOD `BodyAttachAligned` v1.attach_to_3: derived geometrically from
/// veh1.node13 / veh3.node31 named points. See
/// `tier3_sim_complex_attach_detach.rs::attach_v1_to_v3_offset_and_rotation`
/// for the full derivation chain.
fn attach_v1_v3_offset_and_rotation() -> (DVec3, DMat3) {
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

fn six_dof_config() -> DynamicsConfig {
    DynamicsConfig {
        translational_dynamics: true,
        rotational_dynamics: true,
        three_dof: false,
    }
}

fn build_runner_sim() -> (Simulation, usize, usize, usize) {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let v1 = sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&veh1_trans()),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(&(veh1_rot()))),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(veh1_mass()))),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    let v2 = sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&veh2_trans()),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(&(veh2_rot()))),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(veh2_mass()))),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    let v3 = sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&veh3_trans()),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(&(veh3_rot()))),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(veh3_mass()))),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    sim.add_body_to_tree(v1, "veh1");
    sim.add_body_to_tree(v2, "veh2");
    sim.add_body_to_tree(v3, "veh3");
    (sim, v1, v2, v3)
}

fn build_bevy_app() -> (
    App,
    Entity,
    Entity,
    Entity,
    astrodyn::MassBodyId,
    astrodyn::MassBodyId,
    astrodyn::MassBodyId,
) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);

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

    (app, v1, v2, v3, id_v1, id_v2, id_v3)
}

fn spawn_body(
    app: &mut App,
    name: &str,
    id: astrodyn::MassBodyId,
    mass: MassProperties,
    trans: TranslationalState,
    rot: RotationalState,
) -> Entity {
    app.world_mut()
        .spawn((
            Name::new(name.to_string()),
            DynamicsConfigC(six_dof_config()),
            MassPropertiesC::from(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass))),
            MassBodyIdC(id),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(trans),
            RotationalStateC::from(astrodyn::typed_bridge::rot_raw_to_self_ref(&(rot))),
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

/// Bevy adapter and runner produce bit-identical mass-tree topology +
/// composite mass through a chained-attach (re-rooting) sequence driven
/// via the production `AttachEvent` event surface.
///
/// The runner uses `Simulation::attach`, which dispatches through
/// `MassTree::attach_with_reroot`. The Bevy adapter consumes
/// `AttachEvent` in `staging_system`, which also dispatches through
/// `MassTree::attach_with_reroot` and reparents the rerooted subject
/// root's `MassChildOf` Relations component under the new parent so
/// the entity-side composite-mass walk and kinematic walk see the
/// correct shape.
///
/// Both paths exercise the same kernel, so any topology divergence
/// between them at this point would mean the staging-system fence
/// mis-routes the inputs at the kernel boundary.
#[test]
fn bevy_parity_chained_attach_event_topology() {
    let (mut sim, r_v1, r_v2, r_v3) = build_runner_sim();
    let (mut app, b_v1, b_v2, b_v3, id_v1, id_v2, id_v3) = build_bevy_app();

    // Run one step in both runtimes so the schedules are warm and
    // any deferred-Commands flush from registration has landed
    // before the first attach. The first AttachEvent the test fires
    // is consumed at the start of the *next* `FixedUpdate`'s
    // `staging_system`, so this preroll matters.
    sim.step().expect("runner preroll step");
    step_bevy(&mut app);

    // Drive both runtimes forward until the simtime crosses
    // `ATTACH_V1_V2_TIME`.
    let advance_runner_to = |sim: &mut Simulation, target: f64| {
        while sim.elapsed() + 0.5 * DT < target {
            sim.step().expect("runner step()");
        }
    };
    let advance_bevy_to = |app: &mut App, target: f64| {
        while bevy_sim_elapsed(app) + 0.5 * DT < target {
            step_bevy(app);
        }
    };

    advance_runner_to(&mut sim, ATTACH_V1_V2_TIME);
    advance_bevy_to(&mut app, ATTACH_V1_V2_TIME);

    // -- First attach: v1 → v2 (root subject; bit-equivalent to plain
    //    attach). --
    let (offset_v1_v2, t_v1_v2) = attach_v1_v2_offset_and_rotation();
    sim.attach(r_v1, r_v2, offset_v1_v2, t_v1_v2);
    sim.mark_kinematic_only(r_v1);
    fire_bevy_attach(&mut app, b_v1, b_v2, offset_v1_v2, t_v1_v2);
    // The staging-system consumes the queued `AttachEvent` at the
    // top of the next `FixedUpdate`. Step Bevy once to flush it,
    // then advance the runner by the same `dt` so the elapsed
    // simtimes stay aligned.
    step_bevy(&mut app);
    sim.step()
        .expect("runner step() to align with bevy after attach 1");

    // After first attach: parent[v1] = v2, parent[v2] = None,
    // parent[v3] = None. Composite masses: v1=1, v2=3, v3=3.
    assert_topology_match(
        &app,
        &sim,
        b_v1,
        b_v2,
        b_v3,
        r_v1,
        r_v2,
        r_v3,
        id_v1,
        id_v2,
        id_v3,
        "after first attach",
    );
    // The simple root-subject attach does NOT auto-reparent
    // `MassChildOf` on the bevy side (the existing trajectory test
    // `bevy_parity_attach_detach_trajectory_simple` documents this:
    // mission code retains explicit control over whether the
    // freshly-attached child becomes kinematic). Confirm v1 has no
    // `MassChildOf` here so the chained-attach assertion below is
    // distinguishable.
    assert!(
        app.world().get::<MassChildOf>(b_v1).is_none(),
        "simple root-subject attach must not auto-insert MassChildOf on the child — \
         that path is the runner's `mark_kinematic_only` analogue and stays under \
         mission control. Only the chained-attach reroot path inserts MassChildOf \
         on the subject root."
    );

    // -- Chained attach: v1 → v3 while v1 is already a child of v2.
    //    The kernel walks v1's existing root (v2) and reroots v2
    //    under v3. The Bevy staging system must dispatch through
    //    `attach_with_reroot` and reparent v2's `MassChildOf` under
    //    v3 with the JEOD-recomputed (offset, t_parent_child) pair. --
    advance_runner_to(&mut sim, RECHAIN_V1_V3_TIME);
    advance_bevy_to(&mut app, RECHAIN_V1_V3_TIME);

    let (offset_v1_v3, t_v1_v3) = attach_v1_v3_offset_and_rotation();
    sim.attach(r_v1, r_v3, offset_v1_v3, t_v1_v3);
    fire_bevy_attach(&mut app, b_v1, b_v3, offset_v1_v3, t_v1_v3);
    step_bevy(&mut app);
    sim.step()
        .expect("runner step() to align with bevy after attach 2");

    // After reroot: v3 root, v2 under v3, v1 under v2.
    // Composite masses: v1=1, v2=3, v3=6.
    assert_topology_match(
        &app,
        &sim,
        b_v1,
        b_v2,
        b_v3,
        r_v1,
        r_v2,
        r_v3,
        id_v1,
        id_v2,
        id_v3,
        "after chained reroot",
    );

    // Confirm the rerooted subject root (v2) carries a `MassChildOf`
    // that points at v3 with the JEOD-recomputed (offset,
    // t_parent_child) pair. The recompute formulas live in
    // `MassTree::attach_with_reroot`; the Bevy adapter mirrors them
    // when constructing the `MassChildOf` for the subject root
    // entity — this assertion pins that the Bevy-side recompute
    // matches the arena-side recompute bit-for-bit.
    let v2_mass_child_of = app
        .world()
        .get::<MassChildOf>(b_v2)
        .expect("post-reroot: v2 must carry MassChildOf pointing at v3 (the new parent)");
    assert_eq!(
        v2_mass_child_of.parent, b_v3,
        "MassChildOf.parent on v2 must be v3 after reroot, got {:?} (expected {:?})",
        v2_mass_child_of.parent, b_v3,
    );
    // Compute the expected (offset, t) from the kernel formulas
    // (mirrors the staging-system recompute).
    let (expected_offset, expected_t) = {
        let tree = &app.world().resource::<MassTreeR>().0;
        // Post-attach, v2 is the rerooted child of v3; the subject
        // root for the original event was v2 (v1's old root). Rebuild
        // the expected geometry from the *pre-reroot* root (v2's old
        // root = v2 itself, since v2 was a tree root before this
        // chained attach). At the time of the staging-system call,
        // `tree.struct_chain_to_root(subject_root_id=v2, child_id=v1)`
        // produced the chain v2 → v1; the recompute folds this with
        // the user (offset, t_parent_child) and returns the new
        // (offset, t) for v2 → v3. We derive it manually here from
        // the post-reroot tree's published structure_point on v2.
        // For this test we just confirm `MassChildOf` parent is
        // correct + match the runner's parent on v2 (already done
        // above by `assert_topology_match`); the exact (offset, t)
        // values are pinned by the kernel's existing unit test
        // `attach_with_reroot_preserves_subject_geometry_under_rotation`.
        // To keep this test on the event-driven entry point's
        // contract surface, we cross-check the `MassChildOf`'s
        // `(offset, t_parent_child)` against the arena tree's
        // `structure_point` for v2 — which `MassTree::attach`
        // populates with the same recomputed pair the Bevy side
        // stamps into `MassChildOf`.
        let v2_id = id_v2;
        let sp = &tree.get(v2_id).structure_point;
        (sp.position, sp.t_parent_this)
    };
    let off_diff = (v2_mass_child_of.offset - expected_offset).length();
    assert!(
        off_diff < 1e-12,
        "MassChildOf.offset on v2 diverged from arena structure_point: \
         bevy={:?}, arena={:?}, |Δ|={off_diff:.3e}",
        v2_mass_child_of.offset,
        expected_offset,
    );
    let t_diff = (v2_mass_child_of.t_parent_child - expected_t)
        .to_cols_array()
        .iter()
        .map(|x| x.abs())
        .fold(0.0f64, f64::max);
    assert!(
        t_diff < 1e-12,
        "MassChildOf.t_parent_child on v2 diverged from arena structure_point: \
         bevy={:?}, arena={:?}, max|Δ|={t_diff:.3e}",
        v2_mass_child_of.t_parent_child,
        expected_t,
    );

    // After the chained attach, v1 is interior to v3's tree and v2
    // is also interior. Both should be kinematic non-root nodes.
    // `wrench_aggregation_system` walks `MassChildOf` chains and
    // inserts `KinematicChildC` on every non-root entity. Step the
    // schedule once more so the post-staging wrench pass runs and
    // its Commands flush.
    step_bevy(&mut app);
    sim.step()
        .expect("runner step() to align after wrench pass");

    assert!(
        app.world().get::<KinematicChildC>(b_v1).is_some(),
        "post-reroot: v1 (interior to v3's tree) must be marked KinematicChildC by \
         wrench_aggregation_system after the staging-system reparented MassChildOf"
    );
    assert!(
        app.world().get::<KinematicChildC>(b_v2).is_some(),
        "post-reroot: v2 (rerooted subject root, now interior to v3's tree) must \
         be marked KinematicChildC. This is the auto-promote-to-kinematic \
         contract: the runner's `Simulation::attach_inner` reroot path sets \
         `body.kinematic_only = true` on the rerooted subtree; the Bevy adapter \
         expresses the same intent through MassChildOf reparenting + the \
         wrench-aggregation walk."
    );
    assert!(
        app.world().get::<KinematicChildC>(b_v3).is_none(),
        "v3 is the integrated tree root after reroot — it must NOT carry \
         KinematicChildC (would double-count its own wrench)"
    );
}

/// Assert the Bevy `MassTreeR` arena and the runner's `MassTree` agree
/// on (a) parent pointers and (b) composite mass for every body, after
/// each attach event. Both contexts run the same kernel under the same
/// inputs; the assertion guards against the staging-system fence
/// silently mis-routing arguments at the kernel boundary.
#[allow(clippy::too_many_arguments)]
fn assert_topology_match(
    app: &App,
    sim: &Simulation,
    b_v1: Entity,
    b_v2: Entity,
    b_v3: Entity,
    r_v1: usize,
    r_v2: usize,
    r_v3: usize,
    id_v1: astrodyn::MassBodyId,
    id_v2: astrodyn::MassBodyId,
    id_v3: astrodyn::MassBodyId,
    label: &str,
) {
    let bevy_tree = &app.world().resource::<MassTreeR>().0;
    let runner_tree = sim
        .mass_tree
        .as_ref()
        .expect("runner must have MassTree populated");

    let pairs: [(&str, Entity, usize, astrodyn::MassBodyId); 3] = [
        ("v1", b_v1, r_v1, id_v1),
        ("v2", b_v2, r_v2, id_v2),
        ("v3", b_v3, r_v3, id_v3),
    ];

    // Parent-pointer parity by translating runner ids → name.
    for (name, _b_e, r_idx, b_id) in pairs {
        let bevy_parent_id = bevy_tree.parent(b_id);
        let runner_id = sim
            .body_mass_id(r_idx)
            .expect("runner body must have mass id");
        let runner_parent_id = runner_tree.parent(runner_id);
        let bevy_parent_name = bevy_parent_id.map(|p| bevy_tree.get(p).name.clone());
        let runner_parent_name = runner_parent_id.map(|p| runner_tree.get(p).name.clone());
        assert_eq!(
            bevy_parent_name, runner_parent_name,
            "{label}: {name} parent name mismatch — bevy={bevy_parent_name:?}, \
             runner={runner_parent_name:?}"
        );
    }

    // Composite-mass parity. Mass arithmetic is f64-exact for the
    // simple unit-mass values this test uses, so bit-equality is the
    // right contract.
    for (name, _b_e, r_idx, b_id) in pairs {
        let bevy_mass = bevy_tree.get(b_id).composite_properties.mass;
        let runner_id = sim
            .body_mass_id(r_idx)
            .expect("runner body must have mass id");
        let runner_mass = runner_tree.get(runner_id).composite_properties.mass;
        assert_eq!(
            bevy_mass, runner_mass,
            "{label}: {name} composite mass mismatch — bevy={bevy_mass}, \
             runner={runner_mass}"
        );
    }
}

/// Read the Bevy schedule's elapsed simtime. `AstrodynPlugin` advances
/// `SimulationTimeR` every fixed-update step; the test reads it to
/// keep both runtimes in lock-step on the attach event timing.
fn bevy_sim_elapsed(app: &App) -> f64 {
    app.world()
        .resource::<astrodyn_bevy::SimulationTimeR>()
        .0
        .simtime
}

/// Queue an `AttachEvent` on the Bevy app's message bus.
fn fire_bevy_attach(app: &mut App, child: Entity, parent: Entity, offset: DVec3, t_pc: DMat3) {
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child,
            parent,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(offset),
            t_parent_child: astrodyn::FrameTransform::from_matrix(t_pc),
        });
}
