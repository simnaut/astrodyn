//! Bevy ECS mass-tree parity vs the runner's arena `MassTree`.
//!
//! Migrates the `tier3_mass_attach_detach` cases from
//! `crates/jeod_dynamics/tests/tier3_mass_attach_detach.rs` to a
//! Bevy-adapter parity test that exercises the new
//! [`bevy_jeod::MassChildOf`] relation and
//! [`bevy_jeod::composite_mass_system`].
//!
//! For every scenario:
//!
//! 1. build the same parent / child topology in both:
//!    - the runner's `MassTree` arena (`MassTree::add_root` /
//!      `MassTree::add_body` / `MassTree::attach`),
//!    - a Bevy `World` populated with `MassPropertiesC` +
//!      `MassChildOf` components,
//! 2. run composition once (the arena auto-recomputes inside
//!    `attach`; the Bevy app runs `composite_mass_system` once),
//! 3. assert the parent's composite mass / CoM / inertia agree to
//!    `1e-12` between arena and ECS.
//!
//! This is deliberately a **unit-tier** parity test —
//! `composite_mass_system` is a pure read-then-write Bevy system that
//! delegates to the storage-agnostic `jeod_sim::MassStorage` kernel,
//! so the parity is structural: any drift between arena and ECS would
//! mean the kernel mis-walks the relation, not that physics has
//! diverged.
//!
//! Trajectory cross-validation (Tier 3) for *attach during
//! propagation* is sub-issue #273 — it requires the
//! momentum-conservation port (`combine_states_at_attach`) which is
//! out of scope for #271.

use bevy::prelude::*;
use bevy_jeod::{composite_mass_system, AttachEvent, MassBodyIdC, MassChildOf, MassPropertiesC};
use glam::{DMat3, DVec3};
use jeod_sim::{MassProperties, MassTree};

// ── Helpers ──

fn assert_vec3_close(a: DVec3, b: DVec3, tol: f64, msg: &str) {
    let diff = (a - b).length();
    assert!(
        diff < tol,
        "{msg}: diff {diff:.2e} exceeds tolerance {tol:.2e}"
    );
}

fn assert_mat3_close(a: DMat3, b: DMat3, tol: f64, msg: &str) {
    for (i, (ca, cb)) in [
        (a.x_axis, b.x_axis),
        (a.y_axis, b.y_axis),
        (a.z_axis, b.z_axis),
    ]
    .iter()
    .enumerate()
    {
        let diff = (*ca - *cb).length();
        assert!(
            diff < tol,
            "{msg}: column {i} diff {diff:.2e} exceeds tolerance {tol:.2e}"
        );
    }
}

fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_systems(Update, composite_mass_system);
    app
}

/// Spawn a parent + N children in the Bevy `World` and run
/// composition once. Returns `(parent_entity, child_entities)`.
fn spawn_topology(
    app: &mut App,
    parent_core: MassProperties,
    children: &[(MassProperties, DVec3, DMat3)],
) -> (Entity, Vec<Entity>) {
    let parent = app
        .world_mut()
        .spawn(MassPropertiesC::from(parent_core))
        .id();
    let mut child_entities = Vec::with_capacity(children.len());
    for (core, offset, t_parent_child) in children {
        let cid = app
            .world_mut()
            .spawn((
                MassPropertiesC::from(*core),
                MassChildOf::with_rotation(parent, *offset, *t_parent_child),
            ))
            .id();
        child_entities.push(cid);
    }
    app.update();
    (parent, child_entities)
}

/// Read the composite mass / position / inertia from a Bevy entity.
fn read_composite(app: &App, entity: Entity) -> MassProperties {
    app.world()
        .get::<MassPropertiesC>(entity)
        .expect("entity has MassPropertiesC")
        .0
        .to_untyped()
}

// ── Parity scenarios ──

#[test]
fn bevy_parity_mass_single_attach_composite() {
    // Mirror of `tier3_mass_single_attach_composite`.
    let parent_core = MassProperties::new(1000.0);
    let child_core = MassProperties::new(500.0);
    let offset = DVec3::new(1.0, 0.0, 0.0);
    let rot = DMat3::IDENTITY;

    // Bevy
    let mut app = build_app();
    let (parent_e, _) = spawn_topology(&mut app, parent_core, &[(child_core, offset, rot)]);
    let bevy_comp = read_composite(&app, parent_e);

    // Arena
    let mut tree = MassTree::new();
    let pid = tree.add_root("parent".into(), parent_core);
    let cid = tree.add_body("child".into(), child_core);
    tree.attach(cid, pid, offset, rot);
    let arena_comp = tree.get(pid).composite_properties;

    // Composite mass parity
    assert!(
        (bevy_comp.mass - arena_comp.mass).abs() < 1e-12,
        "mass: bevy {} vs arena {}",
        bevy_comp.mass,
        arena_comp.mass
    );
    // Composite CoM parity
    assert_vec3_close(
        bevy_comp.position,
        arena_comp.position,
        1e-12,
        "composite CoM",
    );
    // Composite inertia parity
    assert_mat3_close(
        bevy_comp.inertia,
        arena_comp.inertia,
        1e-10,
        "composite inertia",
    );
}

#[test]
fn bevy_parity_mass_symmetric_children_zero_com_shift() {
    // Mirror of `tier3_mass_symmetric_children_zero_com_shift`.
    let parent_core = MassProperties::new(200.0);
    let child = MassProperties::new(50.0);
    let offset = 3.0;

    let children = [
        (child, DVec3::new(offset, 0.0, 0.0), DMat3::IDENTITY),
        (child, DVec3::new(-offset, 0.0, 0.0), DMat3::IDENTITY),
    ];

    let mut app = build_app();
    let (parent_e, _) = spawn_topology(&mut app, parent_core, &children);
    let bevy_comp = read_composite(&app, parent_e);

    let mut tree = MassTree::new();
    let pid = tree.add_root("parent".into(), parent_core);
    for (core, off, rot) in children {
        let cid = tree.add_body("child".into(), core);
        tree.attach(cid, pid, off, rot);
    }
    let arena_comp = tree.get(pid).composite_properties;

    assert!((bevy_comp.mass - arena_comp.mass).abs() < 1e-12);
    assert_vec3_close(
        bevy_comp.position,
        arena_comp.position,
        1e-12,
        "symmetric CoM",
    );
    assert_mat3_close(
        bevy_comp.inertia,
        arena_comp.inertia,
        1e-10,
        "symmetric inertia",
    );
}

#[test]
fn bevy_parity_mass_parallel_axis_theorem() {
    // Mirror of `tier3_mass_parallel_axis_theorem` (solid sphere
    // attached at offset, parent + child both with explicit
    // inertias).
    let parent_core = MassProperties::new(100.0);
    let sphere_mass = 10.0;
    let sphere_radius = 0.5;
    let i_sphere = 2.0 / 5.0 * sphere_mass * sphere_radius * sphere_radius;
    let sphere_inertia = DMat3::from_diagonal(DVec3::splat(i_sphere));
    let sphere_core = MassProperties::with_inertia(sphere_mass, sphere_inertia, DVec3::ZERO);
    let d = DVec3::new(5.0, 0.0, 0.0);

    let mut app = build_app();
    let (parent_e, _) = spawn_topology(&mut app, parent_core, &[(sphere_core, d, DMat3::IDENTITY)]);
    let bevy_comp = read_composite(&app, parent_e);

    let mut tree = MassTree::new();
    let pid = tree.add_root("parent".into(), parent_core);
    let sid = tree.add_body("sphere".into(), sphere_core);
    tree.attach(sid, pid, d, DMat3::IDENTITY);
    let arena_comp = tree.get(pid).composite_properties;

    assert!((bevy_comp.mass - arena_comp.mass).abs() < 1e-12);
    assert_vec3_close(bevy_comp.position, arena_comp.position, 1e-12, "PA CoM");
    assert_mat3_close(bevy_comp.inertia, arena_comp.inertia, 1e-10, "PA inertia");
    // Inverse inertia parity at the root.
    assert_mat3_close(
        bevy_comp.inverse_inertia,
        arena_comp.inverse_inertia,
        1e-10,
        "PA inverse inertia",
    );
}

#[test]
fn bevy_parity_mass_many_children_composite() {
    // Mirror of `tier3_mass_many_children_composite`: 10 children at
    // helical offsets.
    let parent_mass = 500.0;
    let child_mass = 10.0;
    let n = 10;
    let parent_core = MassProperties::new(parent_mass);

    let mut children: Vec<(MassProperties, DVec3, DMat3)> = Vec::new();
    for i in 0..n {
        let angle = i as f64 * std::f64::consts::TAU / n as f64;
        let offset = DVec3::new(i as f64 * 0.5, 2.0 * angle.cos(), 2.0 * angle.sin());
        children.push((MassProperties::new(child_mass), offset, DMat3::IDENTITY));
    }

    let mut app = build_app();
    let (parent_e, _) = spawn_topology(&mut app, parent_core, &children);
    let bevy_comp = read_composite(&app, parent_e);

    let mut tree = MassTree::new();
    let pid = tree.add_root("parent".into(), parent_core);
    for (core, off, rot) in &children {
        let cid = tree.add_body("child".into(), *core);
        tree.attach(cid, pid, *off, *rot);
    }
    let arena_comp = tree.get(pid).composite_properties;

    assert!((bevy_comp.mass - arena_comp.mass).abs() < 1e-10);
    assert_vec3_close(bevy_comp.position, arena_comp.position, 1e-10, "many CoM");
    assert_mat3_close(bevy_comp.inertia, arena_comp.inertia, 1e-8, "many inertia");
}

#[test]
fn bevy_parity_mass_inertia_tensor_symmetry() {
    // Mirror of `tier3_mass_inertia_tensor_symmetry`: asymmetric
    // offsets + non-trivial rotations.
    let parent_core = MassProperties::with_inertia(
        50.0,
        DMat3::from_diagonal(DVec3::new(100.0, 200.0, 300.0)),
        DVec3::new(0.1, -0.2, 0.3),
    );
    let c1_core = MassProperties::with_inertia(
        20.0,
        DMat3::from_diagonal(DVec3::new(10.0, 30.0, 50.0)),
        DVec3::new(-0.1, 0.05, 0.0),
    );
    let c2_core = MassProperties::with_inertia(
        15.0,
        DMat3::from_diagonal(DVec3::new(5.0, 15.0, 25.0)),
        DVec3::ZERO,
    );

    let angle = 30.0_f64.to_radians();
    let c = angle.cos();
    let s = angle.sin();
    let rot_y_30 = DMat3::from_cols(
        DVec3::new(c, 0.0, -s),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(s, 0.0, c),
    );
    let angle2 = 60.0_f64.to_radians();
    let c2a = angle2.cos();
    let s2a = angle2.sin();
    let rot_x_60 = DMat3::from_cols(
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, c2a, s2a),
        DVec3::new(0.0, -s2a, c2a),
    );

    let children = [
        (c1_core, DVec3::new(1.5, -0.7, 2.1), rot_y_30),
        (c2_core, DVec3::new(-0.3, 1.8, -0.9), rot_x_60),
    ];

    let mut app = build_app();
    let (parent_e, _) = spawn_topology(&mut app, parent_core, &children);
    let bevy_comp = read_composite(&app, parent_e);

    let mut tree = MassTree::new();
    let pid = tree.add_root("parent".into(), parent_core);
    for (core, off, rot) in children {
        let cid = tree.add_body("child".into(), core);
        tree.attach(cid, pid, off, rot);
    }
    let arena_comp = tree.get(pid).composite_properties;

    assert!((bevy_comp.mass - arena_comp.mass).abs() < 1e-12);
    assert_vec3_close(bevy_comp.position, arena_comp.position, 1e-12, "asym CoM");
    assert_mat3_close(bevy_comp.inertia, arena_comp.inertia, 1e-10, "asym inertia");

    // Composite inertia must be symmetric in the ECS path too.
    let i = bevy_comp.inertia;
    assert_mat3_close(i, i.transpose(), 1e-10, "ECS composite inertia symmetric");
}

#[test]
fn bevy_parity_mass_detach_recovers_original() {
    // Mirror of `tier3_mass_detach_recovers_original`. ECS detach is
    // expressed by removing the `MassChildOf` component (the
    // ECS-native equivalent of arena `MassTree::detach`).
    let parent_inertia = DMat3::from_diagonal(DVec3::new(100.0, 200.0, 300.0));
    let parent_core = MassProperties::with_inertia(50.0, parent_inertia, DVec3::new(0.5, 0.0, 0.0));
    let child_core = MassProperties::new(25.0);
    let offset = DVec3::new(3.0, 1.0, -0.5);
    let rot = DMat3::IDENTITY;

    let mut app = build_app();
    let (parent_e, child_es) = spawn_topology(&mut app, parent_core, &[(child_core, offset, rot)]);
    let attached = read_composite(&app, parent_e);

    // Reference (arena attach + detach).
    let mut tree = MassTree::new();
    let pid = tree.add_root("parent".into(), parent_core);
    let cid = tree.add_body("child".into(), child_core);
    tree.attach(cid, pid, offset, rot);
    let arena_attached = tree.get(pid).composite_properties;

    assert!((attached.mass - arena_attached.mass).abs() < 1e-12);

    // Detach: remove the MassChildOf relation in ECS, run system
    // again. The kernel sees only the root, returns
    // `composite == core` for the (now lone) parent.
    let child_e = child_es[0];
    app.world_mut().entity_mut(child_e).remove::<MassChildOf>();
    app.update();
    let after_detach = read_composite(&app, parent_e);

    // Arena reference for the detached state.
    tree.detach(cid);
    let arena_detached = tree.get(pid).composite_properties;

    assert!(
        (after_detach.mass - arena_detached.mass).abs() < 1e-12,
        "post-detach mass: bevy {} vs arena {}",
        after_detach.mass,
        arena_detached.mass
    );
    assert_vec3_close(
        after_detach.position,
        arena_detached.position,
        1e-12,
        "post-detach CoM",
    );
    assert_mat3_close(
        after_detach.inertia,
        arena_detached.inertia,
        1e-10,
        "post-detach inertia",
    );
}

/// Regression for PR #283 review thread PRRT_kwDORtae6c5_KHnH.
///
/// Both composition paths can coexist on the same entity during the
/// migration window: the legacy arena path
/// (`MassBodyIdC` + `AttachEvent` driven by `staging_system`) and the
/// ECS-native path (`MassChildOf` driven by `composite_mass_system`).
/// The ECS path uses `Changed<MassPropertiesC>` to detect mid-sim
/// *core-mass* edits (fuel burn, propellant offload) and refresh its
/// hidden `CoreMassPropertiesC` cache. Before the fix, the legacy
/// `staging_system`'s composite write-back also tripped that filter,
/// so the next tick the ECS path seeded `CoreMassPropertiesC` from a
/// **composite** value — corrupting the core cache so future
/// recompositions Steiner-shifted the already-composed mass on top of
/// itself.
///
/// The fix is `bypass_change_detection` on `staging_system`'s
/// composite write. This test exercises that path: a tick where
/// `staging_system` writes a composite must NOT cause the next
/// tick's `composite_mass_system` to seed `CoreMassPropertiesC` from
/// the composite. We verify by detaching after the staging tick and
/// asserting the parent composite reverts to its core (== parent_core
/// alone). If the bug regresses, the cache holds the composite, the
/// "core" recovered on detach is heavier, and the post-detach
/// composite no longer matches the original core.
#[test]
fn bevy_staging_system_does_not_corrupt_composite_core_cache() {
    use bevy_jeod::{DetachEvent, MassTreeR};
    use jeod_sim::MassBodyId;

    // Build an app with both staging and composite systems wired.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_message::<AttachEvent>();
    app.add_message::<DetachEvent>();
    // composite first, then staging — matches the production schedule
    // ordering in JeodPlugin::build (composite_mass_system runs in
    // the EphemerisUpdate stage, staging_system in the Interaction
    // stage).
    app.add_systems(
        Update,
        (
            composite_mass_system,
            bevy_jeod::staging_system.after(composite_mass_system),
        ),
    );

    // Seed the arena with parent + child so MassBodyIdC entries are
    // valid for staging_system's tree.attach call.
    let mut tree = MassTree::new();
    let parent_core = MassProperties::with_inertia(
        100.0,
        DMat3::from_diagonal(DVec3::new(50.0, 60.0, 70.0)),
        DVec3::ZERO,
    );
    let child_core = MassProperties::new(40.0);
    let parent_arena_id: MassBodyId = tree.add_root("parent".into(), parent_core);
    let child_arena_id: MassBodyId = tree.add_body("child".into(), child_core);
    app.insert_resource(MassTreeR(tree));

    // Spawn parent + child entities. Note: BOTH carry `MassBodyIdC`
    // (legacy path) AND will carry `MassChildOf` (new path) once
    // attached — the migration window scenario.
    let parent_entity = app
        .world_mut()
        .spawn((
            MassPropertiesC::from(parent_core),
            MassBodyIdC(parent_arena_id),
        ))
        .id();
    let child_entity = app
        .world_mut()
        .spawn((
            MassPropertiesC::from(child_core),
            MassBodyIdC(child_arena_id),
        ))
        .id();

    // Tick 1: no attach yet — composite_mass_system seeds
    // CoreMassPropertiesC for both entities from their (still-core)
    // MassPropertiesC.
    app.update();

    // Tick 2: emit AttachEvent. staging_system (legacy) attaches in
    // the arena and writes the composite back into parent's
    // MassPropertiesC. With the fix, this write uses
    // bypass_change_detection so composite_mass_system on the *next*
    // tick does not see it as a Changed core edit.
    let offset = DVec3::new(2.0, 0.0, 0.0);
    app.world_mut().write_message(AttachEvent {
        child: child_entity,
        parent: parent_entity,
        offset: jeod_sim::Vec3Ext::m_at::<jeod_sim::StructuralFrame<jeod_sim::SelfRef>>(offset),
        t_parent_child: DMat3::IDENTITY,
    });
    // Add the ECS-native MassChildOf relation too, so on the next
    // tick composite_mass_system has work to do (otherwise it
    // fast-paths to the empty-parents branch).
    app.world_mut()
        .entity_mut(child_entity)
        .insert(MassChildOf::with_rotation(
            parent_entity,
            offset,
            DMat3::IDENTITY,
        ));
    app.update();

    // Tick 3+: detach the child via the ECS path (remove
    // MassChildOf). With a clean core cache, composite_mass_system
    // reverts the parent's composite to its original core (no
    // children, parent alone). With the bug, the cache holds the
    // staged composite, so the "core" the parent reverts to is the
    // composite — wrong.
    //
    // We run two ticks because the corrupted CoreMassPropertiesC
    // write goes through Commands (deferred to end-of-step). The
    // first post-detach tick sees the correct cached core (from tick
    // 1) and reverts; the SECOND post-detach tick sees the
    // newly-corrupted cache (seeded from the staging composite) and
    // would re-apply the composite. Two ticks therefore exercise the
    // full corruption + replay cycle.
    app.world_mut()
        .entity_mut(child_entity)
        .remove::<MassChildOf>();
    app.update();
    app.update();

    let recovered = read_composite(&app, parent_entity);

    // The recovered parent properties must match the original
    // parent_core, not the composite.
    assert!(
        (recovered.mass - parent_core.mass).abs() < 1e-12,
        "post-detach parent mass: got {}, expected core {}; \
         staging_system's composite write corrupted CoreMassPropertiesC.",
        recovered.mass,
        parent_core.mass,
    );
    assert_vec3_close(
        recovered.position,
        parent_core.position,
        1e-12,
        "post-detach parent CoM (CoreMassPropertiesC corruption check)",
    );
    assert_mat3_close(
        recovered.inertia,
        parent_core.inertia,
        1e-10,
        "post-detach parent inertia (CoreMassPropertiesC corruption check)",
    );
}
