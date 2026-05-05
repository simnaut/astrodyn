//! Bevy parity for the chained-attach re-rooting kernel
//! (`MassTree::attach_with_reroot`, ports JEOD's
//! `dyn_body_attach.cc::attach_child` 521→567 reroot path).
//!
//! Scope: this test pins the composite-mass + parent-pointer
//! invariants of the new mass-tree kernel, exercised through the Bevy
//! adapter's [`MassTreeR`] resource. Specifically:
//!
//! 1. **Composite mass parity.** Build the same chained-attach scenario
//!    in two contexts — (a) a standalone [`MassTree`] (the
//!    [`jeod_runner::Simulation`] backing store), and (b) a Bevy
//!    [`MassTreeR`] resource. After firing the same sequence of
//!    `attach_with_reroot` / `detach` calls in both, every body's
//!    composite mass must match bit-for-bit.
//!
//! 2. **Parent-pointer parity.** Same kernel, asserted on the
//!    `parent` chain — both contexts must end up with identical tree
//!    shapes after each event.
//!
//! ## What is **not** validated here
//!
//! - **`AttachEvent` re-rooting wiring.** The Bevy `staging_system`
//!   does not yet call `attach_with_reroot` on its own — its
//!   `tree.attach(child, parent, ...)` call still panics if the child
//!   already has a parent. Wiring the staging path's pre-mutation
//!   snapshot + `MassChildOf` reparent through the reroot kernel is
//!   tracked separately; this PR ports the kernel + the runner-side
//!   integration, leaving the Bevy `AttachEvent` surface as a
//!   follow-up.
//!
//! - **Trajectory cross-validation.** That lives in the runner-side
//!   Tier 3 tests (`tier3_sim_complex_attach_detach.rs` /
//!   `tier3_sim_compute_child_derivative_*`). The kernel under test
//!   here is identical between runner and adapter, so trajectory
//!   cross-validation in either layer covers the kernel; the parity
//!   test focuses on the storage shape.
//!
//! [`MassTreeR`]: bevy_jeod::MassTreeR

use bevy::prelude::*;
use bevy_jeod::{JeodPlugin, MassTreeR};
use glam::{DMat3, DVec3};
use jeod_sim::{MassProperties, MassTree};

/// Run the chained-attach event sequence in a freshly built
/// `MassTree` and return it. The sequence mirrors the topological
/// timeline of `RUN_complex_attach_detach`:
///   - attach v1 → v2 (root subject)
///   - attach v1 → v3 (chained: v1 already has parent v2, so
///     `attach_with_reroot` re-roots v2 under v3)
///   - detach v1 from v2
///   - re-attach v1 → v2 (v1 is a standalone root again after the
///     detach, so this is a plain root-subject attach under v2,
///     which is interior to v3's tree — no reroot path)
fn run_chained_attach_sequence_on_tree(
    tree: &mut MassTree,
) -> (
    jeod_sim::MassBodyId,
    jeod_sim::MassBodyId,
    jeod_sim::MassBodyId,
) {
    let m1 = MassProperties::with_inertia(
        1.0,
        DMat3::from_diagonal(DVec3::splat(10.0)),
        DVec3::new(5.0, 0.0, 0.0),
    );
    let m2 = MassProperties::with_inertia(
        2.0,
        DMat3::from_diagonal(DVec3::splat(20.0)),
        DVec3::new(5.0, 0.0, 0.0),
    );
    let m3 = MassProperties::with_inertia(
        3.0,
        DMat3::from_diagonal(DVec3::splat(30.0)),
        DVec3::new(5.0, 0.0, 0.0),
    );
    let v1 = tree.add_body("veh1".into(), m1);
    let v2 = tree.add_body("veh2".into(), m2);
    let v3 = tree.add_body("veh3".into(), m3);

    // Attach v1 → v2 (root subject — bit-equivalent to plain attach).
    tree.attach_with_reroot(v1, v2, DVec3::new(-10.0, 0.0, 0.0), DMat3::IDENTITY);
    // Chained attach v1 → v3: subject (v1) already has a parent (v2).
    // The kernel auto-re-roots v2 under v3.
    let _ = tree.attach_with_reroot(v1, v3, DVec3::new(0.0, 0.0, 5.0), DMat3::IDENTITY);
    // Detach v1: removes only the v1 ↔ v2 edge; v2 is still under v3.
    tree.detach(v1);
    // Re-attach v1 → v2 (chained: v1 is now standalone again, v2 is
    // interior to v3's tree). Same code path as the first attach.
    tree.attach_with_reroot(v1, v2, DVec3::new(-10.0, 0.0, 0.0), DMat3::IDENTITY);

    (v1, v2, v3)
}

/// Convenience: assert two trees are bit-identical on every body's
/// composite mass and parent pointer.
fn assert_trees_match(
    a: &MassTree,
    b: &MassTree,
    a_ids: (
        jeod_sim::MassBodyId,
        jeod_sim::MassBodyId,
        jeod_sim::MassBodyId,
    ),
    b_ids: (
        jeod_sim::MassBodyId,
        jeod_sim::MassBodyId,
        jeod_sim::MassBodyId,
    ),
    label: &str,
) {
    let pairs = [
        ("v1", a_ids.0, b_ids.0),
        ("v2", a_ids.1, b_ids.1),
        ("v3", a_ids.2, b_ids.2),
    ];
    // Composite mass.
    for (name, ida, idb) in pairs {
        let ma = a.get(ida).composite_properties.mass;
        let mb = b.get(idb).composite_properties.mass;
        assert_eq!(
            ma, mb,
            "{label}: {name} composite mass mismatch (a={ma}, b={mb})"
        );
    }
    // Parent pointer (translate ids between trees by name match —
    // since both trees register v1, v2, v3 in the same order, the ids
    // happen to be equal, but we don't rely on that: we look up the
    // counterpart's parent id and check by name).
    for (name, ida, idb) in pairs {
        let pa = a.parent(ida);
        let pb = b.parent(idb);
        let pa_name = pa.map(|p| a.get(p).name.clone());
        let pb_name = pb.map(|p| b.get(p).name.clone());
        assert_eq!(
            pa_name, pb_name,
            "{label}: {name} parent name mismatch (a={pa_name:?}, b={pb_name:?})"
        );
    }
}

/// Build a Bevy `App` carrying a `JeodPlugin` + a fresh `MassTreeR`
/// resource, then drive the chained-attach sequence by mutating the
/// resource directly. Compares against an out-of-Bevy `MassTree` that
/// runs the same sequence.
///
/// The kernel under test is identical in both contexts — this test
/// would only fail if the Bevy `MassTreeR` Resource wrapper somehow
/// diverged from the standalone `MassTree` on attach/detach
/// composite-mass arithmetic. It guards against future regressions
/// where a Bevy-side override (e.g. an introspection wrapper) could
/// silently mutate or shadow tree state.
#[test]
fn bevy_parity_chained_attach_reroot_storage() {
    // -- Bevy side --
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(JeodPlugin);
    app.insert_resource(MassTreeR(MassTree::new()));
    let bevy_ids = {
        let mut tree_r = app.world_mut().resource_mut::<MassTreeR>();
        run_chained_attach_sequence_on_tree(&mut tree_r.0)
    };
    let bevy_tree = &app.world().resource::<MassTreeR>().0;

    // -- Runner side (standalone tree) --
    let mut runner_tree = MassTree::new();
    let runner_ids = run_chained_attach_sequence_on_tree(&mut runner_tree);

    // After the full sequence, both trees should be identical:
    //   v3 root
    //   v2 child of v3
    //   v1 child of v2
    // composite masses: v3 = 6, v2 = 3, v1 = 1
    assert_eq!(bevy_tree.get(bevy_ids.2).composite_properties.mass, 6.0);
    assert_eq!(bevy_tree.get(bevy_ids.1).composite_properties.mass, 3.0);
    assert_eq!(bevy_tree.get(bevy_ids.0).composite_properties.mass, 1.0);

    assert_trees_match(
        bevy_tree,
        &runner_tree,
        bevy_ids,
        runner_ids,
        "post-full-sequence",
    );
}

/// Spot-check the intermediate tree shapes the Bevy side passes
/// through, so a regression that affects only the *transient*
/// state (e.g. a misordered recompute_composites pass) is caught here
/// even when the end state happens to converge.
#[test]
fn bevy_parity_chained_attach_reroot_intermediate_states() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(JeodPlugin);
    app.insert_resource(MassTreeR(MassTree::new()));

    let (v1, v2, v3, root_after_first_attach, root_after_reroot, root_after_detach) = {
        let mut tree_r = app.world_mut().resource_mut::<MassTreeR>();
        let tree = &mut tree_r.0;

        let m1 = MassProperties::with_inertia(
            1.0,
            DMat3::from_diagonal(DVec3::splat(10.0)),
            DVec3::new(5.0, 0.0, 0.0),
        );
        let m2 = MassProperties::with_inertia(
            2.0,
            DMat3::from_diagonal(DVec3::splat(20.0)),
            DVec3::new(5.0, 0.0, 0.0),
        );
        let m3 = MassProperties::with_inertia(
            3.0,
            DMat3::from_diagonal(DVec3::splat(30.0)),
            DVec3::new(5.0, 0.0, 0.0),
        );
        let v1 = tree.add_body("veh1".into(), m1);
        let v2 = tree.add_body("veh2".into(), m2);
        let v3 = tree.add_body("veh3".into(), m3);

        // Snapshot intermediate roots after each event.
        tree.attach_with_reroot(v1, v2, DVec3::new(-10.0, 0.0, 0.0), DMat3::IDENTITY);
        let root_after_first_attach = tree.root_of(v1);

        let _ = tree.attach_with_reroot(v1, v3, DVec3::new(0.0, 0.0, 5.0), DMat3::IDENTITY);
        let root_after_reroot = tree.root_of(v1);

        tree.detach(v1);
        let root_after_detach = tree.root_of(v1);

        (
            v1,
            v2,
            v3,
            root_after_first_attach,
            root_after_reroot,
            root_after_detach,
        )
    };

    // After (v1, v2): v1's root is v2.
    assert_eq!(root_after_first_attach, v2);
    // After (v1, v3) with v1 already child of v2: v1's root is now v3
    // (re-rooted whole subtree).
    assert_eq!(root_after_reroot, v3);
    // After detach v1: v1 is now a free root again.
    assert_eq!(root_after_detach, v1);

    // After detach, intermediate composite masses on v3's remaining
    // tree (just v2 + v3) should be 5; v1 should be back to 1.
    let bevy_tree = &app.world().resource::<MassTreeR>().0;
    assert_eq!(bevy_tree.get(v3).composite_properties.mass, 5.0);
    assert_eq!(bevy_tree.get(v2).composite_properties.mass, 2.0);
    assert_eq!(bevy_tree.get(v1).composite_properties.mass, 1.0);
}
