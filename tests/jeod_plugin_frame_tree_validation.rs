//! `JeodPlugin` accepts caller-supplied [`FrameTreeR`] / [`RootFrameIdR`]
//! so mission code can pre-seed custom root-level frames. The two
//! resources describe the same tree and must stay consistent — a
//! mismatched pair would silently attach sources/bodies under the
//! wrong node, panic later in unrelated systems, or corrupt
//! frame-relative state.
//!
//! These tests exercise the five invalid-pair shapes that the plugin
//! must reject loudly:
//!
//! 1. `FrameTreeR` pre-installed without `RootFrameIdR`.
//! 2. `RootFrameIdR` pre-installed without `FrameTreeR`.
//! 3. `RootFrameIdR` is out of range for `FrameTreeR` (stale id from a
//!    different tree).
//! 4. `RootFrameIdR` points at an interior (non-root) frame.
//! 5. `RootFrameIdR` points at a non-`Inertial` root (e.g. a
//!    `PlanetFixed` or `Body` frame as the tree root). The rest of
//!    the plugin assumes the root is non-rotating.
//!
//! And the happy paths:
//!
//! 6. Neither pre-installed: plugin seeds both itself.
//! 7. Both pre-installed and consistent: plugin preserves them.

// `FrameTreeR` is `#[deprecated]` for mission-code use. This test
// pre-installs the resource explicitly to exercise `JeodPlugin`'s
// pre-installation validation path — the test *is* exercising the
// deprecated surface to check it fails-loud on misuse. The
// pre-install path will be retired alongside the resource itself.
#![allow(deprecated)]

use bevy::prelude::*;
use bevy_jeod::{FrameTreeR, JeodPlugin, RootFrameIdR};

#[test]
#[should_panic(expected = "FrameTreeR was pre-installed but RootFrameIdR was not")]
fn jeod_plugin_rejects_frame_tree_without_root_id() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let (frame_tree, _root) = FrameTreeR::new();
    app.insert_resource(frame_tree);
    app.add_plugins(JeodPlugin);
}

#[test]
#[should_panic(expected = "RootFrameIdR was pre-installed but FrameTreeR was not")]
fn jeod_plugin_rejects_root_id_without_frame_tree() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    // FrameId is a `usize` alias; any value at all is "valid" until the
    // plugin checks it against the (absent) FrameTreeR.
    app.insert_resource(RootFrameIdR(0));
    app.add_plugins(JeodPlugin);
}

#[test]
#[should_panic(expected = "out of range for the pre-installed FrameTreeR")]
fn jeod_plugin_rejects_out_of_range_root_id() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let (frame_tree, _root) = FrameTreeR::new();
    // Tree has exactly 1 node (the root at index 0); index 99 is a
    // stale id from somewhere else.
    app.insert_resource(frame_tree);
    app.insert_resource(RootFrameIdR(99));
    app.add_plugins(JeodPlugin);
}

#[test]
#[should_panic(expected = "is not a root of the pre-installed FrameTreeR")]
fn jeod_plugin_rejects_interior_node_as_root_id() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let (mut frame_tree, root) = FrameTreeR::new();
    // Add a child of the real root and advertise *the child* as the
    // root id. The plugin must catch this mismatch.
    let interior = frame_tree.0.add_child(
        root,
        "child.inertial".into(),
        jeod_sim::RefFrameKind::Inertial,
        jeod_sim::RefFrameState::default(),
    );
    app.insert_resource(frame_tree);
    app.insert_resource(RootFrameIdR(interior));
    app.add_plugins(JeodPlugin);
}

#[test]
#[should_panic(expected = "the rest of the plugin assumes the root is")]
fn jeod_plugin_rejects_non_inertial_root_kind() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    // Build a tree whose root is `PlanetFixed` rather than `Inertial`.
    // Source / body registration uses `RefFrameKind::Inertial` for new
    // children of root and the typed Bevy components are tagged
    // `<RootInertial>` — accepting this would let downstream math run
    // against a rotating root and produce silently-wrong physics.
    let mut frame_tree = jeod_sim::FrameTree::new();
    let root = frame_tree.add_root("custom.pfix".into(), jeod_sim::RefFrameKind::PlanetFixed);
    app.insert_resource(FrameTreeR(frame_tree));
    app.insert_resource(RootFrameIdR(root));
    app.add_plugins(JeodPlugin);
}

#[test]
fn jeod_plugin_seeds_both_when_neither_present() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(JeodPlugin);
    let frame_tree = app.world().resource::<FrameTreeR>();
    let root_id = app.world().resource::<RootFrameIdR>().0;
    assert_eq!(frame_tree.0.len(), 1, "plugin should seed exactly one root");
    assert!(
        frame_tree.0.parent(root_id).is_none(),
        "the seeded RootFrameIdR must be a root of the seeded FrameTreeR"
    );
}

#[test]
fn jeod_plugin_preserves_consistent_preseeded_pair() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let (mut frame_tree, root) = FrameTreeR::new();
    // Pre-seed an extra source-frame child to show that mission code's
    // own customizations survive the plugin attachment.
    let _custom = frame_tree.0.add_child(
        root,
        "preseeded_source.inertial".into(),
        jeod_sim::RefFrameKind::Inertial,
        jeod_sim::RefFrameState::default(),
    );
    app.insert_resource(frame_tree);
    app.insert_resource(RootFrameIdR(root));
    app.add_plugins(JeodPlugin);

    let frame_tree = app.world().resource::<FrameTreeR>();
    let root_id = app.world().resource::<RootFrameIdR>().0;
    assert_eq!(
        root_id, root,
        "plugin must not replace the preseeded RootFrameIdR"
    );
    assert_eq!(
        frame_tree.0.len(),
        2,
        "plugin must not add nodes when the caller pre-seeded the tree"
    );
}
