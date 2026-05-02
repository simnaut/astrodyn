//! PR #260 round-10 review fixup: `JeodPlugin` accepts caller-supplied
//! [`FrameTreeR`] / [`RootFrameIdR`] (round-7) so mission code can pre-seed
//! custom root-level frames. The two resources describe the same tree and
//! must stay consistent — a mismatched pair would silently attach
//! sources/bodies under the wrong node, panic later in unrelated systems,
//! or corrupt frame-relative state.
//!
//! These tests exercise the four invalid-pair shapes that the plugin
//! must reject loudly:
//!
//! 1. `FrameTreeR` pre-installed without `RootFrameIdR`.
//! 2. `RootFrameIdR` pre-installed without `FrameTreeR`.
//! 3. `RootFrameIdR` is out of range for `FrameTreeR` (stale id from a
//!    different tree).
//! 4. `RootFrameIdR` points at an interior (non-root) frame.
//!
//! And the happy paths:
//!
//! 5. Neither pre-installed: plugin seeds both itself.
//! 6. Both pre-installed and consistent: plugin preserves them.

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
