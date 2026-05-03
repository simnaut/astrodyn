//! Storage-agnostic frame-tree algorithms ([Frame-Tree-ECS-Native § 7]).
//!
//! Factors the load-bearing tree-walking algorithms
//! (`compose_to_ancestor`, `common_ancestor`, `compute_relative_state`)
//! over a small [`FrameStorage`] trait so the same algorithm serves
//! both backends:
//!
//! - the runner's arena-based [`FrameTree`] (via the `impl FrameStorage
//!   for FrameTree` here), and
//! - the Bevy adapter's ECS-hierarchy view (`bevy_jeod`'s
//!   `RelativeFrameState` `SystemParam`, which `impl`s `FrameStorage`
//!   for itself in the adapter crate).
//!
//! The trait is **internal scaffolding**, not a mission-code surface.
//! Mission code uses `bevy_jeod::frame_param::RelativeFrameState`
//! (`SystemParam`); the trait is the implementation detail that lets
//! the runner and Bevy share one copy of the algorithms.
//!
//! ## What the trait abstracts
//!
//! Pure read-only graph traversal:
//! - `parent(id) -> Option<id>` — walk one link upward.
//! - `state(id) -> RefFrameState` — read the state stored at a node
//!   (relative to its parent).
//!
//! Mutation (e.g. `reparent`) is **not** in the trait: it lives in
//! each backend's native API because mutating Bevy hierarchy goes
//! through `Commands` (deferred), while mutating the arena is direct.
//! Trying to abstract over those two mutation models would force one
//! into the other's idiom.
//!
//! [Frame-Tree-ECS-Native § 7]: https://github.com/simnaut/bevy_jeod/wiki/Frame-Tree-ECS-Native#7-internal-algorithm-sharing-q1

use crate::frame_tree::{FrameId, FrameTree};
use crate::ref_frame_state::RefFrameState;

/// Read-only access to a frame tree, abstracting over the underlying
/// storage. Implementors: [`FrameTree`] (arena), and Bevy adapters
/// such as `bevy_jeod`'s `RelativeFrameState` `SystemParam`.
pub trait FrameStorage {
    /// Identifier type for a frame in this storage. Arena uses
    /// [`FrameId`] (== `usize`); ECS adapter uses `bevy::ecs::entity::Entity`.
    type Id: Copy + Eq + std::fmt::Debug;

    /// Parent of `id`, or `None` for a root.
    fn parent(&self, id: Self::Id) -> Option<Self::Id>;

    /// State stored at `id`, expressed relative to its parent.
    /// Roots return [`RefFrameState::default`] semantically, but the
    /// trait does not require this — `compose_to_ancestor` early-exits
    /// before the state of a root is read in practice.
    fn state(&self, id: Self::Id) -> RefFrameState;
}

/// Compose states from `id` up to `ancestor`, returning the state of
/// `id` relative to `ancestor`. Generic counterpart to
/// [`FrameTree`]'s private `compose_to_ancestor`.
///
/// # Panics
/// Panics if `id` is not a descendant of `ancestor` (the caller is
/// expected to call [`common_ancestor`] first).
pub fn compose_to_ancestor<S: FrameStorage>(
    storage: &S,
    id: S::Id,
    ancestor: S::Id,
) -> RefFrameState {
    if id == ancestor {
        return RefFrameState::default();
    }
    let mut composed = storage.state(id);
    let mut current = id;
    while let Some(parent) = storage.parent(current) {
        if parent == ancestor {
            return composed;
        }
        composed.incr_left(&storage.state(parent));
        current = parent;
    }
    panic!(
        "compose_to_ancestor: frame {id:?} is not a descendant of \
         ancestor {ancestor:?}; common_ancestor walk should have caught this"
    );
}

/// Find the lowest common ancestor of two frames, or `None` if they
/// are in disconnected subtrees. Walks both up to equal depth, then
/// in lockstep until they meet.
pub fn common_ancestor<S: FrameStorage>(storage: &S, a: S::Id, b: S::Id) -> Option<S::Id> {
    let mut da = depth(storage, a);
    let mut db = depth(storage, b);
    let mut ca = a;
    let mut cb = b;
    while da > db {
        ca = storage.parent(ca)?;
        da -= 1;
    }
    while db > da {
        cb = storage.parent(cb)?;
        db -= 1;
    }
    while ca != cb {
        ca = storage.parent(ca)?;
        cb = storage.parent(cb)?;
    }
    Some(ca)
}

/// Depth of `id` in the tree (root = 0).
fn depth<S: FrameStorage>(storage: &S, id: S::Id) -> usize {
    let mut d = 0usize;
    let mut current = id;
    while let Some(parent) = storage.parent(current) {
        d += 1;
        current = parent;
    }
    d
}

/// Compute the state of `to` relative to `from`. Generic counterpart
/// to [`FrameTree::compute_relative_state`].
///
/// # Panics
/// Panics if `from` and `to` do not share a common ancestor.
pub fn compute_relative_state<S: FrameStorage>(
    storage: &S,
    from: S::Id,
    to: S::Id,
) -> RefFrameState {
    if from == to {
        return RefFrameState::default();
    }
    let ancestor = common_ancestor(storage, from, to).unwrap_or_else(|| {
        panic!(
            "compute_relative_state: frames {from:?} and {to:?} do not \
             share a common ancestor"
        )
    });
    let state_from = compose_to_ancestor(storage, from, ancestor);
    let state_to = compose_to_ancestor(storage, to, ancestor);
    let from_negated = RefFrameState::negate(&state_from);
    from_negated.incr_right(&state_to)
}

// -- Arena impl --------------------------------------------------------

impl FrameStorage for FrameTree {
    type Id = FrameId;

    fn parent(&self, id: FrameId) -> Option<FrameId> {
        // Delegates to the inherent `parent` method on FrameTree.
        FrameTree::parent(self, id)
    }

    fn state(&self, id: FrameId) -> RefFrameState {
        self.get(id).state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ref_frame_state::{RefFrameKind, RefFrameRot, RefFrameTrans};
    use glam::DVec3;

    fn build_tree() -> (FrameTree, FrameId, FrameId, FrameId) {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState {
                trans: RefFrameTrans {
                    position: DVec3::new(1.0e6, 2.0e6, 3.0e6),
                    velocity: DVec3::new(10.0, 20.0, 30.0),
                },
                rot: RefFrameRot::default(),
            },
        );
        let b = tree.add_child(
            a,
            "B".into(),
            RefFrameKind::Body,
            RefFrameState {
                trans: RefFrameTrans {
                    position: DVec3::new(100.0, 200.0, 300.0),
                    velocity: DVec3::new(1.0, 2.0, 3.0),
                },
                rot: RefFrameRot::default(),
            },
        );
        (tree, root, a, b)
    }

    #[test]
    fn shared_compute_relative_state_matches_inherent() {
        let (tree, root, _a, b) = build_tree();
        // Inherent (FrameTree's own implementation).
        let inherent = tree.compute_relative_state(root, b);
        // Shared (this module's generic implementation, dispatched via
        // FrameStorage on FrameTree).
        let shared = compute_relative_state(&tree, root, b);
        assert_eq!(inherent.trans.position, shared.trans.position);
        assert_eq!(inherent.trans.velocity, shared.trans.velocity);
        assert_eq!(inherent.rot.t_parent_this, shared.rot.t_parent_this);
        assert_eq!(inherent.rot.ang_vel_this, shared.rot.ang_vel_this);
    }

    #[test]
    fn shared_common_ancestor_matches_inherent() {
        let (tree, root, a, b) = build_tree();
        // Sibling-of-A under root, to make common_ancestor non-trivial.
        let mut tree = tree;
        let _c = tree.add_child(
            root,
            "C".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        assert_eq!(common_ancestor(&tree, b, a), Some(a));
        assert_eq!(common_ancestor(&tree, b, root), Some(root));
        assert_eq!(common_ancestor(&tree, a, b), Some(a));
    }
}
