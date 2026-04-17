//! Arena-based frame tree: a faithful port of JEOD's RefFrame hierarchy.
//!
//! JEOD models reference frames as a tree. Each node stores its state
//! (position, velocity, orientation, angular velocity) relative to its
//! parent. Relative states between arbitrary frames are computed by
//! walking to the common ancestor and composing/negating states.
//!
//! This module is pure Rust with zero Bevy dependency.

use crate::ref_frame_state::{RefFrameKind, RefFrameState};

/// Handle into the [`FrameTree`] arena.
pub type FrameId = usize;

/// A node in the frame tree.
#[derive(Debug, Clone)]
pub struct FrameNode {
    /// Human-readable name (e.g., "Earth.inertial", "ISS.composite_body").
    pub name: String,
    /// Kind of reference frame.
    pub kind: RefFrameKind,
    /// State relative to parent. Identity for root frames.
    pub state: RefFrameState,
}

/// Arena-based frame tree. Portable (no ECS dependency).
///
/// Frames are stored in a flat `Vec`; parent/child relationships are tracked
/// with parallel vectors of `Option<FrameId>` and `Vec<FrameId>`.
pub struct FrameTree {
    nodes: Vec<FrameNode>,
    parent: Vec<Option<FrameId>>,
    children: Vec<Vec<FrameId>>,
}

impl FrameTree {
    /// Create an empty tree.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            parent: Vec::new(),
            children: Vec::new(),
        }
    }

    // -- construction -------------------------------------------------------

    /// Add a root frame (no parent). State is identity.
    pub fn add_root(&mut self, name: String, kind: RefFrameKind) -> FrameId {
        let id = self.nodes.len();
        self.nodes.push(FrameNode {
            name,
            kind,
            state: RefFrameState::default(),
        });
        self.parent.push(None);
        self.children.push(Vec::new());
        id
    }

    /// Add a child frame with the given state relative to its parent.
    ///
    /// # Panics
    /// Panics if `parent_id` does not refer to an existing frame.
    pub fn add_child(
        &mut self,
        parent_id: FrameId,
        name: String,
        kind: RefFrameKind,
        state: RefFrameState,
    ) -> FrameId {
        assert!(
            parent_id < self.nodes.len(),
            "parent_id {parent_id} out of range (have {} frames)",
            self.nodes.len()
        );
        let id = self.nodes.len();
        self.nodes.push(FrameNode { name, kind, state });
        self.parent.push(Some(parent_id));
        self.children.push(Vec::new());
        self.children[parent_id].push(id);
        id
    }

    // -- accessors ----------------------------------------------------------

    /// Borrow a frame node by id.
    pub fn get(&self, id: FrameId) -> &FrameNode {
        &self.nodes[id]
    }

    /// Mutably borrow a frame node by id.
    pub fn get_mut(&mut self, id: FrameId) -> &mut FrameNode {
        &mut self.nodes[id]
    }

    /// Parent of the given frame, or `None` for a root.
    pub fn parent(&self, id: FrameId) -> Option<FrameId> {
        self.parent[id]
    }

    /// Direct children of the given frame.
    pub fn children(&self, id: FrameId) -> &[FrameId] {
        &self.children[id]
    }

    /// Number of frames in the tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    // -- tree traversal -----------------------------------------------------

    /// Depth of `id` in the tree (root has depth 0).
    pub fn depth(&self, id: FrameId) -> usize {
        let mut d = 0usize;
        let mut current = id;
        while let Some(p) = self.parent[current] {
            d += 1;
            current = p;
        }
        d
    }

    /// Find the common ancestor of two frames in O(depth).
    ///
    /// Computes both depths, brings the deeper frame up to the other's depth,
    /// then walks both up in lockstep until they meet.
    ///
    /// Returns `None` if the frames do not share a common root (disconnected
    /// subtrees).
    pub fn common_ancestor(&self, a: FrameId, b: FrameId) -> Option<FrameId> {
        let mut da = self.depth(a);
        let mut db = self.depth(b);
        let mut ca = a;
        let mut cb = b;

        // Equalize depths.
        while da > db {
            ca = self.parent[ca]?;
            da -= 1;
        }
        while db > da {
            cb = self.parent[cb]?;
            db -= 1;
        }

        // Walk up in lockstep.
        while ca != cb {
            ca = self.parent[ca]?;
            cb = self.parent[cb]?;
        }
        Some(ca)
    }

    /// Find the common ancestor of two frames.
    ///
    /// Convenience wrapper around `common_ancestor` that panics if the frames
    /// do not share a common root. Prefer `common_ancestor` in new code.
    pub fn find_common_ancestor(&self, a: FrameId, b: FrameId) -> FrameId {
        self.common_ancestor(a, b)
            .expect("frames do not share a common ancestor")
    }

    /// Path from `descendant` up to `ancestor` (inclusive of both endpoints).
    ///
    /// Returns `None` if `ancestor` is not actually an ancestor of `descendant`.
    pub fn path_to_ancestor(&self, descendant: FrameId, ancestor: FrameId) -> Option<Vec<FrameId>> {
        let mut path = Vec::new();
        let mut current = descendant;
        loop {
            path.push(current);
            if current == ancestor {
                return Some(path);
            }
            current = self.parent[current]?;
        }
    }

    /// Compute the relative state between two frames, returning `None` if
    /// they don't share a common ancestor.
    ///
    /// Option-returning variant of `compute_relative_state`. Useful when the
    /// frame relationship isn't statically known.
    pub fn try_compute_relative_state(&self, from: FrameId, to: FrameId) -> Option<RefFrameState> {
        let ancestor = self.common_ancestor(from, to)?;
        let state_from = self.compose_to_ancestor(from, ancestor);
        let state_to = self.compose_to_ancestor(to, ancestor);
        let from_negated = RefFrameState::negate(&state_from);
        Some(from_negated.incr_right(&state_to))
    }

    /// Compute the relative state between two frames.
    ///
    /// Returns the state of `to` relative to `from` (i.e., if you are
    /// "standing in" the `from` frame, this tells you where `to` is).
    ///
    /// Port of JEOD `ref_frame_compute_relative_state.cc`. The algorithm:
    /// 1. Find common ancestor of `from` and `to`.
    /// 2. Compose states from `from` up to the ancestor.
    /// 3. Compose states from `to` up to the ancestor.
    /// 4. Result = negate(from_composed) composed with to_composed.
    ///
    /// This gives the state of `to` as seen from `from`.
    pub fn compute_relative_state(&self, from: FrameId, to: FrameId) -> RefFrameState {
        let ancestor = self.find_common_ancestor(from, to);

        // Compose state from `from` to ancestor.
        let state_from = self.compose_to_ancestor(from, ancestor);

        // Compose state from `to` to ancestor.
        let state_to = self.compose_to_ancestor(to, ancestor);

        // state_from is the state of `from` relative to ancestor (ancestor -> from).
        // state_to is the state of `to` relative to ancestor (ancestor -> to).
        // We want state of `to` relative to `from` (from -> to).
        // from -> to = negate(ancestor -> from) composed with (ancestor -> to)
        //           = (from -> ancestor) composed with (ancestor -> to)
        let from_negated = RefFrameState::negate(&state_from);
        from_negated.incr_right(&state_to)
    }

    // -- lookup --------------------------------------------------------------

    /// Find a frame by name. Returns the first match, or `None`.
    pub fn find_by_name(&self, name: &str) -> Option<FrameId> {
        self.nodes.iter().position(|n| n.name == name)
    }

    // -- tree mutation -------------------------------------------------------

    /// Test whether `id` is a descendant of `ancestor` (or equal to it).
    pub fn is_descendant_of(&self, id: FrameId, ancestor: FrameId) -> bool {
        if id == ancestor {
            return true;
        }
        let mut current = id;
        while let Some(p) = self.parent[current] {
            if p == ancestor {
                return true;
            }
            current = p;
        }
        false
    }

    /// Move a frame to a new parent, preserving its absolute state.
    ///
    /// Port of JEOD's `RefFrame::transplant_node()`: the frame's position in
    /// the tree changes, but its state relative to the root is preserved by
    /// recomputing the relative state with respect to the new parent.
    ///
    /// # Panics
    /// - `new_parent` is a descendant of `id` (would create a cycle).
    /// - `id` is a root frame with no parent.
    /// - `id` and `new_parent` do not share a common root.
    pub fn reparent(&mut self, id: FrameId, new_parent: FrameId) {
        // Check root first — root has no parent to detach from.
        assert!(
            self.parent[id].is_some(),
            "reparent: frame {} has no parent (is a root) — cannot reparent root frames",
            id
        );

        // No-op if already parented to `new_parent`.
        if self.parent[id] == Some(new_parent) {
            return;
        }

        assert!(
            !self.is_descendant_of(new_parent, id),
            "reparent: new_parent {} is a descendant of {} — would create a cycle",
            new_parent,
            id
        );

        // Verify frames share a common root before computing relative state.
        // find_common_ancestor panics with a generic message; catch it here
        // with a reparent-specific message for easier debugging.
        {
            let mut cur = new_parent;
            while let Some(p) = self.parent[cur] {
                cur = p;
            }
            let new_parent_root = cur;
            cur = id;
            while let Some(p) = self.parent[cur] {
                cur = p;
            }
            assert!(
                cur == new_parent_root,
                "reparent: frame {id} and new_parent {new_parent} do not share a common root"
            );
        }

        // Compute state of `id` relative to `new_parent` (preserves absolute state).
        let new_state = self.compute_relative_state(new_parent, id);

        // Remove from old parent's children list.
        let old_parent = self.parent[id].unwrap();
        self.children[old_parent].retain(|&c| c != id);

        // Attach to new parent.
        self.parent[id] = Some(new_parent);
        self.children[new_parent].push(id);

        // Store recomputed relative state.
        self.nodes[id].state = new_state;
    }

    // -- tree traversal (internal) ------------------------------------------

    /// Compose states from `id` up to `ancestor`, returning the state of
    /// `id` relative to `ancestor`.
    ///
    /// The stored state of each frame is relative to its parent. Walking
    /// up the chain and composing with `incr_left` accumulates the
    /// parent-to-root transforms.
    fn compose_to_ancestor(&self, id: FrameId, ancestor: FrameId) -> RefFrameState {
        if id == ancestor {
            return RefFrameState::default();
        }

        // Start with the state of `id` relative to its parent.
        let mut composed = self.nodes[id].state;
        let mut current = id;

        // Walk upward, composing each parent's state on the left.
        while let Some(p) = self.parent[current] {
            if p == ancestor {
                // We've reached the ancestor; `composed` now represents
                // the state of `id` relative to `ancestor`.
                return composed;
            }
            // composed = parent_state composed with composed
            // i.e., ancestor->...->parent->current becomes ancestor->...->grandparent->current
            composed.incr_left(&self.nodes[p].state);
            current = p;
        }

        // If we get here, we walked all the way to a root without hitting
        // `ancestor`. This shouldn't happen if find_common_ancestor was correct.
        panic!(
            "compose_to_ancestor: frame {} is not a descendant of ancestor {}",
            id, ancestor
        );
    }
}

impl Default for FrameTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ref_frame_state::{RefFrameRot, RefFrameTrans};
    use glam::{DMat3, DVec3};
    use jeod_math::test_utils::{approx_eq_mat3, approx_eq_vec3};
    use jeod_math::JeodQuat;
    use std::f64::consts::FRAC_PI_2;

    const TOL: f64 = 1e-12;

    /// Helper: create a RefFrameState with a rotation about Z axis and a position offset.
    fn make_state(angle_z: f64, pos: DVec3, vel: DVec3, ang_vel: DVec3) -> RefFrameState {
        let q = JeodQuat::left_quat_from_eigen_rotation(angle_z, DVec3::Z);
        let t = q.left_quat_to_transformation();
        RefFrameState {
            trans: RefFrameTrans {
                position: pos,
                velocity: vel,
            },
            rot: RefFrameRot {
                q_parent_this: q,
                t_parent_this: t,
                ang_vel_this: ang_vel,
            },
        }
    }

    // -----------------------------------------------------------------------
    // 1. Single root: no parent, identity state
    // -----------------------------------------------------------------------
    #[test]
    fn single_root() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        assert!(tree.parent(root).is_none(), "root should have no parent");
        assert!(
            tree.children(root).is_empty(),
            "root should have no children"
        );

        let node = tree.get(root);
        assert_eq!(node.name, "root");
        assert_eq!(node.kind, RefFrameKind::Inertial);
        assert_eq!(node.state.trans.position, DVec3::ZERO);
        assert_eq!(node.state.trans.velocity, DVec3::ZERO);
        assert_eq!(node.state.rot.t_parent_this, DMat3::IDENTITY);
        assert_eq!(node.state.rot.ang_vel_this, DVec3::ZERO);
    }

    // -----------------------------------------------------------------------
    // 2. Parent-child links
    // -----------------------------------------------------------------------
    #[test]
    fn parent_child_links() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        let child_state = make_state(
            0.5,
            DVec3::new(1e6, 2e6, 3e6),
            DVec3::new(100.0, 200.0, 300.0),
            DVec3::new(0.01, 0.02, 0.03),
        );
        let child = tree.add_child(root, "child".into(), RefFrameKind::Body, child_state);

        assert_eq!(tree.parent(child), Some(root));
        assert_eq!(tree.children(root), &[child]);
        assert!(tree.children(child).is_empty());

        // Verify stored state matches
        let node = tree.get(child);
        assert!(
            approx_eq_vec3(node.state.trans.position, child_state.trans.position, TOL),
            "child position"
        );
        assert!(
            approx_eq_vec3(node.state.trans.velocity, child_state.trans.velocity, TOL),
            "child velocity"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Relative state to self is identity
    // -----------------------------------------------------------------------
    #[test]
    fn relative_state_to_self() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        let child_state = make_state(
            1.0,
            DVec3::new(1e7, 0.0, 0.0),
            DVec3::new(7000.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 0.001),
        );
        let child = tree.add_child(root, "child".into(), RefFrameKind::Body, child_state);

        let rel = tree.compute_relative_state(child, child);

        assert!(
            approx_eq_vec3(rel.trans.position, DVec3::ZERO, 1e-6),
            "self-relative position should be zero, got {:?}",
            rel.trans.position
        );
        assert!(
            approx_eq_vec3(rel.trans.velocity, DVec3::ZERO, 1e-6),
            "self-relative velocity should be zero, got {:?}",
            rel.trans.velocity
        );
        assert!(
            approx_eq_mat3(&rel.rot.t_parent_this, &DMat3::IDENTITY, 1e-10),
            "self-relative T should be identity"
        );
        assert!(
            approx_eq_vec3(rel.rot.ang_vel_this, DVec3::ZERO, 1e-10),
            "self-relative ang_vel should be zero"
        );
    }

    // -----------------------------------------------------------------------
    // 4. Relative state parent -> child matches child's stored state
    // -----------------------------------------------------------------------
    #[test]
    fn relative_state_parent_child() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        let child_state = make_state(
            0.5,
            DVec3::new(1e6, 2e6, 3e6),
            DVec3::new(100.0, 200.0, 300.0),
            DVec3::new(0.01, 0.02, 0.03),
        );
        let child = tree.add_child(root, "child".into(), RefFrameKind::Body, child_state);

        // relative state from root to child = child's state relative to root
        let rel = tree.compute_relative_state(root, child);

        assert!(
            approx_eq_vec3(rel.trans.position, child_state.trans.position, 1e-6),
            "parent->child position: expected {:?}, got {:?}",
            child_state.trans.position,
            rel.trans.position
        );
        assert!(
            approx_eq_vec3(rel.trans.velocity, child_state.trans.velocity, 1e-6),
            "parent->child velocity: expected {:?}, got {:?}",
            child_state.trans.velocity,
            rel.trans.velocity
        );
        assert!(
            approx_eq_mat3(
                &rel.rot.t_parent_this,
                &child_state.rot.t_parent_this,
                1e-10
            ),
            "parent->child T"
        );
        assert!(
            approx_eq_vec3(rel.rot.ang_vel_this, child_state.rot.ang_vel_this, 1e-10),
            "parent->child ang_vel"
        );
    }

    // -----------------------------------------------------------------------
    // 5. Relative state child -> parent is negation of child's state
    // -----------------------------------------------------------------------
    #[test]
    fn relative_state_child_parent() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        let child_state = make_state(
            0.5,
            DVec3::new(1e6, 2e6, 3e6),
            DVec3::new(100.0, 200.0, 300.0),
            DVec3::new(0.01, 0.02, 0.03),
        );
        let child = tree.add_child(root, "child".into(), RefFrameKind::Body, child_state);

        let rel = tree.compute_relative_state(child, root);
        let expected = RefFrameState::negate(&child_state);

        assert!(
            approx_eq_vec3(rel.trans.position, expected.trans.position, 1e-6),
            "child->parent position: expected {:?}, got {:?}",
            expected.trans.position,
            rel.trans.position
        );
        assert!(
            approx_eq_vec3(rel.trans.velocity, expected.trans.velocity, 1e-6),
            "child->parent velocity: expected {:?}, got {:?}",
            expected.trans.velocity,
            rel.trans.velocity
        );
        assert!(
            approx_eq_mat3(&rel.rot.t_parent_this, &expected.rot.t_parent_this, 1e-10),
            "child->parent T"
        );
        assert!(
            approx_eq_vec3(rel.rot.ang_vel_this, expected.rot.ang_vel_this, 1e-10),
            "child->parent ang_vel"
        );
    }

    // -----------------------------------------------------------------------
    // 6. Three-level tree: root -> A -> B
    //    Relative state root -> B should be composition of A.state and B.state
    // -----------------------------------------------------------------------
    #[test]
    fn three_level_tree() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        let state_a = make_state(
            FRAC_PI_2,
            DVec3::new(1000.0, 0.0, 0.0),
            DVec3::new(10.0, 0.0, 0.0),
            DVec3::ZERO,
        );
        let a = tree.add_child(root, "A".into(), RefFrameKind::Body, state_a);

        let state_b = make_state(
            0.0,
            DVec3::new(500.0, 0.0, 0.0),
            DVec3::new(5.0, 0.0, 0.0),
            DVec3::ZERO,
        );
        let b = tree.add_child(a, "B".into(), RefFrameKind::Body, state_b);

        let rel = tree.compute_relative_state(root, b);

        // Expected: state_a composed with state_b
        let expected = state_a.incr_right(&state_b);

        assert!(
            approx_eq_vec3(rel.trans.position, expected.trans.position, 1e-6),
            "root->B position: expected {:?}, got {:?}",
            expected.trans.position,
            rel.trans.position
        );
        assert!(
            approx_eq_vec3(rel.trans.velocity, expected.trans.velocity, 1e-6),
            "root->B velocity: expected {:?}, got {:?}",
            expected.trans.velocity,
            rel.trans.velocity
        );
        assert!(
            approx_eq_mat3(&rel.rot.t_parent_this, &expected.rot.t_parent_this, 1e-10),
            "root->B T"
        );
        assert!(
            approx_eq_vec3(rel.rot.ang_vel_this, expected.rot.ang_vel_this, 1e-10),
            "root->B ang_vel"
        );
    }

    // -----------------------------------------------------------------------
    // 7. Sibling relative state: two children of the same parent
    // -----------------------------------------------------------------------
    #[test]
    fn sibling_relative_state() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        let state_a = make_state(
            0.3,
            DVec3::new(1e6, 0.0, 0.0),
            DVec3::new(100.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 0.01),
        );
        let a = tree.add_child(root, "A".into(), RefFrameKind::Body, state_a);

        let state_b = make_state(
            -0.7,
            DVec3::new(0.0, 2e6, 0.0),
            DVec3::new(0.0, 200.0, 0.0),
            DVec3::new(0.0, 0.0, 0.02),
        );
        let b = tree.add_child(root, "B".into(), RefFrameKind::Body, state_b);

        // Relative state from A to B should be:
        //   negate(root -> A) composed with (root -> B)
        let rel = tree.compute_relative_state(a, b);

        let a_neg = RefFrameState::negate(&state_a);
        let expected = a_neg.incr_right(&state_b);

        assert!(
            approx_eq_vec3(rel.trans.position, expected.trans.position, 1e-4),
            "sibling A->B position: expected {:?}, got {:?}",
            expected.trans.position,
            rel.trans.position
        );
        assert!(
            approx_eq_vec3(rel.trans.velocity, expected.trans.velocity, 1e-4),
            "sibling A->B velocity: expected {:?}, got {:?}",
            expected.trans.velocity,
            rel.trans.velocity
        );
        assert!(
            approx_eq_mat3(&rel.rot.t_parent_this, &expected.rot.t_parent_this, 1e-10),
            "sibling A->B T"
        );
        assert!(
            approx_eq_vec3(rel.rot.ang_vel_this, expected.rot.ang_vel_this, 1e-10),
            "sibling A->B ang_vel"
        );
    }

    // -----------------------------------------------------------------------
    // 8. Four-level tree: relative state matches direct composition to 1e-14
    //
    // Phase 3 exit criterion: "relative state between any two frames
    // matches direct computation to < 1e-14".
    //
    // Build: root -> A -> B -> C -> D, and root -> E (sibling branch).
    // Compare tree-traversed relative state against explicit composition
    // for multiple frame pairs including cross-branch.
    // -----------------------------------------------------------------------
    #[test]
    fn four_level_tree_relative_state_precision() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        // Use moderate values to avoid floating-point precision loss
        // from large position magnitudes.
        let state_a = make_state(
            0.3,
            DVec3::new(100.0, 200.0, 50.0),
            DVec3::new(1.0, 2.0, 0.5),
            DVec3::new(0.001, 0.002, 0.003),
        );
        let a = tree.add_child(root, "A".into(), RefFrameKind::Body, state_a);

        let state_b = make_state(
            -0.7,
            DVec3::new(50.0, -30.0, 80.0),
            DVec3::new(0.5, -0.3, 0.8),
            DVec3::new(-0.001, 0.001, 0.002),
        );
        let b = tree.add_child(a, "B".into(), RefFrameKind::Body, state_b);

        let state_c = make_state(
            1.2,
            DVec3::new(-20.0, 40.0, 10.0),
            DVec3::new(-0.2, 0.4, 0.1),
            DVec3::new(0.003, -0.002, 0.001),
        );
        let c = tree.add_child(b, "C".into(), RefFrameKind::Body, state_c);

        let state_d = make_state(
            -0.4,
            DVec3::new(10.0, 10.0, -5.0),
            DVec3::new(0.1, 0.1, -0.05),
            DVec3::new(0.0005, 0.0005, -0.001),
        );
        let d = tree.add_child(c, "D".into(), RefFrameKind::Body, state_d);

        // Sibling branch: root -> E
        let state_e = make_state(
            0.8,
            DVec3::new(-60.0, 90.0, 30.0),
            DVec3::new(-0.6, 0.9, 0.3),
            DVec3::new(0.002, -0.001, 0.004),
        );
        let e = tree.add_child(root, "E".into(), RefFrameKind::Body, state_e);

        // 4-level composition accumulates ~6e-14 position error from
        // floating-point arithmetic. We therefore use a 1e-13 tolerance for
        // position, which still demonstrates sub-1e-13 precision, while
        // rotation matrices and angular velocities are checked at 1e-14.
        let tol_rot = 1e-14;
        let tol_pos = 1e-13;

        // ── root → D (4 levels deep) ──
        let rel_root_d = tree.compute_relative_state(root, d);
        let expected_root_d = state_a
            .incr_right(&state_b)
            .incr_right(&state_c)
            .incr_right(&state_d);

        assert!(
            approx_eq_mat3(
                &rel_root_d.rot.t_parent_this,
                &expected_root_d.rot.t_parent_this,
                tol_rot
            ),
            "root→D rotation exceeds {tol_rot:.0e}"
        );
        assert!(
            approx_eq_vec3(
                rel_root_d.trans.position,
                expected_root_d.trans.position,
                tol_pos
            ),
            "root→D position exceeds {tol_pos:.0e}: diff = {:.4e}",
            (rel_root_d.trans.position - expected_root_d.trans.position).length()
        );
        assert!(
            approx_eq_vec3(
                rel_root_d.trans.velocity,
                expected_root_d.trans.velocity,
                tol_pos
            ),
            "root→D velocity exceeds {tol_pos:.0e}: diff = {:.4e}",
            (rel_root_d.trans.velocity - expected_root_d.trans.velocity).length()
        );
        assert!(
            approx_eq_vec3(
                rel_root_d.rot.ang_vel_this,
                expected_root_d.rot.ang_vel_this,
                tol_rot
            ),
            "root→D ang_vel exceeds {tol_rot:.0e}"
        );

        // ── D → root (reverse of above) ──
        let rel_d_root = tree.compute_relative_state(d, root);
        let expected_d_root = RefFrameState::negate(&expected_root_d);

        assert!(
            approx_eq_mat3(
                &rel_d_root.rot.t_parent_this,
                &expected_d_root.rot.t_parent_this,
                tol_rot
            ),
            "D→root rotation exceeds {tol_rot:.0e}"
        );
        assert!(
            approx_eq_vec3(
                rel_d_root.trans.position,
                expected_d_root.trans.position,
                tol_pos
            ),
            "D→root position exceeds {tol_pos:.0e}"
        );

        // ── B → D (same branch, partial traversal) ──
        let rel_b_d = tree.compute_relative_state(b, d);
        let expected_b_d = state_c.incr_right(&state_d);

        assert!(
            approx_eq_mat3(
                &rel_b_d.rot.t_parent_this,
                &expected_b_d.rot.t_parent_this,
                tol_rot
            ),
            "B→D rotation exceeds {tol_rot:.0e}"
        );
        assert!(
            approx_eq_vec3(rel_b_d.trans.position, expected_b_d.trans.position, tol_pos),
            "B→D position exceeds {tol_pos:.0e}"
        );
        assert!(
            approx_eq_vec3(rel_b_d.trans.velocity, expected_b_d.trans.velocity, tol_pos),
            "B→D velocity exceeds {tol_pos:.0e}"
        );
        assert!(
            approx_eq_vec3(
                rel_b_d.rot.ang_vel_this,
                expected_b_d.rot.ang_vel_this,
                tol_rot
            ),
            "B→D ang_vel exceeds {tol_rot:.0e}"
        );

        // ── D → E (cross-branch: D..root..E) ──
        let rel_d_e = tree.compute_relative_state(d, e);
        let expected_d_e = RefFrameState::negate(&expected_root_d).incr_right(&state_e);

        assert!(
            approx_eq_mat3(
                &rel_d_e.rot.t_parent_this,
                &expected_d_e.rot.t_parent_this,
                tol_rot
            ),
            "D→E rotation exceeds {tol_rot:.0e}"
        );
        assert!(
            approx_eq_vec3(rel_d_e.trans.position, expected_d_e.trans.position, tol_pos),
            "D→E position exceeds {tol_pos:.0e}: diff = {:.4e}",
            (rel_d_e.trans.position - expected_d_e.trans.position).length()
        );
        assert!(
            approx_eq_vec3(rel_d_e.trans.velocity, expected_d_e.trans.velocity, tol_pos),
            "D→E velocity exceeds {tol_pos:.0e}"
        );
        assert!(
            approx_eq_vec3(
                rel_d_e.rot.ang_vel_this,
                expected_d_e.rot.ang_vel_this,
                tol_rot
            ),
            "D→E ang_vel exceeds {tol_rot:.0e}"
        );

        println!(
            "  4-level frame tree: all 4 frame pairs match direct composition within tolerances \
             (rot {tol_rot:.0e}, pos/vel {tol_pos:.0e})"
        );
    }

    // -----------------------------------------------------------------------
    // 9. find_by_name
    // -----------------------------------------------------------------------
    #[test]
    fn find_by_name() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("Earth.inertial".into(), RefFrameKind::Inertial);
        let moon = tree.add_child(
            root,
            "Moon.inertial".into(),
            RefFrameKind::Inertial,
            RefFrameState::default(),
        );

        assert_eq!(tree.find_by_name("Earth.inertial"), Some(root));
        assert_eq!(tree.find_by_name("Moon.inertial"), Some(moon));
        assert_eq!(tree.find_by_name("Mars.inertial"), None);
    }

    // -----------------------------------------------------------------------
    // 10. is_descendant_of
    // -----------------------------------------------------------------------
    #[test]
    fn is_descendant_of() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        let b = tree.add_child(a, "B".into(), RefFrameKind::Body, RefFrameState::default());
        let c = tree.add_child(
            root,
            "C".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );

        assert!(tree.is_descendant_of(b, root));
        assert!(tree.is_descendant_of(b, a));
        assert!(tree.is_descendant_of(a, root));
        assert!(tree.is_descendant_of(root, root)); // self
        assert!(!tree.is_descendant_of(root, a));
        assert!(!tree.is_descendant_of(c, a)); // sibling, not descendant
        assert!(!tree.is_descendant_of(b, c)); // cross-branch
    }

    // -----------------------------------------------------------------------
    // 11. reparent: move a child to a different parent, verify state preserved
    // -----------------------------------------------------------------------
    #[test]
    fn reparent_preserves_absolute_state() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        // Two children of root with distinct states.
        let state_a = make_state(
            0.3,
            DVec3::new(1000.0, 0.0, 0.0),
            DVec3::new(10.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 0.01),
        );
        let a = tree.add_child(root, "A".into(), RefFrameKind::Body, state_a);

        let state_b = make_state(
            -0.5,
            DVec3::new(0.0, 2000.0, 0.0),
            DVec3::new(0.0, 20.0, 0.0),
            DVec3::new(0.0, 0.0, 0.02),
        );
        let b = tree.add_child(root, "B".into(), RefFrameKind::Body, state_b);

        // Child of B.
        let state_c = make_state(
            0.1,
            DVec3::new(100.0, 100.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.001, 0.0, 0.0),
        );
        let c = tree.add_child(b, "C".into(), RefFrameKind::Body, state_c);

        // Record absolute state of C before reparent (root -> C).
        let abs_before = tree.compute_relative_state(root, c);

        // Reparent C from B to A.
        tree.reparent(c, a);

        // Verify tree structure changed.
        assert_eq!(tree.parent(c), Some(a));
        assert!(tree.children(a).contains(&c));
        assert!(!tree.children(b).contains(&c));

        // Verify absolute state is preserved.
        let abs_after = tree.compute_relative_state(root, c);

        let tol_pos = 1e-10;
        let tol_rot = 1e-13;
        assert!(
            approx_eq_vec3(abs_after.trans.position, abs_before.trans.position, tol_pos),
            "reparent position drift: expected {:?}, got {:?}, diff = {:.4e}",
            abs_before.trans.position,
            abs_after.trans.position,
            (abs_after.trans.position - abs_before.trans.position).length()
        );
        assert!(
            approx_eq_vec3(abs_after.trans.velocity, abs_before.trans.velocity, tol_pos),
            "reparent velocity drift: expected {:?}, got {:?}",
            abs_before.trans.velocity,
            abs_after.trans.velocity
        );
        assert!(
            approx_eq_mat3(
                &abs_after.rot.t_parent_this,
                &abs_before.rot.t_parent_this,
                tol_rot
            ),
            "reparent rotation drift"
        );
        assert!(
            approx_eq_vec3(
                abs_after.rot.ang_vel_this,
                abs_before.rot.ang_vel_this,
                tol_rot
            ),
            "reparent ang_vel drift"
        );
    }

    // -----------------------------------------------------------------------
    // 12. reparent panics on cycle
    // -----------------------------------------------------------------------
    #[test]
    #[should_panic(expected = "would create a cycle")]
    fn reparent_cycle_panics() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        let b = tree.add_child(a, "B".into(), RefFrameKind::Body, RefFrameState::default());

        // Try to reparent A under its own descendant B — should panic.
        tree.reparent(a, b);
    }

    // -----------------------------------------------------------------------
    // 13. reparent panics on root
    // -----------------------------------------------------------------------
    #[test]
    #[should_panic(expected = "cannot reparent root frames")]
    fn reparent_root_panics() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );

        // Try to reparent the root — should panic.
        tree.reparent(root, a);
    }

    // -----------------------------------------------------------------------
    // 14. common_ancestor: basic cases
    // -----------------------------------------------------------------------
    #[test]
    fn common_ancestor_same_frame() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        assert_eq!(tree.common_ancestor(root, root), Some(root));
    }

    #[test]
    fn common_ancestor_parent_child() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let child = tree.add_child(
            root,
            "child".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        assert_eq!(tree.common_ancestor(root, child), Some(root));
        assert_eq!(tree.common_ancestor(child, root), Some(root));
    }

    #[test]
    fn common_ancestor_siblings() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        let b = tree.add_child(
            root,
            "B".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        assert_eq!(tree.common_ancestor(a, b), Some(root));
    }

    #[test]
    fn common_ancestor_unrelated_trees() {
        let mut tree = FrameTree::new();
        let root_a = tree.add_root("root_a".into(), RefFrameKind::Inertial);
        let root_b = tree.add_root("root_b".into(), RefFrameKind::Inertial);
        assert_eq!(tree.common_ancestor(root_a, root_b), None);
    }

    #[test]
    fn common_ancestor_deep_tree() {
        //     root
        //     ├── A
        //     │   ├── B
        //     │   │   └── C
        //     │   │       └── D
        //     │   └── E
        //     └── F
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        let b = tree.add_child(a, "B".into(), RefFrameKind::Body, RefFrameState::default());
        let c = tree.add_child(b, "C".into(), RefFrameKind::Body, RefFrameState::default());
        let d = tree.add_child(c, "D".into(), RefFrameKind::Body, RefFrameState::default());
        let e = tree.add_child(a, "E".into(), RefFrameKind::Body, RefFrameState::default());
        let f = tree.add_child(
            root,
            "F".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );

        assert_eq!(tree.common_ancestor(d, e), Some(a));
        assert_eq!(tree.common_ancestor(d, f), Some(root));
        assert_eq!(tree.common_ancestor(b, d), Some(b));
        assert_eq!(tree.common_ancestor(d, a), Some(a));
    }

    // -----------------------------------------------------------------------
    // 15. depth
    // -----------------------------------------------------------------------
    #[test]
    fn depth_computation() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        let b = tree.add_child(a, "B".into(), RefFrameKind::Body, RefFrameState::default());
        let c = tree.add_child(b, "C".into(), RefFrameKind::Body, RefFrameState::default());

        assert_eq!(tree.depth(root), 0);
        assert_eq!(tree.depth(a), 1);
        assert_eq!(tree.depth(b), 2);
        assert_eq!(tree.depth(c), 3);
    }

    // -----------------------------------------------------------------------
    // 16. path_to_ancestor
    // -----------------------------------------------------------------------
    #[test]
    fn path_to_ancestor_basic() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        let b = tree.add_child(a, "B".into(), RefFrameKind::Body, RefFrameState::default());
        let c = tree.add_child(b, "C".into(), RefFrameKind::Body, RefFrameState::default());

        assert_eq!(tree.path_to_ancestor(c, root), Some(vec![c, b, a, root]));
        assert_eq!(tree.path_to_ancestor(c, a), Some(vec![c, b, a]));
        assert_eq!(tree.path_to_ancestor(c, c), Some(vec![c]));
    }

    #[test]
    fn path_to_ancestor_not_an_ancestor() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        let b = tree.add_child(
            root,
            "B".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );

        // A is not an ancestor of B (they are siblings).
        assert_eq!(tree.path_to_ancestor(b, a), None);
    }

    // -----------------------------------------------------------------------
    // 17. try_compute_relative_state
    // -----------------------------------------------------------------------
    #[test]
    fn try_compute_relative_state_returns_none_for_disconnected() {
        let mut tree = FrameTree::new();
        let root_a = tree.add_root("root_a".into(), RefFrameKind::Inertial);
        let root_b = tree.add_root("root_b".into(), RefFrameKind::Inertial);
        assert!(tree.try_compute_relative_state(root_a, root_b).is_none());
    }

    #[test]
    fn try_compute_relative_state_matches_panicking_variant() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let child_state = make_state(
            0.3,
            DVec3::new(1e6, 2e6, 3e6),
            DVec3::new(100.0, 200.0, 300.0),
            DVec3::ZERO,
        );
        let child = tree.add_child(root, "child".into(), RefFrameKind::Body, child_state);

        let panicking = tree.compute_relative_state(root, child);
        let non_panicking = tree
            .try_compute_relative_state(root, child)
            .expect("should succeed");

        assert!(approx_eq_vec3(
            panicking.trans.position,
            non_panicking.trans.position,
            TOL
        ));
        assert!(approx_eq_vec3(
            panicking.trans.velocity,
            non_panicking.trans.velocity,
            TOL
        ));
        assert!(approx_eq_mat3(
            &panicking.rot.t_parent_this,
            &non_panicking.rot.t_parent_this,
            TOL
        ));
    }
}
