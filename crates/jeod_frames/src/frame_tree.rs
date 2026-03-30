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

    /// Find the common ancestor of two frames.
    ///
    /// Walks parent pointers from both frames, collecting ancestors of `a`
    /// into a linear list, then walking from `b` until a match is found via
    /// a membership check on that list.
    ///
    /// Panics if the frames do not share a common root.
    pub fn find_common_ancestor(&self, a: FrameId, b: FrameId) -> FrameId {
        // Collect all ancestors of `a` (including `a` itself).
        let mut ancestors_a = Vec::new();
        let mut current = a;
        ancestors_a.push(current);
        while let Some(p) = self.parent[current] {
            ancestors_a.push(p);
            current = p;
        }

        // Walk from `b` upward until we find an ancestor of `a`.
        current = b;
        loop {
            if ancestors_a.contains(&current) {
                return current;
            }
            current = self.parent[current].expect("frames do not share a common ancestor");
        }
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

        println!("  4-level frame tree: all 4 frame pairs match direct composition to 1e-14");
    }
}
