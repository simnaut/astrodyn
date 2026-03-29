//! Arena-based mass tree: a faithful port of JEOD's MassBody hierarchy.
//!
//! JEOD models rigid-body mass as a tree of `MassBody` nodes. Each node has
//! *core* properties (the body alone) and *composite* properties (this body
//! plus all descendants). Composite center of mass and inertia are recomputed
//! bottom-up after every attach/detach using the algorithms from
//! `mass_calc_composite_cm.cc` and `mass_calc_composite_inertia.cc`.
//!
//! This module is pure Rust with zero Bevy dependency.

use glam::{DMat3, DVec3};

use crate::mass::MassProperties;

/// Handle into the [`MassTree`] arena.
pub type MassBodyId = usize;

// ---------------------------------------------------------------------------
// MassPointState
// ---------------------------------------------------------------------------

/// Position and orientation of a mass point relative to a parent frame.
///
/// Maps to JEOD's `MassPointState`: `position` is an offset in the parent
/// structural frame, and `t_parent_this` is the rotation matrix from the
/// parent structural frame to this body's frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MassPointState {
    /// Offset in parent's structural frame (m).
    pub position: DVec3,
    /// Rotation from parent structural frame to this body frame.
    pub t_parent_this: DMat3,
}

impl Default for MassPointState {
    fn default() -> Self {
        Self {
            position: DVec3::ZERO,
            t_parent_this: DMat3::IDENTITY,
        }
    }
}

// ---------------------------------------------------------------------------
// MassBody
// ---------------------------------------------------------------------------

/// A single node in the mass tree.
///
/// Mirrors JEOD's `MassBody` data members. `core_properties` are this body
/// alone; `composite_properties` include all descendants.
#[derive(Debug, Clone)]
pub struct MassBody {
    /// Human-readable name.
    pub name: String,
    /// Mass properties of this body alone.
    pub core_properties: MassProperties,
    /// Mass properties of this body plus all attached children.
    pub composite_properties: MassProperties,
    /// Attachment point: position and orientation of this body's structural
    /// frame origin in the **parent's** structural frame.
    pub structure_point: MassPointState,
    /// Core CoM offset from composite CoM (in structural frame coords).
    pub core_wrt_composite: MassPointState,
    /// Composite CoM position in the **parent's** structural frame.
    pub composite_wrt_pstr: MassPointState,
}

// ---------------------------------------------------------------------------
// MassTree
// ---------------------------------------------------------------------------

/// Arena-based mass tree. Portable, no ECS dependency.
///
/// Bodies are stored in a flat `Vec`; parent/child relationships are tracked
/// with parallel vectors of `Option<MassBodyId>` and `Vec<MassBodyId>`.
pub struct MassTree {
    nodes: Vec<MassBody>,
    parent: Vec<Option<MassBodyId>>,
    children: Vec<Vec<MassBodyId>>,
}

impl MassTree {
    /// Create an empty tree.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            parent: Vec::new(),
            children: Vec::new(),
        }
    }

    // -- accessors ----------------------------------------------------------

    /// Borrow a body by id.
    pub fn get(&self, id: MassBodyId) -> &MassBody {
        &self.nodes[id]
    }

    /// Mutably borrow a body by id.
    pub fn get_mut(&mut self, id: MassBodyId) -> &mut MassBody {
        &mut self.nodes[id]
    }

    /// Parent of the given body, or `None` for a root.
    pub fn parent(&self, id: MassBodyId) -> Option<MassBodyId> {
        self.parent[id]
    }

    /// Direct children of the given body.
    pub fn children(&self, id: MassBodyId) -> &[MassBodyId] {
        &self.children[id]
    }

    // -- construction -------------------------------------------------------

    /// Add a disconnected body (no parent, no children).
    ///
    /// Composite properties are initialised to match core properties.
    pub fn add_body(&mut self, name: String, core: MassProperties) -> MassBodyId {
        let id = self.nodes.len();
        let body = MassBody {
            name,
            composite_properties: core,
            core_properties: core,
            structure_point: MassPointState::default(),
            core_wrt_composite: MassPointState::default(),
            composite_wrt_pstr: MassPointState::default(),
        };
        self.nodes.push(body);
        self.parent.push(None);
        self.children.push(Vec::new());
        id
    }

    /// Add a root body (convenience wrapper around [`add_body`](Self::add_body)).
    pub fn add_root(&mut self, name: String, core: MassProperties) -> MassBodyId {
        self.add_body(name, core)
    }

    // -- attach / detach ----------------------------------------------------

    /// Attach `child_id` to `parent_id` at the given offset and rotation.
    ///
    /// `offset` is the child structural origin in the parent's structural
    /// frame. `t_parent_child` rotates from the parent structural frame to
    /// the child's structural frame.
    ///
    /// Panics if the child already has a parent.
    pub fn attach(
        &mut self,
        child_id: MassBodyId,
        parent_id: MassBodyId,
        offset: DVec3,
        t_parent_child: DMat3,
    ) {
        assert!(
            self.parent[child_id].is_none(),
            "child {} already attached to a parent",
            child_id
        );
        assert_ne!(child_id, parent_id, "cannot attach a body to itself");

        // JEOD_INV: MA.08 — no cycle in mass tree (arena-based, cycles impossible)
        // JEOD_INV: MA.19 — no same-tree attachment (cycle prevention)
        // Prevent creation of cycles: walk up from parent_id to the root
        // and ensure we never encounter child_id. This matches JEOD's
        // attach_validate_parent() (mass_attach.cc:370-388): "the only invalid
        // attachment is one that would make a cyclic graph."
        {
            let mut current = Some(parent_id);
            while let Some(pid) = current {
                assert_ne!(
                    pid, child_id,
                    "cannot attach body {} under its own descendant {} (would create cycle)",
                    child_id, parent_id
                );
                current = self.parent[pid];
            }
        }

        self.parent[child_id] = Some(parent_id);
        self.children[parent_id].push(child_id);

        self.nodes[child_id].structure_point = MassPointState {
            position: offset,
            t_parent_this: t_parent_child,
        };

        self.recompute_composites();
    }

    /// Detach `child_id` from its parent.
    ///
    /// The former parent's composite properties are recomputed. The child's
    /// parent-relative fields (`structure_point`, `composite_wrt_pstr`) are
    /// reset to defaults, matching JEOD's `detach_update_properties()`
    /// (mass_detach.cc:322-324) which calls `initialize_mass_point()` on
    /// all three parent-relative mass points.
    ///
    /// Panics if the child has no parent.
    pub fn detach(&mut self, child_id: MassBodyId) {
        let parent_id = self.parent[child_id]
            .expect("detach called on a body with no parent");

        self.children[parent_id].retain(|&c| c != child_id);
        self.parent[child_id] = None;

        // Reset parent-relative fields on the detached child (JEOD mass_detach.cc:322-324).
        self.nodes[child_id].structure_point = MassPointState::default();
        self.nodes[child_id].composite_wrt_pstr = MassPointState::default();

        // JEOD_INV: MA.15 — detach recomputes inverse inertia for new root
        // Recompute inverse inertia on detached child (JEOD mass_detach.cc:328-335).
        let child = &mut self.nodes[child_id];
        if child.composite_properties.mass > 0.0 {
            let det = child.composite_properties.inertia.determinant();
            assert!(
                det.abs() > 1e-30,
                "Detached child '{}' has singular composite inertia (det={det:.2e})",
                child.name
            );
            child.composite_properties.inverse_inertia =
                child.composite_properties.inertia.inverse();
        } else {
            child.composite_properties.inverse_inertia = DMat3::ZERO;
        }

        // Recompute composites for the tree the parent still belongs to.
        self.recompute_composites();
    }

    // -- composite recomputation (JEOD algorithm) ---------------------------

    /// Recompute composite properties for the entire tree bottom-up.
    ///
    /// JEOD requires computing from leaves to root so that each parent sees
    /// up-to-date child composites. We collect a post-order traversal and
    /// process each node.
    // JEOD_INV: MA.06 — bottom-up mass property update (children first)
    // JEOD_INV: MA.07 — needs_update flag cleared after recomputation (always recomputes)
    pub fn recompute_composites(&mut self) {
        let order = self.post_order();
        for id in order {
            self.update_node(id);
        }
    }

    /// Compute a post-order (leaves-first) traversal of the entire forest.
    fn post_order(&self) -> Vec<MassBodyId> {
        let mut result = Vec::with_capacity(self.nodes.len());
        let mut visited = vec![false; self.nodes.len()];

        for root in 0..self.nodes.len() {
            if self.parent[root].is_none() && !visited[root] {
                self.post_order_walk(root, &mut visited, &mut result);
            }
        }
        result
    }

    fn post_order_walk(
        &self,
        id: MassBodyId,
        visited: &mut [bool],
        result: &mut Vec<MassBodyId>,
    ) {
        if visited[id] {
            return;
        }
        for &child in &self.children[id] {
            self.post_order_walk(child, visited, result);
        }
        visited[id] = true;
        result.push(id);
    }

    /// Update a single node's composite properties.
    ///
    /// Mirrors JEOD `MassBody::update_mass_properties` for one node.
    fn update_node(&mut self, id: MassBodyId) {
        if self.children[id].is_empty() {
            // Atomic body: composite == core (JEOD mass_update.cc lines 59-75).
            let node = &mut self.nodes[id];
            node.composite_properties = node.core_properties;
            node.core_wrt_composite = MassPointState::default();
        } else {
            // First compute composite_wrt_pstr for each child.
            // JEOD mass_update.cc lines 137-143:
            //   composite_wrt_pstr.position =
            //       T_parent_this^T * composite_properties.position
            //       + structure_point.position
            // This transforms the child's composite CoM from child struct frame
            // to parent struct frame.
            let child_ids: Vec<MassBodyId> = self.children[id].clone();
            for &cid in &child_ids {
                let child = &self.nodes[cid];
                let t = child.structure_point.t_parent_this;
                let comp_pos = child.composite_properties.position;
                let struct_pos = child.structure_point.position;
                // T^T * comp_pos + struct_pos
                let pos_in_parent = t.transpose() * comp_pos + struct_pos;

                self.nodes[cid].composite_wrt_pstr.position = pos_in_parent;
                self.nodes[cid].composite_wrt_pstr.t_parent_this =
                    self.nodes[cid].structure_point.t_parent_this;
            }

            // Composite center of mass (JEOD mass_calc_composite_cm.cc).
            self.calc_composite_cm(id);

            // Compute core_wrt_composite (JEOD mass_update.cc lines 104-107):
            // core_wrt_composite.position = core.position - composite.position
            // (both in structural frame, so no rotation needed)
            let core_pos = self.nodes[id].core_properties.position;
            let comp_pos = self.nodes[id].composite_properties.position;
            self.nodes[id].core_wrt_composite.position = core_pos - comp_pos;

            // Composite inertia (JEOD mass_calc_composite_inertia.cc).
            self.calc_composite_inertia(id);
        }

        // For root bodies, compute inverse inertia (JEOD mass_update.cc lines 116-125).
        // JEOD's Matrix3x3::invert_symmetric (dm_invert_symm.cc:86-94) checks
        // for singular matrices via fpclassify(determinant) == FP_ZERO.
        if self.parent[id].is_none() {
            let node = &mut self.nodes[id];
            if node.composite_properties.mass > 0.0 {
                let det = node.composite_properties.inertia.determinant();
                assert!(
                    det.abs() > 1e-30,
                    "Root body '{}' has singular composite inertia (det={det:.2e})",
                    node.name
                );
                node.composite_properties.inverse_inertia =
                    node.composite_properties.inertia.inverse();
            } else {
                node.composite_properties.inverse_inertia = DMat3::ZERO;
            }
        }
    }

    /// Composite center of mass — port of JEOD `mass_calc_composite_cm.cc`.
    ///
    /// Accumulates `mass * position` over the core and all children, where
    /// child positions are their `composite_wrt_pstr.position` (composite
    /// CoM in this body's structural frame).
    fn calc_composite_cm(&mut self, id: MassBodyId) {
        let core = &self.nodes[id].core_properties;
        let mut total_mass = core.mass;
        let mut weighted_pos = core.position * core.mass;

        for &cid in &self.children[id] {
            let child = &self.nodes[cid];
            total_mass += child.composite_properties.mass;
            weighted_pos +=
                child.composite_wrt_pstr.position * child.composite_properties.mass;
        }

        let node = &mut self.nodes[id];
        node.composite_properties.mass = total_mass;
        // JEOD mass_calc_composite_cm.cc:72 / mass_update.cc:64
        node.composite_properties.inverse_mass = if total_mass > 0.0 {
            1.0 / total_mass
        } else {
            0.0
        };
        if total_mass > 0.0 {
            node.composite_properties.position = weighted_pos / total_mass;
        } else {
            node.composite_properties.position = DVec3::ZERO;
        }
    }

    /// Composite inertia — port of JEOD `mass_calc_composite_inertia.cc`.
    ///
    /// Starts with the core body's inertia shifted to the composite CoM via
    /// the parallel axis theorem, then adds each child's composite inertia
    /// (rotated to this body's structural frame) plus the child's parallel
    /// axis contribution.
    fn calc_composite_inertia(&mut self, id: MassBodyId) {
        let cm = self.nodes[id].composite_properties.position;

        // Core contribution: inertia + point-mass shift from core CoM to
        // composite CoM (JEOD mass_calc_composite_inertia.cc lines 61-64).
        let core = &self.nodes[id].core_properties;
        let core_offset = core.position - cm;
        let mut composite_inertia =
            core.inertia + point_mass_inertia(core.mass, core_offset);

        // Child contributions (lines 67-84).
        for &cid in &self.children[id] {
            let child = &self.nodes[cid];
            let child_offset = child.composite_wrt_pstr.position - cm;

            // Rotate child's composite inertia from child struct frame to
            // parent struct frame: T^T * I_child * T
            // This is JEOD's transpose_transform_matrix.
            let t = child.structure_point.t_parent_this;
            let rotated_inertia =
                t.transpose() * child.composite_properties.inertia * t;

            composite_inertia +=
                rotated_inertia + point_mass_inertia(child.composite_properties.mass, child_offset);
        }

        self.nodes[id].composite_properties.inertia = composite_inertia;
        // JEOD's Matrix3x3::invert_symmetric (dm_invert_symm.cc:86-94) checks
        // for singular matrices via fpclassify(determinant) == FP_ZERO.
        if self.nodes[id].composite_properties.mass > 0.0 {
            let det = composite_inertia.determinant();
            assert!(
                det.abs() > 1e-30,
                "Body '{}' has singular composite inertia (det={det:.2e})",
                self.nodes[id].name
            );
            self.nodes[id].composite_properties.inverse_inertia =
                composite_inertia.inverse();
        } else {
            self.nodes[id].composite_properties.inverse_inertia = DMat3::ZERO;
        }
    }
}

impl Default for MassTree {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Parallel axis theorem (Steiner's theorem): inertia of a point mass at
/// offset `r` from the reference point.
///
/// Port of JEOD `MassBody::compute_point_mass_inertia` from
/// `mass_point_mass_inertia.cc`:
///
/// ```text
/// I[i][j] = mass * (r^2 * delta_ij - r[i] * r[j])
/// ```
pub fn point_mass_inertia(mass: f64, offset: DVec3) -> DMat3 {
    let r_sq = offset.length_squared();
    // mass * (r^2 * I - outer(offset, offset))
    let outer = DMat3::from_cols(
        offset * offset.x,
        offset * offset.y,
        offset * offset.z,
    );
    DMat3::from_diagonal(DVec3::splat(r_sq)) * mass - outer * mass
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert two DMat3 are approximately equal.
    fn assert_mat3_close(a: DMat3, b: DMat3, tol: f64, msg: &str) {
        let diff = a - b;
        for col in [diff.x_axis, diff.y_axis, diff.z_axis] {
            assert!(
                col.length() < tol,
                "{msg}: column diff {col:?} exceeds tolerance {tol}"
            );
        }
    }

    /// Helper: assert two DVec3 are approximately equal.
    fn assert_vec3_close(a: DVec3, b: DVec3, tol: f64, msg: &str) {
        let diff = (a - b).length();
        assert!(diff < tol, "{msg}: diff {diff} exceeds tolerance {tol}");
    }

    /// Helper: check that a matrix is symmetric.
    fn assert_symmetric(m: DMat3, tol: f64, msg: &str) {
        let mt = m.transpose();
        assert_mat3_close(m, mt, tol, msg);
    }

    // -----------------------------------------------------------------------
    // 1. Single body: composite == core
    // -----------------------------------------------------------------------

    #[test]
    fn single_body_composite_equals_core() {
        let mut tree = MassTree::new();
        let core = MassProperties::new(42.0);
        let id = tree.add_root("root".into(), core);

        let body = tree.get(id);
        assert_eq!(body.composite_properties.mass, body.core_properties.mass);
        assert_eq!(
            body.composite_properties.inertia,
            body.core_properties.inertia
        );
        assert_eq!(
            body.composite_properties.position,
            body.core_properties.position
        );
    }

    // -----------------------------------------------------------------------
    // 2. Two point masses at known offset
    // -----------------------------------------------------------------------

    #[test]
    fn two_point_masses_composite() {
        let mut tree = MassTree::new();

        // Parent: 10 kg at origin, spherical inertia
        let parent_core = MassProperties::new(10.0);
        let pid = tree.add_root("parent".into(), parent_core);

        // Child: 5 kg at origin of its own structural frame, spherical inertia
        let child_core = MassProperties::new(5.0);
        let cid = tree.add_body("child".into(), child_core);

        // Attach child at offset [3, 0, 0] in parent's struct frame, identity rotation
        tree.attach(cid, pid, DVec3::new(3.0, 0.0, 0.0), DMat3::IDENTITY);

        let parent = tree.get(pid);

        // Composite mass = 15 kg
        assert!(
            (parent.composite_properties.mass - 15.0).abs() < 1e-12,
            "composite mass = {}",
            parent.composite_properties.mass
        );

        // Composite CoM: weighted average = (10*0 + 5*3) / 15 = 1.0 m along x
        let expected_cm = DVec3::new(1.0, 0.0, 0.0);
        assert_vec3_close(
            parent.composite_properties.position,
            expected_cm,
            1e-12,
            "composite CoM",
        );

        // Verify composite inertia via manual parallel axis calculation:
        //   Parent core at [0,0,0], composite CoM at [1,0,0]
        //   Parent offset from composite CoM = [-1, 0, 0]
        //   Child composite CoM in parent struct frame = [3, 0, 0]
        //   Child offset from composite CoM = [2, 0, 0]
        //
        //   I_parent_core = 10 * I_3x3 (spherical)
        //   I_parent_shift = point_mass(10, [-1,0,0]) = 10 * diag(0, 1, 1)
        //
        //   I_child_core = 5 * I_3x3 (spherical, identity rotation so no change)
        //   I_child_shift = point_mass(5, [2,0,0]) = 5 * diag(0, 4, 4) = diag(0, 20, 20)
        //
        //   Total = (10*I + diag(0,10,10)) + (5*I + diag(0,20,20))
        //         = diag(10,10,10) + diag(0,10,10) + diag(5,5,5) + diag(0,20,20)
        //         = diag(15, 45, 45)
        let expected_inertia = DMat3::from_diagonal(DVec3::new(15.0, 45.0, 45.0));
        assert_mat3_close(
            parent.composite_properties.inertia,
            expected_inertia,
            1e-10,
            "composite inertia",
        );
    }

    // -----------------------------------------------------------------------
    // 3. Parallel axis theorem — known values
    // -----------------------------------------------------------------------

    #[test]
    fn parallel_axis_theorem_known_values() {
        let inertia = point_mass_inertia(5.0, DVec3::new(3.0, 0.0, 0.0));
        // I_xx = m * (y^2 + z^2) = 5 * 0 = 0
        // I_yy = m * (x^2 + z^2) = 5 * 9 = 45
        // I_zz = m * (x^2 + y^2) = 5 * 9 = 45
        // Off-diagonals = -m * r_i * r_j = 0 (y = z = 0)
        let expected = DMat3::from_cols(
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(0.0, 45.0, 0.0),
            DVec3::new(0.0, 0.0, 45.0),
        );
        assert_mat3_close(inertia, expected, 1e-12, "point mass inertia at [3,0,0]");
    }

    #[test]
    fn parallel_axis_theorem_off_diagonal() {
        // offset along all three axes to exercise products of inertia
        let r = DVec3::new(1.0, 2.0, 3.0);
        let m = 4.0;
        let inertia = point_mass_inertia(m, r);
        let r_sq = r.length_squared(); // 14

        // Diagonal: m * (r^2 - r_i^2)
        assert!((inertia.x_axis.x - m * (r_sq - r.x * r.x)).abs() < 1e-12);
        assert!((inertia.y_axis.y - m * (r_sq - r.y * r.y)).abs() < 1e-12);
        assert!((inertia.z_axis.z - m * (r_sq - r.z * r.z)).abs() < 1e-12);

        // Off-diagonal: -m * r_i * r_j
        assert!((inertia.y_axis.x - (-m * r.x * r.y)).abs() < 1e-12);
        assert!((inertia.z_axis.x - (-m * r.x * r.z)).abs() < 1e-12);
        assert!((inertia.z_axis.y - (-m * r.y * r.z)).abs() < 1e-12);

        assert_symmetric(inertia, 1e-12, "point mass inertia symmetry");
    }

    // -----------------------------------------------------------------------
    // 4. Attach-detach round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn attach_detach_round_trip() {
        let mut tree = MassTree::new();

        let parent_core = MassProperties::new(10.0);
        let pid = tree.add_root("parent".into(), parent_core);

        // Snapshot of parent composite before attaching child.
        let orig_mass = tree.get(pid).composite_properties.mass;
        let orig_inertia = tree.get(pid).composite_properties.inertia;
        let orig_position = tree.get(pid).composite_properties.position;

        let child_core = MassProperties::new(5.0);
        let cid = tree.add_body("child".into(), child_core);

        // Attach: composite must change.
        tree.attach(cid, pid, DVec3::new(2.0, 0.0, 0.0), DMat3::IDENTITY);
        assert!(
            (tree.get(pid).composite_properties.mass - 15.0).abs() < 1e-12,
            "composite mass after attach"
        );

        // Detach: parent composite should revert to core.
        tree.detach(cid);
        assert!(
            (tree.get(pid).composite_properties.mass - orig_mass).abs() < 1e-12,
            "mass restored after detach"
        );
        assert_mat3_close(
            tree.get(pid).composite_properties.inertia,
            orig_inertia,
            1e-12,
            "inertia restored after detach",
        );
        assert_vec3_close(
            tree.get(pid).composite_properties.position,
            orig_position,
            1e-12,
            "position restored after detach",
        );
    }

    // -----------------------------------------------------------------------
    // 5. Three-body chain A -> B -> C
    // -----------------------------------------------------------------------

    #[test]
    fn three_body_chain() {
        let mut tree = MassTree::new();

        let ma = MassProperties::new(10.0);
        let mb = MassProperties::new(5.0);
        let mc = MassProperties::new(3.0);

        let a = tree.add_root("A".into(), ma);
        let b = tree.add_body("B".into(), mb);
        let c = tree.add_body("C".into(), mc);

        // B attached at [2, 0, 0] on A, C attached at [1, 0, 0] on B
        tree.attach(b, a, DVec3::new(2.0, 0.0, 0.0), DMat3::IDENTITY);
        tree.attach(c, b, DVec3::new(1.0, 0.0, 0.0), DMat3::IDENTITY);

        // A's composite mass = 10 + 5 + 3 = 18
        assert!(
            (tree.get(a).composite_properties.mass - 18.0).abs() < 1e-12,
            "A composite mass = {}",
            tree.get(a).composite_properties.mass
        );

        // B's composite mass = 5 + 3 = 8
        assert!(
            (tree.get(b).composite_properties.mass - 8.0).abs() < 1e-12,
            "B composite mass = {}",
            tree.get(b).composite_properties.mass
        );

        // C's composite mass = 3
        assert!(
            (tree.get(c).composite_properties.mass - 3.0).abs() < 1e-12,
            "C composite mass = {}",
            tree.get(c).composite_properties.mass
        );

        // A's composite CoM:
        //   A core at [0,0,0] mass 10
        //   B's composite CoM in B struct frame:
        //     B core at [0,0,0] mass 5, C at [1,0,0] mass 3 => (5*0+3*1)/8 = 0.375
        //   B's composite CoM in A struct frame: [2+0.375, 0, 0] = [2.375, 0, 0]
        //   A composite CoM: (10*0 + 8*2.375) / 18 = 19/18 ≈ 1.0556
        let expected_a_cm_x = (10.0 * 0.0 + 8.0 * 2.375) / 18.0;
        assert_vec3_close(
            tree.get(a).composite_properties.position,
            DVec3::new(expected_a_cm_x, 0.0, 0.0),
            1e-10,
            "A composite CoM",
        );

        // Detach B from A: A should revert to its own core
        tree.detach(b);
        assert!(
            (tree.get(a).composite_properties.mass - 10.0).abs() < 1e-12,
            "A mass after detach"
        );
        assert_vec3_close(
            tree.get(a).composite_properties.position,
            DVec3::ZERO,
            1e-12,
            "A position after detach",
        );

        // B should still have its composite (B+C)
        assert!(
            (tree.get(b).composite_properties.mass - 8.0).abs() < 1e-12,
            "B mass unchanged after detach from A"
        );
    }

    // -----------------------------------------------------------------------
    // 6. Non-identity rotation
    // -----------------------------------------------------------------------

    #[test]
    fn non_identity_rotation() {
        // Child attached with 90-degree rotation about Z axis.
        // T_parent_child rotates parent X to child Y, parent Y to child -X.
        //
        // If the child has inertia diag(1, 4, 9) in its own frame, then in
        // the parent frame the rotated inertia should be diag(4, 1, 9).
        //
        // T = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  (90 deg about Z)
        // T^T * diag(1,4,9) * T = diag(4,1,9)

        let mut tree = MassTree::new();

        // Parent: 10 kg, spherical inertia
        let parent_core = MassProperties::new(10.0);
        let pid = tree.add_root("parent".into(), parent_core);

        // Child: 5 kg, non-spherical inertia
        let child_inertia = DMat3::from_diagonal(DVec3::new(1.0, 4.0, 9.0));
        let child_core = MassProperties::with_inertia(5.0, child_inertia, DVec3::ZERO);
        let cid = tree.add_body("child".into(), child_core);

        // 90 degrees about Z: T_parent_child
        let t = DMat3::from_cols(
            DVec3::new(0.0, 1.0, 0.0),   // parent X -> child: [0, 1, 0]
            DVec3::new(-1.0, 0.0, 0.0),  // parent Y -> child: [-1, 0, 0]
            DVec3::new(0.0, 0.0, 1.0),   // parent Z -> child: [0, 0, 1]
        );

        // Attach child at [3, 0, 0] with 90-deg rotation
        tree.attach(cid, pid, DVec3::new(3.0, 0.0, 0.0), t);

        let parent = tree.get(pid);

        // Composite mass = 15
        assert!(
            (parent.composite_properties.mass - 15.0).abs() < 1e-12,
            "composite mass"
        );

        // Composite CoM at [1, 0, 0] (same as test 2 — child CoM is at
        // child origin so rotation doesn't move it)
        assert_vec3_close(
            parent.composite_properties.position,
            DVec3::new(1.0, 0.0, 0.0),
            1e-12,
            "composite CoM with rotation",
        );

        // Expected composite inertia:
        //   Parent core offset from composite CoM = [-1, 0, 0]
        //   Parent core inertia = 10*I + point_mass(10, [-1,0,0])
        //     = diag(10,10,10) + diag(0,10,10) = diag(10, 20, 20)
        //
        //   Child rotated inertia: T^T * diag(1,4,9) * T = diag(4, 1, 9)
        //   Child offset from composite CoM = [2, 0, 0]
        //   Child contribution = diag(4,1,9) + point_mass(5, [2,0,0])
        //     = diag(4,1,9) + diag(0,20,20) = diag(4, 21, 29)
        //
        //   Total = diag(10,20,20) + diag(4,21,29) = diag(14, 41, 49)
        let expected_inertia = DMat3::from_diagonal(DVec3::new(14.0, 41.0, 49.0));
        assert_mat3_close(
            parent.composite_properties.inertia,
            expected_inertia,
            1e-10,
            "composite inertia with rotation",
        );
    }

    // -----------------------------------------------------------------------
    // 7. Composite inertia symmetry
    // -----------------------------------------------------------------------

    #[test]
    fn composite_inertia_symmetry() {
        let mut tree = MassTree::new();

        // Use asymmetric offsets and rotations to stress symmetry
        let parent_core = MassProperties::with_inertia(
            10.0,
            DMat3::from_diagonal(DVec3::new(100.0, 200.0, 300.0)),
            DVec3::new(0.1, -0.2, 0.3),
        );
        let pid = tree.add_root("parent".into(), parent_core);

        let child_core = MassProperties::with_inertia(
            5.0,
            DMat3::from_diagonal(DVec3::new(10.0, 20.0, 30.0)),
            DVec3::new(-0.05, 0.1, 0.0),
        );
        let cid = tree.add_body("child".into(), child_core);

        // 45-degree rotation about Y
        let angle = std::f64::consts::FRAC_PI_4;
        let c = angle.cos();
        let s = angle.sin();
        let t = DMat3::from_cols(
            DVec3::new(c, 0.0, -s),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(s, 0.0, c),
        );

        tree.attach(cid, pid, DVec3::new(1.0, 2.0, -0.5), t);

        let inertia = tree.get(pid).composite_properties.inertia;
        assert_symmetric(inertia, 1e-10, "composite inertia after asymmetric attach");

        // Verify inverse is also consistent
        let product = inertia * tree.get(pid).composite_properties.inverse_inertia;
        assert_mat3_close(
            product,
            DMat3::IDENTITY,
            1e-8,
            "I * I^-1 = identity",
        );
    }
}
