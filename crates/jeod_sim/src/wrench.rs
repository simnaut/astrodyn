//! Composite-rigid-body wrench aggregation: walks a [`MassStorage`]
//! tree leaves → root and accumulates each child's `(force, torque)`
//! into the root's totals via the parallel-axis arm
//! ([`shift_wrench_to_parent`]).
//!
//! This is the orchestration half of the composite-rigid-body wrench
//! pipeline. The pure math kernel lives in [`jeod_dynamics::wrench`];
//! this module composes it with the storage-agnostic mass-tree walk
//! pioneered by `recompute_composites_via_storage`.
//!
//! # JEOD precedent
//!
//! Mirrors `DynBody::collect_forces_and_torques` in
//! [`models/dynamics/dyn_body/src/dyn_body_collect.cc`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/dynamics/dyn_body/src/dyn_body_collect.cc):
//! every child node accumulates its own contributions, then transmits
//! the result to its parent in the parent's structural frame.
//! At the root, the final accumulator becomes the body's external
//! force / torque, which the integrator turns into translational /
//! rotational acceleration.
//!
//! # Frame discipline
//!
//! The kernel itself is frame-agnostic — every per-link math operation
//! happens in *the parent's* wrench frame. The orchestration here picks
//! a single canonical frame: each entity's **structural frame**. The
//! caller passes per-node `(force, torque)` already in that node's
//! structural frame; the kernel walks up shifting each child's
//! contribution into the next parent's structural frame; the final
//! per-root output is in the root's structural frame.
//!
//! Bevy / runner adapters that store force / torque in different
//! frames (e.g. inertial-frame force + body-frame torque, the existing
//! `TotalForceC` shape) must convert at the entry boundary
//! (multiply the per-entity `T_inertial_struct` and `T_struct_body^T`
//! to land in the entity's structural frame) and convert back at the
//! root with the inverse rotations. The Bevy adapter's
//! `wrench_aggregation_system` does exactly this.
//!
//! # Out of scope
//!
//! - **Composite-rigid-body integration gating** — only the root
//!   should integrate; children should be propagated kinematically.
//!   This module aggregates the wrenches; gating integration is the
//!   sister system the design-doc Section 15.3 calls
//!   `propagate_state_from_root_system`.
//! - **Frame-attached (kinematic-only) bodies** that ride a parent's
//!   kinematics without contributing mass — those use a separate
//!   relation (`ChildOf` rather than `MassChildOf`), so this walk
//!   correctly skips them.

use std::collections::HashMap;

use glam::DVec3;

use jeod_dynamics::mass_storage::{MassNodeOutputs, MassStorage};
use jeod_dynamics::wrench::{shift_wrench_to_parent, Wrench};

/// Per-edge geometry the wrench-aggregation walk needs at every
/// `child → parent` link.
///
/// Mirrors the JEOD `dyn_body_collect.cc` arithmetic literally:
///
/// - `pcm_to_ccm` is the offset from the **parent's** composite center
///   of mass to the **child's** composite CoM, expressed in the
///   parent's structural frame (matches JEOD line 181:
///   `Vector3::diff(mass.composite_wrt_pstr.position,
///   dyn_parent->mass.composite_properties.position, pcm_to_ccm)`).
/// - `t_parent_child` is the rotation matrix mirroring JEOD's
///   `MassPointState::T_parent_this` for the parent → child link
///   (so `t_parent_child^T · v_child = v_parent`).
///
/// Storage callers fill this in from whatever live shape the backend
/// keeps. The arena returns it from
/// `recompute_composites_via_storage` outputs; the Bevy adapter
/// computes it from `MassChildOf.offset` + the per-entity composite
/// CoM in `MassPropertiesC.position`. Either way, the kernel below
/// does the same math.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeGeometry {
    /// Vector from parent's composite CoM to child's composite CoM,
    /// in parent's structural frame (m).
    pub pcm_to_ccm: DVec3,
    /// Rotation from parent's structural frame to child's structural
    /// frame (matches JEOD `T_parent_this`).
    pub t_parent_child: glam::DMat3,
}

impl Default for EdgeGeometry {
    fn default() -> Self {
        Self {
            pcm_to_ccm: DVec3::ZERO,
            t_parent_child: glam::DMat3::IDENTITY,
        }
    }
}

/// Aggregate per-node wrenches up a [`MassStorage`] tree, returning the
/// root-relative composite wrench for every root.
///
/// `wrenches` is a per-node map from storage id to the node's own
/// `(force, torque)` contribution **in its own structural frame**
/// (torque about the node's composite center of mass, expressed in the
/// node's structural frame). Missing entries default to
/// [`Wrench::zero()`] — i.e., a node not present in the map contributes
/// nothing of its own.
///
/// `edges` is a per-child-edge map carrying the parallel-axis arm
/// `pcm_to_ccm` (parent-CoM → child-CoM, in parent struct) and the
/// `t_parent_child` rotation. Every non-root node in the storage must
/// have an entry; missing entries panic with a "Fail Loudly"
/// diagnostic that names the broken edge so the caller can catch
/// stale topology snapshots before they silently drop forces. The
/// arena and the Bevy adapter both build this map from their live
/// composite + structure-point layout — see
/// [`edge_geometry_from_composites`] for the canonical post-composite
/// helper, and [`EdgeGeometry::default`] for a sentinel that pins
/// the contract on default values.
///
/// Returns a `HashMap<root_id, Wrench>` carrying the aggregated wrench
/// at every root in `storage.roots()`. Non-root nodes' contributions
/// are folded into the corresponding root entry; the map does not
/// expose intermediate per-node accumulators.
///
/// # Algorithm
///
/// Walks every root post-order. For every leaf, the node's own wrench
/// is the leaf accumulator. For every internal node, the accumulator
/// is the node's own wrench plus each child accumulator shifted via
/// [`shift_wrench_to_parent`] using the child's [`EdgeGeometry`]
/// (parent-frame offset and parent → child rotation).
///
/// # Panics
///
/// Panics with a "Fail Loudly" diagnostic if the storage topology is
/// corrupt — a `MassChildOf` cycle, an unreachable subtree, or a
/// child node missing from `edges`.
// JEOD_INV: DB.16 — child forces propagated to parent recursively (via parallel-axis arm)
pub fn aggregate_wrenches_via_storage<S: MassStorage>(
    storage: &S,
    wrenches: &HashMap<S::Id, Wrench>,
    edges: &HashMap<S::Id, EdgeGeometry>,
) -> HashMap<S::Id, Wrench> {
    // Per-node accumulators in the node's *own* structural frame.
    // Filled in post-order: leaf accumulator = leaf own wrench;
    // internal-node accumulator = own + sum(shift(child_acc)).
    let expected = storage.node_count();
    let mut acc: HashMap<S::Id, Wrench> = HashMap::with_capacity(expected);
    let mut visited: std::collections::HashSet<S::Id> =
        std::collections::HashSet::with_capacity(expected);

    let roots = storage.roots();
    let mut out: HashMap<S::Id, Wrench> = HashMap::with_capacity(roots.len());
    for root in &roots {
        walk(storage, *root, wrenches, edges, &mut acc, &mut visited);
        let root_acc = acc.get(root).copied().unwrap_or_else(Wrench::zero);
        out.insert(*root, root_acc);
    }

    // Topology check: every node must have been visited by the
    // root-rooted post-order walk. Mirrors the sibling assert in
    // `recompute_composites_via_storage` and catches the same shape
    // of bug (cycles, orphaned subtrees) per CLAUDE.md "Fail Loudly".
    assert!(
        visited.len() == expected,
        "MassStorage topology has a cycle or orphan: {} of {} nodes unreachable from roots(). \
         Wrench aggregation skipped {} child wrenches; composite-rigid-body integration would \
         silently drop child forces. Check MassChildOf edges.",
        expected.saturating_sub(visited.len()),
        expected,
        expected.saturating_sub(visited.len()),
    );

    out
}

/// Convenience: build an [`EdgeGeometry`] map from the post-order
/// composite output of `recompute_composites_via_storage`.
///
/// Each non-root child gets `pcm_to_ccm = child.composite_wrt_pstr.position
/// − parent.composite.position` (matching JEOD
/// `dyn_body_collect.cc:181`) and `t_parent_child =
/// child.composite_wrt_pstr.t_parent_this`. Roots are absent from
/// the map (they have no parent).
///
/// Used by the arena / runner consumer, which has the composites map
/// to hand. The Bevy adapter computes its `EdgeGeometry` directly
/// from `MassChildOf` + live `MassPropertiesC` to avoid re-running
/// the composite kernel against its already-composed state — see the
/// `wrench_aggregation_system` glue in the `bevy_jeod` root crate.
pub fn edge_geometry_from_composites<S: MassStorage>(
    storage: &S,
    composites: &HashMap<S::Id, MassNodeOutputs>,
) -> HashMap<S::Id, EdgeGeometry> {
    let mut edges: HashMap<S::Id, EdgeGeometry> = HashMap::new();
    for root in storage.roots() {
        edge_walk(storage, root, composites, &mut edges);
    }
    edges
}

fn edge_walk<S: MassStorage>(
    storage: &S,
    id: S::Id,
    composites: &HashMap<S::Id, MassNodeOutputs>,
    out: &mut HashMap<S::Id, EdgeGeometry>,
) {
    let parent_composite = composites.get(&id).unwrap_or_else(|| {
        panic!(
            "edge_geometry_from_composites: parent node missing from composites \
             map. Run `recompute_composites_via_storage` first."
        )
    });
    for &child_id in storage.children(id) {
        let child_composite = composites.get(&child_id).unwrap_or_else(|| {
            panic!(
                "edge_geometry_from_composites: child node missing from composites \
                 map. Run `recompute_composites_via_storage` first."
            )
        });
        out.insert(
            child_id,
            EdgeGeometry {
                pcm_to_ccm: child_composite.composite_wrt_pstr.position
                    - parent_composite.composite.position,
                t_parent_child: child_composite.composite_wrt_pstr.t_parent_this,
            },
        );
        edge_walk(storage, child_id, composites, out);
    }
}

fn walk<S: MassStorage>(
    storage: &S,
    id: S::Id,
    wrenches: &HashMap<S::Id, Wrench>,
    edges: &HashMap<S::Id, EdgeGeometry>,
    acc: &mut HashMap<S::Id, Wrench>,
    visited: &mut std::collections::HashSet<S::Id>,
) {
    if !visited.insert(id) {
        return;
    }
    // 1. Recurse into children first (post-order: leaves accumulate
    //    before parents).
    for &child_id in storage.children(id) {
        walk(storage, child_id, wrenches, edges, acc, visited);
    }

    // 2. Start with this node's own wrench (in the node's structural
    //    frame, torque about the node's composite CoM).
    let mut node_acc = wrenches.get(&id).copied().unwrap_or_else(Wrench::zero);

    // 3. Shift each child accumulator into this node's structural
    //    frame and add, using the per-edge `EdgeGeometry`.
    for &child_id in storage.children(id) {
        let edge = edges.get(&child_id).unwrap_or_else(|| {
            panic!(
                "wrench aggregation: child edge missing from `edges` map. \
                 Every non-root node must have an EdgeGeometry entry — \
                 build it from your live mass-tree state \
                 (e.g. `edge_geometry_from_composites` on a fresh \
                 `recompute_composites_via_storage` output)."
            )
        });
        let child_acc = acc.get(&child_id).copied().unwrap_or_else(Wrench::zero);
        let (df, dtau) = shift_wrench_to_parent(
            child_acc.force,
            child_acc.torque,
            edge.pcm_to_ccm,
            edge.t_parent_child,
        );
        node_acc.force += df;
        node_acc.torque += dtau;
    }

    acc.insert(id, node_acc);
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DMat3, DVec3};
    use jeod_dynamics::mass::MassProperties;
    use jeod_dynamics::mass_body::MassTree;
    use jeod_dynamics::mass_storage::recompute_composites_via_storage;

    fn assert_close(a: DVec3, b: DVec3, tol: f64, label: &str) {
        let d = (a - b).length();
        assert!(d < tol, "{label}: |Δ|={d:.3e}, a={a:?}, b={b:?}");
    }

    fn composites_map<S: MassStorage>(storage: &S) -> HashMap<S::Id, MassNodeOutputs> {
        recompute_composites_via_storage(storage)
            .into_iter()
            .collect()
    }

    fn edges_for<S: MassStorage>(storage: &S) -> HashMap<S::Id, EdgeGeometry> {
        let comps = composites_map(storage);
        edge_geometry_from_composites(storage, &comps)
    }

    #[test]
    fn lone_root_passes_through() {
        let mut tree = MassTree::new();
        let r = tree.add_root("root".into(), MassProperties::new(10.0));
        let comps = edges_for(&tree);

        let mut w = HashMap::new();
        w.insert(
            r,
            Wrench::new(DVec3::new(1.0, 2.0, 3.0), DVec3::new(0.5, -1.0, 0.0)),
        );

        let out = aggregate_wrenches_via_storage(&tree, &w, &comps);
        let agg = out[&r];
        assert_eq!(agg.force, DVec3::new(1.0, 2.0, 3.0));
        assert_eq!(agg.torque, DVec3::new(0.5, -1.0, 0.0));
    }

    #[test]
    fn single_child_force_creates_root_torque() {
        // Parent at origin (mass 10, point at origin). Child at
        // structural offset [1, 0, 0] (mass 5, point at child origin).
        // Apply force [0, 1, 0] on child (in child structural frame).
        // Expected: root force = [0, 1, 0], root torque about root
        // composite CoM = pcm_to_ccm × F.
        //
        // Composite CoM of root: weighted avg of (parent core at 0)
        //   and (child composite at +x): cm = (10·0 + 5·1) / 15 = 1/3 along x.
        // pcm_to_ccm = child composite (in root struct = (1,0,0))
        //              minus root composite CoM (1/3,0,0)
        //            = (2/3, 0, 0).
        // tau = (2/3, 0, 0) × (0, 1, 0) = (0·0 - 0·1, 0·0 - 2/3·0, 2/3·1 - 0·0)
        //     = (0, 0, 2/3).
        let mut tree = MassTree::new();
        let p = tree.add_root("parent".into(), MassProperties::new(10.0));
        let c = tree.add_body("child".into(), MassProperties::new(5.0));
        tree.attach(c, p, DVec3::new(1.0, 0.0, 0.0), DMat3::IDENTITY);

        let comps = edges_for(&tree);

        let mut w = HashMap::new();
        w.insert(c, Wrench::new(DVec3::new(0.0, 1.0, 0.0), DVec3::ZERO));

        let out = aggregate_wrenches_via_storage(&tree, &w, &comps);
        let root_acc = out[&p];

        assert_close(root_acc.force, DVec3::new(0.0, 1.0, 0.0), 1e-15, "force");
        let two_thirds = 2.0 / 3.0;
        assert_close(
            root_acc.torque,
            DVec3::new(0.0, 0.0, two_thirds),
            1e-15,
            "parallel-axis torque",
        );
    }

    #[test]
    fn pure_child_torque_passes_through_with_no_cross_term() {
        // No force, only torque on child — at the root the torque
        // equals the (rotated) child torque, with no parallel-axis
        // contribution (since F = 0).
        let mut tree = MassTree::new();
        let p = tree.add_root("parent".into(), MassProperties::new(10.0));
        let c = tree.add_body("child".into(), MassProperties::new(5.0));
        tree.attach(c, p, DVec3::new(1.0, 0.0, 0.0), DMat3::IDENTITY);

        let comps = edges_for(&tree);

        let mut w = HashMap::new();
        let child_torque = DVec3::new(0.7, -0.3, 1.2);
        w.insert(c, Wrench::new(DVec3::ZERO, child_torque));

        let out = aggregate_wrenches_via_storage(&tree, &w, &comps);
        let root_acc = out[&p];

        assert_eq!(root_acc.force, DVec3::ZERO);
        // Identity rotation per-link, no cross-term, so the parent's
        // torque should equal the child's torque.
        assert_close(root_acc.torque, child_torque, 1e-15, "rotated child torque");
    }

    #[test]
    fn parent_plus_two_children_sum_correctly() {
        // Parent mass 10 with self-wrench (F_p, tau_p).
        // Two children, each mass 5, at offsets +y and -y.
        // Symmetric children → composite CoM at parent struct origin.
        //   total mass = 20, weighted_pos = 0 + 5·(0,1,0) + 5·(0,-1,0) = 0.
        //   composite CoM = (0,0,0).
        //   pcm_to_ccm for child A = (0,1,0)-0 = (0,1,0).
        //   pcm_to_ccm for child B = (0,-1,0)-0 = (0,-1,0).
        // Apply equal +x forces to both children:
        //   shifted force per child: (1,0,0); torques (0,1,0)×(1,0,0)=(0,0,-1)
        //                                          and (0,-1,0)×(1,0,0)=(0,0,+1).
        //   sum of child torques cancels.
        // Plus parent's own force / torque as identity contributions.
        let mut tree = MassTree::new();
        let p = tree.add_root("parent".into(), MassProperties::new(10.0));
        let a = tree.add_body("child_a".into(), MassProperties::new(5.0));
        let b = tree.add_body("child_b".into(), MassProperties::new(5.0));
        tree.attach(a, p, DVec3::new(0.0, 1.0, 0.0), DMat3::IDENTITY);
        tree.attach(b, p, DVec3::new(0.0, -1.0, 0.0), DMat3::IDENTITY);

        let comps = edges_for(&tree);

        let parent_force = DVec3::new(0.0, 0.0, 9.81);
        let parent_torque = DVec3::new(0.5, 0.0, 0.0);
        let child_force = DVec3::new(1.0, 0.0, 0.0);

        let mut w = HashMap::new();
        w.insert(p, Wrench::new(parent_force, parent_torque));
        w.insert(a, Wrench::new(child_force, DVec3::ZERO));
        w.insert(b, Wrench::new(child_force, DVec3::ZERO));

        let out = aggregate_wrenches_via_storage(&tree, &w, &comps);
        let root_acc = out[&p];

        let expected_force = parent_force + child_force + child_force;
        // Torque cancellation: r_a × F = (0,0,-1); r_b × F = (0,0,+1) → sum 0.
        let expected_torque = parent_torque;

        assert_close(root_acc.force, expected_force, 1e-15, "force sum");
        assert_close(
            root_acc.torque,
            expected_torque,
            1e-14,
            "torque cancellation",
        );
    }

    #[test]
    fn two_level_chain_propagates_through() {
        // Three-body chain A → B → C (each mass 1, offsets along +x by 1).
        // Apply a force on the deepest node C; check that it shows up
        // at A with the correct cumulative parallel-axis arm.
        //
        // Composites:
        //   B's composite CoM in B struct = (B core at 0 + C composite at (1,0,0)) / 2 = (0.5,0,0).
        //   C composite_wrt_pstr.position (in B struct) = (1,0,0).
        //   B composite_wrt_pstr.position (in A struct) = (1,0,0)+B's composite.position=(1+0.5,0,0)=(1.5,0,0).
        //   A's composite CoM in A struct = (A at 0 + B composite at (1.5,0,0) (mass 2)) / 3 = (1.0, 0, 0).
        //
        // Force F = (0, 1, 0) applied at C (in C struct = A struct = identity rotations).
        //   pcm_to_ccm at B level: C composite_wrt_pstr − B composite = (1,0,0) − (0.5,0,0) = (0.5,0,0).
        //   torque at B = (0.5,0,0) × (0,1,0) = (0,0,0.5).
        //   pcm_to_ccm at A level: B composite_wrt_pstr − A composite = (1.5,0,0) − (1,0,0) = (0.5,0,0).
        //   torque at A from B's accumulator (which is force=(0,1,0), torque=(0,0,0.5)):
        //        rotated torque = (0,0,0.5).
        //        new cross-term = (0.5,0,0) × (0,1,0) = (0,0,0.5).
        //        sum = (0,0,1.0).
        //   force at A = (0,1,0).
        let mut tree = MassTree::new();
        let a = tree.add_root("A".into(), MassProperties::new(1.0));
        let b = tree.add_body("B".into(), MassProperties::new(1.0));
        let c = tree.add_body("C".into(), MassProperties::new(1.0));
        tree.attach(b, a, DVec3::new(1.0, 0.0, 0.0), DMat3::IDENTITY);
        tree.attach(c, b, DVec3::new(1.0, 0.0, 0.0), DMat3::IDENTITY);

        let comps = edges_for(&tree);

        let mut w = HashMap::new();
        w.insert(c, Wrench::new(DVec3::new(0.0, 1.0, 0.0), DVec3::ZERO));

        let out = aggregate_wrenches_via_storage(&tree, &w, &comps);
        let root_acc = out[&a];

        assert_close(
            root_acc.force,
            DVec3::new(0.0, 1.0, 0.0),
            1e-14,
            "chain force",
        );
        assert_close(
            root_acc.torque,
            DVec3::new(0.0, 0.0, 1.0),
            1e-14,
            "chain torque",
        );
    }

    #[test]
    fn missing_wrench_treated_as_zero() {
        // An entity not present in the wrench map contributes zero —
        // so a one-child tree with an empty wrench map yields zero
        // root accumulator without panicking.
        let mut tree = MassTree::new();
        let p = tree.add_root("parent".into(), MassProperties::new(10.0));
        let c = tree.add_body("child".into(), MassProperties::new(5.0));
        tree.attach(c, p, DVec3::new(1.0, 0.0, 0.0), DMat3::IDENTITY);

        let comps = edges_for(&tree);
        let w = HashMap::new();

        let out = aggregate_wrenches_via_storage(&tree, &w, &comps);
        assert_eq!(out[&p], Wrench::zero());
    }
}
