//! Kinematic state propagation: walks a [`MassStorage`] tree pre-order
//! from each root and derives every kinematic child's instantaneous
//! composite-body state from its parent's state composed with the
//! per-link attach geometry.
//!
//! This is the orchestration half of the `propagate_state_from_root`
//! pipeline ([Frame-Tree-ECS-Native § 15.3](https://github.com/simnaut/bevy_jeod/wiki/Frame-Tree-ECS-Native#15-articulated-bodies-extension-beyond-268s-core-scope)).
//! The pure-math kernel lives in
//! [`jeod_dynamics::kinematic_propagation`]; this module composes it
//! with the storage-agnostic mass-tree walk pioneered by
//! [`recompute_composites_via_storage`](jeod_dynamics::mass_storage::recompute_composites_via_storage)
//! and reused by [`crate::wrench::aggregate_wrenches_via_storage`].
//!
//! # JEOD precedent
//!
//! Mirrors `DynBody::propagate_state_from_structure` in
//! [`models/dynamics/dyn_body/src/dyn_body_propagate_state.cc`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/dynamics/dyn_body/src/dyn_body_propagate_state.cc):
//! starting from the root body's state, recursively derive the
//! structure-frame state for every child, then promote each child's
//! structure state into its composite-body state. The kernel folds the
//! three legs (parent body → parent struct → child struct → child
//! body) into one direct shift; this module only owns the topology
//! walk.
//!
//! # Frame discipline
//!
//! The kernel itself takes raw `glam::DVec3` / `DMat3` inputs in the
//! frame conventions documented on
//! [`jeod_dynamics::kinematic_propagation::KinematicChildInputs`].
//! Storage-side state shapes (`RotationalState`, `TranslationalState`,
//! plus the per-edge `MassPointState`) compose into those inputs at
//! the boundary; outputs are written back in the same shapes. Bevy
//! adapters that store typed `RotationalStateC` /
//! `TranslationalStateC` use this module's
//! [`KinematicNodeState`] as their boundary type and lift / lower
//! typed phantoms at their query layer.
//!
//! # Out of scope
//!
//! - **Joint kinematics drivers** that prescribe relative motion at a
//!   `MassChildOf` link (issue #275, PR #289). When a joint driver
//!   has set a rotation on a *frame* entity associated with a chain
//!   member, that system writes through a parallel surface
//!   (frame-tree `FrameRotC` / `FrameTransC`); this body-side walk
//!   stays passive-rigid.
//! - **Detached subtrees** (`DetachedSubtreeState`) get their own
//!   ballistic propagation in `step_detached_system` — a kinematic
//!   chain that has been torn loose from a root no longer has a
//!   parent in `MassChildOf`, so it falls out of this walk by
//!   construction.

use std::collections::HashMap;

use jeod_dynamics::kinematic_propagation::{compute_kinematic_child_state, KinematicChildInputs};
use jeod_dynamics::mass_storage::MassStorage;
use jeod_dynamics::rotational::RotationalState;
use jeod_dynamics::state::TranslationalState;
use jeod_math::JeodQuat;

use glam::{DMat3, DVec3};

/// Per-node kinematic state read by — and written back from — the
/// propagation walk.
///
/// Combines the per-node fields the kernel needs (rotational state,
/// translational state, struct→body rotation, composite CoM in this
/// node's structural frame). Storage callers fill this in from
/// whatever live shape their backend keeps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinematicNodeState {
    /// Inertial → body rotational state (attitude quaternion +
    /// angular velocity in the node's body frame).
    pub rot: RotationalState,
    /// Inertial-frame translational state (composite-body position +
    /// velocity).
    pub trans: TranslationalState,
    /// Structural → body rotation matrix. Identity when the node has
    /// no per-vehicle structural transform (single-body vehicles, the
    /// default).
    pub t_struct_body: DMat3,
    /// Composite center of mass in **this node's** structural frame.
    /// Mirrors `MassProperties.position` after composite recomputation
    /// — i.e. `mass.composite_properties.position` in JEOD parlance.
    pub composite_in_struct: DVec3,
}

impl Default for KinematicNodeState {
    fn default() -> Self {
        Self {
            rot: RotationalState::default(),
            trans: TranslationalState::default(),
            t_struct_body: DMat3::IDENTITY,
            composite_in_struct: DVec3::ZERO,
        }
    }
}

/// Per-edge geometry the propagation walk needs at every
/// `child → parent` link in a [`MassStorage`].
///
/// Mirrors the JEOD mass tree's `MassPointState` shape but kept here
/// as a dedicated type because the propagation walk reads only the
/// *forward* direction (parent struct → child struct), whereas the
/// wrench walk's [`crate::wrench::EdgeGeometry`] is the same per-edge
/// rotation paired with a different parent-CoM-relative offset.
/// Sharing one type would conflate two separate edge contracts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinematicEdge {
    /// Rotation `T_parent_child` from the parent's structural frame
    /// into the child's structural frame
    /// (`v_child_struct = T_parent_child · v_parent_struct`).
    pub t_parent_child: DMat3,
    /// Child's structural origin in the **parent's** structural
    /// frame (m). Matches `MassChildOf.offset` /
    /// `MassPointState.position`.
    pub offset_in_pstr: DVec3,
}

impl Default for KinematicEdge {
    fn default() -> Self {
        Self {
            t_parent_child: DMat3::IDENTITY,
            offset_in_pstr: DVec3::ZERO,
        }
    }
}

/// Walk a [`MassStorage`] tree pre-order from each root and derive
/// every non-root node's [`KinematicNodeState`] from its parent's
/// state composed with the per-edge geometry.
///
/// `nodes` is a per-storage-id map giving each node's current state —
/// roots are read but never written; non-roots are read for their
/// `t_struct_body` / `composite_in_struct` (which the kernel routes
/// through the parent), and their `rot` / `trans` are recomputed.
/// `edges` is a per-non-root-edge map giving the link rotation and
/// the child's structural offset in the parent's structural frame.
///
/// Returns a fresh `HashMap<S::Id, KinematicNodeState>` keyed by the
/// same storage ids — entries for roots are copied through verbatim,
/// entries for kinematic children are the kernel's outputs. Storage
/// callers handle write-back themselves (the Bevy adapter's
/// `propagate_state_from_root_system` writes the fields back into
/// `RotationalStateC` / `TranslationalStateC`).
///
/// # Panics
///
/// Panics with a "Fail Loudly" diagnostic when:
/// - A non-root node is missing from `nodes` (every chain member
///   must declare its `KinematicNodeState`, even if its rotational
///   state is about to be derived). Without the node entry the
///   kernel cannot read the per-node `t_struct_body` /
///   `composite_in_struct` it needs to land in the child's
///   composite-body frame.
/// - A non-root edge is missing from `edges`. Mirrors the same
///   contract `aggregate_wrenches_via_storage` enforces.
/// - The walk fails to reach every storage node (`MassChildOf` cycle
///   or orphaned subtree).
// JEOD_INV: DB.13 — propagate_state delegates to root body (only roots seed the walk; every kinematic child's state is derived from its parent's)
// JEOD_INV: DB.17 — only the root's state is integrated; non-root state is kinematic
pub fn propagate_state_via_storage<S: MassStorage>(
    storage: &S,
    nodes: &HashMap<S::Id, KinematicNodeState>,
    edges: &HashMap<S::Id, KinematicEdge>,
) -> HashMap<S::Id, KinematicNodeState> {
    let expected = storage.node_count();
    let mut out: HashMap<S::Id, KinematicNodeState> = HashMap::with_capacity(expected);
    let mut state: HashMap<S::Id, VisitState> = HashMap::with_capacity(expected);

    for root in storage.roots() {
        // Roots seed the walk verbatim — their state is the
        // integrator's output and is the source of truth this tick.
        let root_state = *nodes.get(&root).unwrap_or_else(|| {
            panic!(
                "propagate_state_via_storage: root {root:?} missing from `nodes`. Every \
                 storage entity must declare a KinematicNodeState (even roots — their \
                 state seeds the walk). Backend likely failed to populate the per-entity \
                 state map; check the orchestration glue."
            )
        });
        out.insert(root, root_state);
        walk(
            storage,
            root,
            &root_state,
            nodes,
            edges,
            &mut out,
            &mut state,
        );
    }

    let visited_count = state.len();
    assert!(
        visited_count == expected,
        "MassStorage topology has an orphan: {} of {} nodes unreachable from roots(). \
         Kinematic propagation skipped {} children; their RotationalState / \
         TranslationalState would stay frozen at the previous tick's value, silently \
         drifting once the parent moves. Check MassChildOf edges for parents that are \
         not roots and not children of any other node.",
        expected.saturating_sub(visited_count),
        expected,
        expected.saturating_sub(visited_count),
    );

    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisitState {
    Visiting,
    Visited,
}

fn walk<S: MassStorage>(
    storage: &S,
    parent_id: S::Id,
    parent_state: &KinematicNodeState,
    nodes: &HashMap<S::Id, KinematicNodeState>,
    edges: &HashMap<S::Id, KinematicEdge>,
    out: &mut HashMap<S::Id, KinematicNodeState>,
    state: &mut HashMap<S::Id, VisitState>,
) {
    match state.get(&parent_id) {
        Some(VisitState::Visited) => return,
        Some(VisitState::Visiting) => {
            panic!(
                "MassStorage topology has a cycle reachable from roots(): node \
                 {parent_id:?} revisited while still on the active propagation \
                 stack. Kinematic state propagation requires a forest of trees \
                 rooted at MassStorage::roots(); remove the back-edge from \
                 MassChildOf so the node's parent chain terminates at a root."
            );
        }
        None => {}
    }
    state.insert(parent_id, VisitState::Visiting);

    let parent_t_inertial_body = parent_state.rot.quaternion.left_quat_to_transformation();

    for &child_id in storage.children(parent_id) {
        // Per-edge geometry must exist for every non-root child.
        let edge = edges.get(&child_id).unwrap_or_else(|| {
            panic!(
                "kinematic propagation: child edge {child_id:?} missing from `edges` \
                 map. Every non-root node must have a KinematicEdge entry — build it \
                 from the live mass-tree state (e.g. MassChildOf.t_parent_child + \
                 MassChildOf.offset on the Bevy side)."
            )
        });
        // Per-node state must exist for every storage entity (the
        // kernel needs the child's `t_struct_body` and
        // `composite_in_struct`). The output `rot` / `trans` are
        // overwritten — we only consume the structural fields.
        let child_node = nodes.get(&child_id).unwrap_or_else(|| {
            panic!(
                "kinematic propagation: child {child_id:?} missing from `nodes` \
                 map. Every storage entity in the mass tree must declare a \
                 KinematicNodeState; backend likely forgot to seed it. \
                 Even kinematic children whose `rot`/`trans` are about to be \
                 derived must declare their `t_struct_body` and \
                 `composite_in_struct` so the kernel can route through the \
                 child's structural frame."
            )
        });

        let inputs = KinematicChildInputs {
            parent_t_inertial_body,
            parent_ang_vel_body: parent_state.rot.ang_vel_body,
            parent_position_inertial: parent_state.trans.position,
            parent_velocity_inertial: parent_state.trans.velocity,
            parent_t_struct_body: parent_state.t_struct_body,
            parent_composite_in_pstr: parent_state.composite_in_struct,
            t_parent_child: edge.t_parent_child,
            link_offset_in_pstr: edge.offset_in_pstr,
            child_t_struct_body: child_node.t_struct_body,
            child_composite_in_cstr: child_node.composite_in_struct,
        };
        let kernel_out = compute_kinematic_child_state(inputs);

        let derived_rot = RotationalState {
            quaternion: kernel_out.child_q_inertial_body,
            ang_vel_body: kernel_out.child_ang_vel_body,
        };
        let derived_trans = TranslationalState {
            position: kernel_out.child_position_inertial,
            velocity: kernel_out.child_velocity_inertial,
        };
        // Carry the child's structural fields through unchanged — the
        // walk only writes the rotational / translational state.
        let derived = KinematicNodeState {
            rot: derived_rot,
            trans: derived_trans,
            t_struct_body: child_node.t_struct_body,
            composite_in_struct: child_node.composite_in_struct,
        };
        out.insert(child_id, derived);

        // Recurse — pre-order, so the child's derived state is the
        // parent state for its own children.
        walk(storage, child_id, &derived, nodes, edges, out, state);
    }

    state.insert(parent_id, VisitState::Visited);
}

/// Quick sanity helper: is the input quaternion unit-norm to
/// `tol`? Kept module-private; production callers should let the
/// kernel's internal `NormalizedQuat::new` panic surface a non-unit
/// quaternion at the boundary.
#[inline]
fn _unit_norm(q: &JeodQuat, tol: f64) -> bool {
    (q.norm_sq() - 1.0).abs() < tol
}

#[cfg(test)]
mod tests {
    use super::*;
    use jeod_dynamics::mass::MassProperties;
    use jeod_dynamics::mass_body::MassTree;

    /// Simple componentwise vector comparator (kept module-local to
    /// avoid a feature-gated dev-dep on `jeod_math`'s test-utils).
    fn approx_eq_vec3(a: DVec3, b: DVec3, tol: f64) -> bool {
        (a - b).length() < tol
    }

    /// Simple matrix comparator for rotation-orthonormality checks.
    fn approx_eq_mat3(a: &DMat3, b: &DMat3, tol: f64) -> bool {
        (a.x_axis - b.x_axis).length() < tol
            && (a.y_axis - b.y_axis).length() < tol
            && (a.z_axis - b.z_axis).length() < tol
    }

    fn jeod_rot_z(angle: f64) -> DMat3 {
        JeodQuat::left_quat_from_eigen_rotation(angle, DVec3::Z).left_quat_to_transformation()
    }

    /// Build the per-edge `KinematicEdge` map from a `MassTree` arena.
    fn edges_for(tree: &MassTree) -> HashMap<<MassTree as MassStorage>::Id, KinematicEdge> {
        let mut out = HashMap::new();
        for id in 0..tree.len() {
            if MassStorage::parent(tree, id).is_some() {
                let view = MassStorage::node(tree, id);
                out.insert(
                    id,
                    KinematicEdge {
                        t_parent_child: view.structure_point.t_parent_this,
                        offset_in_pstr: view.structure_point.position,
                    },
                );
            }
        }
        out
    }

    /// Identity attach: a chain whose every link is identity must
    /// leave every node's state at its parent's value.
    #[test]
    fn identity_chain_propagates_root_state_unchanged() {
        let mut tree = MassTree::new();
        let root = tree.add_root("root".into(), MassProperties::new(10.0));
        let mid = tree.add_body("mid".into(), MassProperties::new(5.0));
        let leaf = tree.add_body("leaf".into(), MassProperties::new(2.0));
        tree.attach(mid, root, DVec3::ZERO, DMat3::IDENTITY);
        tree.attach(leaf, mid, DVec3::ZERO, DMat3::IDENTITY);

        let root_pos = DVec3::new(7e6, 0.0, 0.0);
        let root_vel = DVec3::new(0.0, 7500.0, 0.0);
        let mut nodes = HashMap::new();
        nodes.insert(
            root,
            KinematicNodeState {
                rot: RotationalState::default(),
                trans: TranslationalState {
                    position: root_pos,
                    velocity: root_vel,
                },
                t_struct_body: DMat3::IDENTITY,
                composite_in_struct: DVec3::ZERO,
            },
        );
        nodes.insert(mid, KinematicNodeState::default());
        nodes.insert(leaf, KinematicNodeState::default());

        let edges = edges_for(&tree);
        let out = propagate_state_via_storage(&tree, &nodes, &edges);

        // Root preserved verbatim.
        assert_eq!(out[&root].trans.position, root_pos);
        assert_eq!(out[&root].trans.velocity, root_vel);
        // mid and leaf inherit root's identity attitude and zero offset
        // ⇒ same position / velocity.
        assert!(approx_eq_vec3(out[&mid].trans.position, root_pos, 1e-12));
        assert!(approx_eq_vec3(out[&mid].trans.velocity, root_vel, 1e-12));
        assert!(approx_eq_vec3(out[&leaf].trans.position, root_pos, 1e-12));
        assert!(approx_eq_vec3(out[&leaf].trans.velocity, root_vel, 1e-12));
    }

    /// Three-body chain root → mid → leaf where every link is
    /// rotated 30°-Z and offset by (1, 0, 0). The leaf's
    /// inertial→body rotation must equal `t_pc · t_pc` (60° about Z).
    #[test]
    fn three_body_chain_composes_rotations_through_middle() {
        let angle = std::f64::consts::PI / 6.0;
        let t_pc = jeod_rot_z(angle);
        let offset = DVec3::new(1.0, 0.0, 0.0);

        let mut tree = MassTree::new();
        let root = tree.add_root("root".into(), MassProperties::new(10.0));
        let mid = tree.add_body("mid".into(), MassProperties::new(5.0));
        let leaf = tree.add_body("leaf".into(), MassProperties::new(2.0));
        tree.attach(mid, root, offset, t_pc);
        tree.attach(leaf, mid, offset, t_pc);

        let mut nodes = HashMap::new();
        nodes.insert(root, KinematicNodeState::default());
        nodes.insert(mid, KinematicNodeState::default());
        nodes.insert(leaf, KinematicNodeState::default());

        let edges = edges_for(&tree);
        let out = propagate_state_via_storage(&tree, &nodes, &edges);

        // Leaf attitude = T_pc · T_pc (60° about Z).
        let expected_t = t_pc * t_pc;
        let leaf_t = out[&leaf].rot.quaternion.left_quat_to_transformation();
        assert!(
            approx_eq_mat3(&leaf_t, &expected_t, 1e-12),
            "leaf T_inertial_body: expected {:?}, got {:?}",
            expected_t,
            leaf_t
        );
        // Position: root at origin + identity-attitude root means
        // mid at offset (1,0,0); the leaf attaches at (1,0,0) in mid's
        // *structural* frame, and mid's struct frame is rotated by
        // t_pc relative to root's struct frame, so the leaf's offset
        // in root frame is t_pc^T · (1,0,0) = (cos30, -sin30, 0).
        // Total leaf position = (1,0,0) + (cos30, -sin30, 0).
        let expected_leaf_pos = offset + t_pc.transpose() * offset;
        assert!(
            approx_eq_vec3(out[&leaf].trans.position, expected_leaf_pos, 1e-12),
            "leaf position: expected {expected_leaf_pos:?}, got {:?}",
            out[&leaf].trans.position
        );
    }

    /// Missing-child guard: a non-root storage entity that lacks a
    /// `KinematicNodeState` must trip the loud diagnostic. Tests the
    /// fail-loudly contract.
    #[test]
    #[should_panic(expected = "missing from `nodes` map")]
    fn missing_child_node_state_panics() {
        let mut tree = MassTree::new();
        let root = tree.add_root("root".into(), MassProperties::new(10.0));
        let child = tree.add_body("child".into(), MassProperties::new(5.0));
        tree.attach(child, root, DVec3::ZERO, DMat3::IDENTITY);

        // Only seed the root — the child is intentionally absent so
        // the walk must trip the diagnostic.
        let mut nodes = HashMap::new();
        nodes.insert(root, KinematicNodeState::default());

        let edges = edges_for(&tree);
        let _ = propagate_state_via_storage(&tree, &nodes, &edges);
    }

    /// Missing-edge guard: a non-root storage entity present in
    /// `nodes` but with no `edges` entry must also fail loudly.
    #[test]
    #[should_panic(expected = "missing from `edges` map")]
    fn missing_child_edge_panics() {
        let mut tree = MassTree::new();
        let root = tree.add_root("root".into(), MassProperties::new(10.0));
        let child = tree.add_body("child".into(), MassProperties::new(5.0));
        tree.attach(child, root, DVec3::ZERO, DMat3::IDENTITY);

        let mut nodes = HashMap::new();
        nodes.insert(root, KinematicNodeState::default());
        nodes.insert(child, KinematicNodeState::default());

        // Empty edges map: the child has no entry, so the walk must
        // trip the diagnostic.
        let edges: HashMap<<MassTree as MassStorage>::Id, KinematicEdge> = HashMap::new();
        let _ = propagate_state_via_storage(&tree, &nodes, &edges);
    }
}
