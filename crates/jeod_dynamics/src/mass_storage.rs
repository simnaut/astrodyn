//! Storage-agnostic mass-tree composition.
//!
//! [`MassStorage`] is the storage-side counterpart of `FrameStorage`
//! (sketched in the [Frame-Tree-ECS-Native design doc Section
//! 7](https://github.com/simnaut/bevy_jeod/wiki/Frame-Tree-ECS-Native)):
//! a thin trait whose implementations expose a parent link plus the
//! per-node *core* mass-properties view, so a single composition
//! kernel can drive both the arena-backed [`MassTree`] used by the
//! `jeod_runner` and the Bevy adapter's `MassChildOf` relation.
//!
//! The kernel ([`recompute_composites_via_storage`] and the lower-level
//! [`compute_node_composite`]) ports the same parallel-axis (Steiner)
//! algorithm encoded in [`MassTree::recompute_composites`]
//! (`mass_calc_composite_cm.cc` + `mass_calc_composite_inertia.cc` from
//! JEOD v5.4):
//!
//! 1. core / atomic node:  `composite ← core`.
//! 2. composite CoM = mass-weighted average over `core` and each
//!    child's `composite_wrt_pstr.position` (the child's composite CoM
//!    expressed in *this* body's structural frame).
//! 3. composite inertia = core inertia shifted to composite CoM
//!    (point-mass shift) + each child's composite inertia rotated
//!    `T^T · I · T` and shifted by the child→composite offset.
//! 4. the post-attach root (and any detached child whose tree just
//!    became its own root) gets `inverse_inertia ← inertia.inverse()`,
//!    matching JEOD's `MassBody::update_mass_properties` and
//!    `mass_detach.cc` lines 322-335.
//!
//! Storage backends supply the *core* properties via
//! [`MassNodeView`]; the kernel returns the recomputed *composite*
//! properties via [`MassNodeOutputs`]. Mutation of underlying storage
//! (i.e. how the new composite values land back in the arena / how
//! they are written into Bevy components) stays at the call site —
//! the trait is read-only on purpose, mirroring the
//! `FrameStorage::state` shape.

use core::fmt::Debug;

use glam::{DMat3, DVec3};

use crate::mass::MassProperties;
use crate::mass_body::{point_mass_inertia, MassBodyId, MassPointState, MassTree};

/// Read-only view of one node in a mass tree, as needed by the
/// composition kernel.
///
/// Mirrors the subset of [`crate::mass_body::MassBody`] the JEOD
/// algorithm actually consults: the body's *core* properties, the
/// `structure_point` (offset + rotation in the parent's structural
/// frame), and a string name used solely for diagnostic messages
/// (singular-inertia panic, cycle-detection diagnostics).
#[derive(Debug, Clone, Copy)]
pub struct MassNodeView<'a> {
    /// This body's core mass / inertia / CoM. Inertia is about the
    /// body axes through the core CoM; position is the core CoM in
    /// the body's own structural frame.
    pub core: MassProperties,
    /// Position + rotation of this body's structural origin in the
    /// **parent's** structural frame. Ignored for roots (the kernel
    /// never reads it for a root node).
    pub structure_point: MassPointState,
    /// Diagnostic name. Used only in panic messages — backends should
    /// supply something meaningful (`"CM"`, `Entity { … }`, …) but
    /// the kernel never compares names.
    pub name: &'a str,
}

/// Output of a single-node composition step.
///
/// The kernel produces this for every node in post-order; the call
/// site decides how to write it back into the underlying storage
/// (mutate the arena `composite_properties` field, or pipe into a
/// Bevy `MassPropertiesC` component, etc.).
#[derive(Debug, Clone, Copy)]
pub struct MassNodeOutputs {
    /// Recomputed composite mass / inertia / CoM (with `inverse_*`
    /// caches kept consistent for roots and detached leaves; see
    /// [`recompute_composites_via_storage`] for the rule).
    pub composite: MassProperties,
    /// `core_wrt_composite`: core CoM minus composite CoM, both in
    /// this body's structural frame. Default for atomic nodes.
    pub core_wrt_composite: MassPointState,
    /// `composite_wrt_pstr`: this node's composite CoM expressed in
    /// the *parent's* structural frame, plus `t_parent_this` copied
    /// from `structure_point`. Default for roots.
    pub composite_wrt_pstr: MassPointState,
}

impl Default for MassNodeOutputs {
    fn default() -> Self {
        Self {
            composite: MassProperties::new(1.0),
            core_wrt_composite: MassPointState::default(),
            composite_wrt_pstr: MassPointState::default(),
        }
    }
}

/// Storage-agnostic read view of a mass tree.
///
/// Implementations supply parent links and a per-node
/// [`MassNodeView`]; the [`recompute_composites_via_storage`] kernel
/// drives composition without knowing whether the storage is an
/// arena, an ECS world, or something else. Designed to mirror the
/// `FrameStorage` trait sketched in the design doc Section 7.
///
/// The trait is intentionally read-only — write-back happens at the
/// call site (the [`MassTree`] arena impl uses
/// [`MassTree::recompute_composites`]; the Bevy adapter's
/// `composite_mass_system` writes outputs into `MassPropertiesC`).
pub trait MassStorage {
    /// Backend-specific node identifier (`MassBodyId` for the arena,
    /// `Entity` for the Bevy adapter).
    type Id: Copy + Eq + Debug;

    /// Direct parent of `id`, or `None` for a root.
    fn parent(&self, id: Self::Id) -> Option<Self::Id>;

    /// Core properties + structure-point view for `id`.
    fn node(&self, id: Self::Id) -> MassNodeView<'_>;

    /// Direct children of `id`. Order matters only for diagnostic
    /// reproducibility — composition is associative + commutative
    /// over Steiner-shifted child contributions.
    fn children(&self, id: Self::Id) -> Vec<Self::Id>;

    /// Every root in the storage (parents whose `parent(...)` is
    /// `None`). The kernel walks each root post-order to honour the
    /// "leaves first" invariant.
    fn roots(&self) -> Vec<Self::Id>;
}

/// Compute the composite properties of one node, given its core view
/// plus the *already-composed* outputs of its children.
///
/// This is the kernel of `mass_calc_composite_cm.cc` +
/// `mass_calc_composite_inertia.cc` factored so it can be driven by
/// any storage that produces children in post-order.
///
/// `inverse_*` caches:
///
/// - For roots (`is_root == true`) the kernel inverts the composite
///   inertia and sets `inverse_inertia` (matching JEOD
///   `mass_update.cc:116-125`); zero composite mass yields
///   `inverse_inertia = 0` (matches JEOD's `Matrix3x3::invert_symmetric`
///   fall-through).
/// - For non-roots `inverse_inertia` is set to a zero matrix —
///   JEOD only inverts at root nodes. The arena's
///   [`MassTree::recompute_composites`] honours the same rule.
///
/// Panics if the composite inertia is singular (matches the existing
/// arena diagnostic in `MassTree::calc_composite_inertia`). The panic
/// message names the body so a mission engineer can diagnose which
/// attachment introduced the degeneracy.
// JEOD_INV: MA.06 — bottom-up mass property update (children first; the kernel takes pre-composed children)
// JEOD_INV: MA.07 — derived quantities recomputed (output composite has fresh inverse caches)
pub fn compute_node_composite(
    node: MassNodeView<'_>,
    children: &[MassNodeOutputs],
    is_root: bool,
) -> MassNodeOutputs {
    if children.is_empty() {
        // Atomic body: composite == core (JEOD mass_update.cc:59-75).
        let mut composite = node.core;
        if is_root {
            // Root with no children still needs `inverse_inertia`
            // populated against the core inertia; the consistency is
            // already maintained by `MassProperties::with_inertia`,
            // so just propagate.
            if composite.mass <= 0.0 {
                composite.inverse_inertia = DMat3::ZERO;
            }
        } else {
            // Non-root: keep the arena's "leaf inverse_inertia is
            // zero" convention so consumers don't accidentally rely
            // on it for non-root nodes.
            composite.inverse_inertia = DMat3::ZERO;
        }
        return MassNodeOutputs {
            composite,
            core_wrt_composite: MassPointState::default(),
            composite_wrt_pstr: MassPointState::default(),
        };
    }

    // 1. Composite CoM (mass-weighted): core + Σ child_composite (in
    //    this body's structural frame).
    let mut total_mass = node.core.mass;
    let mut weighted_pos = node.core.position * node.core.mass;
    for child in children {
        total_mass += child.composite.mass;
        weighted_pos += child.composite_wrt_pstr.position * child.composite.mass;
    }
    let cm = if total_mass > 0.0 {
        weighted_pos / total_mass
    } else {
        DVec3::ZERO
    };

    // 2. Composite inertia: core shifted to composite CoM + each
    //    rotated child shifted by its offset to the composite CoM.
    let core_offset = node.core.position - cm;
    let mut composite_inertia = node.core.inertia + point_mass_inertia(node.core.mass, core_offset);
    for child in children {
        let child_offset = child.composite_wrt_pstr.position - cm;
        let t = child.composite_wrt_pstr.t_parent_this;
        // Rotate child's composite inertia from child structural frame to
        // this body's structural frame:  T^T · I_child · T.
        let rotated_inertia = t.transpose() * child.composite.inertia * t;
        composite_inertia +=
            rotated_inertia + point_mass_inertia(child.composite.mass, child_offset);
    }

    let inverse_mass = if total_mass > 0.0 {
        1.0 / total_mass
    } else {
        0.0
    };

    // 3. inverse_inertia: invert at roots, zero elsewhere (JEOD only
    //    materializes inverse_inertia at the integration root).
    let inverse_inertia = if is_root && total_mass > 0.0 {
        let det = composite_inertia.determinant();
        // JEOD's Matrix3x3::invert_symmetric (dm_invert_symm.cc:86-94)
        // checks for a zero determinant; we panic with a diagnostic
        // that names the body per the "Fail Loudly" rule.
        assert!(
            det.abs() > 1e-30,
            "Body '{}' has singular composite inertia (det={det:.2e}); \
             check mass-tree attach offsets and child inertias.",
            node.name
        );
        composite_inertia.inverse()
    } else {
        DMat3::ZERO
    };

    let composite = MassProperties {
        mass: total_mass,
        inverse_mass,
        inertia: composite_inertia,
        inverse_inertia,
        position: cm,
        // composite.t_parent_this remains the core's struct→body
        // rotation; JEOD propagates this through composite_properties
        // (see `MassBody::update_mass_properties`).
        t_parent_this: node.core.t_parent_this,
        dirty: false,
    };

    let core_wrt_composite = MassPointState {
        position: node.core.position - cm,
        t_parent_this: DMat3::IDENTITY,
    };

    MassNodeOutputs {
        composite,
        core_wrt_composite,
        // composite_wrt_pstr is filled in by the caller — it's a
        // function of the parent's structural frame, which the kernel
        // doesn't see. The driver fills it in before passing this
        // record to the parent.
        composite_wrt_pstr: MassPointState::default(),
    }
}

/// Compute `composite_wrt_pstr` for a child given its post-composition
/// outputs and its `structure_point`.
///
/// JEOD `mass_update.cc:137-143`:
/// ```text
/// composite_wrt_pstr.position =
///     T_parent_this^T · composite_properties.position
///     + structure_point.position
/// composite_wrt_pstr.t_parent_this = structure_point.t_parent_this
/// ```
///
/// This shape lets a driver finish the kernel-emitted
/// [`MassNodeOutputs`] before handing it to the parent.
pub fn finalize_child_in_parent_frame(
    child: &mut MassNodeOutputs,
    structure_point: &MassPointState,
) {
    let t = structure_point.t_parent_this;
    let comp_pos = child.composite.position;
    child.composite_wrt_pstr.position = t.transpose() * comp_pos + structure_point.position;
    child.composite_wrt_pstr.t_parent_this = structure_point.t_parent_this;
}

/// Drive [`compute_node_composite`] in post-order over every root in
/// the storage and return the outputs keyed by node id.
///
/// Storage callers handle write-back themselves (the arena impl
/// mutates `MassTree::nodes[id].composite_properties`; the Bevy
/// adapter writes into `MassPropertiesC`). The kernel only computes
/// — it never mutates.
///
/// Returned vector is in post-order (children before parents) so a
/// caller that needs to write back in dependency-respecting order
/// can iterate in sequence.
pub fn recompute_composites_via_storage<S: MassStorage>(
    storage: &S,
) -> Vec<(S::Id, MassNodeOutputs)> {
    let roots = storage.roots();
    let mut out: Vec<(S::Id, MassNodeOutputs)> = Vec::new();
    let mut seen: Vec<S::Id> = Vec::new();
    for root in roots {
        walk(storage, root, true, &mut out, &mut seen);
    }
    out
}

fn walk<S: MassStorage>(
    storage: &S,
    id: S::Id,
    is_root: bool,
    out: &mut Vec<(S::Id, MassNodeOutputs)>,
    seen: &mut Vec<S::Id>,
) {
    if seen.contains(&id) {
        return;
    }
    let children = storage.children(id);
    let mut child_outputs: Vec<MassNodeOutputs> = Vec::with_capacity(children.len());
    for child_id in children {
        walk(storage, child_id, false, out, seen);
        // Re-find the child's output in `out` (post-order push); the
        // last entry whose id matches is the child we just walked,
        // since each node is pushed exactly once.
        let child_view = storage.node(child_id);
        let mut child_out = out
            .iter()
            .rev()
            .find_map(|(cid, o)| if *cid == child_id { Some(*o) } else { None })
            .expect("child not pushed by post-order walk");
        // Fill in composite_wrt_pstr against this child's
        // structure_point (in *its parent's* structural frame).
        finalize_child_in_parent_frame(&mut child_out, &child_view.structure_point);
        child_outputs.push(child_out);
        // Persist the parent-relative fields back into the stored
        // entry so the caller writing back to storage sees the
        // finalized `composite_wrt_pstr`.
        if let Some(slot) = out.iter_mut().rev().find(|(cid, _)| *cid == child_id) {
            slot.1 = child_out;
        }
    }
    let view = storage.node(id);
    let outputs = compute_node_composite(view, &child_outputs, is_root);
    seen.push(id);
    out.push((id, outputs));
}

// ---------------------------------------------------------------------------
// Arena impl: MassTree
// ---------------------------------------------------------------------------

impl MassStorage for MassTree {
    type Id = MassBodyId;

    fn parent(&self, id: Self::Id) -> Option<Self::Id> {
        MassTree::parent(self, id)
    }

    fn node(&self, id: Self::Id) -> MassNodeView<'_> {
        let body = MassTree::get(self, id);
        MassNodeView {
            core: body.core_properties,
            structure_point: body.structure_point,
            name: &body.name,
        }
    }

    fn children(&self, id: Self::Id) -> Vec<Self::Id> {
        MassTree::children(self, id).to_vec()
    }

    fn roots(&self) -> Vec<Self::Id> {
        let mut roots = Vec::new();
        for id in 0..self.len() {
            if MassTree::parent(self, id).is_none() {
                roots.push(id);
            }
        }
        roots
    }
}

// ---------------------------------------------------------------------------
// Tests — verify the trait-driven kernel reproduces the arena's
// in-place `recompute_composites` bit-for-bit.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mass::MassProperties;

    fn assert_props_close(a: &MassProperties, b: &MassProperties, tol: f64, label: &str) {
        assert!(
            (a.mass - b.mass).abs() < tol,
            "{label}: mass {} vs {}",
            a.mass,
            b.mass
        );
        let dpos = (a.position - b.position).length();
        assert!(dpos < tol, "{label}: position diff {dpos:.3e}");
        for (col_a, col_b) in [
            (a.inertia.x_axis, b.inertia.x_axis),
            (a.inertia.y_axis, b.inertia.y_axis),
            (a.inertia.z_axis, b.inertia.z_axis),
        ] {
            let d = (col_a - col_b).length();
            assert!(d < tol, "{label}: inertia col diff {d:.3e}");
        }
    }

    #[test]
    fn storage_kernel_matches_arena_single_attach() {
        // Single child attached to a root — composite must agree with
        // the arena's in-place recomputation byte-for-byte.
        let mut tree = MassTree::new();
        let parent = tree.add_root(
            "parent".into(),
            MassProperties::with_inertia(
                10.0,
                DMat3::from_diagonal(DVec3::new(50.0, 60.0, 70.0)),
                DVec3::new(0.1, -0.2, 0.0),
            ),
        );
        let child = tree.add_body(
            "child".into(),
            MassProperties::with_inertia(
                4.0,
                DMat3::from_diagonal(DVec3::new(8.0, 9.0, 10.0)),
                DVec3::new(-0.05, 0.0, 0.0),
            ),
        );
        tree.attach(child, parent, DVec3::new(2.0, 0.5, -0.3), DMat3::IDENTITY);

        // Reference: the arena re-runs recompute_composites on attach.
        let arena_parent_composite = tree.get(parent).composite_properties;
        let arena_child_composite = tree.get(child).composite_properties;

        // Kernel via trait.
        let outs = recompute_composites_via_storage(&tree);
        let kernel_parent = outs.iter().find(|(id, _)| *id == parent).unwrap().1;
        let kernel_child = outs.iter().find(|(id, _)| *id == child).unwrap().1;

        assert_props_close(
            &arena_parent_composite,
            &kernel_parent.composite,
            1e-12,
            "parent composite",
        );
        // Non-root entries: kernel zeroes inverse_inertia by design;
        // arena does the same for non-roots.
        assert!((kernel_child.composite.mass - arena_child_composite.mass).abs() < 1e-12);
        assert!(
            (kernel_child.composite.position - arena_child_composite.position).length() < 1e-12
        );
    }

    #[test]
    fn storage_kernel_matches_arena_three_body_chain() {
        // A → B → C with non-trivial offsets and a 90° rotation on C.
        let mut tree = MassTree::new();
        let a = tree.add_root("A".into(), MassProperties::new(10.0));
        let b = tree.add_body("B".into(), MassProperties::new(5.0));
        let c = tree.add_body(
            "C".into(),
            MassProperties::with_inertia(
                3.0,
                DMat3::from_diagonal(DVec3::new(1.0, 4.0, 9.0)),
                DVec3::ZERO,
            ),
        );

        tree.attach(b, a, DVec3::new(2.0, 0.0, 0.0), DMat3::IDENTITY);
        // 90 deg about Z attaching C to B.
        let rot_z90 = DMat3::from_cols(
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        tree.attach(c, b, DVec3::new(1.0, 0.0, 0.0), rot_z90);

        let arena_a = tree.get(a).composite_properties;
        let arena_b = tree.get(b).composite_properties;

        let outs = recompute_composites_via_storage(&tree);
        let ka = outs.iter().find(|(id, _)| *id == a).unwrap().1;
        let kb = outs.iter().find(|(id, _)| *id == b).unwrap().1;

        assert_props_close(&arena_a, &ka.composite, 1e-10, "A composite");
        // Non-root B: compare mass / pos / inertia (arena keeps its
        // own per-row inverse for non-roots — that's a non-load-bearing
        // arena detail; we don't carry it through the kernel).
        assert!((arena_b.mass - kb.composite.mass).abs() < 1e-12);
        assert!((arena_b.position - kb.composite.position).length() < 1e-12);
        for (ca, cb) in [
            (arena_b.inertia.x_axis, kb.composite.inertia.x_axis),
            (arena_b.inertia.y_axis, kb.composite.inertia.y_axis),
            (arena_b.inertia.z_axis, kb.composite.inertia.z_axis),
        ] {
            let d = (ca - cb).length();
            assert!(d < 1e-10, "B inertia diff {d:.3e}");
        }
    }

    #[test]
    fn storage_kernel_atomic_root_round_trip() {
        // Single-node tree: composite must equal core, inverse caches
        // unchanged.
        let mut tree = MassTree::new();
        let r = tree.add_root(
            "alone".into(),
            MassProperties::with_inertia(
                7.0,
                DMat3::from_diagonal(DVec3::new(20.0, 30.0, 40.0)),
                DVec3::new(0.0, 0.1, 0.0),
            ),
        );
        let outs = recompute_composites_via_storage(&tree);
        let k = outs.iter().find(|(id, _)| *id == r).unwrap().1;
        let arena = tree.get(r).composite_properties;
        assert_props_close(&arena, &k.composite, 1e-12, "atomic root");
    }
}
