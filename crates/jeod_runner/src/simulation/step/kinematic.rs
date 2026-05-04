//! Kinematic state propagation (root → leaves) for
//! [`super::super::Simulation`].
//!
//! Drives [`jeod_sim::propagate_state_via_storage`] over the live mass
//! tree once per [`Simulation::step`](super::super::Simulation::step):
//! every non-root [`super::super::types::SimBody`] gets its
//! `composite_body` inertial state (`trans` + `rot`) overwritten from
//! its parent's freshly-derived state composed with the per-link
//! `MassChildOf` rotation + offset.
//!
//! This is the runner-side parallel of Bevy's
//! `propagate_state_from_root_system`
//! ([`crate::kinematic_propagation`](../../../bevy_jeod/src/kinematic_propagation.rs)
//! when read through the workspace). The kernel and the orchestration
//! walk live in `jeod_dynamics` / `jeod_sim`; this module is the thin
//! runner glue.
//!
//! # JEOD precedent
//!
//! Mirrors `DynBody::propagate_state_from_structure` in
//! [`models/dynamics/dyn_body/src/dyn_body_propagate_state.cc`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/dynamics/dyn_body/src/dyn_body_propagate_state.cc):
//! starting from each tree-root body's state, derive every non-root
//! child's `composite_body` state from the per-link attach geometry,
//! then recurse. Only roots integrate (`DB.17`); every other chain
//! member's state is kinematic-derived each tick.
//!
//! # Schedule placement
//!
//! Called from `Simulation::step_internal` **after** stage 3 (per-body
//! mass-property recompute, so `composite_properties.position` is
//! live) and **before** stage 6 (interactions / force collection).
//! This matches the Bevy adapter's `JeodSet::ForceCollection`
//! placement: kinematic children's attitudes are installed before any
//! per-step physics that would consume them. The runner has no upward
//! wrench-aggregation walk yet, so the placement is mostly about
//! schedule symmetry with the Bevy plugin — which means a future
//! wrench walk can simply slot in after this without re-shuffling
//! existing stages.
//!
//! # Out of scope
//!
//! - **Detached subtrees** (`DetachedSubtreeState`) get their own
//!   ballistic propagation in `step_detached_subtrees` — a kinematic
//!   chain that has been torn loose from a root no longer has a
//!   parent in the mass tree, so it falls out of this walk by
//!   construction.
//! - **Joint-kinematics drivers** that prescribe relative motion at
//!   a `MassChildOf` link write through a parallel surface
//!   (frame-tree rotations) and are independent of the body-side
//!   walk here.

use std::collections::HashMap;

use jeod_dynamics::MassBodyId;
use jeod_sim::{
    propagate_state_via_storage, KinematicEdge, KinematicNodeState, MassStorage, RotationalState,
    TranslationalState, TranslationalStateTyped,
};

use super::super::Simulation;
use jeod_sim::IntegrationFrame;

impl Simulation {
    /// Walk the mass tree from each root and overwrite every non-root,
    /// kinematic-only [`SimBody`](super::super::types::SimBody)'s
    /// `trans` + `rot` from the parent's state composed with the link
    /// geometry.
    ///
    /// No-op when there is no mass tree, when the tree has no chains
    /// (every node is a root), or when no `SimBody` is registered as
    /// a mass-tree node.
    ///
    /// # Panics
    ///
    /// Panics with a "Fail Loudly" diagnostic when:
    /// - Two `SimBody` indices share the same `mass_body_id`: writeback
    ///   would silently overwrite one body's state with the other's
    ///   tree-derived value, so the duplicate registration must fail
    ///   immediately.
    /// - A `SimBody` flagged
    ///   [`SimBody::kinematic_only`](super::super::types::SimBody)
    ///   has no [`mass_body_id`](super::super::types::SimBody): the
    ///   flag is meaningless without tree membership.
    /// - A `kinematic_only` body resolves to a tree root: only
    ///   non-root nodes have a parent to derive state from.
    /// - A `kinematic_only` body has no `RotationalState`: the
    ///   propagation walk derives rotational state, so 3-DOF bodies
    ///   cannot be kinematic children.
    /// - The orchestration walk surfaces a topology error (orphan
    ///   subtree, cycle, missing edge — see
    ///   [`propagate_state_via_storage`] for the per-case diagnostics).
    // JEOD_INV: DB.13 — propagate_state delegates to root body
    // JEOD_INV: DB.17 — only the root's state is integrated; non-root state is kinematic-derived
    pub(super) fn propagate_kinematic_state(&mut self) {
        let Some(tree) = self.mass_tree.as_ref() else {
            return;
        };
        if tree.node_count() == 0 {
            return;
        }

        // Build a per-tree-id map from MassBodyId -> SimBody index for
        // the non-root chain members we plan to write back, and validate
        // each kinematic_only flag is paired with a non-root tree node.
        let mut sim_body_for_id: HashMap<MassBodyId, usize> = HashMap::new();
        for (idx, body) in self.bodies.iter().enumerate() {
            if let Some(id) = body.mass_body_id {
                // Two SimBodies sharing a `mass_body_id` would silently
                // race on writeback — the second insert wins, the first
                // body's state ends up reflecting the second's tree
                // node. Fail loudly with both indices so the caller can
                // resolve the duplicate registration.
                if let Some(prev_idx) = sim_body_for_id.insert(id, idx) {
                    panic!(
                        "propagate_kinematic_state: mass_body_id {id:?} is shared by \
                         SimBody {prev_idx} and SimBody {idx}. Each tree node may back \
                         at most one SimBody — call `Simulation::add_body_to_tree` once \
                         per body, or clear the duplicate registration."
                    );
                }
            }
            if body.kinematic_only {
                let id = body.mass_body_id.unwrap_or_else(|| {
                    panic!(
                        "propagate_kinematic_state: SimBody {idx} is flagged \
                         `kinematic_only` but has no `mass_body_id`. Either call \
                         `Simulation::add_body_to_tree({idx}, ...)` and \
                         `Simulation::attach({idx}, ...)` first, or clear the \
                         kinematic flag — kinematic-only bodies must be non-root \
                         nodes in the mass tree."
                    )
                });
                assert!(
                    tree.parent(id).is_some(),
                    "propagate_kinematic_state: SimBody {idx} (mass_body_id {id:?}) is \
                     flagged `kinematic_only` but resolves to a tree root. \
                     Kinematic-only bodies must have a parent — call \
                     `Simulation::attach({idx}, parent_idx, ...)` first."
                );
                assert!(
                    body.rot.is_some(),
                    "propagate_kinematic_state: SimBody {idx} is flagged \
                     `kinematic_only` but has no `RotationalState`. Kinematic \
                     propagation derives both `trans` and `rot`; 3-DOF bodies \
                     cannot be kinematic children. Either set \
                     `VehicleConfig::rot = Some(...)` or clear the flag."
                );
            }
        }

        // Build the per-node KinematicNodeState map. Every storage
        // entity needs an entry — including non-root children whose
        // `rot` / `trans` are about to be overwritten — because the
        // kernel still reads each node's `t_struct_body` /
        // `composite_in_struct` when routing through the child's
        // structural frame.
        let n = tree.node_count();
        let mut nodes: HashMap<MassBodyId, KinematicNodeState> = HashMap::with_capacity(n);
        for id in 0..n {
            // Pull `composite_properties` directly off the arena — the
            // `MassStorage::node()` view exposes only `core_properties` /
            // `structure_point`, so the kinematic walk reads the
            // composite slots through `MassTree::get` instead.
            let composite_in_struct = tree.get(id).composite_properties.position;
            let t_struct_body = tree.get(id).composite_properties.t_parent_this;

            let (rot, trans) = if let Some(&body_idx) = sim_body_for_id.get(&id) {
                // SimBody-backed entries seed the walk from the
                // integrator's most recent output.
                let body = &self.bodies[body_idx];
                let rot = body.rot.unwrap_or_default();
                let trans = body.trans.to_untyped();
                (rot, trans)
            } else {
                // Tree-only nodes (the common case for assemblies like
                // Apollo's launch stack, where only the integrated root
                // lives as a SimBody) have no integrated state of their
                // own; default to zero so the kernel reads them
                // through the parent. They are *output-only* nodes —
                // we propagate to them but don't write back anywhere
                // since no SimBody owns their state.
                (RotationalState::default(), TranslationalState::default())
            };

            nodes.insert(
                id,
                KinematicNodeState {
                    rot,
                    trans,
                    t_struct_body,
                    composite_in_struct,
                },
            );
        }

        // Build the per-edge KinematicEdge map directly from the
        // mass tree's `structure_point` — same edge geometry the
        // Bevy adapter pulls from `MassChildOf` (whose fields are
        // populated from `MassTree::attach`'s offset + rotation
        // arguments).
        let mut edges: HashMap<MassBodyId, KinematicEdge> = HashMap::with_capacity(n);
        for id in 0..n {
            if MassStorage::parent(tree, id).is_some() {
                let view = MassStorage::node(tree, id);
                edges.insert(
                    id,
                    KinematicEdge {
                        t_parent_child: view.structure_point.t_parent_this,
                        offset_in_pstr: view.structure_point.position,
                    },
                );
            }
        }

        // Run the orchestration walk. Pre-order, root → leaves; every
        // non-root child's outputs are derived from the just-written
        // parent state.
        let derived = propagate_state_via_storage(tree, &nodes, &edges);

        // Write back into every kinematic-only SimBody whose tree node
        // is non-root. Roots integrate under their own dynamics and
        // overwriting their state would stomp the integrator's output.
        for (mass_id, state) in &derived {
            // Skip roots (no parent → integrated, not derived).
            if MassStorage::parent(tree, *mass_id).is_none() {
                continue;
            }
            let Some(&body_idx) = sim_body_for_id.get(mass_id) else {
                continue;
            };
            // Only write the body's state if it opted in. A non-root
            // SimBody without the flag (for example, a child with its
            // own integration) keeps whatever the integrator produced —
            // matching JEOD's `compute_point_derivative` semantics where
            // children-of-roots are kinematic by default but a flag
            // upgrades them to fully integrated.
            //
            // We skip the writeback (rather than asserting) because
            // mission code may legitimately register a child SimBody
            // for force introspection without flagging it kinematic-
            // only — the asserts at the top of this method enforce the
            // *forward* contract (kinematic-only ⇒ non-root tree
            // member).
            if !self.bodies[body_idx].kinematic_only {
                continue;
            }
            self.bodies[body_idx].trans =
                TranslationalStateTyped::<IntegrationFrame>::from_untyped_unchecked(&state.trans);
            // body.rot is `Option<RotationalState>` — we already
            // asserted at the top of the method that kinematic-only
            // bodies carry one.
            self.bodies[body_idx].rot = Some(state.rot);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::SimulationBuilderExt;
    use glam::{DMat3, DVec3};
    use jeod_sim::{
        recipes::Mission, GravityControls, JeodQuat, MassProperties, RotationalState,
        TranslationalState, VehicleConfig,
    };

    /// Two-body chain (root + 1 kinematic child) with a non-identity
    /// `t_parent_child` rotation. The child's `body.trans/rot` must
    /// equal the analytical answer of "parent state composed with
    /// link" after the first `step()`.
    #[test]
    fn rotated_attach_child_tracks_parent_through_link() {
        let mut builder = Mission::iss_leo().into_builder();
        builder.dt = 0.1;
        let mut sim = builder.build().expect("Mission::iss_leo must validate");

        // Parent: 60° about Z, ω = (0, 0, 1e-3) rad/s; planet-Earth
        // mass+state already comes from the recipe.
        let parent_q =
            JeodQuat::left_quat_from_eigen_rotation(std::f64::consts::FRAC_PI_3, DVec3::Z);
        let parent_omega = DVec3::new(0.0, 0.0, 1e-3);
        // Mutate the existing root body to install a non-identity
        // rotational state — `iss_leo` ships with identity quat / zero
        // rate by default. The recipe gives us a 6-DOF body already.
        sim.bodies[0].rot = Some(RotationalState {
            quaternion: parent_q,
            ang_vel_body: parent_omega,
        });

        // Register the parent body in the mass tree as a root.
        let parent_id = sim.add_body_to_tree(0, "parent");

        // Add the kinematic child as a *new* SimBody (so propagation
        // has an entity to write back into). Initial state is junk —
        // propagation must overwrite both within the first step.
        let child_idx = sim.add_body(VehicleConfig {
            trans: TranslationalState {
                position: DVec3::new(1e9, 1e9, 1e9),
                velocity: DVec3::new(1e9, 1e9, 1e9),
            },
            rot: Some(RotationalState::default()),
            mass: Some(MassProperties::new(5.0)),
            gravity_controls: GravityControls { controls: vec![] },
            ..Default::default()
        });

        let _ = sim.add_body_to_tree(child_idx, "child");
        // Link: 30° about Z, offset (1, 0, 0).
        let t_pc = JeodQuat::left_quat_from_eigen_rotation(std::f64::consts::FRAC_PI_6, DVec3::Z)
            .left_quat_to_transformation();
        let offset = DVec3::new(1.0, 0.0, 0.0);
        sim.attach(child_idx, 0, offset, t_pc);
        sim.mark_kinematic_only(child_idx);

        // One step exercises `propagate_kinematic_state` end-to-end.
        sim.step().expect("step() must succeed");

        // Expected child T_inertial_body = T_pc · T_inertial_body_parent
        let parent_t_ib = parent_q.left_quat_to_transformation();
        let expected_child_t = t_pc * parent_t_ib;
        let child_q = sim.body(child_idx).rot.expect("child has rot").quaternion;
        let actual_child_t = child_q.left_quat_to_transformation();
        let mat_diff = (actual_child_t.x_axis - expected_child_t.x_axis).length()
            + (actual_child_t.y_axis - expected_child_t.y_axis).length()
            + (actual_child_t.z_axis - expected_child_t.z_axis).length();
        assert!(
            mat_diff < 1e-9,
            "child T_inertial_body mismatch: expected {expected_child_t:?}, got {actual_child_t:?}"
        );

        // Position composition routes through the kernel's
        // parent_struct → child_struct → child_body shift. We build
        // the same `pcm_to_ccm` and `T_inertial_struct.transpose()`
        // chain the kernel uses to verify end-to-end correctness.
        // Parent's composite CoM in parent struct is the mass-weighted
        // CoM of (parent at origin) and (child at `offset`); read both
        // masses from the tree so the test stays valid if the recipe's
        // parent mass changes.
        let parent_mass_kg = sim.bodies[0]
            .mass
            .expect("recipe ships parent with mass")
            .mass;
        let child_mass_kg = sim.bodies[child_idx]
            .mass
            .expect("child config sets mass")
            .mass;
        let composite_total = parent_mass_kg + child_mass_kg;
        // Parent's struct origin holds its own CoM (atomic body), so
        // the only off-origin contribution to the combined CoM is the
        // child at `offset`.
        let parent_composite_in_pstr = offset * (child_mass_kg / composite_total);
        // Child has zero composite_in_cstr (atomic 5 kg point at struct
        // origin), so `T_pc^T · child_composite_in_cstr` vanishes.
        let pcm_to_ccm = offset - parent_composite_in_pstr;

        // Read the parent's *post-step* state for the comparison.
        // Propagation runs both before and after integration in the
        // runner pipeline, and the post-integration call is what
        // `Simulation::body` ultimately observes — so the expected
        // child state must compose with the parent's freshly-
        // integrated state.
        let post_parent = sim.body(0);
        let post_parent_pos = post_parent.trans.position;
        let post_parent_vel = post_parent.trans.velocity;
        let post_parent_q = post_parent.rot.expect("parent has rot").quaternion;
        let post_parent_omega = post_parent.rot.unwrap().ang_vel_body;
        let post_parent_t_ib = post_parent_q.left_quat_to_transformation();
        // Parent has identity struct→body ⇒ T_inertial_struct =
        // T_inertial_body. The kernel rotates pcm_to_ccm into inertial
        // via `T_inertial_struct.transpose()`.
        let expected_offset_inertial = post_parent_t_ib.transpose() * pcm_to_ccm;

        let expected_child_pos = post_parent_pos + expected_offset_inertial;
        let child_pos = sim.body(child_idx).trans.position;
        let pos_err = (child_pos - expected_child_pos).length();
        assert!(
            pos_err < 1e-6,
            "child position mismatch: expected {expected_child_pos:?}, got {child_pos:?}; \
             pcm_to_ccm={pcm_to_ccm:?}, expected_offset={expected_offset_inertial:?}"
        );

        // Velocity composition: v_child = v_parent + ω_inertial × r,
        // with ω_inertial = T_inertial_body^T · ω_body.
        let omega_inertial = post_parent_t_ib.transpose() * post_parent_omega;
        let expected_child_vel = post_parent_vel + omega_inertial.cross(expected_offset_inertial);
        let child_vel = sim.body(child_idx).trans.velocity;
        let vel_err = (child_vel - expected_child_vel).length();
        assert!(
            vel_err < 1e-6,
            "child velocity mismatch: expected {expected_child_vel:?}, got {child_vel:?}; \
             post_parent_vel={post_parent_vel:?}, omega_inertial={omega_inertial:?}, \
             offset={expected_offset_inertial:?}"
        );

        // Suppress unused-variables lints from `parent_t_ib` /
        // `parent_omega` if the optimiser folds the test setup.
        let _ = (parent_id, parent_t_ib, parent_omega);
    }

    /// `mark_kinematic_only` must reject a body that has no
    /// `mass_body_id` (not in the tree).
    #[test]
    #[should_panic(expected = "not in the mass tree")]
    fn mark_kinematic_only_panics_for_body_outside_tree() {
        let mut sim = Mission::iss_leo()
            .into_builder()
            .build()
            .expect("Mission::iss_leo must validate");
        // Body 0 is not yet registered in any mass tree.
        sim.mark_kinematic_only(0);
    }

    /// `mark_kinematic_only` must reject a body that resolves to a tree
    /// root.
    #[test]
    #[should_panic(expected = "is a tree root")]
    fn mark_kinematic_only_panics_for_tree_root() {
        let mut sim = Mission::iss_leo()
            .into_builder()
            .build()
            .expect("Mission::iss_leo must validate");
        sim.add_body_to_tree(0, "root");
        // Body 0 is now a tree root with no parent — flag must reject.
        sim.mark_kinematic_only(0);
    }

    /// `mark_kinematic_only` must reject a 3-DOF body (no rotational
    /// state) — propagation derives `rot` and the body would be left
    /// with a kinematic position but no attitude.
    #[test]
    #[should_panic(expected = "no RotationalState")]
    fn mark_kinematic_only_panics_for_3dof_body() {
        let mut sim = Mission::iss_leo()
            .into_builder()
            .build()
            .expect("Mission::iss_leo must validate");
        let parent_idx = 0;
        // The recipe ships a 6-DOF root, which is fine. Add a 3-DOF
        // child SimBody (rot: None) and try to mark it kinematic.
        let child_idx = sim.add_body(VehicleConfig {
            trans: TranslationalState::default(),
            rot: None,
            mass: Some(MassProperties::new(5.0)),
            gravity_controls: GravityControls { controls: vec![] },
            ..Default::default()
        });
        sim.add_body_to_tree(parent_idx, "parent");
        sim.add_body_to_tree(child_idx, "child");
        sim.attach(child_idx, parent_idx, DVec3::ZERO, DMat3::IDENTITY);
        sim.mark_kinematic_only(child_idx);
    }

    /// Two `SimBody`s sharing a `mass_body_id` would silently race on
    /// writeback inside `propagate_kinematic_state`. The check at the
    /// top of the pass must fail loudly.
    #[test]
    #[should_panic(expected = "is shared by SimBody")]
    fn propagate_kinematic_state_panics_on_duplicate_mass_body_id() {
        let mut sim = Mission::iss_leo()
            .into_builder()
            .build()
            .expect("Mission::iss_leo must validate");
        let parent_id = sim.add_body_to_tree(0, "parent");
        let child_idx = sim.add_body(VehicleConfig {
            trans: TranslationalState::default(),
            rot: Some(RotationalState::default()),
            mass: Some(MassProperties::new(5.0)),
            gravity_controls: GravityControls { controls: vec![] },
            ..Default::default()
        });
        sim.add_body_to_tree(child_idx, "child");
        sim.attach(child_idx, 0, DVec3::ZERO, DMat3::IDENTITY);
        // Force the duplicate-id condition the public API doesn't
        // expose — a future bug in `add_body_to_tree` could leak it,
        // and the propagation pass must catch it regardless.
        sim.bodies[child_idx].mass_body_id = Some(parent_id);
        sim.step()
            .expect("step until propagate_kinematic_state runs");
    }
}
