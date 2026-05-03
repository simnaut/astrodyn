//! Mass-tree topology and detached-subtree machinery for [`super::Simulation`].
//!
//! Carries the bigger attach/detach methods that previously lived in
//! `lib.rs` (~600 lines): `add_body_to_tree`, `attach`, `detach`,
//! `detach_subtree` (~200 lines), `attach_subtree_aligned` (~240 lines,
//! ports JEOD's `DynBody::attach_child` momentum-conservation
//! algorithm), `step_detached_subtrees`, and the
//! `subtree_composite_inertial` chain-walk accessor.

use glam::{DMat3, DVec3};

use jeod_dynamics::{combine_states_at_attach, AttachCombineInputs, MassBodyId, MassPointState};
use jeod_frames::{RefFrameRot, RefFrameState, RefFrameTrans};
use jeod_sim::{IntegrationFrame, RotationalState, TranslationalState, TranslationalStateTyped};

use jeod_dynamics::DetachedSubtreeState;

use super::Simulation;

impl Simulation {
    /// Register a body in the simulation's mass tree.
    ///
    /// Creates (or reuses) a `MassTree` and adds the body's mass as a node.
    /// Returns the `MassBodyId` for use with [`attach`](Self::attach) and
    /// [`detach`](Self::detach). The body's `mass` field must be `Some`.
    ///
    /// # Panics
    /// Panics if the body has no mass properties.
    pub fn add_body_to_tree(
        &mut self,
        body_idx: usize,
        name: impl Into<String>,
    ) -> jeod_dynamics::MassBodyId {
        let mass = self.bodies[body_idx]
            .mass
            .expect("add_body_to_tree requires mass properties");
        let tree = self
            .mass_tree
            .get_or_insert_with(jeod_dynamics::MassTree::new);
        let id = tree.add_body(name.into(), mass);
        self.bodies[body_idx].mass_body_id = Some(id);
        id
    }

    /// Attach a child body to a parent body in the mass tree.
    ///
    /// Both bodies must have been registered via [`add_body_to_tree`](Self::add_body_to_tree).
    /// After attachment, every ancestor's composite mass properties are
    /// updated automatically (via `MassTree::recompute_composites`). The
    /// integrators of every body whose composite changed (the child plus
    /// the parent's full ancestor chain to the root) are then reset to
    /// match JEOD's `dyn_body_attach.cc::reset_integrators()` semantics.
    ///
    /// # Panics
    /// Panics if either body is not in the tree, or if the child already has a parent.
    pub fn attach(
        &mut self,
        child_idx: usize,
        parent_idx: usize,
        offset: DVec3,
        t_parent_child: DMat3,
    ) {
        let child_id = self.bodies[child_idx]
            .mass_body_id
            .expect("attach: child body not in mass tree");
        let parent_id = self.bodies[parent_idx]
            .mass_body_id
            .expect("attach: parent body not in mass tree");

        // ── Site A: mark every body whose composite mass is about to
        //    change as topology-dirty. JEOD_INV: IG.37 — this is bound
        //    to the topology mutation itself; if we ever do this in a
        //    new method but forget the matching reset (Site B below),
        //    the dirty flag remains set and `integrate()` panics on
        //    the next step with the IG.37 diagnostic. The set of
        //    affected bodies after `attach` is the child plus the
        //    parent's full ancestor chain (since `MassTree::attach`
        //    walks `recompute_composites` from leaves to every root,
        //    so any ancestor of the new parent is touched).
        //
        // The set is sorted + deduped so the helpers below (and the
        // mass-sync pass) can use `binary_search` for O(log n)
        // membership instead of a linear `Vec::contains` scan,
        // mirroring the Bevy path's affected-id discipline (issue
        // #274 / PR #282 review thread `PRRT_kwDORtae6c5_KoAT`).
        let mut affected_ids: Vec<jeod_dynamics::MassBodyId> = vec![child_id];
        {
            let tree_ro = self.mass_tree.as_ref().expect("attach: no mass tree");
            affected_ids.extend(tree_ro.ancestors_inclusive(parent_id));
        }
        affected_ids.sort_unstable();
        affected_ids.dedup();
        Self::mark_body_integrators_dirty_by_id(&mut self.bodies, &affected_ids);

        // ── Mutate the tree itself. ──
        let tree = self.mass_tree.as_mut().expect("attach: no mass tree");
        tree.attach(child_id, parent_id, offset, t_parent_child);
        // Sync every affected body's composite mass from the tree.
        // `affected_ids` is sorted + deduped above; binary_search keeps
        // this O(n_bodies · log n_affected) instead of O(n²).
        for body in self.bodies.iter_mut() {
            if let Some(id) = body.mass_body_id {
                if affected_ids.binary_search(&id).is_ok() {
                    body.mass = Some(tree.get(id).composite_properties);
                }
            }
        }

        // ── Site B: reset the integrator history. Separate from Site A
        //    so a regression that drops this call leaves the dirty flag
        //    set (IG.37 panics on next integrate). Mirrors JEOD's
        //    `dyn_body_attach.cc::reset_integrators()` (lines 860, 871).
        Self::reset_body_integrators_by_id(&mut self.bodies, &affected_ids);
    }

    /// Detach a child body from its parent in the mass tree.
    ///
    /// After detachment, the former parent's *and every one of its
    /// ancestors'* composite mass properties are updated from the
    /// tree's recomputed composites (mirroring
    /// `MassTree::recompute_composites`). The child becomes a root and
    /// its own composite is also updated.
    ///
    /// # Panics
    /// Panics if the body is not in the tree or has no parent.
    pub fn detach(&mut self, child_idx: usize) {
        let child_id = self.bodies[child_idx]
            .mass_body_id
            .expect("detach: child body not in mass tree");

        // ── Site A: mark every affected body's integrators dirty
        //    BEFORE we mutate the tree. The set of bodies whose
        //    composite changes after detach is the (former) child plus
        //    the former parent's full ancestor chain, since
        //    `MassTree::detach` recomputes composites bottom-up over
        //    every tree root (the new tree containing the parent and
        //    the new tree containing the freshly detached child).
        //
        // The set is sorted + deduped so the helpers below (and the
        // mass-sync pass) can use `binary_search` for O(log n)
        // membership instead of a linear `Vec::contains` scan,
        // mirroring the Bevy path's affected-id discipline (issue
        // #274 / PR #282 review thread `PRRT_kwDORtae6c5_KoAT`).
        let mut affected_ids: Vec<jeod_dynamics::MassBodyId> = vec![child_id];
        let parent_id = {
            let tree_ro = self.mass_tree.as_ref().expect("detach: no mass tree");
            let pid = tree_ro
                .parent(child_id)
                .expect("detach: child body has no parent in tree");
            affected_ids.extend(tree_ro.ancestors_inclusive(pid));
            pid
        };
        affected_ids.sort_unstable();
        affected_ids.dedup();
        Self::mark_body_integrators_dirty_by_id(&mut self.bodies, &affected_ids);

        // ── Mutate the tree. ──
        let tree = self.mass_tree.as_mut().expect("detach: no mass tree");
        tree.detach(child_id);
        // Sync mass on every affected body from the recomputed tree.
        // `affected_ids` is sorted + deduped above; binary_search keeps
        // this O(n_bodies · log n_affected) instead of O(n²).
        for body in self.bodies.iter_mut() {
            if let Some(id) = body.mass_body_id {
                if affected_ids.binary_search(&id).is_ok() {
                    body.mass = Some(tree.get(id).composite_properties);
                }
            }
        }
        let _ = parent_id; // silence unused if no parent_idx lookup is needed below

        // ── Site B: reset integrator history. Mirrors JEOD's
        //    `dyn_body_detach.cc:271-273` `reset_integrators()` call.
        //    Separated from Site A so a future regression that drops
        //    this call leaves the dirty bit set on every affected body
        //    and panics in IG.37 on the next integrate.
        Self::reset_body_integrators_by_id(&mut self.bodies, &affected_ids);
    }

    /// Mark every Simulation body whose `mass_body_id` is in
    /// `affected_ids` as having stale multi-step integrator history.
    ///
    /// Called from each topology-mutation site (attach / detach /
    /// detach_subtree / attach_subtree_aligned) **before** the
    /// matching `reset_body_integrators_by_id` call. The two-step
    /// pattern is deliberate: if a future code path adds a new
    /// topology mutation and remembers to mark dirty but forgets the
    /// reset, the dirty flag stays set and `integrate()` panics on
    /// the next step with the IG.37 diagnostic. RK4 / RKF4(5) bodies
    /// have no integrator state and are silently skipped.
    ///
    /// Mirrors JEOD's `dyn_body_attach.cc::reset_integrators()` (lines
    /// 860, 871) and `dyn_body_detach.cc:271-273`.
    ///
    /// `affected_ids` **must be sorted in ascending order and
    /// deduplicated** so the inner membership check can use
    /// `binary_search` (O(log n)) instead of `Vec::contains` (O(n)).
    /// All four call sites in this module construct it via
    /// `sort_unstable + dedup`; a `debug_assert` enforces the
    /// invariant in debug builds. Issue #274 / PR #282 review thread
    /// `PRRT_kwDORtae6c5_KoAT`.
    // JEOD_INV: IG.37 — multi-step integrator history must be reset on topology change
    fn mark_body_integrators_dirty_by_id(
        bodies: &mut [super::types::SimBody],
        affected_ids: &[jeod_dynamics::MassBodyId],
    ) {
        debug_assert!(
            affected_ids.windows(2).all(|w| w[0] < w[1]),
            "mark_body_integrators_dirty_by_id requires affected_ids \
             sorted ascending and deduplicated for binary_search lookup"
        );
        for body in bodies.iter_mut() {
            let Some(id) = body.mass_body_id else {
                continue;
            };
            if affected_ids.binary_search(&id).is_err() {
                continue;
            }
            if let Some(ref mut gj) = body.gj_state {
                gj.mark_topology_dirty();
            }
            if let Some(ref mut abm) = body.abm4_state {
                abm.mark_topology_dirty();
            }
        }
    }

    /// Reset multi-step integrator history on every Simulation body
    /// whose `mass_body_id` is in `affected_ids` and clear the dirty
    /// flag. Pair this with `mark_body_integrators_dirty_by_id` at the
    /// same call site — never collapse the two into one helper, since
    /// the temporal separation is what makes IG.37 fail-loud.
    ///
    /// Mirrors JEOD's `dyn_body_attach.cc::reset_integrators()` and
    /// `dyn_body_detach.cc:271-273`.
    ///
    /// `affected_ids` **must be sorted in ascending order and
    /// deduplicated** (same precondition as
    /// `mark_body_integrators_dirty_by_id`).
    // JEOD_INV: IG.37 — multi-step integrator history must be reset on topology change
    fn reset_body_integrators_by_id(
        bodies: &mut [super::types::SimBody],
        affected_ids: &[jeod_dynamics::MassBodyId],
    ) {
        debug_assert!(
            affected_ids.windows(2).all(|w| w[0] < w[1]),
            "reset_body_integrators_by_id requires affected_ids \
             sorted ascending and deduplicated for binary_search lookup"
        );
        for body in bodies.iter_mut() {
            let Some(id) = body.mass_body_id else {
                continue;
            };
            if affected_ids.binary_search(&id).is_err() {
                continue;
            }
            jeod_sim::reset_integrators(body.gj_state.as_mut(), body.abm4_state.as_mut());
        }
    }

    /// Detach a tree-only subtree from its parent in the mass tree,
    /// capturing the subtree's composite-body inertial state at the
    /// moment of separation.
    ///
    /// The parent of the subtree may be either the integrated body's
    /// own mass-tree node or another *already-detached* subtree (whose
    /// state lives in [`Simulation::detached_subtrees`]). The method
    /// locates the parent automatically by walking up the tree from
    /// `subtree_root_id` to its root.
    ///
    /// `integrated_body_idx` is consulted only to identify the
    /// Simulation body whose mass-tree id matches the parent's tree
    /// root (when the parent is the integrated body).
    /// `subtree_root_id` is the [`MassBodyId`] being detached.
    ///
    /// At detach time the subtree shared the parent's rigid motion, so
    /// its composite-CoM inertial state is computed from the parent's
    /// state plus the offset between the two composites (drawn from
    /// the mass tree). The state is then stored on
    /// [`Simulation::detached_subtrees`] and propagated each step
    /// until [`attach_subtree_aligned`](Self::attach_subtree_aligned)
    /// re-attaches the subtree.
    ///
    /// If the parent is a detached subtree, its tracked composite state
    /// is updated to reflect the loss of mass (the parent's
    /// composite-CoM in inertial shifts when its mass distribution
    /// changes — even though the underlying structure point doesn't
    /// move). If the parent is the integrated body, `body.trans`
    /// (the integrated `composite_body` inertial state, post-`bd279c2`)
    /// is shifted by the inertial-frame composite-CoM delta so it
    /// continues to track the new (smaller) composite, and the body's
    /// mass is re-synced from the recomputed composite_properties.
    ///
    /// # Panics
    /// Panics if no mass tree is configured, the subtree id is not in
    /// the tree, the subtree has no parent, the parent's root has no
    /// tracked state, or a subtree with the same id is already in the
    /// detached map.
    pub fn detach_subtree(&mut self, integrated_body_idx: usize, subtree_root_id: MassBodyId) {
        let tree = self
            .mass_tree
            .as_ref()
            .expect("detach_subtree: no mass tree configured");
        // Walk up to find the root of subtree_root_id's current tree.
        tree.parent(subtree_root_id)
            .expect("detach_subtree: subtree has no parent in tree");
        let mut tree_root_id = subtree_root_id;
        while let Some(p) = tree.parent(tree_root_id) {
            tree_root_id = p;
        }

        // The parent's pre-detach composite-CoM offset in its own struct frame.
        let parent_pre_composite_props = tree.get(tree_root_id).composite_properties;

        // Determine where the parent's state lives — either an integrated
        // Simulation body or a detached subtree.
        let integrated_mass_body_id = self.bodies[integrated_body_idx].mass_body_id;
        let parent_is_integrated = integrated_mass_body_id == Some(tree_root_id);

        // Pre-detach inertial composite_body state of the parent.
        // body.trans / body.rot represent the integrated body's
        // composite_body inertial state (matching JEOD's integration
        // variable; see `attach_subtree_aligned` and the "Integration
        // target" note on `Simulation`).
        let parent_composite_state: RefFrameState = if parent_is_integrated {
            let body_trans = self.bodies[integrated_body_idx].trans;
            let body_rot = self.bodies[integrated_body_idx]
                .rot
                .expect("detach_subtree: 6-DOF integrated body required");
            RefFrameState {
                trans: RefFrameTrans {
                    position: body_trans.position.raw_si(),
                    velocity: body_trans.velocity.raw_si(),
                },
                rot: RefFrameRot {
                    q_parent_this: body_rot.quaternion,
                    t_parent_this: body_rot.quaternion.left_quat_to_transformation(),
                    ang_vel_this: body_rot.ang_vel_body,
                },
            }
        } else {
            // Parent is itself a detached subtree.
            let detached = self
                .detached_subtrees
                .get(&tree_root_id)
                .unwrap_or_else(|| {
                    panic!(
                        "detach_subtree: parent tree-root {tree_root_id:?} of \
                         subtree {subtree_root_id:?} has no tracked state — \
                         did you forget to call detach_subtree on it first?"
                    )
                });
            detached.to_ref_frame_state()
        };

        // Walk down the tree from the root to the subtree, applying
        // propagate_forward at each level. This handles arbitrary tree
        // depth (e.g. cm → sm → s3 → lm where the subtree being
        // detached is several levels below the root). Each level uses
        // the immediate-parent-struct-frame `composite_wrt_pstr` from
        // the mass tree.
        let mut chain = Vec::<MassBodyId>::new();
        let mut current_id = subtree_root_id;
        while current_id != tree_root_id {
            chain.push(current_id);
            current_id = tree
                .parent(current_id)
                .expect("detach_subtree: chain walk hit a parentless intermediate");
        }
        chain.reverse(); // tree_root → ... → subtree
        let mut current_state = parent_composite_state;
        let mut current_node_id = tree_root_id;
        for next_id in chain {
            let next_node = tree.get(next_id);
            let current_node = tree.get(current_node_id);
            // Body-aware step:
            //   offset_in_current_body = T_current_struct_to_body
            //                          · (next.composite_wrt_pstr.position
            //                             − current.composite_properties.position)
            //   T_current_body_to_next_body = T_next_struct_to_body
            //                               · next.structure_point.t_parent_this
            //                               · T_current_body_to_struct
            // For axis-aligned bodies (struct == body) this collapses to
            // the simple struct-frame difference and the bare structure_point
            // rotation.
            let t_current_struct_to_body = current_node.composite_properties.t_parent_this;
            let t_next_struct_to_body = next_node.composite_properties.t_parent_this;
            let offset_struct =
                next_node.composite_wrt_pstr.position - current_node.composite_properties.position;
            let offset_in_current_body = t_current_struct_to_body * offset_struct;
            let t_current_body_to_next_body = t_next_struct_to_body
                * next_node.structure_point.t_parent_this
                * t_current_struct_to_body.transpose();
            let rel = MassPointState {
                position: offset_in_current_body,
                t_parent_this: t_current_body_to_next_body,
            };
            current_state = jeod_dynamics::propagate_forward(&current_state, &rel);
            current_node_id = next_id;
        }
        let subtree_state = current_state;

        // Apply the topology change — this also recomputes parent's
        // composite_properties (now without the subtree).
        let tree = self.mass_tree.as_mut().unwrap();
        tree.detach(subtree_root_id);
        let parent_post_composite_props = tree.get(tree_root_id).composite_properties;

        if parent_is_integrated {
            // body.trans/body.rot are the integrated composite_body
            // state. JEOD's detach handler preserves core_body
            // (Pos_Vel_Att_Rate source = core_body) and rederives the
            // post-detach composite from it. The composite-CoM offset
            // in the parent's struct frame shifts when the subtree
            // leaves, so the inertial composite_body position +
            // velocity must shift by the corresponding kinematic
            // offset. Rotation/ang_vel are unchanged because
            // composite_properties.t_parent_this == core_properties
            // .t_parent_this throughout (see mass tree recompute).
            let cm_delta_struct =
                parent_post_composite_props.position - parent_pre_composite_props.position;
            // composite_properties.t_parent_this is the struct→body
            // rotation; compose with body.rot's t_parent_this to map
            // struct → inertial.
            let t_struct_to_body = parent_pre_composite_props.t_parent_this;
            let cm_delta_body = t_struct_to_body * cm_delta_struct;
            let t_inertial_to_body = parent_composite_state.rot.t_parent_this;
            let cm_delta_inertial = t_inertial_to_body.transpose() * cm_delta_body;
            // Velocity offset from rigid-body rotation: ω × Δr in body
            // frame, then rotated to inertial.
            let omega_body = parent_composite_state.rot.ang_vel_this;
            let dvel_inertial = t_inertial_to_body.transpose() * omega_body.cross(cm_delta_body);
            self.bodies[integrated_body_idx].trans =
                TranslationalStateTyped::<IntegrationFrame>::from_untyped_unchecked(
                    &TranslationalState {
                        position: parent_composite_state.trans.position + cm_delta_inertial,
                        velocity: parent_composite_state.trans.velocity + dvel_inertial,
                    },
                );
            // body.rot unchanged — composite/core share body axes.
            self.bodies[integrated_body_idx].mass = Some(parent_post_composite_props);
        } else {
            // Parent is a detached subtree — update its tracked
            // composite-body state to reflect the new (smaller) composite.
            // The parent's struct origin in inertial is unchanged (rigid
            // body); only the composite-CoM has moved within the struct
            // frame. Convert the struct-frame CoM-delta to inertial via
            //
            //   T_inertial_to_struct = T_struct_to_body^T · T_inertial_to_body
            //
            // (matching `jeod_dynamics::compute_t_inertial_struct`). The
            // earlier form `T_struct_to_body * T_inertial_to_body` was
            // only correct when `T_struct_to_body` is symmetric (identity
            // or yaw_180) and silently produced wrong CoM-shift directions
            // for non-symmetric mass-tree orientations.
            let cm_delta_struct =
                parent_post_composite_props.position - parent_pre_composite_props.position;
            let t_struct_to_body = parent_pre_composite_props.t_parent_this;
            let t_inertial_struct = jeod_dynamics::compute_t_inertial_struct(
                &t_struct_to_body,
                &parent_composite_state.rot.t_parent_this,
            );
            let r_struct_to_inertial = t_inertial_struct.transpose();
            let cm_delta_inertial = r_struct_to_inertial * cm_delta_struct;
            // Velocity contribution from rotation: ω × Δr expressed in
            // body frame, then rotated to inertial.
            let w_body = parent_composite_state.rot.ang_vel_this;
            let cm_delta_body = t_struct_to_body * cm_delta_struct;
            let dvel_inertial =
                parent_composite_state.rot.t_parent_this.transpose() * w_body.cross(cm_delta_body);
            let updated = DetachedSubtreeState {
                composite_position: parent_composite_state.trans.position + cm_delta_inertial,
                composite_velocity: parent_composite_state.trans.velocity + dvel_inertial,
                composite_attitude: DetachedSubtreeState::attitude_from_raw_jeod_quat(
                    parent_composite_state.rot.q_parent_this,
                ),
                composite_ang_vel_body: parent_composite_state.rot.ang_vel_this,
            };
            self.detached_subtrees.insert(tree_root_id, updated);
        }

        if std::env::var("APOLLO_TRACE").is_ok() {
            eprintln!(
                "DETACH: subtree {subtree_root_id:?} state stored:\n  pos={:?}\n  vel={:?}\n  ω={:?}",
                subtree_state.trans.position,
                subtree_state.trans.velocity,
                subtree_state.rot.ang_vel_this
            );
        }

        // Insert the new subtree's state into the detached map.
        let prior = self.detached_subtrees.insert(
            subtree_root_id,
            DetachedSubtreeState::from_ref_frame_state(&subtree_state),
        );
        assert!(
            prior.is_none(),
            "detach_subtree: subtree {subtree_root_id:?} was already in detached_subtrees \
             — call attach_subtree_aligned first or use a fresh subtree id"
        );

        // JEOD_INV: IG.37 — Multi-step integrators (GJ, ABM4) carry predictor
        // history that is invalidated by the topology change. Reset the
        // integrated body's integrators (it just lost the subtree's mass,
        // so its dynamics changed and `body.trans` was shifted).
        //
        // Mark + reset are split into two distinct call sites (rather
        // than one bundled helper) so a future code path that adds a
        // new subtree-mutation method and remembers Site A but forgets
        // Site B leaves the dirty bit set, panicking on next integrate.
        if parent_is_integrated {
            let body = &mut self.bodies[integrated_body_idx];
            // Site A: mark dirty.
            if let Some(ref mut gj) = body.gj_state {
                gj.mark_topology_dirty();
            }
            if let Some(ref mut abm) = body.abm4_state {
                abm.mark_topology_dirty();
            }
            // Site B: reset history (separate observation site).
            jeod_sim::reset_integrators(body.gj_state.as_mut(), body.abm4_state.as_mut());
        }
    }

    /// Re-attach a previously-detached subtree to the integrated body's
    /// mass tree using named mass points (matching JEOD's
    /// `attach_aligned`), then update the integrated body's state via
    /// JEOD's [`combine_states_at_attach`] momentum-conservation
    /// algorithm.
    ///
    /// The integrated body's `body.trans` / `body.rot` represent the
    /// *composite_body* inertial state of the whole mass tree rooted
    /// at the integrated body — i.e. the integration variable that
    /// JEOD's `DynamicsIntegrationGroup::gravitation()` evaluates
    /// gravity at and that `DynBody::trans_integ()` integrates. The
    /// subtree state from [`Simulation::detached_subtrees`] is the
    /// subtree's composite_body frame. After the algorithm runs, the
    /// integrated body's `trans` / `rot` are set to the new combined
    /// composite_body inertial state.
    ///
    /// To compare against JEOD's logged core_body, derive core via
    /// [`Simulation::body_core_inertial`].
    ///
    /// # Panics
    /// Panics if the integrated body has no rotational state, no mass
    /// tree is configured, the parent or subtree id is not in the tree,
    /// either named mass point is missing on its body, or the subtree
    /// is not in [`Self::detached_subtrees`].
    pub fn attach_subtree_aligned(
        &mut self,
        integrated_body_idx: usize,
        subtree_root_id: MassBodyId,
        subtree_point: &str,
        parent_id: MassBodyId,
        parent_point: &str,
    ) {
        let tree = self
            .mass_tree
            .as_ref()
            .expect("attach_subtree_aligned: no mass tree configured");
        let integrated_mass_body_id = self.bodies[integrated_body_idx]
            .mass_body_id
            .expect("attach_subtree_aligned: integrated body not registered in mass tree");

        // Read pre-attach composite mass props of the integrated body
        // (= the whole pre-attach tree without the subtree).
        let parent_pre_composite_props = tree.get(integrated_mass_body_id).composite_properties;
        let orig_parent_cm_struct = parent_pre_composite_props.position;
        let core_wrt_composite_pre = tree.get(integrated_mass_body_id).core_wrt_composite;
        // Pre-attach subtree composite mass props.
        let subtree_composite_props = tree.get(subtree_root_id).composite_properties;

        // Read the integrated body's pre-attach composite_body state
        // directly from body.trans/body.rot (post-refactor convention).
        let body_trans = self.bodies[integrated_body_idx].trans;
        let body_rot = self.bodies[integrated_body_idx]
            .rot
            .expect("attach_subtree_aligned: 6-DOF body required");
        let parent_composite_pre = RefFrameState {
            trans: RefFrameTrans {
                position: body_trans.position.raw_si(),
                velocity: body_trans.velocity.raw_si(),
            },
            rot: RefFrameRot {
                q_parent_this: body_rot.quaternion,
                t_parent_this: body_rot.quaternion.left_quat_to_transformation(),
                ang_vel_this: body_rot.ang_vel_body,
            },
        };
        let _ = core_wrt_composite_pre; // unused under composite-body convention

        // Borrow the subtree's free-flight composite state (don't remove
        // yet — we want the entry to survive if any of the operations
        // below panic, so the caller can retry / recover instead of
        // silently losing state).
        let subtree_state = *self
            .detached_subtrees
            .get(&subtree_root_id)
            .unwrap_or_else(|| {
                panic!(
                    "attach_subtree_aligned: subtree {subtree_root_id:?} is not in \
                     detached_subtrees — call detach_subtree first or pre-populate"
                )
            });
        let child_composite = subtree_state.to_ref_frame_state();

        // Apply the topology change (also recomputes composite props).
        let tree_mut = self.mass_tree.as_mut().unwrap();
        tree_mut.attach_aligned(subtree_root_id, subtree_point, parent_id, parent_point);
        // Read post-attach composite props.
        let combined_composite_props = tree_mut.get(integrated_mass_body_id).composite_properties;

        // The combine algorithm needs `T_inertial_to_struct`, which by
        // the standard frame-chain rule is
        //
        //   T_inertial_to_struct = T_struct_to_body^T · T_inertial_to_body
        //
        // (matching `jeod_dynamics::compute_t_inertial_struct` and JEOD's
        // `dyn_body_collect.cc` lines 219-221). composite_properties
        // .t_parent_this is the struct→body rotation. The earlier form
        // `T_struct_to_body * T_inertial_to_body` was only valid for
        // symmetric struct-to-body rotations (identity, yaw_180 — the
        // ones Apollo happens to use); non-symmetric vehicle orientations
        // would silently get a wrong torque arm in the combine algorithm.
        let t_struct_to_body = parent_pre_composite_props.t_parent_this;
        let parent_t_inertial_struct = jeod_dynamics::compute_t_inertial_struct(
            &t_struct_to_body,
            &parent_composite_pre.rot.t_parent_this,
        );

        // APOLLO_TRACE diagnostic: dump every input to combine_states_at_attach
        // so we can diff against JEOD ground truth. Gated by env var so the
        // regular test path is unaffected. (See #248 attach-bug investigation.)
        if std::env::var("APOLLO_TRACE").is_ok() {
            eprintln!("=== ATTACH TRACE (integrated body {integrated_body_idx} → subtree {subtree_root_id:?}) ===");
            eprintln!("  PARENT COMPOSITE (= our body.trans/body.rot):");
            eprintln!(
                "    pos    = [{:.10e} {:.10e} {:.10e}]",
                parent_composite_pre.trans.position.x,
                parent_composite_pre.trans.position.y,
                parent_composite_pre.trans.position.z
            );
            eprintln!(
                "    vel    = [{:.10e} {:.10e} {:.10e}]",
                parent_composite_pre.trans.velocity.x,
                parent_composite_pre.trans.velocity.y,
                parent_composite_pre.trans.velocity.z
            );
            eprintln!(
                "    q      = [{:.10e} {:.10e} {:.10e} {:.10e}]",
                parent_composite_pre.rot.q_parent_this.scalar(),
                parent_composite_pre.rot.q_parent_this.vector().x,
                parent_composite_pre.rot.q_parent_this.vector().y,
                parent_composite_pre.rot.q_parent_this.vector().z
            );
            eprintln!(
                "    ω_body = [{:.10e} {:.10e} {:.10e}]",
                parent_composite_pre.rot.ang_vel_this.x,
                parent_composite_pre.rot.ang_vel_this.y,
                parent_composite_pre.rot.ang_vel_this.z
            );
            eprintln!("  CHILD COMPOSITE (= detached_subtrees[{subtree_root_id:?}]):");
            eprintln!(
                "    pos    = [{:.10e} {:.10e} {:.10e}]",
                child_composite.trans.position.x,
                child_composite.trans.position.y,
                child_composite.trans.position.z
            );
            eprintln!(
                "    vel    = [{:.10e} {:.10e} {:.10e}]",
                child_composite.trans.velocity.x,
                child_composite.trans.velocity.y,
                child_composite.trans.velocity.z
            );
            eprintln!(
                "    q      = [{:.10e} {:.10e} {:.10e} {:.10e}]",
                child_composite.rot.q_parent_this.scalar(),
                child_composite.rot.q_parent_this.vector().x,
                child_composite.rot.q_parent_this.vector().y,
                child_composite.rot.q_parent_this.vector().z
            );
            eprintln!(
                "    ω_body = [{:.10e} {:.10e} {:.10e}]",
                child_composite.rot.ang_vel_this.x,
                child_composite.rot.ang_vel_this.y,
                child_composite.rot.ang_vel_this.z
            );
            eprintln!("  PARENT MASS (pre-attach):");
            eprintln!("    mass={:.10e}", parent_pre_composite_props.mass);
            eprintln!(
                "    pos_struct=[{:.10e} {:.10e} {:.10e}]",
                parent_pre_composite_props.position.x,
                parent_pre_composite_props.position.y,
                parent_pre_composite_props.position.z
            );
            eprintln!(
                "    inertia.diag=[{:.10e} {:.10e} {:.10e}]",
                parent_pre_composite_props.inertia.x_axis.x,
                parent_pre_composite_props.inertia.y_axis.y,
                parent_pre_composite_props.inertia.z_axis.z
            );
            eprintln!("  CHILD MASS (pre-attach):");
            eprintln!("    mass={:.10e}", subtree_composite_props.mass);
            eprintln!(
                "    pos_struct=[{:.10e} {:.10e} {:.10e}]",
                subtree_composite_props.position.x,
                subtree_composite_props.position.y,
                subtree_composite_props.position.z
            );
            eprintln!(
                "    inertia.diag=[{:.10e} {:.10e} {:.10e}]",
                subtree_composite_props.inertia.x_axis.x,
                subtree_composite_props.inertia.y_axis.y,
                subtree_composite_props.inertia.z_axis.z
            );
            eprintln!("  COMBINED MASS (post-attach):");
            eprintln!("    mass={:.10e}", combined_composite_props.mass);
            eprintln!(
                "    pos_struct=[{:.10e} {:.10e} {:.10e}]",
                combined_composite_props.position.x,
                combined_composite_props.position.y,
                combined_composite_props.position.z
            );
            eprintln!(
                "    inertia.diag=[{:.10e} {:.10e} {:.10e}]",
                combined_composite_props.inertia.x_axis.x,
                combined_composite_props.inertia.y_axis.y,
                combined_composite_props.inertia.z_axis.z
            );
            eprintln!(
                "  orig_parent_cm_struct=[{:.10e} {:.10e} {:.10e}]",
                orig_parent_cm_struct.x, orig_parent_cm_struct.y, orig_parent_cm_struct.z
            );
            eprintln!("=== end ATTACH TRACE ===");
        }

        // Run the JEOD combine algorithm.
        let combined = combine_states_at_attach(AttachCombineInputs {
            parent_composite: parent_composite_pre,
            parent_mass: parent_pre_composite_props,
            parent_t_inertial_struct,
            child_composite,
            child_mass: subtree_composite_props,
            combined_mass: combined_composite_props,
            orig_parent_cm_struct,
        });

        // The new whole-tree composite state is the integration target —
        // matches JEOD's `composite_body` post-attach (the source for
        // `Vel_Rate` per `set_state_source_internal` at the end of
        // `DynBody::attach_update_properties`).
        self.bodies[integrated_body_idx].trans =
            TranslationalStateTyped::<IntegrationFrame>::from_untyped_unchecked(
                &TranslationalState {
                    position: combined.composite_state.trans.position,
                    velocity: combined.composite_state.trans.velocity,
                },
            );
        self.bodies[integrated_body_idx].rot = Some(RotationalState {
            quaternion: combined.composite_state.rot.q_parent_this,
            ang_vel_body: combined.composite_state.rot.ang_vel_this,
        });
        self.bodies[integrated_body_idx].mass = Some(combined_composite_props);

        // Combine succeeded — only now remove the subtree's detached
        // entry. If any earlier step panicked (missing mass points,
        // combine preconditions, etc.) the entry survives so callers
        // can retry or inspect rather than silently losing state.
        self.detached_subtrees.remove(&subtree_root_id);

        // JEOD_INV: IG.37 — Multi-step integrators (GJ, ABM4) carry predictor
        // history that is invalidated by the topology + state combine.
        // Mirror JEOD's `dyn_body_attach.cc::reset_integrators()` for the
        // integrated body, whose `body.trans` / `body.rot` were just
        // overwritten by the combine.
        //
        // Split mark + reset across two adjacent-but-distinct call
        // sites: a future code path that adds another integrated-body
        // mutation and remembers Site A but forgets Site B leaves the
        // dirty flag set, so the next `integrate()` panics with the
        // IG.37 diagnostic instead of silently propagating stale
        // predictor history.
        let body = &mut self.bodies[integrated_body_idx];
        // Site A: mark integrators dirty.
        if let Some(ref mut gj) = body.gj_state {
            gj.mark_topology_dirty();
        }
        if let Some(ref mut abm) = body.abm4_state {
            abm.mark_topology_dirty();
        }
        // Site B: reset integrator history (separate observation site).
        jeod_sim::reset_integrators(body.gj_state.as_mut(), body.abm4_state.as_mut());

        if std::env::var("APOLLO_TRACE").is_ok() {
            eprintln!(
                "  COMBINE OUTPUT: pos=[{:.4e} {:.4e} {:.4e}] ω_body=[{:.6e} {:.6e} {:.6e}]",
                combined.composite_state.trans.position.x,
                combined.composite_state.trans.position.y,
                combined.composite_state.trans.position.z,
                combined.composite_state.rot.ang_vel_this.x,
                combined.composite_state.rot.ang_vel_this.y,
                combined.composite_state.rot.ang_vel_this.z
            );
        }
    }

    /// Advance every entry in [`Simulation::detached_subtrees`] by `dt`
    /// seconds. Each subtree propagates ballistically — no gravity, no
    /// torque — matching JEOD's behavior for tree roots whose
    /// `grav_interaction.controls` is empty (which is the case for
    /// every non-LES vehicle in `SIM_Apollo`).
    pub fn step_detached_subtrees(&mut self, dt: f64) {
        for state in self.detached_subtrees.values_mut() {
            state.step_ballistic(dt);
        }
    }
}

#[cfg(test)]
mod tests {
    //! Integration-level tests for the IG.37 wiring. These tests live in
    //! the runner crate so they can poke at `SimBody::gj_state` /
    //! `abm4_state` directly — the public `body()` accessor only exposes
    //! `VehicleOutput`, which omits integrator state.

    use super::*;
    use crate::Simulation;
    use jeod_dynamics::MassProperties;
    use jeod_sim::{
        GaussJacksonConfig, GravityControl, GravityControls, GravityModel, GravitySource,
        GravitySourceEntry, IntegratorType, SimulationTime, TranslationalState, VehicleConfig,
    };

    /// JEOD's `dyn_body_attach.cc::reset_integrators()` precedent: after an
    /// attach, both bodies' Gauss-Jackson predictor / corrector history
    /// must be reinitialized. We verify that:
    ///   1. After enough steps to leave priming, the GJ states are past
    ///      priming (`is_priming() == false`).
    ///   2. After `Simulation::attach`, both bodies' GJ states are back
    ///      in priming and the topology-dirty flag is cleared (which is
    ///      what `reset_for_topology_change` guarantees).
    ///   3. The simulation can take another step without tripping the
    ///      IG.37 assertion in `GaussJacksonState::integrate` — proving
    ///      the wiring closes the gap end-to-end.
    #[test]
    fn attach_resets_gauss_jackson_state_on_both_bodies() {
        const MU: f64 = 5.76e14;
        let dt = 1.0_f64;

        let trans_a = TranslationalState {
            position: DVec3::new(9e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 8000.0, 0.0),
        };
        // Slightly different orbit for body B so its predictor history
        // is non-trivially distinct from body A.
        let trans_b = TranslationalState {
            position: DVec3::new(9e6, 1.0, 0.0),
            velocity: DVec3::new(0.0, 7900.0, 0.0),
        };

        let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
        let mut sim = Simulation::new(time, dt);

        let earth = sim.add_source(
            "Earth",
            GravitySourceEntry {
                source: GravitySource {
                    mu: MU,
                    model: GravityModel::PointMass,
                },
                position: jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
                velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
                t_inertial_pfix: None,
                delta_c20: 0.0,
                rotation_model: crate::RotationModel::default(),
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
            },
        );

        let gj_cfg = GaussJacksonConfig::with_order(8);
        let make_cfg = |trans: TranslationalState| VehicleConfig {
            trans,
            integrator: IntegratorType::GaussJackson(gj_cfg),
            mass: Some(MassProperties::new(1000.0)),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, false)],
            },
            ..Default::default()
        };
        let body_a = sim.add_body(make_cfg(trans_a));
        let body_b = sim.add_body(make_cfg(trans_b));

        // Register both in the mass tree so we can attach later.
        let id_a = sim.add_body_to_tree(body_a, "BodyA");
        let id_b = sim.add_body_to_tree(body_b, "BodyB");

        sim.validate().expect("validate failed");

        // ── Step long enough to leave GJ priming on both bodies. ──
        // GJ8 needs ~50 stages to fully bootstrap; 200 sim steps is
        // comfortably past that.
        sim.step_n(200).expect("step_n failed");

        let gj_a_pre = sim.bodies[body_a]
            .gj_state
            .as_ref()
            .expect("body A must have gj_state after validate");
        let gj_b_pre = sim.bodies[body_b]
            .gj_state
            .as_ref()
            .expect("body B must have gj_state after validate");
        assert!(
            !gj_a_pre.is_priming(),
            "test setup expected body A's GJ state past priming after 200 steps"
        );
        assert!(
            !gj_b_pre.is_priming(),
            "test setup expected body B's GJ state past priming after 200 steps"
        );
        assert!(!gj_a_pre.is_topology_dirty());
        assert!(!gj_b_pre.is_topology_dirty());
        let _ = (id_a, id_b);

        // ── Attach: triggers IG.37 reset on both bodies. ──
        sim.attach(body_a, body_b, DVec3::ZERO, DMat3::IDENTITY);

        let gj_a_post = sim.bodies[body_a]
            .gj_state
            .as_ref()
            .expect("body A must still have gj_state");
        let gj_b_post = sim.bodies[body_b]
            .gj_state
            .as_ref()
            .expect("body B must still have gj_state");
        assert!(
            gj_a_post.is_priming(),
            "body A's GJ state must be back in priming after attach (IG.37)"
        );
        assert!(
            gj_b_post.is_priming(),
            "body B's GJ state must be back in priming after attach (IG.37)"
        );
        assert!(!gj_a_post.is_topology_dirty());
        assert!(!gj_b_post.is_topology_dirty());

        // ── Step once more: the IG.37 assertion in `integrate()` must
        // not fire. With our wiring it's cleared; without it, this would
        // panic with the "stale predictor/corrector history" message.
        sim.step_n(1).expect("post-attach step failed");
    }

    /// `Simulation::detach` must reset GJ state on both the parent and
    /// the detaching child. Mirrors JEOD's `dyn_body_detach.cc:271-273`.
    #[test]
    fn detach_resets_gauss_jackson_state_on_both_bodies() {
        const MU: f64 = 5.76e14;
        let dt = 1.0_f64;
        let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
        let mut sim = Simulation::new(time, dt);

        let earth = sim.add_source(
            "Earth",
            GravitySourceEntry {
                source: GravitySource {
                    mu: MU,
                    model: GravityModel::PointMass,
                },
                position: jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
                velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
                t_inertial_pfix: None,
                delta_c20: 0.0,
                rotation_model: crate::RotationModel::default(),
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
            },
        );

        let gj_cfg = GaussJacksonConfig::with_order(8);
        let trans = TranslationalState {
            position: DVec3::new(9e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 8000.0, 0.0),
        };
        let make = |trans: TranslationalState| VehicleConfig {
            trans,
            integrator: IntegratorType::GaussJackson(gj_cfg),
            mass: Some(MassProperties::new(1000.0)),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, false)],
            },
            ..Default::default()
        };
        let body_a = sim.add_body(make(trans));
        let body_b = sim.add_body(make(TranslationalState {
            position: trans.position + DVec3::new(0.0, 1.0, 0.0),
            velocity: trans.velocity,
        }));
        sim.add_body_to_tree(body_a, "BodyA");
        sim.add_body_to_tree(body_b, "BodyB");

        // Pre-attach and step into operational mode, then detach.
        sim.attach(body_b, body_a, DVec3::ZERO, DMat3::IDENTITY);
        sim.validate().expect("validate failed");
        sim.step_n(200).expect("step_n failed");
        // After 200 post-attach steps both should be operational again.
        assert!(!sim.bodies[body_a].gj_state.as_ref().unwrap().is_priming());
        assert!(!sim.bodies[body_b].gj_state.as_ref().unwrap().is_priming());

        // Detach → both GJ states must reset.
        sim.detach(body_b);
        let gj_a = sim.bodies[body_a].gj_state.as_ref().unwrap();
        let gj_b = sim.bodies[body_b].gj_state.as_ref().unwrap();
        assert!(gj_a.is_priming(), "parent's GJ must reset on detach");
        assert!(gj_b.is_priming(), "child's GJ must reset on detach");
        assert!(!gj_a.is_topology_dirty());
        assert!(!gj_b.is_topology_dirty());
    }

    /// ABM4 sibling of `attach_resets_gauss_jackson_state_on_both_bodies`.
    /// `Simulation::attach` must also reset ABM4 history on both bodies —
    /// the underlying `mark_body_integrators_dirty_by_id` /
    /// `reset_body_integrators_by_id` helpers cover both integrator
    /// kinds, but without an ABM4-specific test a regression that only
    /// breaks the ABM4 arm would slip through (PR #282 review thread
    /// `PRRT_kwDORtae6c5_J-p_`).
    #[test]
    fn attach_resets_abm4_state_on_both_bodies() {
        const MU: f64 = 5.76e14;
        let dt = 1.0_f64;

        let trans_a = TranslationalState {
            position: DVec3::new(9e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 8000.0, 0.0),
        };
        let trans_b = TranslationalState {
            position: DVec3::new(9e6, 1.0, 0.0),
            velocity: DVec3::new(0.0, 7900.0, 0.0),
        };

        let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
        let mut sim = Simulation::new(time, dt);

        let earth = sim.add_source(
            "Earth",
            GravitySourceEntry {
                source: GravitySource {
                    mu: MU,
                    model: GravityModel::PointMass,
                },
                position: jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
                velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
                t_inertial_pfix: None,
                delta_c20: 0.0,
                rotation_model: crate::RotationModel::default(),
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
            },
        );

        let make_cfg = |trans: TranslationalState| VehicleConfig {
            trans,
            integrator: IntegratorType::Abm4,
            mass: Some(MassProperties::new(1000.0)),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, false)],
            },
            ..Default::default()
        };
        let body_a = sim.add_body(make_cfg(trans_a));
        let body_b = sim.add_body(make_cfg(trans_b));
        sim.add_body_to_tree(body_a, "BodyA");
        sim.add_body_to_tree(body_b, "BodyB");

        sim.validate().expect("validate failed");

        // ── Step long enough to leave ABM4 priming on both bodies. ──
        // ABM4 primes after `HIST_LEN - 1 = 3` steps; 5 is comfortably past.
        sim.step_n(5).expect("step_n failed");

        let abm_a_pre = sim.bodies[body_a]
            .abm4_state
            .as_ref()
            .expect("body A must have abm4_state after validate");
        let abm_b_pre = sim.bodies[body_b]
            .abm4_state
            .as_ref()
            .expect("body B must have abm4_state after validate");
        assert!(
            !abm_a_pre.is_priming(),
            "test setup expected body A's ABM4 state past priming after 5 steps"
        );
        assert!(
            !abm_b_pre.is_priming(),
            "test setup expected body B's ABM4 state past priming after 5 steps"
        );
        assert!(!abm_a_pre.is_topology_dirty());
        assert!(!abm_b_pre.is_topology_dirty());

        // ── Attach: triggers IG.37 reset on both bodies. ──
        sim.attach(body_a, body_b, DVec3::ZERO, DMat3::IDENTITY);

        let abm_a_post = sim.bodies[body_a].abm4_state.as_ref().unwrap();
        let abm_b_post = sim.bodies[body_b].abm4_state.as_ref().unwrap();
        assert!(
            abm_a_post.is_priming(),
            "body A's ABM4 state must be back in priming after attach (IG.37)"
        );
        assert!(
            abm_b_post.is_priming(),
            "body B's ABM4 state must be back in priming after attach (IG.37)"
        );
        assert!(!abm_a_post.is_topology_dirty());
        assert!(!abm_b_post.is_topology_dirty());

        // ── Step once more: the IG.37 assertion in `abm4_translational_step`
        // must not fire. ──
        sim.step_n(1).expect("post-attach step failed");
    }

    /// ABM4 sibling of `detach_resets_gauss_jackson_state_on_both_bodies`.
    #[test]
    fn detach_resets_abm4_state_on_both_bodies() {
        const MU: f64 = 5.76e14;
        let dt = 1.0_f64;
        let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
        let mut sim = Simulation::new(time, dt);

        let earth = sim.add_source(
            "Earth",
            GravitySourceEntry {
                source: GravitySource {
                    mu: MU,
                    model: GravityModel::PointMass,
                },
                position: jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
                velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
                t_inertial_pfix: None,
                delta_c20: 0.0,
                rotation_model: crate::RotationModel::default(),
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
            },
        );

        let trans = TranslationalState {
            position: DVec3::new(9e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 8000.0, 0.0),
        };
        let make = |trans: TranslationalState| VehicleConfig {
            trans,
            integrator: IntegratorType::Abm4,
            mass: Some(MassProperties::new(1000.0)),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, false)],
            },
            ..Default::default()
        };
        let body_a = sim.add_body(make(trans));
        let body_b = sim.add_body(make(TranslationalState {
            position: trans.position + DVec3::new(0.0, 1.0, 0.0),
            velocity: trans.velocity,
        }));
        sim.add_body_to_tree(body_a, "BodyA");
        sim.add_body_to_tree(body_b, "BodyB");

        sim.attach(body_b, body_a, DVec3::ZERO, DMat3::IDENTITY);
        sim.validate().expect("validate failed");
        sim.step_n(5).expect("step_n failed");
        assert!(!sim.bodies[body_a].abm4_state.as_ref().unwrap().is_priming());
        assert!(!sim.bodies[body_b].abm4_state.as_ref().unwrap().is_priming());

        sim.detach(body_b);
        let abm_a = sim.bodies[body_a].abm4_state.as_ref().unwrap();
        let abm_b = sim.bodies[body_b].abm4_state.as_ref().unwrap();
        assert!(abm_a.is_priming(), "parent's ABM4 must reset on detach");
        assert!(abm_b.is_priming(), "child's ABM4 must reset on detach");
        assert!(!abm_a.is_topology_dirty());
        assert!(!abm_b.is_topology_dirty());
    }

    /// `Simulation::attach`/`detach` must reset integrators on the
    /// **full ancestor chain**, not just the directly-named bodies.
    /// `MassTree::attach`/`detach` recompute composite mass properties
    /// all the way to the root; an integrator on any ancestor is
    /// invalidated by that recompute. Builds a 3-body chain
    /// `top → middle → leaf`, attaches a fourth body underneath
    /// `middle`, and verifies that `top`'s GJ state is reset (in
    /// addition to `middle` and the new attachee). Mirrors PR #282
    /// review threads `PRRT_kwDORtae6c5_J-qF` (attach) and
    /// `PRRT_kwDORtae6c5_J-qI` (detach).
    #[test]
    fn attach_and_detach_reset_full_ancestor_chain() {
        const MU: f64 = 5.76e14;
        let dt = 1.0_f64;
        let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
        let mut sim = Simulation::new(time, dt);

        let earth = sim.add_source(
            "Earth",
            GravitySourceEntry {
                source: GravitySource {
                    mu: MU,
                    model: GravityModel::PointMass,
                },
                position: jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
                velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
                t_inertial_pfix: None,
                delta_c20: 0.0,
                rotation_model: crate::RotationModel::default(),
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
            },
        );

        // Four bodies on similar orbits — only `top` is integrated; the
        // others sit on the same orbit and are attached in a chain to
        // build an ancestor relationship inside `MassTree`. Only `top`
        // needs a working integrator, but every body whose
        // `composite_properties` is recomputed by `MassTree::attach`
        // / `detach` should still see its (otherwise unused) integrator
        // reset.
        let gj_cfg = GaussJacksonConfig::with_order(8);
        let trans = TranslationalState {
            position: DVec3::new(9e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 8000.0, 0.0),
        };
        let make_cfg = || VehicleConfig {
            trans,
            integrator: IntegratorType::GaussJackson(gj_cfg),
            mass: Some(MassProperties::new(1000.0)),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, false)],
            },
            ..Default::default()
        };
        let top = sim.add_body(make_cfg());
        let middle = sim.add_body(make_cfg());
        let leaf = sim.add_body(make_cfg());
        let new_attachee = sim.add_body(make_cfg());
        sim.add_body_to_tree(top, "Top");
        sim.add_body_to_tree(middle, "Middle");
        sim.add_body_to_tree(leaf, "Leaf");
        sim.add_body_to_tree(new_attachee, "NewAttachee");

        // Build the chain: middle → top, leaf → middle. After this the
        // tree has root=top, with middle as child and leaf as grandchild.
        sim.attach(middle, top, DVec3::ZERO, DMat3::IDENTITY);
        sim.attach(leaf, middle, DVec3::ZERO, DMat3::IDENTITY);

        sim.validate().expect("validate failed");

        // ── Step long enough to leave GJ priming on `top`. ──
        sim.step_n(200).expect("step_n failed");
        assert!(
            !sim.bodies[top].gj_state.as_ref().unwrap().is_priming(),
            "test setup: top's GJ must be past priming"
        );

        // ── Attach NewAttachee underneath middle. This recomputes
        //    middle's *and* top's composite properties (top is an
        //    ancestor of middle). top's GJ state must therefore reset.
        //
        //    Pre-fix, only `middle` and `new_attachee` would be reset,
        //    leaving `top` with stale predictor history that
        //    references the pre-attach mass distribution. ──
        sim.attach(new_attachee, middle, DVec3::ZERO, DMat3::IDENTITY);
        assert!(
            sim.bodies[top].gj_state.as_ref().unwrap().is_priming(),
            "ancestor `top`'s GJ must be reset when a body is attached \
             under its descendant `middle` (IG.37 ancestor coverage)"
        );
        assert!(
            !sim.bodies[top]
                .gj_state
                .as_ref()
                .unwrap()
                .is_topology_dirty(),
            "ancestor `top`'s GJ must be reset (dirty bit cleared)"
        );

        // Step past priming again so we can test the detach branch.
        sim.step_n(200).expect("step_n failed (post-attach)");
        assert!(!sim.bodies[top].gj_state.as_ref().unwrap().is_priming());

        // ── Detach `new_attachee` — same ancestor coverage requirement. ──
        sim.detach(new_attachee);
        assert!(
            sim.bodies[top].gj_state.as_ref().unwrap().is_priming(),
            "ancestor `top`'s GJ must be reset when a descendant of \
             `middle` is detached (IG.37 ancestor coverage)"
        );
        assert!(
            !sim.bodies[top]
                .gj_state
                .as_ref()
                .unwrap()
                .is_topology_dirty(),
            "ancestor `top`'s GJ dirty bit must be cleared by reset"
        );

        // Confirm a subsequent step doesn't trip the IG.37 assertion.
        sim.step_n(1).expect("post-detach step failed");
    }

    /// Stress test the affected-id lookup in attach/detach: build a
    /// 100-body chain (one integrated GJ body + 99 tree-only nodes,
    /// linearly chained as ancestors), then attach a new body
    /// underneath the deepest ancestor and detach it again.
    /// `MassTree::attach` recomputes composites along the entire
    /// 100-deep ancestor chain, so `affected_ids` contains 100
    /// entries — and the helpers must do membership lookup against
    /// that set for every Simulation body in `self.bodies`. With the
    /// sort + binary_search pattern (PR #282 review thread
    /// `PRRT_kwDORtae6c5_KoAT`) the per-body cost is O(log 100); a
    /// regression back to `Vec::contains` would be O(100) per body,
    /// O(n²) overall.
    ///
    /// We don't time the test — what we assert is functional
    /// correctness on a deep chain: the integrated body's GJ state
    /// resets exactly once per attach / detach (never observed dirty
    /// after the call), and a follow-up `step_n(1)` doesn't trip the
    /// IG.37 panic. A scaling regression would compile and produce
    /// the same observable end-state, but on a JEOD-scale sim with
    /// thousands of bodies the cost would dominate; the named
    /// invariant — "exactly one mark + one reset per affected body
    /// per topology change" — is documented at the helper sites.
    #[test]
    fn attach_detach_scales_to_deep_ancestor_chain() {
        const MU: f64 = 5.76e14;
        const CHAIN_LEN: usize = 100;
        let dt = 1.0_f64;
        let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
        let mut sim = Simulation::new(time, dt);

        let earth = sim.add_source(
            "Earth",
            GravitySourceEntry {
                source: GravitySource {
                    mu: MU,
                    model: GravityModel::PointMass,
                },
                position: jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
                velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
                t_inertial_pfix: None,
                delta_c20: 0.0,
                rotation_model: crate::RotationModel::default(),
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
            },
        );

        // One integrated GJ8 body (the only one with integrator
        // state — the rest of the chain is tree-only nodes).
        let gj_cfg = GaussJacksonConfig::with_order(8);
        let trans = TranslationalState {
            position: DVec3::new(9e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 8000.0, 0.0),
        };
        let cm = sim.add_body(VehicleConfig {
            trans,
            integrator: IntegratorType::GaussJackson(gj_cfg),
            mass: Some(MassProperties::new(1000.0)),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, false)],
            },
            ..Default::default()
        });
        let cm_id = sim.add_body_to_tree(cm, "cm");

        // Build a 100-deep ancestor chain rooted at cm. Each link is
        // a tree-only node attached underneath the previous one.
        let mut chain_ids: Vec<MassBodyId> = Vec::with_capacity(CHAIN_LEN);
        chain_ids.push(cm_id);
        {
            let tree = sim
                .mass_tree
                .as_mut()
                .expect("mass tree must exist after add_body_to_tree");
            for i in 1..CHAIN_LEN {
                let id = tree.add_body(format!("node_{i}"), MassProperties::new(10.0));
                let parent = chain_ids[i - 1];
                tree.attach(id, parent, DVec3::new(0.1, 0.0, 0.0), DMat3::IDENTITY);
                chain_ids.push(id);
            }
        }
        sim.sync_body_mass_from_tree(cm);

        sim.validate().expect("validate failed");
        sim.step_n(200).expect("step_n failed");
        assert!(
            !sim.bodies[cm].gj_state.as_ref().unwrap().is_priming(),
            "test setup: cm's GJ must be past priming"
        );

        // Attach a new body underneath the deepest tree-only node.
        // `MassTree::attach` recomputes composites for the new node
        // plus the entire 100-deep ancestor chain, so `affected_ids`
        // has 101 entries. Every Simulation body (just `cm` here)
        // does one binary_search per call — but the same code path
        // applies on a sim with thousands of bodies.
        let deepest = *chain_ids.last().expect("chain must be non-empty");
        let attachee = {
            let tree = sim.mass_tree.as_mut().unwrap();
            let id = tree.add_body("attachee".into(), MassProperties::new(5.0));
            // We need a Simulation body whose mass_body_id == this
            // attachee, so attach via `Simulation::attach` (which
            // requires both ends to be Simulation bodies). For this
            // test we instead use the tree-only attach to keep the
            // setup small — the helpers fan over Simulation bodies,
            // so what matters is the affected-id list size, not
            // whether the new node is Simulation-tracked.
            tree.attach(id, deepest, DVec3::new(0.1, 0.0, 0.0), DMat3::IDENTITY);
            id
        };
        // Manually drive the mark / reset path so we exercise the
        // helpers directly — Simulation::attach takes Simulation body
        // indices, but we want to test the affected-id discipline
        // with a long ancestor chain that includes tree-only nodes.
        let mut affected_ids: Vec<MassBodyId> = vec![attachee];
        affected_ids.extend(sim.mass_tree.as_ref().unwrap().ancestors_inclusive(deepest));
        affected_ids.sort_unstable();
        affected_ids.dedup();
        assert_eq!(
            affected_ids.len(),
            CHAIN_LEN + 1,
            "affected_ids should include the new attachee plus the full {CHAIN_LEN}-deep ancestor chain"
        );
        Simulation::mark_body_integrators_dirty_by_id(&mut sim.bodies, &affected_ids);
        Simulation::reset_body_integrators_by_id(&mut sim.bodies, &affected_ids);

        // After the helpers run, cm's GJ must be primed and clean.
        let gj_post = sim.bodies[cm].gj_state.as_ref().unwrap();
        assert!(
            gj_post.is_priming(),
            "cm's GJ must be back in priming after a topology change \
             affecting its full ancestor chain"
        );
        assert!(
            !gj_post.is_topology_dirty(),
            "cm's GJ topology-dirty flag must be cleared (binary_search lookup ran)"
        );

        // Confirm a subsequent step doesn't trip the IG.37 assertion.
        sim.step_n(1)
            .expect("post-deep-chain step failed (IG.37 must not fire)");
    }

    // ── detach_subtree / attach_subtree_aligned IG.37 wiring ────────
    //
    // The next four tests cover the inline mark+reset blocks at the
    // end of `Simulation::detach_subtree` (`parent_is_integrated`
    // branch) and `Simulation::attach_subtree_aligned`. Without them,
    // a regression that drops either reset call would still pass
    // `attach_resets_*` / `detach_resets_*` above, since those only
    // exercise the simpler `Simulation::attach` / `detach` (no
    // subtree state propagation, no momentum-conservation combine).
    // PR #282 review threads `PRRT_kwDORtae6c5_KoAQ` (detach_subtree)
    // and `PRRT_kwDORtae6c5_KoAR` (attach_subtree_aligned).
    //
    // ## Why we splice integrator state in by hand
    //
    // `Simulation::detach_subtree` and `attach_subtree_aligned` both
    // require a 6-DOF integrated body (they read `body.rot` to
    // propagate the subtree's composite-CoM state).
    // `Simulation::validate` forbids the only integrators that own
    // multi-step history (GJ / ABM4) on 6-DOF bodies — see
    // `ValidationError::GaussJacksonWith6Dof` / `Abm4With6Dof`. So
    // an end-to-end test of the inline reset block can't go through
    // `validate` + `step_n()` today.
    //
    // The reset block IS still on the production code path: it
    // guards future re-enablement of GJ/ABM4 with rotational
    // dynamics, *and* it guards external callers who construct a
    // 6-DOF SimBody and splice in `gj_state` / `abm4_state`
    // themselves (e.g. from integration-tests like these, or from a
    // downstream crate that bypasses our validator). The fail-loud
    // safety net for *those* callers is the IG.37 panic in
    // `integrate()`.
    //
    // Each test below:
    //   1. Builds a 3-body mass-tree chain `cm → middle → leaf` with
    //      a 6-DOF RK4-integrated `cm` (so it passes `validate`).
    //   2. Splices a `GaussJacksonState` / `Abm4State` directly onto
    //      `cm` after validate, then advances the state past priming
    //      using the integrator's own public API at a constant
    //      zero-accel function (we don't need realistic dynamics —
    //      only a state that is observably *past* priming).
    //   3. Asserts pre-state: `is_priming() == false` AND
    //      `is_topology_dirty() == false`.
    //   4. Triggers the topology mutation (`detach_subtree` /
    //      `attach_subtree_aligned`).
    //   5. Asserts post-state: integrator is back in priming AND not
    //      topology-dirty. Both flags moving together is the
    //      signature of `reset_for_topology_change` having run —
    //      since `mark_topology_dirty` alone never touches
    //      `is_priming`. If a regression drops the inline reset
    //      call, this assertion fails (the test fails on the
    //      `is_priming` check).
    //
    // The accompanying `dirty_*_state_panics_on_integrate` tests
    // independently verify the IG.37 fail-loud panic that backs the
    // mark site: pre-mark dirty (simulating the case where mark
    // fired but reset was forgotten), then run `integrate()` and
    // assert the panic message naming the IG.37 diagnostic.

    /// Set up a 3-body mass-tree chain `cm → middle → leaf` with a
    /// 6-DOF RK4-integrated `cm`. The chain has a non-trivial
    /// `propagate_forward` walk inside `detach_subtree(cm_idx, leaf)`
    /// (the `chain` loop iterates over both `middle` and `leaf`).
    /// Returns the simulation, the integrated-body index, and the
    /// mass-tree ids of `cm`, `middle`, `leaf`.
    fn build_three_body_chain_with_rot() -> (Simulation, usize, MassBodyId, MassBodyId, MassBodyId)
    {
        const MU: f64 = 5.76e14;
        let dt = 1.0_f64;

        let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
        let mut sim = Simulation::new(time, dt);

        let earth = sim.add_source(
            "Earth",
            GravitySourceEntry {
                source: GravitySource {
                    mu: MU,
                    model: GravityModel::PointMass,
                },
                position: jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
                velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
                t_inertial_pfix: None,
                delta_c20: 0.0,
                rotation_model: crate::RotationModel::default(),
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
            },
        );

        // 6-DOF + RK4 (so it passes validate). We splice GJ/ABM4 state
        // in by hand after validate to exercise detach_subtree's IG.37
        // reset block — see the module-level comment above for why.
        let cm_idx = sim.add_body(VehicleConfig {
            trans: TranslationalState {
                position: DVec3::new(9e6, 0.0, 0.0),
                velocity: DVec3::new(0.0, 8000.0, 0.0),
            },
            rot: Some(jeod_sim::RotationalState::default()),
            integrator: IntegratorType::Rk4,
            mass: Some(MassProperties::new(1000.0)),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, false)],
            },
            ..Default::default()
        });

        let cm_id = sim.add_body_to_tree(cm_idx, "cm");

        // Tree-only subtree nodes (no Simulation body / integrator).
        let tree = sim
            .mass_tree
            .as_mut()
            .expect("mass tree must be initialised by add_body_to_tree above");
        let middle = tree.add_body("middle".into(), MassProperties::new(500.0));
        let leaf = tree.add_body("leaf".into(), MassProperties::new(250.0));

        // cm → middle → leaf at unit offsets, identity rotations.
        tree.attach(middle, cm_id, DVec3::new(1.0, 0.0, 0.0), DMat3::IDENTITY);
        tree.attach(leaf, middle, DVec3::new(1.0, 0.0, 0.0), DMat3::IDENTITY);

        sim.sync_body_mass_from_tree(cm_idx);

        (sim, cm_idx, cm_id, middle, leaf)
    }

    /// Drive a `GaussJacksonState` past priming using its public
    /// `integrate()` API at constant zero acceleration. The post-step
    /// state values aren't used — only the priming flag is.
    fn drive_gj_past_priming(gj: &mut jeod_dynamics::GaussJacksonState) {
        let dt = 1.0_f64;
        let mut state = TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        };
        // GJ8 needs ~50 stages to bootstrap; 200 stages is comfortably past.
        for _ in 0..200 {
            let _ = gj.integrate(dt, 1.0, DVec3::ZERO, &mut state);
        }
        assert!(
            !gj.is_priming(),
            "test setup: drive_gj_past_priming did not exit priming after 200 stages"
        );
    }

    /// Drive an `Abm4State` past priming using `abm4_translational_step`
    /// at constant zero acceleration. ABM4 primes after `HIST_LEN - 1 = 3`
    /// steps; 5 is comfortably past.
    fn drive_abm4_past_priming(abm: &mut jeod_dynamics::Abm4State) {
        let dt = 1.0_f64;
        let mut state = TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        };
        for _ in 0..5 {
            state = jeod_dynamics::abm4_translational_step(&state, |_s, _t| DVec3::ZERO, dt, abm);
        }
        assert!(
            !abm.is_priming(),
            "test setup: drive_abm4_past_priming did not exit priming after 5 stages"
        );
    }

    /// `Simulation::detach_subtree` must reset the integrated body's
    /// Gauss-Jackson history after the topology mutation. Build a
    /// 3-body chain, splice GJ state onto the integrated body, drive
    /// past priming, detach the deepest leaf (a non-trivial chain walk
    /// inside `detach_subtree`), and verify the integrator is reset.
    /// PR #282 review thread `PRRT_kwDORtae6c5_KoAQ`.
    #[test]
    fn detach_subtree_resets_gauss_jackson_state_on_integrated_body() {
        let (mut sim, cm_idx, _cm_id, _middle, leaf) = build_three_body_chain_with_rot();
        sim.validate().expect("validate failed");

        // Splice GJ8 state onto cm post-validate (validate forbids
        // GJ+6DOF; we're exercising the inline reset block defensively).
        let cfg = GaussJacksonConfig::with_order(8);
        let mut gj = jeod_dynamics::GaussJacksonState::new(cfg);
        drive_gj_past_priming(&mut gj);
        assert!(!gj.is_topology_dirty());
        sim.bodies[cm_idx].gj_state = Some(gj);

        // ── detach_subtree: drops the leaf node. The chain walk
        //    `cm → middle → leaf` exercises the `propagate_forward`
        //    loop inside detach_subtree (lines 320–358). ──
        sim.detach_subtree(cm_idx, leaf);

        let gj_post = sim.bodies[cm_idx]
            .gj_state
            .as_ref()
            .expect("integrated body must still have gj_state");
        // is_priming flips back to true ONLY through
        // reset_for_topology_change — proving the inline reset call
        // inside detach_subtree fires. If a regression drops that
        // call, this assertion fails.
        assert!(
            gj_post.is_priming(),
            "cm's GJ state must be back in priming after detach_subtree (IG.37)"
        );
        assert!(
            !gj_post.is_topology_dirty(),
            "cm's GJ topology-dirty flag must be cleared by reset on detach_subtree (IG.37)"
        );
    }

    /// ABM4 sibling of `detach_subtree_resets_gauss_jackson_state_*`.
    /// PR #282 review thread `PRRT_kwDORtae6c5_KoAQ`.
    #[test]
    fn detach_subtree_resets_abm4_state_on_integrated_body() {
        let (mut sim, cm_idx, _cm_id, _middle, leaf) = build_three_body_chain_with_rot();
        sim.validate().expect("validate failed");

        let mut abm = jeod_dynamics::Abm4State::new();
        drive_abm4_past_priming(&mut abm);
        assert!(!abm.is_topology_dirty());
        sim.bodies[cm_idx].abm4_state = Some(abm);

        sim.detach_subtree(cm_idx, leaf);

        let abm_post = sim.bodies[cm_idx]
            .abm4_state
            .as_ref()
            .expect("integrated body must still have abm4_state");
        assert!(
            abm_post.is_priming(),
            "cm's ABM4 state must be back in priming after detach_subtree (IG.37)"
        );
        assert!(
            !abm_post.is_topology_dirty(),
            "cm's ABM4 topology-dirty flag must be cleared by reset on detach_subtree (IG.37)"
        );
    }

    // ── attach_subtree_aligned IG.37 wiring ─────────────────────────
    //
    // The next two tests cover the inline mark+reset block at the end
    // of `Simulation::attach_subtree_aligned`. Same shape as the
    // `detach_subtree` tests above (and same constraint — `validate`
    // forbids GJ/ABM4 + 6-DOF, so we splice integrator state in by
    // hand). PR #282 review thread `PRRT_kwDORtae6c5_KoAR`.
    //
    // Setup is asymmetric: we need a *detached* subtree to attach.
    // We call `detach_subtree` first to populate
    // `Simulation::detached_subtrees`, then splice the integrator
    // state and run `attach_subtree_aligned` so its inline reset
    // block has something to clear.

    /// `Simulation::attach_subtree_aligned` must reset the integrated
    /// body's Gauss-Jackson history after the topology + state combine.
    /// PR #282 review thread `PRRT_kwDORtae6c5_KoAR`.
    #[test]
    fn attach_subtree_aligned_resets_gauss_jackson_state_on_integrated_body() {
        let (mut sim, cm_idx, _cm_id, middle, leaf) = build_three_body_chain_with_rot();

        // Add named docking points needed by attach_aligned.
        {
            let tree = sim
                .mass_tree
                .as_mut()
                .expect("mass tree must exist after build_three_body_chain_with_rot");
            tree.add_mass_point(middle, "middle.dock", DVec3::ZERO, DMat3::IDENTITY);
            tree.add_mass_point(leaf, "leaf.dock", DVec3::ZERO, DMat3::IDENTITY);
        }

        sim.validate().expect("validate failed");

        // Detach `leaf` to populate `detached_subtrees` so we have
        // something to re-attach. detach_subtree does its own IG.37
        // reset, but cm has no integrator state spliced in yet — so
        // detach is a no-op on integrators here.
        sim.detach_subtree(cm_idx, leaf);

        // Splice GJ8 state onto cm AFTER the detach so
        // attach_subtree_aligned's reset block has something to clear.
        let cfg = GaussJacksonConfig::with_order(8);
        let mut gj = jeod_dynamics::GaussJacksonState::new(cfg);
        drive_gj_past_priming(&mut gj);
        assert!(!gj.is_topology_dirty());
        sim.bodies[cm_idx].gj_state = Some(gj);

        // ── Re-attach. `attach_subtree_aligned` runs the
        //    combine-states algorithm, overwrites body.trans /
        //    body.rot, and must reset the integrator at the end. ──
        sim.attach_subtree_aligned(cm_idx, leaf, "leaf.dock", middle, "middle.dock");

        let gj_post = sim.bodies[cm_idx].gj_state.as_ref().unwrap();
        // `is_priming() == true` only happens through
        // `reset_for_topology_change` — so this fails if the inline
        // reset call inside `attach_subtree_aligned` is removed.
        assert!(
            gj_post.is_priming(),
            "cm's GJ state must be back in priming after attach_subtree_aligned (IG.37)"
        );
        assert!(
            !gj_post.is_topology_dirty(),
            "cm's GJ topology-dirty flag must be cleared on attach_subtree_aligned (IG.37)"
        );
    }

    /// ABM4 sibling of `attach_subtree_aligned_resets_gauss_jackson_*`.
    /// PR #282 review thread `PRRT_kwDORtae6c5_KoAR`.
    #[test]
    fn attach_subtree_aligned_resets_abm4_state_on_integrated_body() {
        let (mut sim, cm_idx, _cm_id, middle, leaf) = build_three_body_chain_with_rot();

        {
            let tree = sim
                .mass_tree
                .as_mut()
                .expect("mass tree must exist after build_three_body_chain_with_rot");
            tree.add_mass_point(middle, "middle.dock", DVec3::ZERO, DMat3::IDENTITY);
            tree.add_mass_point(leaf, "leaf.dock", DVec3::ZERO, DMat3::IDENTITY);
        }

        sim.validate().expect("validate failed");
        sim.detach_subtree(cm_idx, leaf);

        let mut abm = jeod_dynamics::Abm4State::new();
        drive_abm4_past_priming(&mut abm);
        assert!(!abm.is_topology_dirty());
        sim.bodies[cm_idx].abm4_state = Some(abm);

        sim.attach_subtree_aligned(cm_idx, leaf, "leaf.dock", middle, "middle.dock");

        let abm_post = sim.bodies[cm_idx].abm4_state.as_ref().unwrap();
        assert!(
            abm_post.is_priming(),
            "cm's ABM4 state must be back in priming after attach_subtree_aligned (IG.37)"
        );
        assert!(
            !abm_post.is_topology_dirty(),
            "cm's ABM4 topology-dirty flag must be cleared on attach_subtree_aligned (IG.37)"
        );
    }

    /// Verify the IG.37 fail-loud safety net actually fires when an
    /// integrator is left dirty: simulate a regression where mark
    /// fired but reset was forgotten by manually flipping the dirty
    /// flag to true on a primed integrator, then call `integrate()`.
    /// The test passes only if the integrator panics with the IG.37
    /// diagnostic. This is the pair-half of the inline mark sites in
    /// detach_subtree / attach_subtree_aligned — together they make
    /// any regression that drops a Site B (reset) loud rather than
    /// silent.
    #[test]
    #[should_panic(expected = "topology")]
    fn dirty_gauss_jackson_state_panics_on_integrate() {
        let cfg = GaussJacksonConfig::with_order(8);
        let mut gj = jeod_dynamics::GaussJacksonState::new(cfg);
        drive_gj_past_priming(&mut gj);

        // Simulate a regression where Site A (mark) fired but Site B
        // (reset_for_topology_change) was forgotten.
        gj.mark_topology_dirty();
        assert!(gj.is_topology_dirty());

        // The next integrate() must fail-loud per IG.37.
        let mut state = TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        };
        let _ = gj.integrate(1.0, 1.0, DVec3::ZERO, &mut state);
    }

    /// ABM4 sibling of `dirty_gauss_jackson_state_panics_on_integrate`.
    #[test]
    #[should_panic(expected = "topology")]
    fn dirty_abm4_state_panics_on_integrate() {
        let mut abm = jeod_dynamics::Abm4State::new();
        drive_abm4_past_priming(&mut abm);

        abm.mark_topology_dirty();
        assert!(abm.is_topology_dirty());

        let state = TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        };
        let _ = jeod_dynamics::abm4_translational_step(&state, |_, _| DVec3::ZERO, 1.0, &mut abm);
    }
}
