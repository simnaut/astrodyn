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
    /// After attachment, the parent's composite mass properties are updated
    /// automatically. The parent body's `mass` is synced from the tree.
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
        let tree = self.mass_tree.as_mut().expect("attach: no mass tree");
        tree.attach(child_id, parent_id, offset, t_parent_child);
        // Sync parent's composite mass from tree
        self.bodies[parent_idx].mass = Some(tree.get(parent_id).composite_properties);

        // JEOD_INV: IG.37 — Multi-step integrators (GJ, ABM4) carry predictor
        // history that is invalidated by the topology change. Mirror JEOD's
        // `dyn_body_attach.cc::reset_integrators()` (lines 860, 871) by
        // resetting both bodies' integrator state. RK4 / RKF4(5) carry no
        // history; the helper no-ops for them.
        Self::mark_and_reset_body_integrators(&mut self.bodies[child_idx]);
        Self::mark_and_reset_body_integrators(&mut self.bodies[parent_idx]);
    }

    /// Detach a child body from its parent in the mass tree.
    ///
    /// After detachment, both the former parent's and the child's mass
    /// properties are updated from the tree's recomputed composites.
    ///
    /// # Panics
    /// Panics if the body is not in the tree or has no parent.
    pub fn detach(&mut self, child_idx: usize) {
        let child_id = self.bodies[child_idx]
            .mass_body_id
            .expect("detach: child body not in mass tree");
        let tree = self.mass_tree.as_mut().expect("detach: no mass tree");
        let parent_id = tree
            .parent(child_id)
            .expect("detach: child body has no parent in tree");
        tree.detach(child_id);
        // Sync both bodies' mass from tree
        self.bodies[child_idx].mass = Some(tree.get(child_id).composite_properties);
        // Find parent body index and sync
        let mut parent_idx_opt: Option<usize> = None;
        for (idx, body) in self.bodies.iter_mut().enumerate() {
            if body.mass_body_id == Some(parent_id) {
                body.mass = Some(tree.get(parent_id).composite_properties);
                parent_idx_opt = Some(idx);
                break;
            }
        }

        // JEOD_INV: IG.37 — Multi-step integrators (GJ, ABM4) carry predictor
        // history that is invalidated by the topology change. Mirror JEOD's
        // `dyn_body_detach.cc:271-273` reset of both the parent and the
        // child, since both bodies' dynamics have changed (the parent lost
        // mass, the child became a free root).
        Self::mark_and_reset_body_integrators(&mut self.bodies[child_idx]);
        if let Some(parent_idx) = parent_idx_opt {
            Self::mark_and_reset_body_integrators(&mut self.bodies[parent_idx]);
        }
    }

    /// Reset a body's multi-step integrator state in response to a
    /// topology change (attach / detach / subtree swap).
    ///
    /// Marks the integrator dirty (so any caller that bypasses this
    /// helper still trips the IG.37 assertion in `integrate()`), then
    /// resets it. For RK4 / RKF4(5) bodies — which carry no integrator
    /// state — both arms are `None` and this is a no-op.
    ///
    /// Mirrors JEOD's `dyn_body_attach.cc::reset_integrators()` (lines
    /// 860, 871) and `dyn_body_detach.cc:271-273`.
    // JEOD_INV: IG.37 — multi-step integrator history must be reset on topology change
    fn mark_and_reset_body_integrators(body: &mut super::types::SimBody) {
        if let Some(ref mut gj) = body.gj_state {
            gj.mark_topology_dirty();
        }
        if let Some(ref mut abm) = body.abm4_state {
            abm.mark_topology_dirty();
        }
        jeod_sim::reset_integrators(body.gj_state.as_mut(), body.abm4_state.as_mut());
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
        if parent_is_integrated {
            Self::mark_and_reset_body_integrators(&mut self.bodies[integrated_body_idx]);
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
        Self::mark_and_reset_body_integrators(&mut self.bodies[integrated_body_idx]);

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
}
