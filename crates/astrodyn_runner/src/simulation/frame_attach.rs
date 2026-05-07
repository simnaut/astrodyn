//! Frame-attached body API for [`super::Simulation`].
//!
//! Implements JEOD's `DynBody::attach_to_frame` /
//! `DynBody::detach()`-from-frame semantics for the runner's owned-state
//! harness, as the runner-side counterpart of the kernel that lives in
//! `astrodyn_dynamics::frame_attach`. The Bevy adapter has the equivalent
//! glue in `astrodyn_bevy::frame_attach_system`.
//!
//! ### What `attach_to_frame` does
//!
//! After [`Simulation::attach_to_frame`] returns, the body's
//! translational and rotational state stops being integrated under its
//! own dynamics. Each subsequent `step()` derives the body's
//! composite-body state from the parent reference frame's state plus
//! the captured offset (see
//! [`Simulation::propagate_frame_attached_state`]). The kernel is
//! [`astrodyn_dynamics::derive_frame_attached_state`] — the same algebra
//! `propagate_forward` runs for mass-tree kinematic propagation, just
//! seeded from the frame tree instead of the mass tree.
//!
//! Multi-step integrator history (Gauss–Jackson, ABM4) is reset on
//! both attach and detach so the predictor cache doesn't carry stale
//! pre-attach derivatives across the topology change. Mirrors JEOD's
//! `dyn_body_attach.cc::reset_integrators()` (lines 860, 871) and
//! `dyn_body_detach.cc:271-273`. RK4 / RKF4(5) bodies have no
//! integrator state and are silently no-op'd by
//! `Self::reset_body_integrators_by_id`.
//!
//! ### What `detach()`-from-frame does
//!
//! Releases the body's frame attachment. The body's `trans` / `rot`
//! retain whatever value the frame-attached propagation produced last
//! tick, so when integration resumes, it picks up from the body's
//! instantaneous state at separation — the rigid-body invariant that
//! makes a "release" indistinguishable from a continuous trajectory at
//! the release instant.
//!
//! ### Out of scope here
//!
//! - Mass-tree attach/detach (`Simulation::attach` /
//!   `Simulation::detach`) is unchanged. A body cannot be both
//!   mass-tree-attached and frame-attached; the API gates this at the
//!   call site.
//! - Velocity-resume / state override convenience methods (the
//!   SIM_dyncomp `RUN_attach_to_ref_frame` Python pattern of writing a
//!   pre-attach velocity back after detach) are not provided here —
//!   callers that need this can mutate `body.trans.velocity` directly
//!   via [`Simulation::set_body_velocity`] after `detach_from_frame`.

use glam::DMat3;

use astrodyn::{RotationalState, TranslationalState, TranslationalStateTyped};
use astrodyn_frames::FrameId;
use astrodyn_quantities::frame::IntegrationFrame;

use super::types::FrameAttachState;
use super::Simulation;

impl Simulation {
    /// Attach a body to a non-body reference frame.
    ///
    /// After this call, the body's `trans` / `rot` are derived each
    /// `step()` from the parent reference frame's state composed with
    /// the captured offset, and translational + rotational integration
    /// is suppressed for this body until [`detach_from_frame`](Self::detach_from_frame)
    /// is called. Mirrors JEOD `DynBody::attach_to_frame(RefFrame &
    /// parent, RefFrameState X_pframe_to_struct)`
    /// (`models/dynamics/dyn_body/src/dyn_body_attach.cc:367-379`).
    ///
    /// `parent_frame_id` must reference a frame in the simulation's
    /// frame tree. `attach_offset_position` is the structure-frame
    /// origin's location in parent-frame coordinates;
    /// `t_parent_struct` is the rotation matrix from parent-frame axes
    /// to the body's structural-frame axes (which equals the body
    /// frame here, since structure and composite-body frames share an
    /// orientation in the simulation's body model).
    ///
    /// Multi-step integrator history is reset (mirrors JEOD
    /// `reset_integrators()` at `dyn_body_attach.cc:860`).
    ///
    /// # Panics
    /// Panics if:
    /// - `body_idx` is out of range.
    /// - The body is already frame-attached. Call `detach_from_frame`
    ///   first; double-attach without detach would silently overwrite
    ///   the previous attachment's offset and lose the original
    ///   parent-frame relationship.
    /// - The body has a mass-tree parent (`MassChildOf`) — JEOD's
    ///   `attach_to_frame` walks `get_root_body_internal()` first and
    ///   stores the attachment on the root body. Mixing mass-tree and
    ///   frame-tree attachment on the same node is not modelled here;
    ///   detach the body from its mass-tree parent first if it should
    ///   be frame-attached.
    /// - `parent_frame_id` is not a node in the simulation's frame
    ///   tree.
    // JEOD_INV: DB.21 — only unattached bodies integrate (frame-attach gate)
    // JEOD_INV: DB.13 — composite-body propagation delegated to parent frame
    // JEOD_INV: IG.37 — multi-step integrator history reset on topology change
    pub fn attach_to_frame(
        &mut self,
        body_idx: usize,
        parent_frame_id: FrameId,
        attach_offset_position: glam::DVec3,
        t_parent_struct: DMat3,
    ) {
        assert!(
            body_idx < self.bodies.len(),
            "attach_to_frame: body index {body_idx} out of range (have {} bodies)",
            self.bodies.len()
        );
        assert!(
            self.bodies[body_idx].frame_attach.is_none(),
            "attach_to_frame: body {body_idx} is already frame-attached. \
             Call `Simulation::detach_from_frame({body_idx})` before \
             re-attaching to a different parent frame; silent overwrite \
             would lose the original frame-tree relationship and leave \
             the body's offset desynchronized from its actual position."
        );
        // Mass-tree attachment is mutually exclusive with frame
        // attachment in JEOD: `attach_to_frame` resolves the root body
        // first and writes `frame_attach` on that root, while the
        // mass-tree `attach_to(parent_dyn_body)` populates the body's
        // own parent pointer. Allowing both simultaneously would let
        // the per-step pass derive a contradiction (mass-tree parent
        // says one position, frame-tree parent says another); enforce
        // mutual exclusion explicitly.
        if let Some(mass_id) = self.bodies[body_idx].mass_body_id {
            if let Some(tree) = self.mass_tree.as_ref() {
                assert!(
                    tree.parent(mass_id).is_none(),
                    "attach_to_frame: body {body_idx} (mass_body_id {mass_id:?}) \
                     has a mass-tree parent already. Detach from the mass tree \
                     first via `Simulation::detach({body_idx})` — frame attachment \
                     and mass-tree attachment are mutually exclusive (JEOD's \
                     `attach_to_frame` writes `frame_attach` on the integrated \
                     root, not on a child body)."
                );
            }
        }
        // Verify the parent frame is in the tree. `FrameTree::get`
        // panics on out-of-range; we want a more actionable diagnostic
        // (FrameId is a `usize` index into the arena).
        assert!(
            parent_frame_id < self.frame_tree.len(),
            "attach_to_frame: parent_frame_id {parent_frame_id} is not in the \
             simulation's frame tree (frame tree has {} nodes). The frame must \
             be added via `FrameTree::add_root` / `add_child` before attachment.",
            self.frame_tree.len()
        );

        self.bodies[body_idx].frame_attach = Some(FrameAttachState {
            parent_frame_id,
            attach_offset: astrodyn_dynamics::MassPointState {
                position: attach_offset_position,
                t_parent_this: t_parent_struct,
            },
        });

        // Reset multi-step integrator history on the attached body —
        // GJ / ABM4 predictor cache from before the attach is no
        // longer applicable; the next detach will resume integration
        // and the predictor must rebuild from scratch.
        let mass_id = self.bodies[body_idx].mass_body_id;
        if let Some(id) = mass_id {
            Self::mark_body_integrators_dirty_by_id(&mut self.bodies, &[id]);
            Self::reset_body_integrators_by_id(&mut self.bodies, &[id]);
        } else {
            // Body has no mass-tree node (the common case for a
            // single-body sim attached to a planet pfix frame). Reset
            // its integrator state directly without the by-id helper —
            // those helpers are keyed on `MassBodyId` because
            // mass-tree topology changes touch a sorted set of nodes.
            // Here the set is just the one body.
            let body = &mut self.bodies[body_idx];
            if let Some(state) = body.gj_state.as_mut() {
                state.reset();
            }
            if let Some(state) = body.abm4_state.as_mut() {
                state.reset();
            }
        }
    }

    /// Release a body's frame attachment.
    ///
    /// Subsequent `step()` calls integrate the body's translational +
    /// rotational state under its own dynamics, picking up from the
    /// body's instantaneous state at the moment of detach (the
    /// composite-body state last written by
    /// `Self::propagate_frame_attached_state`). Mirrors
    /// `DynBody::detach()` at
    /// `models/dynamics/dyn_body/src/dyn_body_detach.cc:141-143` (the
    /// `frame_attach.isAttached()` branch that calls
    /// `frame_attach.clear_attachment()`).
    ///
    /// Multi-step integrator history is reset (mirrors JEOD
    /// `reset_integrators()` at `dyn_body_detach.cc:271-273`) so the
    /// resumed integration rebuilds its predictor cache from the new
    /// initial condition rather than carrying stale derivatives from
    /// before the attach.
    ///
    /// # Panics
    /// Panics if `body_idx` is out of range or if the body is not
    /// currently frame-attached. A no-op detach would silently mask
    /// caller bugs (e.g., detaching twice in response to a single
    /// release event).
    // JEOD_INV: DB.21 — only unattached bodies integrate (release gate)
    // JEOD_INV: IG.37 — multi-step integrator history reset on topology change
    pub fn detach_from_frame(&mut self, body_idx: usize) {
        assert!(
            body_idx < self.bodies.len(),
            "detach_from_frame: body index {body_idx} out of range (have {} bodies)",
            self.bodies.len()
        );
        assert!(
            self.bodies[body_idx].frame_attach.is_some(),
            "detach_from_frame: body {body_idx} is not currently frame-attached. \
             Call `Simulation::attach_to_frame(...)` before detaching, or \
             remove the duplicate detach to avoid masking caller bugs."
        );
        self.bodies[body_idx].frame_attach = None;

        // Reset integrator history. Symmetric to attach: GJ / ABM4
        // predictor cache from before the attach window expired the
        // moment we called `attach_to_frame`; on detach the body
        // resumes integration and the predictor must rebuild from the
        // new state.
        let mass_id = self.bodies[body_idx].mass_body_id;
        if let Some(id) = mass_id {
            Self::mark_body_integrators_dirty_by_id(&mut self.bodies, &[id]);
            Self::reset_body_integrators_by_id(&mut self.bodies, &[id]);
        } else {
            let body = &mut self.bodies[body_idx];
            if let Some(state) = body.gj_state.as_mut() {
                state.reset();
            }
            if let Some(state) = body.abm4_state.as_mut() {
                state.reset();
            }
        }
    }

    /// Attach a body to a non-body reference frame using a named
    /// subject mass-point and the JEOD `BodyAttachAligned` 180°-yaw
    /// docking convention.
    ///
    /// Port of JEOD `BodyAttachAligned`'s ref-parent branch
    /// (`models/dynamics/body_action/src/body_attach_aligned.cc:111-126`)
    /// composed with the named-point variant of
    /// `DynBody::attach_to_frame`
    /// (`models/dynamics/dyn_body/src/dyn_body_attach.cc:302-365`):
    /// the body's mass-point pose in struct coordinates is composed
    /// with the hardcoded `(zero offset, diag(-1,-1,1))` parent-point
    /// alignment to derive the parent-frame-to-struct offset, then the
    /// existing [`attach_to_frame`](Self::attach_to_frame) populates
    /// `frame_attach` with that pair.
    ///
    /// The body must have a registered mass-tree node
    /// ([`add_body_to_tree`](Self::add_body_to_tree)) and the named
    /// subject point must already exist on it (added via
    /// `MassTree::add_mass_point` or equivalent — `Simulation` does
    /// not yet take mass-point definitions through `VehicleConfig`,
    /// so test / mission setup must reach into `sim.mass_tree` and
    /// add the points after `add_body_to_tree` returns).
    ///
    /// # Panics
    /// All of [`attach_to_frame`](Self::attach_to_frame)'s
    /// preconditions, plus:
    /// - The body has no mass-tree node yet.
    /// - The named mass-point does not exist on the subject body.
    // JEOD_INV: MA.16 — 180° yaw convention for attach-by-point
    // JEOD_INV: MA.21 — named points must exist on body
    pub fn attach_to_frame_aligned(
        &mut self,
        body_idx: usize,
        subject_point_name: &str,
        parent_frame_id: FrameId,
    ) {
        assert!(
            body_idx < self.bodies.len(),
            "attach_to_frame_aligned: body index {body_idx} out of range (have {} bodies)",
            self.bodies.len()
        );
        let mass_id = self.bodies[body_idx].mass_body_id.unwrap_or_else(|| {
            panic!(
                "attach_to_frame_aligned: body {body_idx} has no mass-tree node. \
                 Call `Simulation::add_body_to_tree({body_idx}, ...)` first so the \
                 named mass-point lookup has a body to look up against."
            )
        });
        let tree = self.mass_tree.as_ref().unwrap_or_else(|| {
            panic!(
                "attach_to_frame_aligned: simulation has no mass tree. \
                 Call `Simulation::add_body_to_tree(...)` for the subject body \
                 before requesting a named-point alignment."
            )
        });
        let (offset_pframe_struct, t_pframe_struct) =
            tree.attach_to_frame_offset_aligned(mass_id, subject_point_name);
        self.attach_to_frame(
            body_idx,
            parent_frame_id,
            offset_pframe_struct,
            t_pframe_struct,
        );
    }

    /// Attach a body to a non-body reference frame using a named
    /// subject mass-point with an explicit `(offset, T)` parent-frame
    /// pose for that point.
    ///
    /// Port of JEOD's named-point variant of
    /// `DynBody::attach_to_frame(this_point_name, parent_frame_name,
    /// offset_pframe_cpt, T_pframe_cpt)`
    /// (`models/dynamics/dyn_body/src/dyn_body_attach.cc:302-365`),
    /// without the 180°-yaw alignment hardcode that
    /// [`attach_to_frame_aligned`](Self::attach_to_frame_aligned)
    /// applies. Used by JEOD's `SIM_dyncomp/RUN_attach_to_ref_frame`
    /// `attach_wrap_child_parent_pos_rot` helper, which spells out the
    /// pose of the named subject point in the parent frame and lets the
    /// runner compose with the body's own struct-to-named-point geometry
    /// to derive the parent-frame-to-struct pair the matrix attach takes.
    ///
    /// `offset_pframe_cpt` and `t_pframe_cpt` describe the pose of the
    /// named subject point (`subject_point_name`) in the parent reference
    /// frame's coordinates: the position of the named point and the
    /// rotation matrix from parent-frame axes to the named point's body
    /// axes. The runner composes that with the named point's pose in the
    /// body's struct frame (via [`MassTree::attach_to_frame_offset`]) to
    /// produce the `(offset_pframe_struct, t_pframe_struct)` pair the
    /// underlying [`attach_to_frame`](Self::attach_to_frame) records.
    ///
    /// The body must have a registered mass-tree node
    /// ([`add_body_to_tree`](Self::add_body_to_tree)) and the named
    /// subject point must already exist on it (added via
    /// `MassTree::add_mass_point`).
    ///
    /// # Panics
    /// All of [`attach_to_frame`](Self::attach_to_frame)'s
    /// preconditions, plus:
    /// - The body has no mass-tree node yet.
    /// - The named mass-point does not exist on the subject body.
    ///
    /// [`MassTree::attach_to_frame_offset`]: astrodyn_dynamics::MassTree::attach_to_frame_offset
    // JEOD_INV: MA.21 — named points must exist on body
    pub fn attach_to_frame_named_point(
        &mut self,
        body_idx: usize,
        subject_point_name: &str,
        parent_frame_id: FrameId,
        offset_pframe_cpt: glam::DVec3,
        t_pframe_cpt: DMat3,
    ) {
        assert!(
            body_idx < self.bodies.len(),
            "attach_to_frame_named_point: body index {body_idx} out of range \
             (have {} bodies)",
            self.bodies.len()
        );
        let mass_id = self.bodies[body_idx].mass_body_id.unwrap_or_else(|| {
            panic!(
                "attach_to_frame_named_point: body {body_idx} has no mass-tree \
                 node. Call `Simulation::add_body_to_tree({body_idx}, ...)` first \
                 so the named mass-point lookup has a body to look up against."
            )
        });
        let tree = self.mass_tree.as_ref().unwrap_or_else(|| {
            panic!(
                "attach_to_frame_named_point: simulation has no mass tree. \
                 Call `Simulation::add_body_to_tree(...)` for the subject body \
                 before requesting a named-point attach."
            )
        });
        let (offset_pframe_struct, t_pframe_struct) = tree.attach_to_frame_offset(
            mass_id,
            subject_point_name,
            offset_pframe_cpt,
            t_pframe_cpt,
        );
        self.attach_to_frame(
            body_idx,
            parent_frame_id,
            offset_pframe_struct,
            t_pframe_struct,
        );
    }

    /// Whether the given body is currently attached to a reference
    /// frame.
    ///
    /// # Panics
    /// Panics if `body_idx` is out of range. Mirrors the same guard on
    /// [`Self::attach_to_frame`] / [`Self::detach_from_frame`] so a
    /// stale index produces an actionable diagnostic naming the
    /// misconfiguration rather than a generic slice-bounds panic.
    pub fn is_frame_attached(&self, body_idx: usize) -> bool {
        assert!(
            body_idx < self.bodies.len(),
            "is_frame_attached: body index {body_idx} out of range (have {} bodies)",
            self.bodies.len()
        );
        self.bodies[body_idx].frame_attach.is_some()
    }

    /// Per-step pass that derives every frame-attached body's
    /// composite-body state from its parent reference frame's current
    /// state composed with the captured offset.
    ///
    /// Drives [`astrodyn_dynamics::derive_frame_attached_state`] on each
    /// flagged body. The kernel returns a state in *root-inertial*
    /// coordinates (the parent frame state was lifted via
    /// `frame_tree.compute_relative_state(root, parent)` first); the
    /// writeback lowers back through the body's `IntegOrigin` so the
    /// typed `TranslationalStateTyped<IntegrationFrame>` slot stays
    /// internally consistent.
    ///
    /// Schedule placement in `step_internal` mirrors the kinematic-mass
    /// walk: pre- *and* post-integration. Pre-integration so any
    /// per-step physics that reads `body.trans` (gravity, atmosphere,
    /// derived states) sees the current frame-attached value rather
    /// than the previous tick's; post-integration so callers reading
    /// `Simulation::body(idx)` after `step()` see the freshly-derived
    /// value (and so any frame-attached body's state stays glued to
    /// the parent through stage 8b's frame-switch evaluation, which
    /// can move the parent frame in the same tick).
    ///
    /// Free-flying (non-attached) bodies are skipped — their state is
    /// owned by the integrator. The pass is a no-op when no body is
    /// frame-attached.
    // JEOD_INV: DB.13 — propagate_state delegates to parent frame
    // JEOD_INV: DB.14 — integration-frame switch on attach: the
    // attached body's state is owned by the parent frame, not the
    // integrator
    // JEOD_INV: RF.10 — frame-attach is a shift site: lift parent
    // state to root inertial via `compute_relative_state(root,
    // parent)`, then lower the body's writeback through its own
    // `IntegOrigin`
    pub(crate) fn propagate_frame_attached_state(
        &mut self,
        body_integ_origins: &[astrodyn_quantities::IntegOrigin],
    ) {
        // Snapshot whether any body is attached at all so we can
        // bail before doing the per-body sweep on the common
        // non-attached case (callers that never use `attach_to_frame`
        // pay nothing here).
        if !self.bodies.iter().any(|b| b.frame_attach.is_some()) {
            return;
        }

        let root = self.root_frame_id;
        // The body loop reads `body_integ_origins[body_idx]` and
        // mutates `self.bodies[body_idx]` in the same iteration; the
        // index form is the cleanest way to pair the two without
        // borrowing both as iterators at once. Suppress
        // needless_range_loop because the suggested rewrite (zip /
        // enumerate over either side) reintroduces a borrow conflict
        // with `self.frame_tree` later in the body.
        #[allow(clippy::needless_range_loop)]
        for body_idx in 0..self.bodies.len() {
            let Some(attach) = self.bodies[body_idx].frame_attach else {
                continue;
            };
            // Read the parent frame's current state in root-inertial
            // coordinates (the frame tree's relative-state walk starts
            // from `root` and ends at the parent frame).
            let parent_state = self
                .frame_tree
                .compute_relative_state(root, attach.parent_frame_id);

            // Pull the body's structure → composite-CoM offset so
            // the kernel can apply JEOD's struct → composite step
            // (`dyn_body_propagate_state.cc:571`'s
            // `compute_derived_state_forward(structure,
            // mass.composite_properties, composite_body)`). The
            // captured `attach_offset` is the parent → struct pose
            // (mirrors JEOD `attach_to_frame`'s
            // `frame_attach.initialize_attachment(parent,
            // X_pframe_to_struct)`), so without this step the body
            // would land `|composite_properties.position|` away from
            // JEOD's composite-body trajectory whenever the body's
            // CoM is offset from the structural origin. Atomic
            // bodies whose CoM coincides with the struct origin
            // (`mass.position = ZERO`, the default) reduce the
            // second `propagate_forward` to a numerical no-op.
            //
            // Fall back to a zero offset for 3-DOF bodies that lack
            // a `MassProperties` block entirely (the same `None` path
            // the integrator handles in `step::integrate.rs`); JEOD
            // models 3-DOF bodies with a default-constructed
            // composite_properties whose `position` and
            // `T_parent_this` are zero/identity, matching this
            // fallback.
            let composite_offset = self.bodies[body_idx]
                .mass
                .map(|mp| astrodyn_dynamics::MassPointState {
                    position: mp.position,
                    t_parent_this: mp.t_parent_this,
                })
                .unwrap_or_default();
            // Compose with the captured offset and the body's
            // structure → composite CoM offset to get the body's
            // composite-body inertial state.
            // JEOD_INV: DB.31 — frame-attach derives composite from
            // captured struct offset.
            let derived = astrodyn_dynamics::derive_frame_attached_state(
                astrodyn_dynamics::FrameAttachInputs {
                    parent_frame: parent_state,
                    attach_offset: attach.attach_offset,
                    composite_offset,
                },
            );

            // Lower the kernel's root-inertial output back into
            // integration-frame storage. Bit-identical no-op when the
            // body integrates in the root frame (the common case);
            // for a body in `PlanetInertial<P>` the shift is the only
            // thing that keeps the typed phantom and the actual
            // coordinates consistent. RF.10 shift site.
            let integ_origin = &body_integ_origins[body_idx];
            let trans_root =
                TranslationalStateTyped::<astrodyn::RootInertial>::from_untyped_unchecked(
                    &TranslationalState {
                        position: derived.trans.position,
                        velocity: derived.trans.velocity,
                    },
                );
            self.bodies[body_idx].trans =
                TranslationalStateTyped::<IntegrationFrame>::from_inertial(
                    trans_root,
                    integ_origin,
                );

            // Rotational state writeback is unconditional for now:
            // the kernel always produces an attitude (the parent's
            // rotation composed with the captured `t_parent_struct`).
            // 3-DOF bodies have `rot = None` and we'd otherwise need
            // to either silently drop the kernel's rotational result
            // or panic; mirror the existing `kinematic_only` walk by
            // writing only when the body already carries a `rot` slot.
            if self.bodies[body_idx].rot.is_some() {
                self.bodies[body_idx].rot = Some(RotationalState {
                    quaternion: derived.rot.q_parent_this,
                    ang_vel_body: derived.rot.ang_vel_this,
                });
            }

            // Sync body frame entity in the frame tree so consumers
            // that read the body's frame entity (e.g. another body
            // attaching to *this* body's frame, or a
            // `compute_relative_state` walk that traverses through it)
            // see the same value as `body.trans` / `body.rot`. JEOD's
            // `RefFrameState` carries both translational and rotational
            // state (`models/utils/ref_frames/include/ref_frame_state.hh`
            // — `RefFrameState` is `trans` + `rot`), so the rotational
            // sync is required for correctness whenever the body owns a
            // 6-DOF `rot` slot. Otherwise the node's `q_parent_this` /
            // `t_parent_this` / `ang_vel_this` would carry initialization
            // defaults instead of the freshly-derived attitude.
            let body_frame_id = self.bodies[body_idx].body_frame_id;
            let node = self.frame_tree.get_mut(body_frame_id);
            node.state.trans.position = self.bodies[body_idx].trans.position.raw_si();
            node.state.trans.velocity = self.bodies[body_idx].trans.velocity.raw_si();
            // JEOD_INV: RF.04 — recompute T_parent_this from the new
            // q_parent_this so callers reading either form see the same
            // attitude.
            if let Some(body_rot) = self.bodies[body_idx].rot {
                node.state.rot.q_parent_this = body_rot.quaternion;
                node.state.rot.t_parent_this = body_rot.quaternion.left_quat_to_transformation();
                node.state.rot.ang_vel_this = body_rot.ang_vel_body;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimulationBuilderExt;
    use astrodyn::recipes::Mission;
    use glam::DVec3;

    /// Attach the ISS-LEO recipe vehicle to Earth's *inertial* frame.
    /// Because the parent frame's state is a pure no-op (Earth.inertial
    /// IS the simulation's root frame in `Mission::iss_leo`), the
    /// attached body's composite-body state must equal the captured
    /// offset on every step — the body has stopped integrating and is
    /// frozen at the attach offset.
    #[test]
    fn attach_to_inertial_frame_freezes_body_at_offset() {
        let mut sim = Mission::iss_leo()
            .into_builder()
            .build()
            .expect("Mission::iss_leo must validate");

        // Earth's inertial frame == root in this recipe; pick any
        // canonical attach offset that's not the body's current state.
        let earth_inertial = sim.source_inertial_frame_id(0);
        let attach_offset_pos = DVec3::new(7e6, 0.0, 0.0);
        let attach_t_parent_struct = DMat3::IDENTITY;

        sim.attach_to_frame(0, earth_inertial, attach_offset_pos, attach_t_parent_struct);

        // Step 100 times. The integrator must not run on body 0; the
        // per-step propagation must overwrite its state with the
        // (frozen) parent-frame composition every tick. Earth.inertial
        // is the root, so the parent-frame state is identically zero
        // and the body should sit exactly at the offset position with
        // zero velocity for the entire run.
        for _ in 0..100 {
            sim.step().expect("step must not fail");
            let out = sim.body(0);
            assert!(
                (out.trans.position - attach_offset_pos).length() < 1e-9,
                "frame-attached body drifted from offset: got {:?}",
                out.trans.position
            );
            assert!(
                out.trans.velocity.length() < 1e-9,
                "frame-attached body picked up spurious velocity: got {:?}",
                out.trans.velocity
            );
        }
    }

    /// Attach to a *rotating* frame: the body picks up the parent
    /// frame's `ω × r` velocity each tick. Verifies the runner's
    /// per-step propagation actually reads frame state (rather than
    /// caching a one-time value at attach), and that the math kernel
    /// flows through end-to-end. Uses Earth's pfix frame, which the
    /// `Mission::iss_leo` recipe configures with an Earth-rotation
    /// model.
    #[test]
    fn attach_to_pfix_frame_picks_up_omega_cross_r_velocity() {
        let mut sim = Mission::iss_leo()
            .into_builder()
            .build()
            .expect("Mission::iss_leo must validate");

        // Step once to sync ephemeris / planet rotation so pfix has a
        // populated state.
        sim.step().expect("seed step must not fail");

        let pfix_id = sim
            .source_pfix_frame_id(0)
            .expect("Earth source has a pfix frame in iss_leo recipe");

        // Attach 6.371e6 m along the parent-frame +x axis (equator,
        // prime meridian for an identity rotation between pfix and
        // Earth.inertial in this snapshot).
        let r_eq = 6.371e6_f64;
        let offset_pos = DVec3::new(r_eq, 0.0, 0.0);
        sim.attach_to_frame(0, pfix_id, offset_pos, DMat3::IDENTITY);

        sim.step().expect("post-attach step must not fail");

        let out = sim.body(0);
        // The pfix frame moves the offset point at ω × r in inertial
        // coordinates. Even with the small step the velocity should be
        // dominated by the rotation contribution (~465 m/s at the
        // equator), and crucially nonzero — the integrator-only path
        // would have left the body at its initial orbit velocity
        // (~7.6 km/s along y, way larger). What we are checking is
        // that the body's velocity was *replaced* by the parent-frame
        // composition, not just adjusted by an integrator step.
        let speed = out.trans.velocity.length();
        assert!(
            (200.0..=600.0).contains(&speed),
            "expected ~465 m/s sidereal-rotation contribution at the equator, got {speed} m/s"
        );
    }

    /// `is_frame_attached` reports the current attachment state, and
    /// `detach_from_frame` clears it. After detach, the integrator
    /// resumes producing nonzero velocity changes on each step (the
    /// body falls back into its own dynamics, with whatever state the
    /// frame-attached propagation left it in).
    #[test]
    fn detach_resumes_integration() {
        let mut sim = Mission::iss_leo()
            .into_builder()
            .build()
            .expect("Mission::iss_leo must validate");

        let earth_inertial = sim.source_inertial_frame_id(0);
        let attach_offset = DVec3::new(7e6, 0.0, 0.0);
        sim.attach_to_frame(0, earth_inertial, attach_offset, DMat3::IDENTITY);
        assert!(sim.is_frame_attached(0));

        for _ in 0..10 {
            sim.step().expect("attach-phase step");
        }
        // Body should still be frozen at the offset, with zero velocity.
        let frozen = sim.body(0);
        assert!((frozen.trans.position - attach_offset).length() < 1e-9);
        assert!(frozen.trans.velocity.length() < 1e-9);

        sim.detach_from_frame(0);
        assert!(!sim.is_frame_attached(0));

        // Integrate one step. The body has zero velocity from the
        // attach window, so a single step under Earth's gravity must
        // produce a nonzero velocity change in the radial direction
        // (gravity points toward origin, so velocity gains a -x
        // component over the step).
        sim.step().expect("detached step");
        let after = sim.body(0);
        assert!(
            after.trans.velocity.length() > 0.0,
            "after detach the integrator should have produced nonzero velocity, \
             got {:?}",
            after.trans.velocity
        );
    }

    /// Double-attach must panic — the second `attach_to_frame` would
    /// silently overwrite the captured offset and the parent-frame
    /// reference, masking caller bugs that re-issue an attach without
    /// detaching first.
    #[test]
    #[should_panic(expected = "already frame-attached")]
    fn double_attach_panics() {
        let mut sim = Mission::iss_leo()
            .into_builder()
            .build()
            .expect("Mission::iss_leo must validate");
        let earth_inertial = sim.source_inertial_frame_id(0);
        sim.attach_to_frame(0, earth_inertial, DVec3::ZERO, DMat3::IDENTITY);
        sim.attach_to_frame(0, earth_inertial, DVec3::ZERO, DMat3::IDENTITY);
    }

    /// `attach_to_frame_aligned` produces the same frozen-at-offset
    /// behaviour as the explicit-offset path when the parent frame is
    /// inertial (zero state) and the named mass-point yields offset
    /// `(10, 0, 0)`. Pins that the named-point alignment plumbing
    /// produces the JEOD-derived `(offset, T)` pair end-to-end (mass
    /// tree lookup → algebra → `attach_to_frame` writeback) without
    /// regressing the captured-offset semantics.
    #[test]
    fn attach_to_frame_aligned_freezes_at_named_point_offset() {
        let mut sim = Mission::iss_leo()
            .into_builder()
            .build()
            .expect("Mission::iss_leo must validate");

        // Register the body in the mass tree and add the SIM_ref_attach
        // pt2pt mass-point: `attach1` at (10, 0, 0) with identity
        // orientation. Passing identity orientation reduces the
        // `BodyAttachAligned` algebra to `(offset = (10, 0, 0),
        // T = diag(-1, -1, 1))`, which is the same pair as the matrix
        // form (verified by the unit test in astrodyn_dynamics).
        sim.add_body_to_tree(0, "target");
        let mass_id = sim
            .body_mass_id(0)
            .expect("just registered the body in the mass tree");
        sim.mass_tree.as_mut().unwrap().add_mass_point(
            mass_id,
            "attach1",
            DVec3::new(10.0, 0.0, 0.0),
            DMat3::IDENTITY,
        );

        let earth_inertial = sim.source_inertial_frame_id(0);
        sim.attach_to_frame_aligned(0, "attach1", earth_inertial);

        // Same freeze property as `attach_to_inertial_frame_freezes_body_at_offset`,
        // but the offset came from the named-point alignment instead
        // of being passed in by the caller.
        for _ in 0..50 {
            sim.step().expect("step must not fail");
            let out = sim.body(0);
            assert!(
                (out.trans.position - DVec3::new(10.0, 0.0, 0.0)).length() < 1e-9,
                "frame-attached body drifted from (10,0,0): got {:?}",
                out.trans.position
            );
            assert!(
                out.trans.velocity.length() < 1e-9,
                "frame-attached body picked up spurious velocity: got {:?}",
                out.trans.velocity
            );
        }
    }

    /// `attach_to_frame_aligned` panics when the body has no mass-tree
    /// node — the named-point alignment cannot run without a body to
    /// look points up on, and a silent fallback to the explicit-offset
    /// path would mask caller misconfiguration.
    #[test]
    #[should_panic(expected = "has no mass-tree node")]
    fn attach_to_frame_aligned_without_mass_tree_panics() {
        let mut sim = Mission::iss_leo()
            .into_builder()
            .build()
            .expect("Mission::iss_leo must validate");
        let earth_inertial = sim.source_inertial_frame_id(0);
        sim.attach_to_frame_aligned(0, "attach1", earth_inertial);
    }

    /// Detach without a prior attach must panic — silently no-op'ing
    /// a stale detach would mask paired-event bugs in mission code.
    #[test]
    #[should_panic(expected = "not currently frame-attached")]
    fn unpaired_detach_panics() {
        let mut sim = Mission::iss_leo()
            .into_builder()
            .build()
            .expect("Mission::iss_leo must validate");
        sim.detach_from_frame(0);
    }

    /// Frame-attached propagation must sync rotational state to the
    /// body's frame-tree node, not just translational state. Otherwise
    /// any frame-tree consumer that reads the body's frame node — a
    /// `compute_relative_state` walk that traverses through it, or a
    /// future body attaching to *this* body's frame — would observe
    /// `node.state.rot` carrying initialization defaults
    /// (identity quaternion, zero ω) instead of the freshly-derived
    /// attitude. JEOD's `RefFrameState` carries both `trans` and
    /// `rot`; the runner's sync must too.
    #[test]
    fn frame_attached_body_syncs_rot_to_frame_tree_node() {
        // 6-DOF body so the body has a `rot` slot at all.
        let mut sim = Mission::iss_leo_drag()
            .into_builder()
            .build()
            .expect("Mission::iss_leo_drag must validate");

        // One step to populate pfix rotation state.
        sim.step().expect("seed step must not fail");

        let pfix_id = sim
            .source_pfix_frame_id(0)
            .expect("Earth source has a pfix frame in iss_leo_drag recipe");

        // Attach with a non-identity rotation between pfix and body so
        // a missed sync (leaving the node at identity) would show up
        // as a numerically distinguishable drift between the body's
        // quaternion and the frame node's quaternion.
        let theta = 0.7_f64;
        let (s, c) = (theta.sin(), theta.cos());
        let t_pfix_to_body = DMat3::from_cols(
            DVec3::new(c, s, 0.0),
            DVec3::new(-s, c, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        sim.attach_to_frame(0, pfix_id, DVec3::new(6.371e6, 0.0, 0.0), t_pfix_to_body);

        // Step under frame-attached propagation. The body's rot is
        // overwritten by the kernel's `parent ⊕ offset` composition;
        // the frame-tree node sync must follow.
        sim.step().expect("post-attach step must not fail");

        let body = &sim.bodies[0];
        let body_rot = body
            .rot
            .expect("iss_leo_drag is 6-DOF — body must carry rot");
        let body_frame_id = body.body_frame_id;
        let node_state = sim.frame_tree().get(body_frame_id).state;

        // Quaternion + transformation cache must equal the body's rot.
        assert!(
            (node_state.rot.q_parent_this.scalar() - body_rot.quaternion.scalar()).abs() < 1e-12
                && (node_state.rot.q_parent_this.vector() - body_rot.quaternion.vector()).length()
                    < 1e-12,
            "frame-tree node q_parent_this {:?} desync from body rot {:?}",
            node_state.rot.q_parent_this,
            body_rot.quaternion
        );
        let t_expected = body_rot.quaternion.left_quat_to_transformation();
        for col in 0..3 {
            assert!(
                (node_state.rot.t_parent_this.col(col) - t_expected.col(col)).length() < 1e-12,
                "frame-tree node t_parent_this column {col} desync from quaternion-derived T"
            );
        }
        assert!(
            (node_state.rot.ang_vel_this - body_rot.ang_vel_body).length() < 1e-12,
            "frame-tree node ang_vel_this {:?} desync from body ang_vel_body {:?}",
            node_state.rot.ang_vel_this,
            body_rot.ang_vel_body
        );
    }
}
