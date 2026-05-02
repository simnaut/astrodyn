//! Body lifecycle, accessors, setters, and contact-pair registration
//! for [`super::Simulation`].
//!
//! Methods: `register_contact_pair`, `num_contact_pairs`, `add_body`,
//! `body`, `convert_body_trans_core_to_composite`, `body_core_inertial`,
//! `subtree_composite_inertial`, `srp_plate_temperatures`, body setters
//! (`set_body_external_force`, `set_body_external_torque`,
//! `set_body_position`, `set_body_velocity`, `set_body_mass`),
//! `sync_body_mass_from_tree`.

use glam::DVec3;

use jeod_dynamics::{MassBodyId, MassPointState};
use jeod_frames::{RefFrameKind, RefFrameRot, RefFrameState, RefFrameTrans};
use jeod_sim::{
    evaluate_ground_contact_pair, ContactFacet, GroundFacet, IntegrationFrame, MassProperties,
    Phase, Position, VehicleConfig, Velocity,
};

use super::types::{
    ContactPairConfig, GroundContactImpulse, GroundContactPairConfig, SimBody, VehicleOutput,
};
use super::Simulation;

impl Simulation {
    /// Register a contact interaction between two bodies.
    ///
    /// Once registered, contact forces between these facets are evaluated at
    /// each RK4 stage of [`step`](Self::step). This matches JEOD's
    /// derivative-class `check_contact()` scheduling in `SIM_contact`.
    ///
    /// The force acts on body A (`facet_a`); the equal-and-opposite force
    /// acts on body B. Torques are accumulated about each body's CoM.
    ///
    /// # Panics
    /// * Called after the first [`step`](Self::step). Contact-pair
    ///   registration is initialization-only — JEOD wires it at
    ///   `P_BODY("initialization")` (JEOD_INV: IN.38).
    /// * Either `body_a` or `body_b` is out of range for the registered bodies.
    /// * `body_a == body_b` — contact pair bodies must be distinct
    ///   (JEOD_INV: IN.30, matching JEOD's `unique_pair` invariant).
    /// * `facet_a.material != facet_b.material`. JEOD parks the
    ///   spring/damper/friction parameters on a single `SpringPairInteraction`
    ///   per pair, so both facets must carry identical
    ///   [`ContactMaterial`](jeod_interactions::ContactMaterial) values.
    ///   Panic here instead of deferring until the first integrator step.
    pub fn register_contact_pair(
        &mut self,
        body_a: usize,
        facet_a: ContactFacet,
        body_b: usize,
        facet_b: ContactFacet,
    ) {
        // JEOD_INV: IN.38 — JEOD wires `Contact::register_contact` at
        // `P_BODY("initialization")` (`SIM_ground_contact/S_modules/sv_dyn.sm:130-133`)
        // so it runs exactly once before integration starts. Our API
        // surface allows the call at any time, so guard explicitly: the
        // first step computes init-phase impulses against `t=0` body
        // state, and silently mixing those into a running trajectory
        // would inject a spurious impulse.
        assert!(
            !self.has_stepped,
            "register_contact_pair: contact-pair registration is initialization-only — \
             must be called before the first `step()` (JEOD wires this at \
             P_BODY(\"initialization\") in sv_dyn.sm)"
        );
        assert!(
            body_a < self.bodies.len(),
            "register_contact_pair: body_a index {body_a} out of range ({} bodies)",
            self.bodies.len()
        );
        assert!(
            body_b < self.bodies.len(),
            "register_contact_pair: body_b index {body_b} out of range ({} bodies)",
            self.bodies.len()
        );
        // JEOD_INV: IN.30 — contact pair bodies must be distinct (JEOD `unique_pair`)
        assert_ne!(
            body_a, body_b,
            "register_contact_pair: body A and body B must be distinct (got both = {body_a})"
        );
        // JEOD pairs a single `SpringPairInteraction` to each facet pair, so
        // both facets must carry identical material parameters. Enforce here
        // rather than inside `compute_contact_force` at first step.
        assert_eq!(
            facet_a.material, facet_b.material,
            "register_contact_pair: facet_a.material and facet_b.material must be equal \
             (JEOD pairs a single SpringPairInteraction to each facet pair)"
        );
        self.contact_pairs.push(ContactPairConfig {
            body_a,
            facet_a,
            body_b,
            facet_b,
        });
    }

    /// Number of registered contact pairs.
    pub fn num_contact_pairs(&self) -> usize {
        self.contact_pairs.len()
    }

    /// Register a ground-contact interaction between a vehicle and a
    /// planetary surface.
    ///
    /// Once registered, ground contact forces on `body_a` are evaluated
    /// at each RK4 stage of [`step`](Self::step) — matching JEOD's
    /// derivative-class `check_contact_ground()` job in
    /// `SIM_ground_contact/S_modules/contact.sm`.
    ///
    /// The first call also pins the planet source whose `pfix` rotation
    /// will be queried for terrain lookups; subsequent registrations must
    /// use the same `planet_source`. For [`SphericalTerrain`](jeod_sim::SphericalTerrain)
    /// the pfix rotation cancels in the ground-point computation and
    /// `planet_source` is documentation-only — but we still validate
    /// consistency to keep ground-contact registrations explicit.
    ///
    /// # Panics
    /// * Called after the first [`step`](Self::step). Ground-contact-pair
    ///   registration is initialization-only — JEOD wires it at
    ///   `P_BODY/P_DYN("initialization")` (JEOD_INV: IN.38).
    /// * `body_a` is out of range for the registered bodies.
    /// * `body_a` lacks a `RotationalState` or [`MassProperties`]
    ///   (ground contact requires 6-DOF + mass — checked here so the
    ///   coupled-RK4 path can rely on it without re-checking per stage).
    /// * `vehicle_facet.material != ground_facet.material` (JEOD pairs a
    ///   single `SpringPairInteraction` per facet pair).
    /// * `ground_facet.active == false` (JEOD_INV: IN.35).
    /// * `ground_facet.alt_offset` is not finite (JEOD_INV: IN.36).
    /// * `planet_source` is out of range for the registered gravity
    ///   sources, or differs from a previously-registered ground-contact
    ///   pair's `planet_source` (all ground pairs must share one
    ///   `pfix` rotation).
    pub fn register_ground_contact_pair(
        &mut self,
        body_a: usize,
        vehicle_facet: ContactFacet,
        ground_facet: GroundFacet,
        planet_source: usize,
    ) {
        // JEOD_INV: IN.38 — JEOD wires `Contact::register_contact` /
        // `ContactGround::register_ground_facet` at
        // `P_BODY("initialization")` / `P_DYN("initialization")`
        // (`SIM_ground_contact/S_modules/sv_dyn.sm:130-133` and
        // `contact.sm:70-72`) so registration runs exactly once before
        // integration. The init-phase impulse stored in
        // `pending_initial_impulse` is computed against `t=0` body
        // state and consumed at stage 1 of the first step; injecting
        // that into an already-stepped run would corrupt the
        // trajectory with a spurious impulse independent of vehicle
        // altitude. Mirror JEOD's structural guarantee with a runtime
        // assert.
        assert!(
            !self.has_stepped,
            "register_ground_contact_pair: ground-contact-pair registration is \
             initialization-only — must be called before the first `step()` \
             (JEOD wires this at P_BODY/P_DYN(\"initialization\") in \
             SIM_ground_contact's sv_dyn.sm and contact.sm)"
        );
        assert!(
            body_a < self.bodies.len(),
            "register_ground_contact_pair: body_a index {body_a} out of range ({} bodies)",
            self.bodies.len()
        );
        // JEOD pairs a single SpringPairInteraction per facet pair. Both
        // sides carry identical material (vehicle "steel" vs ground
        // "dirt" reduce to a single pair material in JEOD's lookup).
        assert_eq!(
            vehicle_facet.material, ground_facet.material,
            "register_ground_contact_pair: vehicle_facet.material and \
             ground_facet.material must be equal (JEOD pairs a single \
             SpringPairInteraction per facet pair)"
        );
        // JEOD_INV: IN.35 — only active GroundFacets contribute force.
        assert!(
            ground_facet.active,
            "register_ground_contact_pair: ground_facet.active must be true"
        );
        // JEOD_INV: IN.36 — alt_offset must be finite.
        assert!(
            ground_facet.alt_offset.is_finite(),
            "register_ground_contact_pair: ground_facet.alt_offset must be finite, got {}",
            ground_facet.alt_offset
        );
        assert!(
            planet_source < self.gravity_data.len(),
            "register_ground_contact_pair: planet_source index {planet_source} out of range \
             ({} sources)",
            self.gravity_data.len()
        );
        match self.ground_contact_planet_source {
            None => self.ground_contact_planet_source = Some(planet_source),
            Some(prev) => assert_eq!(
                prev, planet_source,
                "register_ground_contact_pair: all ground-contact pairs must reference the \
                 same planet source (got {planet_source}, previously registered with {prev})"
            ),
        }
        // Compute JEOD's initialization-time impulse (pre-propagation
        // `GroundInteraction::initialize → in_contact()` with
        // `vp.state.trans.position == (0, 0, 0)`). This is the impulsive
        // force JEOD records on `subject->force` during init and that
        // the integrator consumes at stage 1 of the first step.
        //
        // For non-spherical Terrain implementations, the pfix rotation
        // matters in the ground-point computation, so we fetch the
        // current value from the frame tree. SphericalTerrain happens
        // to cancel the rotation out, but we don't special-case it
        // here — the matrix is correct for whatever Terrain the caller
        // provides. Callers using non-trivial planet rotation should
        // ensure ephemeris/RNP has been propagated before
        // registering ground-contact pairs; for SphericalTerrain it
        // doesn't matter.
        let t_inertial_pfix = self.source_frame_ids[planet_source]
            .pfix
            .map(|pfix_id| self.frame_tree.get(pfix_id).state.rot.t_parent_this)
            .unwrap_or(glam::DMat3::IDENTITY);
        let body = &self.bodies[body_a];
        let body_rot = body.rot.as_ref().unwrap_or_else(|| {
            panic!(
                "register_ground_contact_pair: body_a={body_a} has no RotationalState; \
                 ground contact requires 6-DOF (set `rot: Some(...)` on the VehicleConfig)"
            )
        });
        let body_mass = body.mass.as_ref().unwrap_or_else(|| {
            panic!(
                "register_ground_contact_pair: body_a={body_a} has no MassProperties; \
                 set `mass: Some(...)` on the VehicleConfig"
            )
        });
        // `body.trans` is `TranslationalStateTyped<IntegrationFrame>` after
        // #258; `evaluate_ground_contact_pair` takes the untyped form.
        let trans_untyped = body.trans.to_untyped();
        let pending_initial_impulse = evaluate_ground_contact_pair(
            &vehicle_facet,
            &ground_facet,
            &trans_untyped,
            body_rot,
            body.t_struct_body,
            body_mass,
            t_inertial_pfix,
            Phase::Initialization,
        )
        .map(|eval| GroundContactImpulse {
            force_inertial: eval.force_on_a,
            torque_body: eval.torque_a_body,
        });
        self.ground_contact_pairs.push(GroundContactPairConfig {
            body_a,
            vehicle_facet,
            ground_facet,
            pending_initial_impulse,
        });
    }

    /// Number of registered ground-contact pairs.
    pub fn num_ground_contact_pairs(&self) -> usize {
        self.ground_contact_pairs.len()
    }

    /// Add a dynamic body from a [`VehicleConfig`]. Returns its index.
    ///
    /// The config is consumed and converted into internal state. Creates a
    /// body frame in the frame tree under the integration frame. Use
    /// [`body`](Simulation::body) to access results after stepping.
    pub fn add_body(&mut self, config: VehicleConfig) -> usize {
        let idx = self.bodies.len();

        // Resolve integration frame from source index.
        let integ_frame_id = config
            .integ_source
            .map(|src| {
                self.source_frame_ids
                    .get(src)
                    .unwrap_or_else(|| {
                        panic!(
                            "VehicleConfig::integ_source index {src} is out of range; \
                             {} source frame(s) configured",
                            self.source_frame_ids.len()
                        )
                    })
                    .inertial
            })
            .unwrap_or(self.root_frame_id);

        // Create body frame in tree under the integration frame.
        let body_frame_id = self.frame_tree.add_child(
            integ_frame_id,
            format!("body_{idx}.integ"),
            RefFrameKind::Body,
            RefFrameState {
                trans: RefFrameTrans {
                    // VehicleConfig::trans is integration-frame; the frame
                    // tree node lives in the parent (integration) frame, so
                    // values copy directly with no shift.
                    position: config.trans.position,
                    velocity: config.trans.velocity,
                },
                rot: RefFrameRot::default(),
            },
        );

        self.bodies
            .push(SimBody::from_config(config, integ_frame_id, body_frame_id));
        idx
    }

    // JEOD_INV: DS.01 — derived state config immutable after init; read-only access only
    /// Get the current output state of a body by index.
    ///
    /// Returns a [`VehicleOutput`] containing the current integrated state
    /// plus any derived states that were configured.
    pub fn body(&self, idx: usize) -> VehicleOutput {
        self.bodies[idx].output()
    }

    /// Adjust an integrated body's `trans` from a `core_body` inertial
    /// state to the corresponding `composite_body` inertial state,
    /// using the current mass tree's `core_wrt_composite` offset.
    ///
    /// JEOD's integration variable is `composite_body`; at init time,
    /// however, callers often have JEOD-published values that were
    /// logged from `core_body` (as in our `tier3_sim_apollo_trajectory`
    /// reference CSV). Use this once after the mass tree has reached
    /// its initial topology to flip the interpretation.
    ///
    /// `body.rot` is unchanged — composite and core share body axes
    /// (see `body_core_inertial` for the full convention).
    ///
    /// # Panics
    /// Panics if the body is not registered in the mass tree, or has
    /// no rotational state.
    pub fn convert_body_trans_core_to_composite(&mut self, idx: usize) {
        let cw_inertial;
        let dvel_inertial;
        {
            let body = &self.bodies[idx];
            let mass_body_id = body.mass_body_id.expect(
                "convert_body_trans_core_to_composite: body is not registered in the mass tree",
            );
            let tree = self
                .mass_tree
                .as_ref()
                .expect("convert_body_trans_core_to_composite: no mass tree configured");
            let node = tree.get(mass_body_id);
            let cw_struct = node.core_wrt_composite.position;
            let t_struct_to_body = node.composite_properties.t_parent_this;
            let cw_body = t_struct_to_body * cw_struct;
            let body_rot = body
                .rot
                .expect("convert_body_trans_core_to_composite: 6-DOF body required");
            let t_inertial_to_body = body_rot.quaternion.left_quat_to_transformation();
            let t_body_to_inertial = t_inertial_to_body.transpose();
            cw_inertial = t_body_to_inertial * cw_body;
            dvel_inertial = t_body_to_inertial * body_rot.ang_vel_body.cross(cw_body);
        }
        // composite = core − cw_inertial; subtract the rigid-body
        // ω × r contribution on velocity. All values stay in the body's
        // integration frame.
        let trans = &mut self.bodies[idx].trans;
        trans.position =
            Position::<IntegrationFrame>::from_raw_si(trans.position.raw_si() - cw_inertial);
        trans.velocity =
            Velocity::<IntegrationFrame>::from_raw_si(trans.velocity.raw_si() - dvel_inertial);
    }

    /// Derive the integrated body's `core_body` inertial position +
    /// velocity from its `composite_body` integration state and the
    /// current mass tree.
    ///
    /// JEOD integrates `composite_body` (matching
    /// `DynamicsIntegrationGroup::gravitation()` and
    /// `DynBody::trans_integ()`), so during stages 1–6 of the JEOD
    /// integration loop `body.trans` represents the composite. The
    /// `core_body` frame is the per-body CoM, derived via the mass
    /// tree's `core_wrt_composite` offset rotated to inertial.
    ///
    /// In our mass tree `composite_properties.t_parent_this` is left
    /// equal to `core_properties.t_parent_this` (set per-body at init,
    /// not re-derived for the merged composite), so composite and core
    /// share body axes — only position and velocity differ. Returns
    /// `(position, velocity)` in the body's integration-frame inertial
    /// coordinates (same frame as `body.trans`).
    ///
    /// # Panics
    /// Panics if the body is not registered in the mass tree, or if
    /// the body has no rotational state (6-DOF required to derive the
    /// kinematic offset).
    pub fn body_core_inertial(&self, idx: usize) -> (DVec3, DVec3) {
        let body = &self.bodies[idx];
        let mass_body_id = body
            .mass_body_id
            .expect("body_core_inertial: body is not registered in the mass tree");
        let tree = self
            .mass_tree
            .as_ref()
            .expect("body_core_inertial: no mass tree configured");
        let node = tree.get(mass_body_id);
        let core_wrt_composite_struct = node.core_wrt_composite.position;
        // Struct → composite-body rotation (composite shares core's body axes).
        let t_struct_to_body = node.composite_properties.t_parent_this;
        let cw_body = t_struct_to_body * core_wrt_composite_struct;
        // Body → inertial via T_inertial_to_body⁻¹ = T_inertial_to_body.transpose().
        let body_rot = body.rot.expect("body_core_inertial: 6-DOF body required");
        let t_inertial_to_body = body_rot.quaternion.left_quat_to_transformation();
        let t_body_to_inertial = t_inertial_to_body.transpose();
        let cw_inertial = t_body_to_inertial * cw_body;
        let core_position = body.trans.position.raw_si() + cw_inertial;
        // v_core = v_composite + ω × r (in inertial frame). ω in body
        // frame is body.rot.ang_vel_body; rotate the cross-product to
        // inertial.
        let omega_body = body_rot.ang_vel_body;
        let core_velocity =
            body.trans.velocity.raw_si() + t_body_to_inertial * omega_body.cross(cw_body);
        (core_position, core_velocity)
    }

    /// Resolve the `composite_body` inertial state of any mass-tree body,
    /// regardless of whether it is the integrated body, attached as a
    /// child of the integrated body, or the root of a detached subtree.
    ///
    /// Walks to the tree root, reads the root's inertial composite state
    /// from either the integrated-body slot ([`Self::body`]) or the
    /// [`Self::detached_subtrees`] map (when the root is detached), then
    /// chain-walks down to `target_id` using the same body-aware step as
    /// [`Self::detach_subtree`] (with [`jeod_dynamics::propagate_forward`]
    /// at each level).
    ///
    /// This mirrors what JEOD's truth recorder logs for `lm_dyn.composite_body`
    /// regardless of the LM's current attach state — the
    /// `tier3_sim_apollo_lm_state_vs_truth` diagnostic compares against
    /// `apollo_attach_truth.csv` rows produced by that same recorder.
    ///
    /// # Panics
    /// Panics if no mass tree is configured, `target_id` is not in the
    /// tree, the root has no integrated body and no detached-subtree
    /// entry, or the integrated root is missing rotational state.
    pub fn subtree_composite_inertial(&self, target_id: MassBodyId) -> RefFrameState {
        let tree = self
            .mass_tree
            .as_ref()
            .expect("subtree_composite_inertial: no mass tree configured");
        // Walk to root.
        let mut root_id = target_id;
        while let Some(p) = tree.parent(root_id) {
            root_id = p;
        }

        // Resolve the root's inertial composite_body state.
        let integrated_idx = self
            .bodies
            .iter()
            .position(|b| b.mass_body_id == Some(root_id));
        let root_state: RefFrameState = if let Some(idx) = integrated_idx {
            let body = &self.bodies[idx];
            let body_rot = body.rot.expect(
                "subtree_composite_inertial: integrated root has no rotational state \
                 (6-DOF required)",
            );
            RefFrameState {
                trans: RefFrameTrans {
                    position: body.trans.position.raw_si(),
                    velocity: body.trans.velocity.raw_si(),
                },
                rot: RefFrameRot {
                    q_parent_this: body_rot.quaternion,
                    t_parent_this: body_rot.quaternion.left_quat_to_transformation(),
                    ang_vel_this: body_rot.ang_vel_body,
                },
            }
        } else if let Some(detached) = self.detached_subtrees.get(&root_id) {
            detached.to_ref_frame_state()
        } else {
            panic!(
                "subtree_composite_inertial: tree root {root_id:?} for target \
                 {target_id:?} has no integrated body and no detached-subtree entry"
            );
        };

        // Build the chain root → target and walk down with the body-aware step.
        let mut chain = Vec::<MassBodyId>::new();
        let mut cur = target_id;
        while cur != root_id {
            chain.push(cur);
            cur = tree.parent(cur).expect(
                "subtree_composite_inertial: chain walk hit a parentless intermediate \
                 (target lost its parent during traversal)",
            );
        }
        chain.reverse();

        let mut current_state = root_state;
        let mut current_node_id = root_id;
        for next_id in chain {
            let next_node = tree.get(next_id);
            let current_node = tree.get(current_node_id);
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
        current_state
    }

    /// Read the current per-plate temperatures (K) for a body's flat-plate
    /// SRP configuration, or `None` if the body has no flat-plate SRP.
    ///
    /// Useful for unit tests and data recording; returns a reference so no
    /// allocation is needed for the common steady-state case.
    pub fn srp_plate_temperatures(&self, body_idx: usize) -> Option<&[f64]> {
        self.bodies[body_idx]
            .flat_plate_state
            .as_ref()
            .map(|fps| fps.temperatures.as_slice())
    }

    /// Set the externally applied force (inertial frame, N) for a body.
    ///
    /// Added to `total_force.force` each step after force collection.
    pub fn set_body_external_force(&mut self, idx: usize, force: DVec3) {
        self.bodies[idx].external_force = force;
    }

    /// Set the externally applied torque (body frame, N*m) for a body.
    ///
    /// Added to `total_force.torque` each step after force collection.
    pub fn set_body_external_torque(&mut self, idx: usize, torque: DVec3) {
        self.bodies[idx].external_torque = torque;
    }

    /// Set a body's translational position (inertial frame, m).
    ///
    /// Used for prescribed-motion tests where position is set externally
    /// at each timestep (e.g., SIM_2A_SHADOW_CALC).
    pub fn set_body_position(&mut self, idx: usize, position: DVec3) {
        self.bodies[idx].trans.position = Position::<IntegrationFrame>::from_raw_si(position);
        let fid = self.bodies[idx].body_frame_id;
        self.frame_tree.get_mut(fid).state.trans.position = position;
    }

    /// Set a body's translational velocity (inertial frame, m/s).
    ///
    /// Used for impulsive maneuvers (e.g., Apollo TLI delta-V).
    pub fn set_body_velocity(&mut self, idx: usize, velocity: DVec3) {
        self.bodies[idx].trans.velocity = Velocity::<IntegrationFrame>::from_raw_si(velocity);
        let fid = self.bodies[idx].body_frame_id;
        self.frame_tree.get_mut(fid).state.trans.velocity = velocity;
    }

    /// Replace a body's mass properties.
    ///
    /// Used for discrete mass changes (e.g., post-burn fuel consumption,
    /// stage separation). Recomputes `inverse_mass` and `inverse_inertia`.
    ///
    /// **Warning:** If the body is registered in the mass tree, calling this
    /// method will desynchronize the body's mass from the tree's copy. Use
    /// [`sync_body_mass_from_tree`](Self::sync_body_mass_from_tree) instead
    /// when the mass tree has been modified via `attach`/`detach`.
    pub fn set_body_mass(&mut self, idx: usize, mut mass: MassProperties) {
        mass.dirty = true;
        mass.recompute_derived();
        self.bodies[idx].mass = Some(mass);
    }

    /// Sync a body's mass properties from the mass tree's composite.
    ///
    /// After modifying the mass tree via `attach`/`detach`, call this to
    /// update the body's mass from the tree's composite properties.
    ///
    /// # Panics
    /// Panics if the body is not registered in the mass tree.
    pub fn sync_body_mass_from_tree(&mut self, idx: usize) {
        let id = self.bodies[idx]
            .mass_body_id
            .expect("sync_body_mass_from_tree requires body registered in mass tree");
        let tree = self
            .mass_tree
            .as_ref()
            .expect("sync_body_mass_from_tree requires a mass tree");
        let mut composite = tree.get(id).composite_properties;
        composite.dirty = true;
        composite.recompute_derived();
        self.bodies[idx].mass = Some(composite);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jeod_interactions::{ContactMaterial, SphericalTerrain};
    use jeod_time::leap_second::default_leap_second_table;
    use jeod_time::SimulationTime;
    use std::sync::Arc;

    fn empty_sim() -> Simulation {
        let time = SimulationTime::new(0.0, default_leap_second_table());
        Simulation::new(time, 1.0)
    }

    fn dummy_material() -> ContactMaterial {
        ContactMaterial::jeod_spring(1.0, 1.0, 0.5)
    }

    #[test]
    #[should_panic(expected = "initialization-only")]
    fn register_contact_pair_after_step_panics() {
        // JEOD_INV: IN.38 — registration must precede the first step.
        let mut sim = empty_sim();
        sim.has_stepped = true;
        let mat = dummy_material();
        let facet = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        sim.register_contact_pair(0, facet, 1, facet);
    }

    #[test]
    #[should_panic(expected = "initialization-only")]
    fn register_ground_contact_pair_after_step_panics() {
        // JEOD_INV: IN.38 — registration must precede the first step.
        let mut sim = empty_sim();
        sim.has_stepped = true;
        let mat = dummy_material();
        let veh = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let ground = GroundFacet::new(Arc::new(SphericalTerrain::new(6_378_137.0)), 0.0, mat);
        sim.register_ground_contact_pair(0, veh, ground, 0);
    }
}
