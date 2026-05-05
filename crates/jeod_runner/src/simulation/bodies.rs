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
    ///   registration must precede integration — see
    ///   [`Simulation::has_stepped`](super::Simulation) for the rationale.
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
        // Reject late registration: see `Simulation::has_stepped`.
        assert!(
            !self.has_stepped,
            "register_contact_pair: contact-pair registration must precede the first \
             `step()` — registering after integration starts would inject a t=0 init-phase \
             impulse into a running trajectory"
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
    ///   registration must precede integration — see
    ///   [`Simulation::has_stepped`](super::Simulation) for the rationale.
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
        // Reject late registration: `pending_initial_impulse` is
        // computed against `t=0` body state below and consumed at
        // stage 1 of the first step. Registering mid-run would inject
        // that impulse into an already-propagating trajectory. See
        // `Simulation::has_stepped`.
        assert!(
            !self.has_stepped,
            "register_ground_contact_pair: ground-contact-pair registration must precede \
             the first `step()` — registering after integration starts would inject a t=0 \
             init-phase impulse into a running trajectory"
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

    /// Mass-tree id of the body at index `idx`, or `None` if the body
    /// was added without a mass-tree registration (e.g. a fixed
    /// gravity source). Test code that needs to read live mass-tree
    /// state for assertions on composite mass / parent topology after
    /// runtime attach/detach uses this to translate the body index it
    /// holds into the [`MassBodyId`] the [`MassTree`] indexes by.
    ///
    /// [`MassTree`]: jeod_dynamics::MassTree
    pub fn body_mass_id(&self, idx: usize) -> Option<MassBodyId> {
        self.bodies[idx].mass_body_id
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

    /// Set the externally applied force in the body's structural frame
    /// (N).
    ///
    /// Mirrors JEOD's `Force` model
    /// (`models/dynamics/dyn_body/include/force.hh`): the force vector
    /// is expressed in the structural frame of the body it acts on, and
    /// rotated to inertial at force-collection time using the body's
    /// current attitude. This is what the Tier 3
    /// `SIM_verif_attach_detach` `RUN_compute_child_derivative` Trick
    /// `add_read` snippets actually toggle (`veh1.force.force = […]`
    /// in `input.py`).
    ///
    /// The previous inertial-frame
    /// [`set_body_external_force`](Self::set_body_external_force) entry
    /// point is preserved for callers that already produce
    /// inertial-frame contributions (custom thrusters, debugging, etc.);
    /// the two contribute additively at force collection.
    pub fn set_body_external_force_struct(&mut self, idx: usize, force_struct: DVec3) {
        self.bodies[idx].external_force_struct = force_struct;
    }

    /// Set the externally applied torque in the body's structural frame
    /// (N·m).
    ///
    /// Mirrors JEOD's `Torque` model. Rotated to body frame at force
    /// collection via the body's structural-to-body transform; companion
    /// to [`set_body_external_force_struct`](Self::set_body_external_force_struct).
    pub fn set_body_external_torque_struct(&mut self, idx: usize, torque_struct: DVec3) {
        self.bodies[idx].external_torque_struct = torque_struct;
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

    /// Replace a body's rotational state (attitude quaternion + body
    /// angular velocity).
    ///
    /// Companion to [`set_body_position`](Self::set_body_position) /
    /// [`set_body_velocity`](Self::set_body_velocity) for tests and
    /// scenarios that need to override the recipe-derived initial
    /// rotational state (e.g., the `tier3_sim_ref_attach` Tier 3 test
    /// initializes the SIM_ref_attach target vehicle's YPR attitude).
    /// Panics if the body has no rotational state slot — call
    /// [`add_body`](Self::add_body) with `VehicleConfig::rot = Some(...)`
    /// to register a 6-DOF body first.
    pub fn set_body_rot(&mut self, idx: usize, rot: jeod_sim::RotationalState) {
        assert!(
            self.bodies[idx].rot.is_some(),
            "set_body_rot: body {idx} is 3-DOF (no `RotationalState` slot). \
             Spawn the body with `VehicleConfig::rot = Some(...)` first if \
             rotational integration is needed; otherwise leave the body \
             3-DOF and don't call this setter."
        );
        self.bodies[idx].rot = Some(rot);
        // Mirror the rotational state onto the body's frame-tree node so
        // downstream consumers — `compute_relative_state` walks that
        // traverse through it, or another body attaching to *this*
        // body's frame — see the freshly written attitude / angular
        // velocity instead of the previous step's value. JEOD's
        // `RefFrameState` carries both translational and rotational
        // state by definition; the per-component setters
        // `set_body_position` / `set_body_velocity` already mirror their
        // half, and the frame-attached integration sync in
        // `propagate_frame_attached_state` mirrors both halves at the
        // end of every step. Leaving `rot` out of this setter would
        // silently desync the node until the next integration tick,
        // producing wrong relative-state lookups against this body in
        // the meantime.
        let fid = self.bodies[idx].body_frame_id;
        let node = self.frame_tree.get_mut(fid);
        node.state.rot.q_parent_this = rot.quaternion;
        // JEOD_INV: RF.04 — recompute T_parent_this from the new
        // q_parent_this so callers reading either form see the same
        // attitude.
        node.state.rot.t_parent_this = rot.quaternion.left_quat_to_transformation();
        node.state.rot.ang_vel_this = rot.ang_vel_body;
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
    /// After modifying the mass tree via direct `tree.attach` / `tree.detach`
    /// calls (i.e., obtained via `sim.mass_tree.as_mut()`), call this to
    /// update the body's mass from the tree's composite properties.
    ///
    /// This method is the **documented sync point** for the
    /// `tree.attach`/`tree.detach` + `sync_body_mass_from_tree` path, which
    /// is the lower-level alternative to the high-level
    /// [`attach`](Self::attach) / [`detach`](Self::detach) /
    /// [`detach_subtree`](Self::detach_subtree) /
    /// [`attach_subtree_aligned`](Self::attach_subtree_aligned) methods.
    /// JEOD's `dyn_body_attach.cc::reset_integrators()` (lines 860, 871)
    /// and `dyn_body_detach.cc:271-273` reset multi-step integrator
    /// (Gauss-Jackson, ABM4) predictor/corrector history on every topology
    /// change; this method mirrors that behavior so the lower-level path
    /// stays IG.37-safe by construction. Without this, a Gauss-Jackson or
    /// ABM4 caller using `tree.attach` + `sync_body_mass_from_tree` would
    /// silently propagate stale predictor history across the topology
    /// change, producing wrong physics with no panic.
    ///
    /// Single-step integrators (RK4, RKF4(5)) carry no per-step history
    /// and are no-ops in the reset, so applying the reset unconditionally
    /// here is safe regardless of the body's chosen integrator.
    ///
    /// # Sync every affected body, not just the mutated node
    ///
    /// `MassTree::attach`/`detach` recomputes composite mass properties
    /// up the parent's full ancestor chain to the root. Callers using the
    /// low-level path **must** invoke `sync_body_mass_from_tree` once for
    /// **every** Simulation body whose composite was touched — that is,
    /// the directly mutated child *plus every ancestor of the (former)
    /// parent that is registered as a Simulation body*. Syncing only the
    /// child or only the immediate parent leaves higher ancestors with
    /// stale composite mass and stale multi-step integrator history,
    /// silently producing wrong physics on the next `step()`. The
    /// high-level [`attach`](Self::attach) / [`detach`](Self::detach) /
    /// [`detach_subtree`](Self::detach_subtree) /
    /// [`attach_subtree_aligned`](Self::attach_subtree_aligned) methods
    /// already do this ancestor walk for you (see
    /// `simulation/mass_tree.rs` `affected_ids` collection); this section
    /// applies only when callers bypass them.
    ///
    /// # Panics
    /// Panics if the body is not registered in the mass tree.
    // JEOD_INV: IG.37 — multi-step integrator history must be reset on topology change
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

        // ── IG.37: mark + reset the body's multi-step integrator history.
        //    The two-step pattern (mark dirty, then reset) is deliberate —
        //    it mirrors the high-level attach/detach methods so a future
        //    refactor that drops the reset would leave the dirty flag set
        //    and trigger the IG.37 panic on the next `integrate()` call.
        //    JEOD's `dyn_body_attach.cc::reset_integrators()` (lines 860,
        //    871) and `dyn_body_detach.cc:271-273` are the corresponding
        //    JEOD sites.
        let body = &mut self.bodies[idx];
        if let Some(ref mut gj) = body.gj_state {
            gj.mark_topology_dirty();
        }
        if let Some(ref mut abm) = body.abm4_state {
            abm.mark_topology_dirty();
        }
        jeod_sim::reset_integrators(body.gj_state.as_mut(), body.abm4_state.as_mut());
    }

    /// Clear the kinematic-only flag on a body, returning ownership of
    /// its trans/rot to the integrator.
    ///
    /// `Simulation::detach` already clears this flag automatically on
    /// the freshly-detached child (a kinematic child without a parent
    /// would panic on the next `step()` in `propagate_kinematic_state`,
    /// so detach is the single fail-loud entry point). This method
    /// remains for the rarer case where a caller wants to revoke the
    /// kinematic-only assignment without changing the tree topology
    /// — e.g. swapping a fully-integrated child in place of a
    /// kinematic one mid-mission.
    ///
    /// No-op if the flag is already clear. Does not validate
    /// mass-tree topology; this method is purely a flag flip.
    pub fn clear_kinematic_only(&mut self, idx: usize) {
        self.bodies[idx].kinematic_only = false;
    }

    /// Flag a body as a kinematic child of its mass-tree parent.
    ///
    /// After this call, the body's `trans` + `rot` are derived each
    /// [`step`](Self::step) by
    /// [`propagate_state_via_storage`](jeod_sim::propagate_state_via_storage)
    /// from the parent's freshly-derived state composed with the
    /// link's `MassChildOf` rotation + offset (mirrors JEOD's
    /// `DynBody::propagate_state_from_structure`). The integrator on
    /// this body is **skipped** — only the tree root integrates per
    /// `JEOD_INV: DB.17`.
    ///
    /// # Panics
    /// * The body is not registered in the mass tree (call
    ///   [`add_body_to_tree`](Self::add_body_to_tree) first).
    /// * The body resolves to a tree root (kinematic-only bodies must
    ///   be non-root — call [`attach`](Self::attach) first).
    /// * The body has no [`RotationalState`](jeod_sim::RotationalState):
    ///   kinematic propagation derives both `trans` and `rot`, so
    ///   3-DOF bodies cannot be kinematic children.
    // JEOD_INV: DB.17 — only the root's state is integrated; non-root state is kinematic-derived
    pub fn mark_kinematic_only(&mut self, idx: usize) {
        let body = &self.bodies[idx];
        let id = body.mass_body_id.unwrap_or_else(|| {
            panic!(
                "mark_kinematic_only: SimBody {idx} is not in the mass tree. \
                 Call `add_body_to_tree({idx}, ...)` and `attach({idx}, parent_idx, ...)` first."
            )
        });
        let tree = self
            .mass_tree
            .as_ref()
            .expect("mark_kinematic_only: no mass tree configured");
        assert!(
            tree.parent(id).is_some(),
            "mark_kinematic_only: SimBody {idx} (mass_body_id {id:?}) is a tree root. \
             Kinematic-only bodies must have a parent — call \
             `attach({idx}, parent_idx, offset, t_parent_child)` first."
        );
        assert!(
            body.rot.is_some(),
            "mark_kinematic_only: SimBody {idx} has no RotationalState. \
             Kinematic propagation derives both trans and rot; 3-DOF bodies cannot \
             be kinematic children. Set `VehicleConfig::rot = Some(...)` when \
             constructing the body."
        );
        self.bodies[idx].kinematic_only = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimulationBuilderExt;
    use jeod_interactions::{ContactMaterial, SphericalTerrain};
    use jeod_math::JeodQuat;
    use jeod_sim::recipes::Mission;
    use jeod_sim::RotationalState;
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
    #[should_panic(expected = "must precede the first `step()`")]
    fn register_contact_pair_after_step_panics() {
        let mut sim = empty_sim();
        sim.has_stepped = true;
        let mat = dummy_material();
        let facet = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        sim.register_contact_pair(0, facet, 1, facet);
    }

    #[test]
    #[should_panic(expected = "must precede the first `step()`")]
    fn register_ground_contact_pair_after_step_panics() {
        let mut sim = empty_sim();
        sim.has_stepped = true;
        let mat = dummy_material();
        let veh = ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let ground = GroundFacet::new(Arc::new(SphericalTerrain::new(6_378_137.0)), 0.0, mat);
        sim.register_ground_contact_pair(0, veh, ground, 0);
    }

    /// `set_body_rot` must mirror the new attitude / angular velocity
    /// onto the body's frame-tree node, in the same way
    /// `set_body_position` / `set_body_velocity` mirror their halves.
    /// Otherwise frame-tree consumers (e.g. `compute_relative_state`
    /// walks that traverse through this node, or another body attaching
    /// to *this* body's frame) would silently observe the previous
    /// quaternion / ω until the next integration tick re-syncs the
    /// node — producing wrong relative-state lookups in between.
    #[test]
    fn set_body_rot_syncs_frame_tree_node() {
        // 6-DOF body so the body has a `rot` slot at all.
        let mut sim = Mission::iss_leo_drag()
            .into_builder()
            .build()
            .expect("Mission::iss_leo_drag must validate");

        // Build a non-identity attitude + non-zero angular velocity so a
        // missed sync (leaving the node at the previous step's value)
        // would show up as a numerically distinguishable drift between
        // the body's quaternion and the frame node's quaternion.
        let new_quat =
            JeodQuat::left_quat_from_eigen_rotation(0.7, DVec3::new(1.0, 2.0, 3.0).normalize());
        let new_ang_vel = DVec3::new(0.01, -0.02, 0.03);
        let new_rot = RotationalState {
            quaternion: new_quat,
            ang_vel_body: new_ang_vel,
        };

        sim.set_body_rot(0, new_rot);

        let body = &sim.bodies[0];
        let body_frame_id = body.body_frame_id;
        let node_state = sim.frame_tree().get(body_frame_id).state;

        assert!(
            (node_state.rot.q_parent_this.scalar() - new_quat.scalar()).abs() < 1e-12
                && (node_state.rot.q_parent_this.vector() - new_quat.vector()).length() < 1e-12,
            "frame-tree node q_parent_this {:?} desync from set_body_rot input {:?}",
            node_state.rot.q_parent_this,
            new_quat
        );
        let t_expected = new_quat.left_quat_to_transformation();
        for col in 0..3 {
            assert!(
                (node_state.rot.t_parent_this.col(col) - t_expected.col(col)).length() < 1e-12,
                "frame-tree node t_parent_this column {col} desync from quaternion-derived T"
            );
        }
        assert!(
            (node_state.rot.ang_vel_this - new_ang_vel).length() < 1e-12,
            "frame-tree node ang_vel_this {:?} desync from set_body_rot input {:?}",
            node_state.rot.ang_vel_this,
            new_ang_vel
        );

        // Confirm the body slot itself was also updated.
        let body_rot = sim.bodies[0]
            .rot
            .expect("iss_leo_drag is 6-DOF — body must carry rot");
        assert_eq!(body_rot.quaternion, new_quat);
        assert_eq!(body_rot.ang_vel_body, new_ang_vel);
    }
}
