//! Bevy-side [`SimContext`] adapter that lets a
//! [`astrodyn_verif_jeod::verification::PreStepClosure`] mutate
//! a Bevy [`App`]'s world in lockstep with the runner-side
//! `astrodyn_runner::Simulation`.
//!
//! JEOD_INV: TS.01 — this module sits at the per-entity storage
//! boundary between the runtime-typed runner (`Simulation` carrying
//! `<V: Vehicle>` parameters per body) and the runtime-typed Bevy
//! world (`AttachEvent<SelfRef, SelfRef>` events on the message bus,
//! `m_at::<StructuralFrame<SelfRef>>`-tagged offsets, etc.). Every
//! `SelfRef` use below sits at this boundary; see
//! `docs/JEOD_invariants.md` row TS.01 for the full rule and the
//! lint at `tests/self_ref_self_planet_discipline.rs`.
//!
//! The parity trait drives both runtimes from the same scenario factory
//! and the same [`astrodyn_verif_jeod::verification::PreStepClosure`];
//! on each per-tick iteration the closure is invoked twice — once with
//! `&mut Simulation`, once with a freshly-constructed
//! [`BevySimContext`] borrowing the app's world.
//!
//! ## Scope
//!
//! - **Source-state injection** ([`SimContext::set_source_position`],
//!   [`SimContext::set_source_state`],
//!   [`SimContext::set_tidal_body_position`]) — direct `World::get_mut`
//!   writes mirroring `astrodyn_bevy::SourceMutator`.
//! - **Mass-tree mid-flight attach/detach** ([`SimContext::attach`],
//!   [`SimContext::detach`]) — write
//!   `AttachEvent<SelfRef, SelfRef>` / `DetachEvent` onto the
//!   message bus. The next `app.world_mut().run_schedule(FixedUpdate)`
//!   drains the queue at the top of `staging_system`, before
//!   integration runs that tick — so both runtimes feed the same
//!   pre-attach state into the same `combine_states_at_attach` kernel
//!   and the same integrator-reset path. Bit-identity holds.
//! - **Kinematic-only gating** ([`SimContext::mark_kinematic_only`]) —
//!   insert `KinematicChildC` directly on the child entity. The runner sets
//!   `bodies[idx].kinematic_only = true` synchronously; on the Bevy
//!   side `wrench_aggregation_system` would also install
//!   `KinematicChildC` once it observes the new mass-tree topology,
//!   but explicit insertion here matches the runner's
//!   call-site-synchronous semantics so the integrator-skip is in
//!   place before the next FixedUpdate runs.
//!
//! All five methods preserve the runner's "pre-step only" contract:
//! the closure runs before `step_n`/`run_schedule(FixedUpdate)`, so a
//! mid-step reentrant attach is structurally impossible.

use astrodyn::{
    BodyFrame, Force, FrameTransform, Planet, PlanetInertial, Position, RootInertial, SelfRef,
    StructuralFrame, Torque, Vec3Ext,
};
use astrodyn_bevy::{
    AttachEvent, DetachEvent, ExternalForceC, ExternalTorqueC, FrameAttachEvent, FrameEntityC,
    FrameTransC, KinematicChildC, MassChildOf, PfixFrameEntityC, RotationalStateC,
    SourceInertialPositionC, SourceInertialVelocityC, TidalConfigC, TranslationalStateC,
};
use astrodyn_verif_jeod::verification::{SimContext, SourceFrameKind};
use bevy::ecs::message::Messages;
use bevy::prelude::*;
use glam::{DMat3, DQuat, DVec3};

/// `SimContext` adapter over a Bevy [`World`].
///
/// Holds a mutable world borrow plus the parallel `source_entities` slice
/// from [`astrodyn_bevy::ScenarioHandles`], so per-index source lookups
/// are O(1) and match the runner-side `source_idx` convention used by
/// `astrodyn_runner::Simulation::set_source_*`.
///
/// The `<P: Planet>` parameter pins the same planet-inertial frame the
/// scenario integrates in — `TranslationalStateC<P>` writes are relabel-
/// only (no numeric change) since the public `SimContext` API frame is
/// `RootInertial` but storage is `PlanetInertial<P>`. This mirrors
/// `astrodyn_bevy::SourceMutator<P>`'s relabel-at-storage-boundary
/// convention; the system instantiation's `<P>` parameter pins the
/// storage convention identically.
pub struct BevySimContext<'w, P: Planet> {
    world: &'w mut World,
    source_entities: &'w [Entity],
    body_entities: &'w [Entity],
    _planet: std::marker::PhantomData<P>,
}

impl<'w, P: Planet> BevySimContext<'w, P> {
    /// Construct a context borrowing the given world plus the
    /// source-entity and body-entity slices for the lifetime of one
    /// `pre_step` invocation. Both slices are parallel to the
    /// runner-side `source_idx` / `body_idx` conventions.
    pub fn new(
        world: &'w mut World,
        source_entities: &'w [Entity],
        body_entities: &'w [Entity],
    ) -> Self {
        Self {
            world,
            source_entities,
            body_entities,
            _planet: std::marker::PhantomData,
        }
    }

    fn source_entity(&self, source_idx: usize) -> Entity {
        *self.source_entities.get(source_idx).unwrap_or_else(|| {
            panic!(
                "BevySimContext: source_idx {source_idx} out of range \
                 (have {} sources)",
                self.source_entities.len()
            )
        })
    }

    fn body_entity(&self, body_idx: usize) -> Entity {
        *self.body_entities.get(body_idx).unwrap_or_else(|| {
            panic!(
                "BevySimContext: body_idx {body_idx} out of range \
                 (have {} bodies)",
                self.body_entities.len()
            )
        })
    }

    fn frame_entity(&self, source: Entity) -> Entity {
        self.world
            .get::<FrameEntityC>(source)
            .unwrap_or_else(|| {
                panic!(
                    "BevySimContext: source entity {source:?} is missing \
                     FrameEntityC (was the source registered via \
                     populate_app / register_source_frames_system?)"
                )
            })
            .0
    }
}

impl<P: Planet> SimContext for BevySimContext<'_, P> {
    fn set_source_position(&mut self, source_idx: usize, position: DVec3) {
        let source = self.source_entity(source_idx);
        let frame = self.frame_entity(source);

        let typed_pos = position.m_at::<RootInertial>();

        // Frame-entity FrameTransC: the source's frame-tree node holds
        // the canonical position read by gravity / integration.
        let mut frame_trans = self.world.get_mut::<FrameTransC>(frame).unwrap_or_else(|| {
            panic!(
                "BevySimContext::set_source_position: source {source_idx} \
                 has FrameEntityC({frame:?}) but the frame entity has no \
                 FrameTransC."
            )
        });
        frame_trans.position = position;
        // NLL releases the `frame_trans` mutable borrow at its last use
        // above so the next `world.get_mut::<…>` call below typechecks.

        // SourceInertialPositionC on the source entity.
        let mut pos_c = self
            .world
            .get_mut::<SourceInertialPositionC>(source)
            .unwrap_or_else(|| {
                panic!(
                    "BevySimContext::set_source_position: source {source_idx} \
                     ({source:?}) is missing SourceInertialPositionC."
                )
            });
        pos_c.0 = typed_pos;

        // TranslationalStateC<P> on the source entity: relabel root→planet.
        let mut ts = self
            .world
            .get_mut::<TranslationalStateC<P>>(source)
            .unwrap_or_else(|| {
                panic!(
                    "BevySimContext::set_source_position: source {source_idx} \
                     ({source:?}) is missing TranslationalStateC<{}>.",
                    std::any::type_name::<P>(),
                )
            });
        ts.0.position = typed_pos.relabel_to::<PlanetInertial<P>>();
    }

    fn set_source_state(&mut self, source_idx: usize, position: DVec3, velocity: DVec3) {
        let source = self.source_entity(source_idx);
        let frame = self.frame_entity(source);

        let typed_pos = position.m_at::<RootInertial>();
        let typed_vel = velocity.m_per_s_at::<RootInertial>();

        let mut frame_trans = self.world.get_mut::<FrameTransC>(frame).unwrap_or_else(|| {
            panic!(
                "BevySimContext::set_source_state: source {source_idx} has \
                 FrameEntityC({frame:?}) but the frame entity has no \
                 FrameTransC."
            )
        });
        frame_trans.position = position;
        frame_trans.velocity = velocity;

        let mut pos_c = self
            .world
            .get_mut::<SourceInertialPositionC>(source)
            .unwrap_or_else(|| {
                panic!(
                    "BevySimContext::set_source_state: source {source_idx} \
                     ({source:?}) is missing SourceInertialPositionC."
                )
            });
        pos_c.0 = typed_pos;

        // SourceInertialVelocityC: auto-insert if missing (mirrors
        // SourceMutator::set_source_state behaviour).
        if let Some(mut vc) = self.world.get_mut::<SourceInertialVelocityC>(source) {
            vc.0 = typed_vel;
        } else {
            self.world
                .entity_mut(source)
                .insert(SourceInertialVelocityC(typed_vel));
        }

        let mut ts = self
            .world
            .get_mut::<TranslationalStateC<P>>(source)
            .unwrap_or_else(|| {
                panic!(
                    "BevySimContext::set_source_state: source {source_idx} \
                     ({source:?}) is missing TranslationalStateC<{}>.",
                    std::any::type_name::<P>(),
                )
            });
        ts.0.position = typed_pos.relabel_to::<PlanetInertial<P>>();
        ts.0.velocity = typed_vel.relabel_to::<PlanetInertial<P>>();
    }

    fn set_tidal_body_position(
        &mut self,
        source_idx: usize,
        tidal_body_idx: usize,
        position: DVec3,
    ) {
        let source = self.source_entity(source_idx);
        let mut tidal = self
            .world
            .get_mut::<TidalConfigC>(source)
            .unwrap_or_else(|| {
                panic!(
                    "BevySimContext::set_tidal_body_position: source \
                 {source_idx} ({source:?}) is missing TidalConfigC. \
                 Wire `tidal_config: Some(...)` on the GravitySourceEntry \
                 so populate_app inserts TidalConfigC."
                )
            });
        let len = tidal.0.tidal_bodies.len();
        assert!(
            tidal_body_idx < len,
            "BevySimContext::set_tidal_body_position: source {source_idx} \
             tidal_body_idx {tidal_body_idx} out of bounds (len={len})"
        );
        tidal.0.tidal_bodies[tidal_body_idx].position_inertial = position.m_at::<RootInertial>();
    }

    fn attach(
        &mut self,
        child_idx: usize,
        parent_idx: usize,
        offset: DVec3,
        t_parent_child: DMat3,
    ) {
        let child = self.body_entity(child_idx);
        let parent = self.body_entity(parent_idx);

        // Write `AttachEvent<SelfRef, SelfRef>` onto the message bus
        // **only**. `MassChildOf` is deliberately NOT inserted here:
        // composite_mass_system races staging_system on the attach
        // tick (the file-level "Tick-1 / steady-state separation"
        // contract in `bevy_parity_kinematic_propagation.rs`), so
        // mass-tree edge insertion must wait until after
        // staging_system has finished the combine. Mission code
        // installs `MassChildOf` later — typically alongside
        // `mark_kinematic_only` (which we mirror below).
        //
        // The next `app.world_mut().run_schedule(FixedUpdate)` drains
        // the queue at the top of `staging_system`, before
        // integration — so the runner's synchronous
        // `Simulation::attach` and the Bevy adapter's deferred attach
        // both run their combine kernel on the same pre-attach pair,
        // and the merged composite-body state plus integrator reset
        // land before that tick's integration. The message payload
        // uses `<SelfRef, SelfRef>` because the canonical Bevy
        // adapter registers the runtime-resolved instantiation
        // (vehicle identity is decided by per-entity storage, not at
        // message-bus dispatch).
        let event = AttachEvent::<SelfRef, SelfRef> {
            child,
            parent,
            offset: offset.m_at::<StructuralFrame<SelfRef>>(),
            t_parent_child:
                FrameTransform::<StructuralFrame<SelfRef>, StructuralFrame<SelfRef>>::from_matrix(
                    t_parent_child,
                ),
        };
        let mut messages = self
            .world
            .resource_mut::<Messages<AttachEvent<SelfRef, SelfRef>>>();
        messages.write(event);
    }

    fn detach(&mut self, child_idx: usize) {
        let child = self.body_entity(child_idx);
        let mut messages = self.world.resource_mut::<Messages<DetachEvent>>();
        messages.write(DetachEvent { child });
        // Remove the ECS-native parent ↔ child edge. `staging_system`
        // updates the mass-tree resource; removing `MassChildOf` here
        // keeps the component-level edge in sync so the next tick's
        // wrench / propagation systems observe a root-equivalent
        // topology for the detached body. Detach has no
        // composite_mass_system race because the topology change
        // *removes* an edge — the post-detach composites recompute
        // independently of when the system runs relative to the
        // event drain.
        self.world.entity_mut(child).remove::<MassChildOf>();
    }

    fn mark_kinematic_only(&mut self, child_idx: usize) {
        let child = self.body_entity(child_idx);
        // Insert `MassChildOf` (the ECS-native parent ↔ child edge)
        // and `KinematicChildC` (the integrator gating marker)
        // together. Recipes call `attach` first (writes AttachEvent
        // only), then on a later pre_step call `mark_kinematic_only`
        // — which lands here, post-staging — to install the steady-
        // state kinematic-chain handles. Mirrors the hand-rolled
        // `bevy_parity_kinematic_propagation.rs` pattern that splits
        // the attach event from the kinematic-edge installation
        // across two ticks (the file's "Tick-1 / steady-state
        // separation" docstring) to avoid composite_mass_system
        // racing staging_system within one tick.
        //
        // The mass-tree resource was updated by `staging_system` at
        // the top of the previous FixedUpdate, so reading the
        // child's parent + edge geometry from there gives us the
        // post-attach topology. `MassTreeR` carries the offset and
        // rotation in raw glam form — pull them out and reuse for
        // `MassChildOf` so the ECS edge mirrors the resource edge
        // exactly.
        let (parent_entity, offset, t_parent_child) = {
            let tree_r = self.world.resource::<astrodyn_bevy::MassTreeR>();
            let mass_id = self
                .world
                .get::<astrodyn_bevy::MassBodyIdC>(child)
                .unwrap_or_else(|| {
                    panic!(
                        "BevySimContext::mark_kinematic_only: child entity {child:?} \
                         has no MassBodyIdC — call `attach` first."
                    )
                })
                .0;
            let parent_id = tree_r.0.parent(mass_id).unwrap_or_else(|| {
                panic!(
                    "BevySimContext::mark_kinematic_only: child mass id {mass_id:?} \
                     has no parent in the mass tree — staging_system has not yet \
                     processed the AttachEvent? Call `attach` and run one \
                     FixedUpdate before `mark_kinematic_only`."
                )
            });
            // `MassBody::structure_point` is the per-attach struct
            // origin (`offset`) + struct-frame rotation
            // (`t_parent_this`) JEOD's `MassTree::attach` records.
            let sp = &tree_r.0.get(mass_id).structure_point;
            let offset = sp.position;
            let t_pc = sp.t_parent_this;
            // Map parent mass id back to entity by scanning
            // body_entities for the matching MassBodyIdC. Bridge
            // currently provides no direct mass-id ↔ entity index
            // (issue tracked at #389); the linear scan is fine for
            // the scenario sizes parity tests cover (≤ ~10 bodies).
            let mut parent_entity = None;
            for &candidate in self.body_entities {
                if let Some(id) = self.world.get::<astrodyn_bevy::MassBodyIdC>(candidate) {
                    if id.0 == parent_id {
                        parent_entity = Some(candidate);
                        break;
                    }
                }
            }
            let parent_entity = parent_entity.unwrap_or_else(|| {
                panic!(
                    "BevySimContext::mark_kinematic_only: parent mass id {parent_id:?} \
                     not found among body_entities."
                )
            });
            (parent_entity, offset, t_pc)
        };
        self.world.entity_mut(child).insert((
            MassChildOf {
                parent: parent_entity,
                offset,
                t_parent_child,
            },
            KinematicChildC,
        ));
    }

    fn set_body_external_force(&mut self, body_idx: usize, force: DVec3) {
        // Match the runner's `Simulation::set_body_external_force`
        // bit-for-bit: overwrite the body's external-force component
        // with the new value. The runner's storage frame is
        // `RootInertial` (the recipe wires `external_force` typed
        // against root-inertial on `VehicleConfig`), and the Bevy
        // `ExternalForceC` carries the same phantom — so this is a
        // direct typed write with no relabel.
        let entity = self.body_entity(body_idx);
        let typed = Force::<RootInertial>::from_raw_si(force);
        // Auto-insert when missing: `populate_app` only inserts
        // `ExternalForceC` when `VehicleConfig.external_force` is
        // non-zero, so a scenario that starts with zero force but
        // schedules a non-zero pulse needs the component installed
        // here (mirrors the `SourceInertialVelocityC` auto-insert in
        // `set_source_state` above).
        if let Some(mut fc) = self.world.get_mut::<ExternalForceC>(entity) {
            fc.0 = typed;
        } else {
            self.world.entity_mut(entity).insert(ExternalForceC(typed));
        }
    }

    fn set_body_external_torque(&mut self, body_idx: usize, torque: DVec3) {
        // Match the runner: overwrite the body's external-torque
        // component with the new value. The torque carries the
        // body-frame phantom against the runtime-resolved
        // `SelfRef` vehicle wildcard (JEOD_INV: TS.01), same as
        // `VehicleConfig.external_torque` and `ExternalTorqueC`.
        let entity = self.body_entity(body_idx);
        let typed = Torque::<BodyFrame<SelfRef>>::from_raw_si(torque);
        if let Some(mut tc) = self.world.get_mut::<ExternalTorqueC>(entity) {
            tc.0 = typed;
        } else {
            self.world.entity_mut(entity).insert(ExternalTorqueC(typed));
        }
    }

    fn body_q_inertial_body(&self, body_idx: usize) -> DQuat {
        // Mirror the runner: panic with a descriptive message rather
        // than returning identity when the body has no rotational
        // state. The closure must rotate body→inertial via the same
        // quaternion the integrator reads, so a 3-DOF misuse is a
        // programmer error caught here.
        let entity = self.body_entity(body_idx);
        let rot = self
            .world
            .get::<RotationalStateC>(entity)
            .unwrap_or_else(|| {
                panic!(
                    "BevySimContext::body_q_inertial_body: body {body_idx} ({entity:?}) \
                 has no RotationalStateC (3-DOF body). Add `rot: Some(...)` to its \
                 VehicleConfig if the pre_step closure needs an inertial-body \
                 quaternion."
                )
            });
        rot.0.q_inertial_body.as_witness().inner().to_glam()
    }

    fn attach_to_frame(
        &mut self,
        body_idx: usize,
        source_idx: usize,
        frame_kind: SourceFrameKind,
        offset: DVec3,
        t_parent_child: DMat3,
    ) {
        // Resolve the parent reference frame entity for this source +
        // frame kind. The runner-side analog walks
        // `source_inertial_frame_id` / `source_pfix_frame_id`; the Bevy
        // adapter exposes the same pair as components on the source
        // entity (`FrameEntityC` for the inertial frame,
        // `PfixFrameEntityC` for the rotating planet-fixed frame). The
        // pfix entity is only present when the source carries a
        // rotation model — surface the absence as a fail-loud panic
        // instead of a silent no-op so the recipe's misconfiguration
        // is named at the call site.
        let body = self.body_entity(body_idx);
        let source = self.source_entity(source_idx);
        let parent_frame = match frame_kind {
            SourceFrameKind::Inertial => self.frame_entity(source),
            SourceFrameKind::Pfix => {
                self.world
                    .get::<PfixFrameEntityC>(source)
                    .unwrap_or_else(|| {
                        panic!(
                            "BevySimContext::attach_to_frame: source {source_idx} ({source:?}) \
                             has no PfixFrameEntityC; either the source lacks a rotation model \
                             (wire `rotation_model: Some(...)` on the GravitySourceEntry) or \
                             use SourceFrameKind::Inertial to attach to the non-rotating frame."
                        )
                    })
                    .0
            }
        };
        // Match the runner's `Simulation::attach_to_frame`: write a
        // `FrameAttachEvent` onto the message bus. `frame_attach_system`
        // drains the queue at the top of the next `FixedUpdate`, before
        // integration runs — so both runtimes observe the same
        // pre-attach state and emit the same captured-offset
        // attachment, and the integrator-reset path lands before that
        // tick's integration. Bit-identity across both runtimes is
        // the target; sub-ULP drift in the Earth.pfix rotation matrix
        // at a handful of post-attach records is a known Bevy-side
        // schedule investigation (see `bevy_parity_ref_attach_matrix`,
        // currently `#[ignore]`'d).
        let event = FrameAttachEvent {
            body,
            parent_frame,
            offset,
            t_parent_body: t_parent_child,
        };
        let mut messages = self.world.resource_mut::<Messages<FrameAttachEvent>>();
        messages.write(event);
    }
}

// We use `Position::<StructuralFrame<SelfRef>>` indirectly via
// `m_at::<StructuralFrame<SelfRef>>()`; the unused-import lint sees
// `Position` only through that path, so name it explicitly to silence
// the transitive `unused_imports` check on direct use.
const _: fn() = || {
    let _: Position<StructuralFrame<SelfRef>> = DVec3::ZERO.m_at::<StructuralFrame<SelfRef>>();
};
