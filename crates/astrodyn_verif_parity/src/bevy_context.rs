//! Bevy-side [`SimContext`] adapter that lets a [`PreStepClosure`] mutate
//! a Bevy [`App`]'s world in lockstep with the runner-side
//! `astrodyn_runner::Simulation`.
//!
//! The parity trait drives both runtimes from the same scenario factory
//! and the same [`PreStepClosure`]; on each per-tick iteration the
//! closure is invoked twice — once with `&mut Simulation`, once with a
//! freshly-constructed [`BevySimContext`] borrowing the app's world.
//!
//! ## Scope
//!
//! - **Source-state injection** ([`set_source_position`],
//!   [`set_source_state`], [`set_tidal_body_position`]) — direct
//!   `World::get_mut` writes mirroring `astrodyn_bevy::SourceMutator`.
//! - **Mass-tree mid-flight attach/detach** ([`attach`], [`detach`])
//!   — write `AttachEvent<SelfRef, SelfRef>` / `DetachEvent` onto the
//!   message bus. The next `app.world_mut().run_schedule(FixedUpdate)`
//!   drains the queue at the top of `staging_system`, before
//!   integration runs that tick — so both runtimes feed the same
//!   pre-attach state into the same `combine_states_at_attach` kernel
//!   and the same integrator-reset path. Bit-identity holds.
//! - **Kinematic-only gating** ([`mark_kinematic_only`]) — insert
//!   `KinematicChildC` directly on the child entity. The runner sets
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
    FrameTransform, Planet, PlanetInertial, Position, RootInertial, SelfRef, StructuralFrame,
    Vec3Ext,
};
use astrodyn_bevy::{
    AttachEvent, DetachEvent, FrameEntityC, FrameTransC, KinematicChildC, MassChildOf,
    SourceInertialPositionC, SourceInertialVelocityC, TidalConfigC, TranslationalStateC,
};
use astrodyn_verif_jeod::verification::SimContext;
use bevy::ecs::message::Messages;
use bevy::prelude::*;
use glam::{DMat3, DVec3};

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

        // Write `AttachEvent<SelfRef, SelfRef>` onto the message bus.
        // The next `app.world_mut().run_schedule(FixedUpdate)` drains
        // the queue at the top of `staging_system`, before integration
        // — so the runner's synchronous `Simulation::attach` and the
        // Bevy adapter's deferred attach both run their combine kernel
        // on the same pre-attach pair, and the merged composite-body
        // state plus integrator reset land before that tick's
        // integration. The message payload uses `<SelfRef, SelfRef>`
        // because the canonical Bevy adapter registers the
        // runtime-resolved instantiation (vehicle identity is decided
        // by per-entity storage, not at message-bus dispatch).
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

        // Also insert `MassChildOf` on the child entity so the
        // ECS-native parent ↔ child edge mirrors the topology
        // `staging_system` is about to record on the mass-tree
        // resource. Mirrors the hand-rolled
        // `bevy_parity_attach_detach_trajectory.rs` pattern; without
        // it kinematic-propagation reads the wrong parent on the
        // attach-event tick.
        let _ = (parent, child); // silence unused if reordered later
        self.world.entity_mut(child).insert(MassChildOf {
            parent,
            offset,
            t_parent_child,
        });
    }

    fn detach(&mut self, child_idx: usize) {
        let child = self.body_entity(child_idx);
        let mut messages = self.world.resource_mut::<Messages<DetachEvent>>();
        messages.write(DetachEvent { child });
        // Remove the ECS-native parent ↔ child edge. `staging_system`
        // updates the mass-tree resource; removing `MassChildOf` here
        // keeps the component-level edge in sync so the next tick's
        // wrench / propagation systems observe a root-equivalent
        // topology for the detached body.
        self.world.entity_mut(child).remove::<MassChildOf>();
    }

    fn mark_kinematic_only(&mut self, child_idx: usize) {
        let child = self.body_entity(child_idx);
        // `wrench_aggregation_system` would also install
        // `KinematicChildC` once it observes a non-root mass-tree
        // node, but explicit insertion here matches the runner's
        // synchronous `mark_kinematic_only` semantics — the marker is
        // present *before* the next FixedUpdate's integration runs.
        self.world.entity_mut(child).insert(KinematicChildC);
    }
}

// We use `Position::<StructuralFrame<SelfRef>>` indirectly via
// `m_at::<StructuralFrame<SelfRef>>()`; the unused-import lint sees
// `Position` only through that path, so name it explicitly to silence
// the transitive `unused_imports` check on direct use.
const _: fn() = || {
    let _: Position<StructuralFrame<SelfRef>> = DVec3::ZERO.m_at::<StructuralFrame<SelfRef>>();
};
