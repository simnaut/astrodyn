//! Bevy-side [`SimContext`] implementation for parity tests.
//!
//! The runner crate forwards `SimContext` directly to
//! `astrodyn_runner::Simulation`. The Bevy adapter has no comparable
//! single-handle wrapper — state is spread across components and the
//! source mutation surface is a [`SourceMutator`] `SystemParam`, while
//! attach/detach is event-driven through `AttachEvent` / `DetachEvent`.
//! [`AppSimContext`] bridges the two: it holds a `&mut App` plus the
//! [`ScenarioHandles`] returned by `populate_app`, and routes each
//! `SimContext` method to the correct Bevy surface so a `pre_step`
//! closure can drive the bevy half of a parity test the same way it
//! drives the runner half.
//!
//! The bridge is parity-test scaffolding — it lives in this crate
//! (which already depends on both `astrodyn_bevy` and
//! `astrodyn_verif_jeod`) rather than in `astrodyn_bevy` proper, since
//! the `SimContext` trait sits in `astrodyn_verif_jeod` (which
//! `astrodyn_bevy` only sees as a dev-dependency).

use std::marker::PhantomData;

use astrodyn::{Planet, SelfRef, StructuralFrame, Vec3Ext};
use astrodyn_bevy::{AttachEvent, DetachEvent, ScenarioHandles, SourceMutator};
use astrodyn_verif_jeod::verification::SimContext;
use bevy::prelude::*;
use glam::{DMat3, DVec3};

/// Bevy-side [`SimContext`] implementation backed by an `App` and the
/// handles returned by `SimulationBuilderBevyExt::populate_app`.
///
/// The runner-side `impl SimContext for Simulation` mutates the
/// arena in place; this impl performs the analogous mutations on the
/// ECS world (source positions via [`SourceMutator`] one-shot
/// systems, attach/detach via the message bus, kinematic-only as a
/// no-op since the Bevy side derives the kinematic marker from tree
/// topology automatically).
pub struct AppSimContext<'a, P: Planet> {
    app: &'a mut App,
    handles: &'a ScenarioHandles,
    _planet: PhantomData<P>,
}

impl<'a, P: Planet> AppSimContext<'a, P> {
    pub fn new(app: &'a mut App, handles: &'a ScenarioHandles) -> Self {
        Self {
            app,
            handles,
            _planet: PhantomData,
        }
    }
}

impl<P: Planet> SimContext for AppSimContext<'_, P> {
    fn set_source_position(&mut self, source_idx: usize, position: DVec3) {
        let entity = self.handles.source_entities[source_idx];
        self.app
            .world_mut()
            .run_system_cached_with(
                |In((source, pos)): In<(Entity, DVec3)>, mut mutator: SourceMutator<P>| {
                    mutator.set_source_position(source, pos);
                },
                (entity, position),
            )
            .expect(
                "AppSimContext::set_source_position: \
                 run_system_cached_with failed",
            );
    }

    fn set_source_state(&mut self, source_idx: usize, position: DVec3, velocity: DVec3) {
        let entity = self.handles.source_entities[source_idx];
        self.app
            .world_mut()
            .run_system_cached_with(
                |In((source, pos, vel)): In<(Entity, DVec3, DVec3)>,
                 mut mutator: SourceMutator<P>| {
                    mutator.set_source_state(source, pos, vel);
                },
                (entity, position, velocity),
            )
            .expect(
                "AppSimContext::set_source_state: \
                 run_system_cached_with failed",
            );
    }

    fn attach(
        &mut self,
        child_idx: usize,
        parent_idx: usize,
        offset: DVec3,
        t_parent_child: DMat3,
    ) {
        let child = self.handles.body_entities[child_idx];
        let parent = self.handles.body_entities[parent_idx];
        // The canonical staging-system message bus is the runtime-
        // resolved `AttachEvent<SelfRef, SelfRef>` pair; both
        // structural-frame phantom slots are SelfRef wildcards at the
        // bridge boundary. JEOD_INV: TS.01.
        self.app
            .world_mut()
            .write_message(AttachEvent::<SelfRef, SelfRef> {
                child,
                parent,
                offset: offset.m_at::<StructuralFrame<SelfRef>>(),
                t_parent_child: astrodyn::FrameTransform::<
                    StructuralFrame<SelfRef>,
                    StructuralFrame<SelfRef>,
                >::from_matrix(t_parent_child),
            });
    }

    fn detach(&mut self, child_idx: usize) {
        let child = self.handles.body_entities[child_idx];
        self.app.world_mut().write_message(DetachEvent { child });
    }

    fn mark_kinematic_only(&mut self, _child_idx: usize) {
        // No-op on the Bevy side. `wrench_aggregation_system`
        // auto-inserts `KinematicChildC` on every non-root in the
        // mass tree (`crates/astrodyn_bevy/src/wrench.rs`), so any
        // attached child is already kinematic without an explicit
        // marker call. The runner exposes this as a separate flag
        // (`Simulation::mark_kinematic_only`) because it integrates
        // attached children by default unless explicitly opted out;
        // Bevy's model is the inverse and matches JEOD_INV: DB.17
        // ("only the root integrates").
    }
}
