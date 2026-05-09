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
//! Today this implements the source-position injection methods:
//! [`set_source_position`], [`set_source_state`], and
//! [`set_tidal_body_position`]. These are the only methods the
//! `pre_step`-using recipes in
//! `astrodyn_verif_jeod::run_verification::sim_*` actually call (third-body
//! ephemeris updates, tidal Sun/Moon position injection).
//!
//! [`attach`], [`detach`], [`mark_kinematic_only`] inherit the trait's
//! default-panic behaviour for now. Wiring them up requires careful
//! coordination with the Bevy-side `staging_system` to preserve
//! bit-identity with the runner's synchronous `Simulation::attach`
//! combine kernel + integrator reset; that is sub-task A's runtime
//! attach/detach work, tracked separately.

use astrodyn::{Planet, PlanetInertial, RootInertial, Vec3Ext};
use astrodyn_bevy::{
    FrameEntityC, FrameTransC, SourceInertialPositionC, SourceInertialVelocityC, TidalConfigC,
    TranslationalStateC,
};
use astrodyn_verif_jeod::verification::SimContext;
use bevy::prelude::*;
use glam::DVec3;

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
    _planet: std::marker::PhantomData<P>,
}

impl<'w, P: Planet> BevySimContext<'w, P> {
    /// Construct a context borrowing the given world and source-entity
    /// slice for the lifetime of one `pre_step` invocation.
    pub fn new(world: &'w mut World, source_entities: &'w [Entity]) -> Self {
        Self {
            world,
            source_entities,
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
}
