//! Bevy-native `SystemParam` for cross-frame state computation
//! ([Frame-Tree-ECS-Native § 13][1] — additive infrastructure, PR 1).
//!
//! Replaces `FrameTreeR`-backed arena lookups (`compute_relative_state`,
//! `frame_origin`) with ECS hierarchy walks over Bevy's `ChildOf` /
//! `Children` relationship and the new
//! [`FrameTransC`] / [`FrameRotC`] / [`FrameAngVelC`] components.
//!
//! Mission code asks for cross-frame state by passing entity handles,
//! never `FrameId`s — the surface looks like any other Bevy
//! `SystemParam`. The shared algorithm lives in
//! [`jeod_sim::frame_compute_relative_state_via_storage`] (the
//! `FrameStorage` trait abstraction described in [§ 7][2]); this
//! `SystemParam` only supplies the storage adapter.
//!
//! At PR 1 (this module) the dual-write path keeps the new components
//! exactly in sync with the arena, so this `SystemParam` and
//! `FrameTreeR.compute_relative_state` return bit-identical numerics
//! (see `tests/frame_storage_relative_frame_state.rs`). Mission code may
//! adopt the new surface incrementally; the arena is removed in
//! Section 13 PR 4.
//!
//! [1]: https://github.com/simnaut/bevy_jeod/wiki/Frame-Tree-ECS-Native#13-migration-sequencing
//! [2]: https://github.com/simnaut/bevy_jeod/wiki/Frame-Tree-ECS-Native#7-internal-algorithm-sharing-q1

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use glam::DVec3;
use jeod_sim::{FrameStorage, RefFrameRot, RefFrameState, RefFrameTrans};

use crate::components::{FrameAngVelC, FrameRotC, FrameTransC};

/// Compute relative state between two frame entities by walking
/// Bevy's hierarchy and composing per-node states with the pure-state
/// math from `jeod_frames` (`incr_left`, `incr_right`, `negate`),
/// dispatched through the storage-agnostic
/// [`jeod_sim::frame_compute_relative_state_via_storage`] algorithm.
///
/// ECS-native replacement for
/// `FrameTreeR.compute_relative_state(from_id, to_id)` and
/// `frame_origin(tree, root, frame_id)`. Issue #277.
///
/// During the dual-write phase (PR 1) the underlying components are
/// kept in lockstep with the arena; this `SystemParam` is therefore a
/// drop-in alternative with identical numerics. Mission code that
/// migrates first sees an `Entity`-keyed surface (no `FrameId`s, no
/// `Res<FrameTreeR>`).
#[derive(SystemParam)]
pub struct RelativeFrameState<'w, 's> {
    parents: Query<'w, 's, &'static ChildOf>,
    states: Query<
        'w,
        's,
        (
            &'static FrameTransC,
            &'static FrameRotC,
            &'static FrameAngVelC,
        ),
    >,
}

impl<'w, 's> RelativeFrameState<'w, 's> {
    /// `(position, velocity)` of `to` relative to `from`, both in
    /// `from`-frame coordinates.
    pub fn position_velocity(&self, from: Entity, to: Entity) -> (DVec3, DVec3) {
        let rel = self.relative_state(from, to);
        (rel.trans.position, rel.trans.velocity)
    }

    /// Position of `to` relative to `from`, in `from`-frame
    /// coordinates.
    pub fn position(&self, from: Entity, to: Entity) -> DVec3 {
        self.relative_state(from, to).trans.position
    }

    /// Full [`RefFrameState`] of `to` relative to `from`. Delegates
    /// to the storage-agnostic
    /// [`jeod_sim::frame_compute_relative_state_via_storage`]
    /// algorithm via this `SystemParam`'s [`FrameStorage`] impl —
    /// the same code path the runner's arena uses, so the algorithm
    /// is single-sourced (see [Frame-Tree-ECS-Native § 7][1]).
    ///
    /// [1]: https://github.com/simnaut/bevy_jeod/wiki/Frame-Tree-ECS-Native#7-internal-algorithm-sharing-q1
    pub fn relative_state(&self, from: Entity, to: Entity) -> RefFrameState {
        jeod_sim::frame_compute_relative_state_via_storage(self, from, to)
    }
}

/// `FrameStorage` impl: lets the storage-agnostic algorithms in
/// `jeod_frames::frame_storage` operate over the ECS hierarchy + the
/// new frame-state components. Issue #277.
impl<'w, 's> FrameStorage for RelativeFrameState<'w, 's> {
    type Id = Entity;

    fn parent(&self, id: Entity) -> Option<Entity> {
        self.parents.get(id).ok().map(|child_of| child_of.parent())
    }

    fn state(&self, id: Entity) -> RefFrameState {
        let (trans, rot, ang_vel) = self.states.get(id).unwrap_or_else(|err| {
            panic!(
                "RelativeFrameState::state: frame entity {id:?} is missing \
                 FrameTransC / FrameRotC / FrameAngVelC components ({err:?}). \
                 Frame entities must be spawned with all three (or use the \
                 register_*_frames_system path that inserts them)."
            )
        });
        RefFrameState {
            trans: RefFrameTrans {
                position: trans.position,
                velocity: trans.velocity,
            },
            rot: RefFrameRot {
                q_parent_this: rot.q_parent_this,
                t_parent_this: rot.t_parent_this,
                ang_vel_this: ang_vel.0,
            },
        }
    }
}
