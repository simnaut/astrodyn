//! Bevy-native `SystemParam`s for cross-frame state computation
//! ([Frame-Tree-ECS-Native § 13][1]).
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
//! Two SystemParams are exposed (mirroring [§ 6][3]'s mission-code
//! catalog):
//!
//! - [`RelativeFrameState`] — general "state of `to` relative to
//!   `from`" query. Returns raw `DVec3` (and the full
//!   [`RefFrameState`] when the rotation/angular-velocity portion is
//!   needed). The drop-in replacement for
//!   `FrameTreeR.compute_relative_state(from, to)`.
//! - [`FrameOrigin`] — specialized "origin of a frame in an ancestor
//!   frame" query. Returns
//!   `(Position<RootInertial>, Velocity<RootInertial>)` typed at the
//!   root-inertial phantom when called against the root frame entity.
//!   Sugar over `RelativeFrameState::position_velocity(root, frame)`
//!   that makes the resulting frame phantom explicit in the
//!   signature. The drop-in replacement for
//!   `frame_origin(tree, root, frame_id)` /
//!   `frame_origin_typed::<RootInertial>(tree, root, frame_id)`.
//!
//! During the dual-write phase (PR 1–3) the underlying components are
//! kept in lockstep with [`crate::FrameTreeR`], so these `SystemParam`s
//! and the arena helpers return bit-identical numerics
//! (see `tests/frame_storage_relative_frame_state.rs`). Mission code
//! may adopt the new surface incrementally; the arena is removed in
//! Section 13 PR 4.
//!
//! [1]: https://github.com/simnaut/bevy_jeod/wiki/Frame-Tree-ECS-Native#13-migration-sequencing
//! [2]: https://github.com/simnaut/bevy_jeod/wiki/Frame-Tree-ECS-Native#7-internal-algorithm-sharing-q1
//! [3]: https://github.com/simnaut/bevy_jeod/wiki/Frame-Tree-ECS-Native#6-mission-code-surface--systemparam-catalog-q7

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use glam::DVec3;
use jeod_sim::{
    Frame, FrameStorage, Position, RefFrameRot, RefFrameState, RefFrameTrans, RootInertial,
    Velocity,
};

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

/// Compute the origin (position + velocity) of a frame entity expressed
/// in a chosen ancestor frame's coordinates. ECS-native replacement for
/// `jeod_sim::frame_origin` / `jeod_sim::frame_origin_typed::<F>`.
/// Issue #278 (Frame-Tree-ECS-Native § 6, PR 2).
///
/// Internally backed by the same `Query<&ChildOf>` +
/// `Query<(&FrameTransC, &FrameRotC, &FrameAngVelC)>` walks as
/// [`RelativeFrameState`]; in fact `FrameOrigin` wraps a
/// [`RelativeFrameState`]. `FrameOrigin` is a specialized variant for
/// the common "origin of frame F in an ancestor frame" query — most
/// frequently "in the root frame," which is the form gravity,
/// integration, and `IntegOrigin`-shift sites need.
///
/// The three-method shape mirrors the design doc's catalog:
///
/// - [`FrameOrigin::origin_in_root`] —
///   `(Position<RootInertial>, Velocity<RootInertial>)`. The typed
///   surface that lifts the result into the root-inertial phantom
///   without an `from_raw_si` boundary at the call site.
///   Equivalent to
///   `frame_origin_typed::<RootInertial>(tree, root, frame)` /
///   `(rel.position_velocity(root, frame))`-then-wrap.
/// - [`FrameOrigin::origin_in`] — raw `DVec3` form for callers whose
///   ancestor isn't the root (e.g. an integration frame that's a
///   child of root, not root itself). Caller chooses the
///   ancestor entity, mirroring the arena helper's `(root, frame)`
///   parameter shape.
/// - [`FrameOrigin::origin_in_typed`] — generic-typed sibling of
///   `origin_in_root` for callers whose ancestor frame's marker is
///   some other `F: Frame` (e.g. a `PlanetInertial<P>` integration
///   frame). Caller asserts that `ancestor`'s marker is `F`; the
///   phantom-tag attachment is unchecked at runtime, mirroring
///   `jeod_sim::frame_origin_typed`.
///
/// # Example
/// ```ignore
/// use bevy::prelude::*;
/// use bevy_jeod::prelude::*;
///
/// fn read_origin_in_root(
///     origin: FrameOrigin,
///     root: Res<RootFrameEntityR>,
///     bodies: Query<&FrameEntityC, With<MyBody>>,
/// ) -> Position<RootInertial> {
///     let body_e = bodies.single().unwrap().0;
///     let (pos, _vel) = origin.origin_in_root(root.0, body_e);
///     pos
/// }
/// ```
#[derive(SystemParam)]
pub struct FrameOrigin<'w, 's> {
    rel: RelativeFrameState<'w, 's>,
}

impl<'w, 's> FrameOrigin<'w, 's> {
    /// `(position, velocity)` of `frame`'s origin expressed in
    /// `ancestor`-frame coordinates. Raw `DVec3` form for callers
    /// whose ancestor isn't the root (e.g. an `IntegrationFrame` that
    /// is a child of root, not root itself).
    ///
    /// When `frame == ancestor`, returns `(DVec3::ZERO, DVec3::ZERO)` —
    /// the same identity short-circuit as
    /// `jeod_sim::frame_origin(tree, root, root)`.
    pub fn origin_in(&self, ancestor: Entity, frame: Entity) -> (DVec3, DVec3) {
        self.rel.position_velocity(ancestor, frame)
    }

    /// Typed `(Position<RootInertial>, Velocity<RootInertial>)` for the
    /// common "origin in the root inertial frame" query. Caller passes
    /// the root frame [`Entity`] (typically `Res<RootFrameEntityR>.0`)
    /// — the typed phantom is `RootInertial` by convention, asserted
    /// at the call site by passing the root frame entity.
    ///
    /// Equivalent to
    /// `jeod_sim::frame_origin_typed::<RootInertial>(tree, root, frame)`
    /// without the per-call `from_raw_si` lift at the consumer's site.
    pub fn origin_in_root(
        &self,
        root: Entity,
        frame: Entity,
    ) -> (Position<RootInertial>, Velocity<RootInertial>) {
        let (pos_raw, vel_raw) = self.origin_in(root, frame);
        (
            Position::<RootInertial>::from_raw_si(pos_raw), // allowed: SystemParam typed boundary — relative-frame walk returns raw DVec3 (storage-agnostic algorithm shared with the runner's arena); the caller asserts the ancestor is the root by passing the root frame entity, so RootInertial is the correct phantom by construction.
            Velocity::<RootInertial>::from_raw_si(vel_raw), // allowed: same SystemParam typed boundary as `pos_raw` above.
        )
    }

    /// Generic-typed sibling of [`origin_in_root`](Self::origin_in_root)
    /// for callers whose ancestor frame's marker is some other
    /// `F: Frame` (e.g. a `PlanetInertial<P>` integration frame). The
    /// caller asserts that `ancestor`'s marker is `F`; no runtime
    /// check is performed (mirroring `jeod_sim::frame_origin_typed`,
    /// which is also a phantom-tag attachment).
    pub fn origin_in_typed<F: Frame>(
        &self,
        ancestor: Entity,
        frame: Entity,
    ) -> (Position<F>, Velocity<F>) {
        let (pos_raw, vel_raw) = self.origin_in(ancestor, frame);
        (
            Position::<F>::from_raw_si(pos_raw), // allowed: SystemParam typed boundary — caller asserts that `ancestor`'s frame marker is `F` (no runtime check, mirroring `jeod_sim::frame_origin_typed`); the relative-frame walk returns raw DVec3 in `ancestor`-frame coordinates by construction.
            Velocity::<F>::from_raw_si(vel_raw), // allowed: same SystemParam typed boundary as `pos_raw` above.
        )
    }
}
