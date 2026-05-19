//! Bevy-native `SystemParam`s for cross-frame state computation
//! ([Frame-Tree-ECS-Native § 13][1]).
//!
//! ECS-native cross-frame queries via Bevy's `ChildOf` / `Children`
//! relationship and the [`FrameTransC`] / [`FrameRotC`] /
//! [`FrameAngVelC`] components on frame entities. Mission code asks
//! for cross-frame state by passing entity handles; the
//! `SystemParam`s walk the hierarchy and dispatch through the
//! storage-agnostic [`astrodyn::FrameStorage`] algorithm shared with
//! `astrodyn_runner`.
//!
//! Mission code asks for cross-frame state by passing entity handles,
//! never `FrameId`s — the surface looks like any other Bevy
//! `SystemParam`. The shared algorithm lives in
//! [`astrodyn::frame_compute_relative_state_via_storage`] (the
//! `FrameStorage` trait abstraction described in [§ 7][2]); this
//! `SystemParam` only supplies the storage adapter.
//!
//! Two SystemParams are exposed (mirroring [§ 6][3]'s mission-code
//! catalog):
//!
//! - [`RelativeFrameState`] — general "state of `to` relative to
//!   `from`" query. Returns raw `DVec3` (and the full
//!   [`RefFrameState`] when the rotation/angular-velocity portion is
//!   needed).
//! - [`FrameOrigin`] — specialized "origin of a frame in an ancestor
//!   frame" query. Returns
//!   `(Position<RootInertial>, Velocity<RootInertial>)` typed at the
//!   root-inertial phantom when called against the root frame entity.
//!   Sugar over `RelativeFrameState::position_velocity(root, frame)`
//!   that makes the resulting frame phantom explicit in the
//!   signature.
//!
//! Both `SystemParam`s walk the ECS hierarchy
//! (`Query<&ChildOf>`) over the
//! [`crate::components::FrameTransC`] / [`crate::components::FrameRotC`] /
//! [`crate::components::FrameAngVelC`] components on frame entities,
//! delegating the per-step compose to the storage-agnostic algorithm
//! shared with `astrodyn_runner`'s arena via the
//! [`astrodyn::FrameStorage`] trait.
//!
//! [1]: https://github.com/simnaut/astrodyn/wiki/Frame-Tree-ECS-Native#13-migration-sequencing
//! [2]: https://github.com/simnaut/astrodyn/wiki/Frame-Tree-ECS-Native#7-internal-algorithm-sharing-q1
//! [3]: https://github.com/simnaut/astrodyn/wiki/Frame-Tree-ECS-Native#6-mission-code-surface--systemparam-catalog-q7

use astrodyn::{
    Frame, FrameStorage, Position, RefFrameRot, RefFrameState, RefFrameTrans, RootInertial,
    Velocity,
};
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use glam::DVec3;

use crate::components::{FrameAngVelC, FrameRotC, FrameTransC};

/// Compute relative state between two frame entities by walking
/// Bevy's hierarchy and composing per-node states with the pure-state
/// math from `astrodyn_frames` (`incr_left`, `incr_right`, `negate`),
/// dispatched through the storage-agnostic
/// [`astrodyn::frame_compute_relative_state_via_storage`] algorithm.
///
/// Numerically identical to `astrodyn_runner::Simulation`'s arena-backed
/// `compute_relative_state` for the same scenario — both consumers
/// share the storage-agnostic algorithm via the
/// [`astrodyn::FrameStorage`] trait. Mission code reads an
/// `Entity`-keyed surface (no `FrameId`s, no frame-tree resource).
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
    /// Typed `(Position<F>, Velocity<F>)` of `to` relative to `from`,
    /// both in `from`-frame coordinates. The caller asserts via the
    /// turbofish that `from`'s frame marker is `F` — phantom-attachment
    /// is unchecked at runtime, matching [`FrameOrigin::origin_in_typed`].
    ///
    /// Callers whose `from`'s marker isn't known at compile time should
    /// read [`relative_state`](Self::relative_state) and project the
    /// `.trans.position` / `.trans.velocity` fields directly.
    pub fn position_velocity_typed<F: astrodyn::Frame>(
        &self,
        from: Entity,
        to: Entity,
    ) -> (astrodyn::Position<F>, astrodyn::Velocity<F>) {
        let trans = self.relative_state(from, to).trans;
        (
            // allowed: typed-sibling accessor boundary; phantom attached unchecked.
            astrodyn::Position::<F>::from_raw_si(trans.position),
            // allowed: typed-sibling accessor boundary; phantom attached unchecked.
            astrodyn::Velocity::<F>::from_raw_si(trans.velocity),
        )
    }

    /// Typed `Position<F>` of `to` relative to `from`, in `from`-frame
    /// coordinates. See [`position_velocity_typed`](Self::position_velocity_typed)
    /// for the phantom-attachment semantics.
    pub fn position_typed<F: astrodyn::Frame>(
        &self,
        from: Entity,
        to: Entity,
    ) -> astrodyn::Position<F> {
        // allowed: typed-sibling accessor boundary; phantom attached unchecked.
        astrodyn::Position::<F>::from_raw_si(self.relative_state(from, to).trans.position)
    }

    /// Full [`RefFrameState`] of `to` relative to `from`. Delegates
    /// to the storage-agnostic
    /// [`astrodyn::frame_compute_relative_state_via_storage`]
    /// algorithm via this `SystemParam`'s [`FrameStorage`] impl —
    /// the same code path the runner's arena uses, so the algorithm
    /// is single-sourced (see [Frame-Tree-ECS-Native § 7][1]).
    ///
    /// [1]: https://github.com/simnaut/astrodyn/wiki/Frame-Tree-ECS-Native#7-internal-algorithm-sharing-q1
    pub fn relative_state(&self, from: Entity, to: Entity) -> RefFrameState {
        astrodyn::frame_compute_relative_state_via_storage(self, from, to)
    }
}

/// `FrameStorage` impl: lets the storage-agnostic algorithms in
/// `astrodyn_frames::frame_storage` operate over the ECS hierarchy + the
/// frame-state components.
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
/// `astrodyn::frame_origin` / `astrodyn::frame_origin_typed::<F>`
/// ([Frame-Tree-ECS-Native § 6][1]).
///
/// [1]: https://github.com/simnaut/astrodyn/wiki/Frame-Tree-ECS-Native#6-mission-code-surface--systemparam-catalog-q7
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
///   without a `from_raw_si` boundary at the call site.
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
///   `astrodyn::frame_origin_typed`.
///
/// # Example
/// ```ignore
/// use bevy::prelude::*;
/// use astrodyn_bevy::prelude::*;
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
    /// `astrodyn::frame_origin(tree, root, root)`.
    ///
    /// Prefer [`origin_in_root`](Self::origin_in_root) for the common
    /// "in the root inertial frame" query and
    /// [`origin_in_typed`](Self::origin_in_typed) for callers whose
    /// ancestor frame has a statically-known phantom marker. This raw
    /// variant is the deliberate escape hatch for the case where the
    /// ancestor's marker isn't known at compile time (frame-tree walks
    /// driven by runtime entity references).
    // ESCAPE_HATCH: documented raw return for the non-static-F case —
    // the typed siblings origin_in_root / origin_in_typed handle the
    // static-marker majority of callers. See #485 H4 / T2.
    pub fn origin_in(&self, ancestor: Entity, frame: Entity) -> (DVec3, DVec3) {
        let trans = self.rel.relative_state(ancestor, frame).trans;
        (trans.position, trans.velocity)
    }

    /// Typed `(Position<RootInertial>, Velocity<RootInertial>)` for the
    /// common "origin in the root inertial frame" query. Caller passes
    /// the root frame [`Entity`] (typically `Res<RootFrameEntityR>.0`)
    /// — the typed phantom is `RootInertial` by convention, asserted
    /// at the call site by passing the root frame entity.
    ///
    /// Equivalent to
    /// `astrodyn::frame_origin_typed::<RootInertial>(tree, root, frame)`
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
    /// check is performed (mirroring `astrodyn::frame_origin_typed`,
    /// which is also a phantom-tag attachment).
    pub fn origin_in_typed<F: Frame>(
        &self,
        ancestor: Entity,
        frame: Entity,
    ) -> (Position<F>, Velocity<F>) {
        let (pos_raw, vel_raw) = self.origin_in(ancestor, frame);
        (
            Position::<F>::from_raw_si(pos_raw), // allowed: SystemParam typed boundary — caller asserts that `ancestor`'s frame marker is `F` (no runtime check, mirroring `astrodyn::frame_origin_typed`); the relative-frame walk returns raw DVec3 in `ancestor`-frame coordinates by construction.
            Velocity::<F>::from_raw_si(vel_raw), // allowed: same SystemParam typed boundary as `pos_raw` above.
        )
    }
}
