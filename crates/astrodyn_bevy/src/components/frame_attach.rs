//! Bevy `Component` and `Message` newtypes for non-body reference-frame
//! attachment and detached-subtree free-flight propagation.

use bevy::prelude::*;
use glam::DVec3;

#[allow(unused_imports)] // intra-doc-link resolution
use super::{
    frame_tree::{FrameAngVelC, FrameRotC, FrameTransC},
    mass_tree::{AttachEvent, KinematicChildC},
    state::{RotationalStateC, TranslationalStateC},
};

/// Component: this body is attached to a non-body **reference frame**
/// (not to another body in the mass tree).
///
/// Port of JEOD's `DynBody::frame_attach` member, populated by
/// [`DynBody::attach_to_frame`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/dynamics/dyn_body/src/dyn_body_attach.cc#L271).
/// While present, the body's [`TranslationalStateC`] +
/// [`RotationalStateC`] are derived each tick by the
/// [`crate::frame_attach_system::propagate_frame_attached_state_system`]
/// from the parent frame entity's state composed with the captured
/// offset, and the integration system skips this body (mirrors the
/// `frame_attach.isAttached()` branch in JEOD
/// `dyn_body_integration.cc:309-333`).
///
/// Distinct from [`KinematicChildC`], which gates the same skip path
/// on a parent **body** in the mass tree. A body cannot be both at
/// once — JEOD's `attach_to_frame` writes `frame_attach` on the
/// integrated tree root, not on a child body, and the runner's
/// `Simulation::attach_to_frame` (`astrodyn_runner::Simulation::attach_to_frame`)
/// gate refuses an entity that already has a mass-tree parent. The
/// Bevy adapter's [`crate::frame_attach_system::frame_attach_system`]
/// enforces the same exclusion.
///
/// Mission code MUST NOT insert this component manually — use
/// [`FrameAttachEvent`] / [`FrameDetachEvent`] so the integrator
/// history reset and frame-tree coupling stay consistent. The
/// [`crate::frame_attach_system::frame_attach_system`] inserts and
/// removes the marker.
// JEOD_INV: DB.21 — only unattached bodies integrate (frame-attach gate)
// JEOD_INV: DB.13 — composite-body propagation delegated to parent frame
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct FrameAttachedC {
    /// Entity of the parent reference frame (`FrameEntityC.0` for the
    /// frame). Must point at a frame entity that carries
    /// [`FrameTransC`] / [`FrameRotC`] / [`FrameAngVelC`] — typically a
    /// gravity source's `inertial` or `pfix` frame entity, or any
    /// frame the mission has spawned in the ECS hierarchy.
    pub parent_frame: Entity,
    /// Rigid-body offset from the parent frame to this body's
    /// composite-body frame, in parent-frame coordinates. Frozen at
    /// attach time and never mutated until the body is detached.
    pub offset: DVec3,
    /// Rotation matrix from parent-frame axes to this body's body-frame
    /// axes (`t_parent_struct` in the runner API). Frozen at attach time.
    pub t_parent_body: glam::DMat3,
}

/// Message: attach a body to a non-body reference frame.
///
/// Bevy adapter for JEOD's `DynBody::attach_to_frame`. The
/// `frame_attach_system` inserts a [`FrameAttachedC`] component on
/// `body`, captures the offset, and resets multi-step integrator
/// history. Subsequent ticks derive the body's state from
/// `parent_frame`'s current state plus `offset`. See
/// `Simulation::attach_to_frame` (`astrodyn_runner::Simulation::attach_to_frame`)
/// for the runner-side equivalent.
#[derive(Message, Debug, Clone)]
pub struct FrameAttachEvent {
    /// Entity of the body to attach.
    pub body: Entity,
    /// Entity of the parent reference frame (a frame entity carrying
    /// [`FrameTransC`] / [`FrameRotC`] / [`FrameAngVelC`]).
    pub parent_frame: Entity,
    /// Body structural origin in parent-frame coordinates (m). Frozen
    /// at attach time.
    pub offset: DVec3,
    /// Rotation matrix from parent-frame axes to body-frame axes.
    /// Frozen at attach time.
    pub t_parent_body: glam::DMat3,
}

/// Message: release a body's reference-frame attachment.
///
/// Bevy adapter for JEOD `DynBody::detach()` (the
/// `frame_attach.isAttached()` branch in
/// `models/dynamics/dyn_body/src/dyn_body_detach.cc:141-143`). The
/// `frame_attach_system` removes the [`FrameAttachedC`] component;
/// integration resumes on the next step from whatever state the
/// frame-attached propagation left in [`TranslationalStateC`] /
/// [`RotationalStateC`].
#[derive(Message, Debug, Clone)]
pub struct FrameDetachEvent {
    /// Entity of the body to detach.
    pub body: Entity,
}

/// Composite-body inertial state of a free-flying mass-tree subtree
/// that has been detached from its parent and is coasting
/// ballistically (no force, no torque) until [`AttachEvent`] re-attaches
/// it.
///
/// Inserted on the child entity by `staging_system`'s `DetachEvent`
/// branch; removed by the `AttachEvent` branch when the same entity
/// is re-attached. While present, [`step_detached_system`](crate::step_detached_system)
/// advances the contained state by the schedule's fixed `dt` each
/// tick — position drifts at `composite_velocity`, attitude rotates
/// under `composite_ang_vel_body`. Also synchronizes the entity's
/// own [`TranslationalStateC`] / [`RotationalStateC`] each tick so
/// downstream consumers (gravity, derived state, mission code)
/// continue to read the body's current inertial state from the
/// canonical components rather than having to special-case detached
/// vs integrated bodies.
///
/// Bevy mirror of `astrodyn_runner::Simulation::detached_subtrees`. Wraps
/// [`astrodyn::DetachedSubtreeState`] (which owns the JEOD scalar-first
/// left-multiply attitude convention via
/// [`astrodyn::BodyAttitude<astrodyn::SelfRef>`](astrodyn::BodyAttitude)).
#[derive(Component, Debug, Clone, Copy)]
pub struct DetachedSubtreeStateC(pub astrodyn::DetachedSubtreeState);
