//! Bevy `Component` newtypes for the frame-tree-as-entities
//! infrastructure: per-frame translational / rotational /
//! angular-velocity state, frame-kind markers, and the bidirectional
//! handles linking body / source / planet entities to their frame
//! entities.

use bevy::prelude::*;
use glam::DVec3;

#[allow(unused_imports)] // intra-doc-link resolution
use super::gravity::{PlanetFixedRotationC, RotationModelC};

// ── Frames-as-entities components ──
//
// Live on **frame entities** (not body or source entities) and carry
// the per-frame state described in the [Frame-Tree-ECS-Native wiki
// page][1] (Section 13 sequencing). The ECS hierarchy is the single
// source of truth for all frame-tree state — gravity, integration,
// frame-switch, and mission code via [`crate::frame_param`] all read
// from the ECS hierarchy directly via `ChildOf` / `Children`.
//
// Component split rationale: the three pieces of `RefFrameState` are
// independently mutated in practice. `FrameTransC` is rewritten by
// integration / source-position updates; `FrameRotC` is rewritten by
// planet-fixed rotation updates; `FrameAngVelC` is rewritten alongside
// `FrameRotC` for pfix frames but stays at zero for inertial / body
// frames. Splitting lets change-detection fire on the right writers
// only.
//
// [1]: https://github.com/simnaut/bevy_jeod/wiki/Frame-Tree-ECS-Native#13-migration-sequencing

/// Translational state (position + velocity) of a frame entity
/// relative to its parent frame entity. Mirrors
/// [`jeod_sim::RefFrameTrans`]. Stored raw (`DVec3`) at the additive-
/// infrastructure stage; later PRs in the sequence may carry typed
/// `Position<P>` / `Velocity<P>` keyed off the parent frame entity's
/// marker. Issue #277.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct FrameTransC {
    /// Position relative to parent frame, in parent-frame coordinates (m).
    pub position: DVec3,
    /// Velocity relative to parent frame, in parent-frame coordinates (m/s).
    pub velocity: DVec3,
}

/// Rotational state of a frame entity relative to its parent: the
/// left-transformation quaternion and the cached transformation matrix
/// `t_parent_this`. Mirrors the rotation portion of
/// [`jeod_sim::RefFrameRot`] minus `ang_vel_this`, which lives in
/// [`FrameAngVelC`] for change-detection granularity. Issue #277.
#[derive(Component, Debug, Clone, Copy)]
pub struct FrameRotC {
    /// Left-transformation quaternion (parent → this).
    pub q_parent_this: jeod_sim::JeodQuat,
    /// Transformation matrix `t_parent_this` derived from the quaternion (cache).
    pub t_parent_this: glam::DMat3,
}

impl Default for FrameRotC {
    fn default() -> Self {
        Self {
            q_parent_this: jeod_sim::JeodQuat::identity(),
            t_parent_this: glam::DMat3::IDENTITY,
        }
    }
}

/// Angular velocity of a frame entity relative to its parent, expressed
/// in this-frame coordinates (rad/s). Mirrors
/// [`jeod_sim::RefFrameRot::ang_vel_this`]. Split from [`FrameRotC`] so
/// pfix-rotation systems that only rewrite angular velocity (or body
/// integration that only rewrites attitude) get fine-grained
/// change-detection. Issue #277.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct FrameAngVelC(pub DVec3);

// Marker components — mark the kind of frame an entity represents.
// Bevy idiom: query keying via `With<…>`. Replaces the runtime
// `RefFrameKind` enum on the arena side. Suffix `Marker` avoids
// colliding with `jeod_sim`'s phantom-frame types (`BodyFrame`,
// `PlanetFixed`, `IntegrationFrame`, `RootInertial`).

/// Marker: this entity is a root or planet inertial frame. Issue #277.
#[derive(Component, Debug, Default, Clone, Copy)]
pub struct InertialFrameMarker;

/// Marker: this entity is a planet-fixed (pfix) frame, child of an
/// inertial frame, rotating with the planet. Issue #277.
#[derive(Component, Debug, Default, Clone, Copy)]
pub struct PlanetFixedFrameMarker;

/// Marker: this entity is a body's body-frame (composite_body in JEOD).
/// Issue #277.
#[derive(Component, Debug, Default, Clone, Copy)]
pub struct BodyFrameMarker;

/// Marker: this entity has been registered as an integration frame for
/// at least one body. Inserted (idempotently) by
/// `register_body_frames_system` when a body is spawned with this
/// frame as its integration frame; never removed. The marker has
/// **sticky** semantics — `frame_switch_system` reparents a body's
/// frame entity (via `commands.entity(...).insert(ChildOf(...))`)
/// when the body switches frames, but does not touch this marker,
/// because (a) one integration frame entity can serve many bodies
/// and tracking a "currently in use" predicate would require
/// ref-counting, and (b) downstream SystemParam consumers in later
/// PRs of the [Section 13 sequence][1] only need to know whether a
/// frame entity *can* serve as an integration frame, which the
/// registration-time signal answers correctly. The authoritative
/// "this body's integration frame is X" lookup is the body frame
/// entity's `ChildOf` parent. A frame entity may carry both
/// `InertialFrameMarker` and `IntegrationFrameMarker` simultaneously
/// — they describe orthogonal properties of the frame. Issue #277.
///
/// [1]: https://github.com/simnaut/bevy_jeod/wiki/Frame-Tree-ECS-Native#13-migration-sequencing
#[derive(Component, Debug, Default, Clone, Copy)]
pub struct IntegrationFrameMarker;

/// Bidirectional handle linking a body / source / planet entity to its
/// frame entity in the ECS hierarchy. Inserted by
/// `register_*_frames_system` for every entity that carries dynamics
/// state. Internal physics consumers (gravity, integration,
/// frame-switch) and mission code via [`crate::frame_param`] read this
/// handle and walk `Query<&ChildOf>` from the frame entity to recover
/// the body's integration frame, the source's child frames, etc. The
/// frame entity itself carries [`FrameTransC`] / [`FrameRotC`] /
/// [`FrameAngVelC`] (the per-node state).
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
pub struct FrameEntityC(pub Entity);

/// Frame entity for a source's planet-fixed (pfix) child frame.
///
/// Inserted by `register_pfix_frames_system` for every gravity source
/// that carries [`PlanetFixedRotationC`] and a non-`None`
/// [`RotationModelC`]. Removed when the rotation model toggles to
/// `None` (in which case the underlying ECS entity is retained as
/// [`RetiredPfixFrameEntityC`] for reuse on the next toggle back to a
/// rotating model — see that component's docs).
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
pub struct PfixFrameEntityC(pub Entity);

/// Hidden component that stashes a previously-spawned pfix *frame
/// entity* (the canonical [`PfixFrameEntityC`]) on a source whose
/// [`RotationModelC`] just toggled to
/// [`RotationModel::None`](jeod_sim::RotationModel::None). The
/// public [`PfixFrameEntityC`] is removed at the same time so any
/// reader branching on its presence correctly observes "no
/// planet-fixed frame", but the orphan ECS entity itself is kept
/// alive — its `Name` is renamed to a `.retired` sentinel and its
/// `FrameRotC`/`FrameAngVelC` are reset to identity — so the next
/// toggle back to a rotating model can reuse it instead of spawning
/// a fresh entity.
///
/// Without this, every `None → rotating → None → rotating …` toggle
/// cycle would leak a fresh `<name>.frame.pfix` entity per cycle,
/// since [`crate::systems::register_pfix_frames_system`] filters by
/// `Without<PfixFrameEntityC>` and unconditionally spawns a new
/// entity for any source missing the public component.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
pub struct RetiredPfixFrameEntityC(pub Entity);
