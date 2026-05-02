//! Issue #268 prototype: Bevy-native `SystemParam`s for cross-frame
//! state computation.
//!
//! Replaces `FrameTreeR`-backed arena lookups (`compute_relative_state`,
//! `frame_origin`) with ECS hierarchy walks over Bevy 0.18's
//! `ChildOf` / `Children` relationship and the new
//! [`FrameTransC`](crate::components::FrameTransC) /
//! [`FrameRotC`](crate::components::FrameRotC) /
//! [`FrameAngVelC`](crate::components::FrameAngVelC) components.
//!
//! Mission code asks for cross-frame state by passing entity handles,
//! never `FrameId`s — the surface looks like any other Bevy
//! SystemParam.

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use glam::DVec3;
use jeod_sim::{FrameStorage, RefFrameRot, RefFrameState, RefFrameTrans};

use crate::components::{FrameAngVelC, FrameRotC, FrameTransC};

/// Compute relative state between two frame entities by walking
/// Bevy's hierarchy and composing per-node states with the pure-state
/// math from `jeod_frames` (`incr_left`, `incr_right`, `negate`).
///
/// ECS-native replacement for
/// `FrameTreeR.compute_relative_state(from_id, to_id)` /
/// `frame_origin(tree, root, frame_id)`. Issue #268.
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
    /// is single-sourced.
    pub fn relative_state(&self, from: Entity, to: Entity) -> RefFrameState {
        jeod_sim::frame_compute_relative_state_via_storage(self, from, to)
    }
}

// `FrameStorage` impl: lets the storage-agnostic algorithms in
// `jeod_frames::frame_storage` operate over the ECS hierarchy + the
// new frame-state components. Issue #268 trait-experiment.
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

// ──────────────────────────────────────────────────────────────────────
// Issue #268 design-doc evidence: Sketch 2 — frame switch handler
// rewritten in ECS-native shape using `ChildOf` + `RelativeFrameState`.
//
// Compiles but is not wired into any schedule. Exists to demonstrate
// the "After" snippet from the plan reads cleanly and that the
// `RelativeFrameState` SystemParam composes with `Commands` and
// hierarchy mutation for the canonical frame-switch use case. The
// production `frame_switch_system` (`crate::systems::frame_switch_system`)
// stays on the lifted `jeod_sim::evaluate_and_apply_frame_switch`
// helper for now — this sketch is the design-doc reference for what
// the eventual replacement looks like.
// ──────────────────────────────────────────────────────────────────────

use crate::components::{BodyFrameMarker, FrameEntityC, FrameSwitchesC};
use jeod_sim::{Position, RootInertial, SwitchSense, Velocity};

/// Sketch: Bevy-native frame-switch handler. Walks each body's
/// `FrameSwitchesC`, evaluates switch distance via the
/// [`RelativeFrameState`] SystemParam, and on trigger reparents the
/// body's frame entity via `commands.entity(child).insert(ChildOf(p))`
/// — the canonical ECS-native pattern.
///
/// **Not wired into the schedule.** The production frame-switch system
/// uses the lifted arena helper `jeod_sim::evaluate_and_apply_frame_switch`.
/// This sketch is design-doc evidence per issue #268: it demonstrates
/// that the "After" snippet in the plan compiles in real Rust.
#[allow(dead_code)]
pub fn frame_switch_system_ecs_native_sketch(
    mut commands: Commands,
    rel: RelativeFrameState,
    sources: Query<&FrameEntityC, Without<BodyFrameMarker>>,
    mut bodies: Query<
        (
            Entity,
            &mut crate::components::TranslationalStateC,
            &FrameEntityC,
            &mut FrameSwitchesC,
            &mut crate::components::GravityControlsC,
        ),
        With<crate::components::DynamicsConfigC>,
    >,
    parents: Query<&ChildOf>,
) {
    for (body_entity, mut trans, body_frame, mut switches, mut gravity_controls) in &mut bodies {
        if switches.0.is_empty() {
            continue;
        }

        // Current integ frame is the body frame's parent in the ECS
        // hierarchy — no `IntegFrameIdC` lookup needed.
        let current_integ_frame = match parents.get(body_frame.0) {
            Ok(child_of) => child_of.parent(),
            Err(_) => continue,
        };

        // Evaluate each active switch using `RelativeFrameState`.
        let mut triggered: Option<(usize, Entity)> = None;
        for (idx, sw) in switches.0.iter().enumerate() {
            if !sw.active {
                continue;
            }
            let target_frame = match sources.get(sw.target_source).ok() {
                Some(fe) => fe.0,
                None => continue,
            };
            // Body position expressed in the target frame: if it's
            // within the switch distance, fire the switch.
            let body_in_target = rel.position(target_frame, body_frame.0).length_squared();
            let body_in_current = rel
                .position(current_integ_frame, body_frame.0)
                .length_squared();
            let threshold_sq = sw.switch_distance * sw.switch_distance;
            let fire = match sw.switch_sense {
                SwitchSense::OnApproach => body_in_target < threshold_sq,
                SwitchSense::OnDeparture => body_in_current > threshold_sq,
            };
            if fire {
                triggered = Some((idx, target_frame));
                break;
            }
        }

        let Some((idx, new_parent_frame)) = triggered else {
            continue;
        };

        // Reproject body's translational state into the new integ
        // frame. `rel.position`/`relative_state` operate over the
        // current ECS hierarchy — exactly what we need to compute the
        // body's position vs. the new parent before reparenting.
        let new_state = rel.relative_state(new_parent_frame, body_frame.0);
        trans.0.position = Position::<RootInertial>::from_raw_si(new_state.trans.position);
        trans.0.velocity = Velocity::<RootInertial>::from_raw_si(new_state.trans.velocity);

        // Reparent the body's frame entity. This is the load-bearing
        // ECS-native operation: replaces `FrameTree::reparent` and
        // sidesteps the entire arena `FrameId` machinery. Bevy's
        // hierarchy plugin updates `Children` on both old and new
        // parents automatically.
        commands
            .entity(body_frame.0)
            .insert(ChildOf(new_parent_frame));

        // Flip gravity controls: target source becomes central; all
        // others differential.
        let target_source = switches.0[idx].target_source;
        switches.0[idx].active = false;
        for ctrl in &mut gravity_controls.0.controls {
            ctrl.differential = ctrl.source_name != target_source;
        }
        let _ = body_entity; // kept for symmetry with production system's diagnostic strings
    }
}
