//! Frame-tree orchestration helpers shared by every consumer of `astrodyn`.
//!
//! These functions were lifted from `astrodyn_runner::simulation` (issue #71).
//! Both `astrodyn_runner` (standalone harness) and ECS adapters (the
//! `astrodyn_bevy` root crate) need to update planet-fixed rotations and
//! evaluate distance-based integration-frame switches against a frame
//! tree, so the logic lives at the orchestration layer where every
//! consumer can call it. Per CLAUDE.md, `astrodyn_runner` is a peer of
//! `astrodyn_bevy`, not a layer above it; both consume `astrodyn`.

use astrodyn_frames::{FrameId, FrameTree, RefFrameStateTyped};
use astrodyn_math::JeodQuat;
use astrodyn_quantities::aliases::{Position, Velocity};
use astrodyn_quantities::frame::Frame;
use glam::{DMat3, DVec3};

use crate::vehicle_config::{FrameSwitchConfig, SwitchSense};
use crate::GravityControls;
use crate::TranslationalState;

/// Sync a planet-fixed frame node's rotation state from a computed matrix.
///
/// Sets `t_parent_this`, derives `q_parent_this` from it, and sets
/// `ang_vel_this = [0, 0, planet_omega]` matching JEOD's `planet_rnp.cc`.
/// The `planet_omega` value typically comes from
/// [`crate::PlanetConfig::omega`].
pub fn sync_pfix_rotation(
    frame_tree: &mut FrameTree,
    pfix_id: FrameId,
    rotation: DMat3,
    planet_omega: f64,
) {
    let node = frame_tree.get_mut(pfix_id);
    node.state.rot.t_parent_this = rotation;
    node.state.rot.q_parent_this = JeodQuat::left_quat_from_transformation(&rotation);
    // JEOD sets ang_vel_this = [0, 0, planet_omega] in planet_rnp.cc.
    // This is used by compute_relative_state velocity composition.
    node.state.rot.ang_vel_this = DVec3::new(0.0, 0.0, planet_omega);
}

/// Get the position and velocity of a frame relative to the root inertial
/// frame. Mirrors `astrodyn_runner::Simulation::frame_origin` so consumers that
/// don't own a `Simulation` (the Bevy adapter) can compute frame origins
/// using only the frame tree.
pub fn frame_origin(
    frame_tree: &FrameTree,
    root_frame_id: FrameId,
    frame_id: FrameId,
) -> (DVec3, DVec3) {
    if frame_id == root_frame_id {
        return (DVec3::ZERO, DVec3::ZERO);
    }
    let state = frame_tree.compute_relative_state(root_frame_id, frame_id);
    (state.trans.position, state.trans.velocity)
}

/// Typed sibling of [`frame_origin`] returning the frame's position and
/// velocity already wrapped as `Position<F>` / `Velocity<F>`. The frame
/// marker `F` is the **root frame's** marker (the result expresses the
/// query frame's origin in root-relative coordinates) — caller asserts
/// the root is `F`. For the Bevy adapter and `astrodyn_runner` today, the
/// root is by construction inertial, so callers pass `F = Inertial`.
///
/// Phase C5 of issue #71: lets consumers compute the integration-frame
/// origin without re-lifting raw `DVec3` through `from_raw_si` at the
/// hot-path boundary.
pub fn frame_origin_typed<F: Frame>(
    frame_tree: &FrameTree,
    root_frame_id: FrameId,
    frame_id: FrameId,
) -> (Position<F>, Velocity<F>) {
    let (pos_raw, vel_raw) = frame_origin(frame_tree, root_frame_id, frame_id);
    (
        Position::<F>::from_raw_si(pos_raw),
        Velocity::<F>::from_raw_si(vel_raw),
    )
}

/// Typed sibling of [`FrameTree::compute_relative_state`] returning a
/// [`RefFrameStateTyped<From, To>`]. The caller asserts the supplied
/// frame IDs correspond to frames whose markers are `From` and `To` —
/// the arena is heterogeneous and stores untyped `RefFrameState` per
/// node, so this is a `from_untyped_unchecked` boundary lift.
///
/// Phase C5 of issue #71.
pub fn compute_relative_state_typed<From: Frame, To: Frame>(
    frame_tree: &FrameTree,
    from: FrameId,
    to: FrameId,
) -> RefFrameStateTyped<From, To> {
    let untyped = frame_tree.compute_relative_state(from, to);
    RefFrameStateTyped::<From, To>::from_untyped_unchecked(&untyped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use astrodyn_frames::{RefFrameKind, RefFrameRot, RefFrameState, RefFrameTrans};
    use astrodyn_quantities::frame::{Ecef, RootInertial};

    #[test]
    fn frame_origin_typed_matches_untyped() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let child = tree.add_child(
            root,
            "child".into(),
            RefFrameKind::Inertial,
            RefFrameState {
                trans: RefFrameTrans {
                    position: DVec3::new(1.0e7, 2.0e7, 3.0e7),
                    velocity: DVec3::new(100.0, 200.0, 300.0),
                },
                rot: RefFrameRot::default(),
            },
        );

        let (untyped_pos, untyped_vel) = frame_origin(&tree, root, child);
        let (typed_pos, typed_vel) = frame_origin_typed::<RootInertial>(&tree, root, child);

        assert_eq!(typed_pos.raw_si(), untyped_pos);
        assert_eq!(typed_vel.raw_si(), untyped_vel);
    }

    #[test]
    fn frame_origin_typed_root_is_zero() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let (typed_pos, typed_vel) = frame_origin_typed::<RootInertial>(&tree, root, root);
        assert_eq!(typed_pos.raw_si(), DVec3::ZERO);
        assert_eq!(typed_vel.raw_si(), DVec3::ZERO);
    }

    #[test]
    fn compute_relative_state_typed_matches_untyped() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let child = tree.add_child(
            root,
            "ecef".into(),
            RefFrameKind::PlanetFixed,
            RefFrameState {
                trans: RefFrameTrans {
                    position: DVec3::new(1.0, 2.0, 3.0),
                    velocity: DVec3::new(0.1, 0.2, 0.3),
                },
                rot: RefFrameRot::default(),
            },
        );

        let untyped = tree.compute_relative_state(root, child);
        let typed = compute_relative_state_typed::<RootInertial, Ecef>(&tree, root, child);

        assert_eq!(typed.trans.position.raw_si(), untyped.trans.position);
        assert_eq!(typed.trans.velocity.raw_si(), untyped.trans.velocity);
        assert_eq!(typed.rot.t_parent_this(), untyped.rot.t_parent_this);
    }
}

/// Error returned by [`evaluate_and_apply_frame_switch`] when a configured
/// switch references a source identifier that the caller-supplied
/// `resolve_source` closure couldn't map to a frame.
///
/// Generic over `SourceId` so the runner (`SourceId = usize`) and the
/// Bevy adapter (`SourceId = Entity`) both surface meaningful diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSwitchTargetMissing<SourceId = usize> {
    /// Index of the body whose `frame_switches` referenced the missing target.
    pub body_idx: usize,
    /// Source identifier requested by the switch.
    pub target_source: SourceId,
    /// Number of sources currently registered (for diagnostics; the
    /// caller passes this in since the closure-based source lookup
    /// doesn't expose a count).
    pub num_sources: usize,
}

/// Evaluate distance-based integration-frame switches for a single body and,
/// if a switch triggers, reparent the body's frame in the tree, copy the
/// post-switch translational state out of the tree, and flip the body's
/// gravity-controls `differential` flags (target source becomes central, all
/// others become differential).
///
/// JEOD reference: `dyn_body_frame_switch.cc:173-182`. Applied AFTER
/// integration so the body integrates in its current frame for this step
/// and transforms to the new frame for the next step.
///
/// Generic over the source identifier type used by [`FrameSwitchConfig`]
/// and [`GravityControls`] — `usize` for `astrodyn_runner` (sources indexed by
/// registration order), `bevy::ecs::entity::Entity` for the Bevy adapter
/// (sources identified by ECS entity). The caller supplies a
/// `resolve_source` closure that maps a `SourceId` to its inertial
/// [`FrameId`] in the frame tree.
///
/// Returns `Ok(true)` if a switch fired, `Ok(false)` if no switch triggered,
/// or `Err(FrameSwitchTargetMissing)` if a configured target source could
/// not be resolved.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_and_apply_frame_switch<SourceId, F>(
    frame_tree: &mut FrameTree,
    root_frame_id: FrameId,
    body_frame_id: FrameId,
    integ_frame_id: &mut FrameId,
    trans: &mut TranslationalState,
    frame_switches: &mut [FrameSwitchConfig<SourceId>],
    gravity_controls: &mut GravityControls<SourceId>,
    resolve_source: F,
    num_sources: usize,
    body_idx: usize,
) -> Result<bool, FrameSwitchTargetMissing<SourceId>>
where
    SourceId: Clone + PartialEq,
    F: Fn(&SourceId) -> Option<FrameId>,
{
    if frame_switches.is_empty() {
        return Ok(false);
    }

    let mut switch_idx = None;
    for (idx, sw) in frame_switches.iter().enumerate() {
        if !sw.active {
            continue;
        }
        let target_fid =
            resolve_source(&sw.target_source).ok_or_else(|| FrameSwitchTargetMissing {
                body_idx,
                target_source: sw.target_source.clone(),
                num_sources,
            })?;
        let (target_origin, _) = frame_origin(frame_tree, root_frame_id, target_fid);
        let (current_origin, _) = frame_origin(frame_tree, root_frame_id, *integ_frame_id);
        let body_pos_eci = trans.position + current_origin;
        let threshold_sq = sw.switch_distance * sw.switch_distance;

        // JEOD dyn_body_frame_switch.cc:173-182:
        // OnApproach: compute_position_from(*integ_frame) → distance to target
        // OnDeparture: state.trans.position magnitude → distance from current origin
        let triggered = match sw.switch_sense {
            SwitchSense::OnApproach => {
                (body_pos_eci - target_origin).length_squared() < threshold_sq
            }
            SwitchSense::OnDeparture => trans.position.length_squared() > threshold_sq,
        };
        if triggered {
            switch_idx = Some(idx);
            break;
        }
    }

    let Some(idx) = switch_idx else {
        return Ok(false);
    };

    let target_source = frame_switches[idx].target_source.clone();
    frame_switches[idx].active = false;

    // Resolve again — the closure was already proven to return Some for
    // this target above, so unwrap is safe (and a re-failure here would
    // be a caller-side data race we want to surface).
    let new_integ_fid = resolve_source(&target_source).expect(
        "evaluate_and_apply_frame_switch: target source resolved during evaluation \
         but failed during application — caller-side mutation between lookups",
    );

    // Reparent body frame in tree (preserves absolute state).
    frame_tree.reparent(body_frame_id, new_integ_fid);
    let new_state = frame_tree.get(body_frame_id).state;
    trans.position = new_state.trans.position;
    trans.velocity = new_state.trans.velocity;
    *integ_frame_id = new_integ_fid;

    // Flip gravity controls: target source becomes non-differential
    // (central body), all others become differential.
    for ctrl in &mut gravity_controls.controls {
        ctrl.differential = ctrl.source_name != target_source;
    }
    Ok(true)
}
