//! Frame-tree orchestration helpers shared by every consumer of `astrodyn`.
//!
//! These functions were lifted from `astrodyn_runner::simulation` (issue #71).
//! Both `astrodyn_runner` (standalone harness) and ECS adapters (the
//! `astrodyn_bevy` root crate) need to update planet-fixed rotations and
//! evaluate distance-based integration-frame switches against a frame
//! tree, so the logic lives at the orchestration layer where every
//! consumer can call it. Per CLAUDE.md, `astrodyn_runner` is a peer of
//! `astrodyn_bevy`, not a layer above it; both consume `astrodyn`.

use astrodyn_frames::{FrameId, FrameTree};
use astrodyn_quantities::frame_descriptor::FrameUid;

pub use astrodyn_frames::{
    compute_relative_state_typed, frame_origin, frame_origin_typed, sync_pfix_rotation,
};

use crate::vehicle_config::{FrameSwitchConfig, SwitchSense};
use crate::GravityControls;
use crate::TranslationalState;

/// Error returned by [`evaluate_and_apply_frame_switch`] when a configured
/// switch references a source identity that the caller-supplied
/// `resolve_target` closure couldn't map to a frame.
///
/// Carries the missing [`FrameUid`] so the diagnostic names the identity
/// (issue #668) — host-agnostic, unlike the pre-collapse generic that
/// surfaced a bare index or entity id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameSwitchTargetMissing {
    /// Index of the body whose `frame_switches` referenced the missing target.
    pub body_idx: usize,
    /// Source identity requested by the switch.
    pub target: FrameUid,
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
/// Switch targets and gravity controls reference sources by
/// [`FrameUid`] (issue #668). The caller supplies a `resolve_target`
/// closure that maps an identity to its inertial [`FrameId`] in the
/// frame tree — host knowledge stays at the host: the runner restricts
/// valid targets to its registered sources' inertial frames, which a
/// bare tree lookup could not express (a body-frame identity must not
/// silently become a reparent target).
///
/// Returns `Ok(true)` if a switch fired, `Ok(false)` if no switch triggered,
/// or `Err(FrameSwitchTargetMissing)` if a configured target could not be
/// resolved.
///
/// # Panics
/// Panics if a triggering switch's target identity is not an
/// inertial-flavor class — the body would integrate there, and only
/// root/planet/barycenter-inertial frames may host integration (the
/// same class eligibility enforced at tree construction).
#[allow(clippy::too_many_arguments)]
pub fn evaluate_and_apply_frame_switch<F>(
    frame_tree: &mut FrameTree,
    root_frame_id: FrameId,
    body_frame_id: FrameId,
    integ_frame_id: &mut FrameId,
    trans: &mut TranslationalState,
    frame_switches: &mut [FrameSwitchConfig],
    gravity_controls: &mut GravityControls,
    resolve_target: F,
    num_sources: usize,
    body_idx: usize,
) -> Result<bool, FrameSwitchTargetMissing>
where
    F: Fn(&FrameUid) -> Option<FrameId>,
{
    if frame_switches.is_empty() {
        return Ok(false);
    }

    let mut switch_idx = None;
    for (idx, sw) in frame_switches.iter().enumerate() {
        if !sw.active {
            continue;
        }
        // JEOD_INV: RF.15 — only inertial-flavor classes may host
        // integration; a frame-switch target is the body's next
        // integration frame, so the same class eligibility applies at
        // the switch boundary (e.g. a Body-class identity as target is
        // a misconfiguration, even if it resolves in the tree).
        assert!(
            sw.target.class.may_be_root_or_integ(),
            "evaluate_and_apply_frame_switch: body {body_idx} switch target `{}` has \
             class {:?}, which may not host integration — switch targets must be a \
             gravity source's inertial frame (RootInertial / PlanetInertial / \
             BarycenterInertial). Fix the FrameSwitchConfig target identity.",
            sw.target,
            sw.target.class,
        );
        let target_fid = resolve_target(&sw.target).ok_or_else(|| FrameSwitchTargetMissing {
            body_idx,
            target: sw.target.clone(),
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

    let target = frame_switches[idx].target.clone();
    frame_switches[idx].active = false;

    // Resolve again — the closure was already proven to return Some for
    // this target above, so unwrap is safe (and a re-failure here would
    // be a caller-side data race we want to surface).
    let new_integ_fid = resolve_target(&target).expect(
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
    // (central body), all others become differential. Identity match
    // by FrameUid — the same convention in every host.
    for ctrl in &mut gravity_controls.controls {
        ctrl.differential = ctrl.source != target;
    }
    Ok(true)
}
