//! Source-state mutators that operate on a [`FrameTree`] +
//! [`SourceFrameIds`] slice.
//!
//! Lifted from `jeod_runner::Simulation::set_source_position` and
//! `set_source_state` (issue #71). These free functions are the
//! frame-tree-touching primitives both consumers (`jeod_runner` and the
//! Bevy adapter) call when a mission needs to retarget a gravity source
//! at runtime — e.g. switching from analytic ephemerides to spacecraft-
//! supplied state, or replaying recorded planet positions.
//!
//! `set_source_ephemeris` is intentionally not lifted: it records a
//! `(target, observer)` mapping on a runner-private vector, with no
//! frame-tree mutation. The Bevy adapter expresses the same intent via
//! its `EphemerisBodyC` component.

use glam::{DMat3, DVec3};
use jeod_frames::{FrameId, FrameTree};

use crate::source_frames::SourceFrameIds;

/// Get the inertial-frame ID for a gravity source. Panics if `source_idx`
/// is out of range.
pub fn source_frame_id(source_frame_ids: &[SourceFrameIds], source_idx: usize) -> FrameId {
    source_frame_ids
        .get(source_idx)
        .unwrap_or_else(|| {
            panic!(
                "source_frame_id: source index {source_idx} is out of range; \
                 {} source frame(s) configured",
                source_frame_ids.len()
            )
        })
        .inertial
}

/// Get the current position of a gravity source relative to the root
/// inertial frame. Returns `DVec3::ZERO` for the root-mapped central source.
pub fn source_position(
    frame_tree: &FrameTree,
    source_frame_ids: &[SourceFrameIds],
    root_frame_id: FrameId,
    source_idx: usize,
) -> DVec3 {
    let fid = source_frame_id(source_frame_ids, source_idx);
    if fid == root_frame_id {
        DVec3::ZERO
    } else {
        frame_tree.get(fid).state.trans.position
    }
}

/// Get the planet-fixed rotation matrix for a gravity source. Returns `None`
/// if the source has no rotation model (no pfix frame).
pub fn source_pfix_rotation(
    frame_tree: &FrameTree,
    source_frame_ids: &[SourceFrameIds],
    source_idx: usize,
) -> Option<DMat3> {
    source_frame_ids
        .get(source_idx)
        .unwrap_or_else(|| {
            panic!(
                "source_pfix_rotation: source index {source_idx} out of range; \
                 {} source(s) configured",
                source_frame_ids.len()
            )
        })
        .pfix
        .map(|pfix_id| frame_tree.get(pfix_id).state.rot.t_parent_this)
}

/// Set the position of a gravity source relative to the root inertial frame.
/// Panics if `source_idx` is out of range or if the source maps to the root
/// (central body).
pub fn set_source_position(
    frame_tree: &mut FrameTree,
    source_frame_ids: &[SourceFrameIds],
    root_frame_id: FrameId,
    source_idx: usize,
    position: DVec3,
) {
    assert!(
        source_idx < source_frame_ids.len(),
        "set_source_position: source index {source_idx} out of range; \
         {} source(s) configured",
        source_frame_ids.len()
    );
    let fid = source_frame_ids[source_idx].inertial;
    assert_ne!(
        fid, root_frame_id,
        "set_source_position: cannot set position of the root (central body) source"
    );
    frame_tree.get_mut(fid).state.trans.position = position;
}

/// Set position and velocity of a gravity source relative to the root
/// inertial frame on the frame tree. Callers that maintain parallel
/// structures (e.g. relativistic-correction state, or ECS components
/// like `SourceInertialVelocityC` in the Bevy adapter) should sync
/// those structures themselves with the same `velocity` argument they
/// passed in — `jeod_runner::Simulation::set_source_state` does this
/// for `gravity_data[idx].velocity`, and the Bevy adapter's
/// `SourceMutator` does it for `SourceInertialVelocityC`.
///
/// Prefer this over [`set_source_position`] when velocity is also
/// available, to keep position and velocity consistent.
pub fn set_source_state(
    frame_tree: &mut FrameTree,
    source_frame_ids: &[SourceFrameIds],
    root_frame_id: FrameId,
    source_idx: usize,
    position: DVec3,
    velocity: DVec3,
) {
    assert!(
        source_idx < source_frame_ids.len(),
        "set_source_state: source index {source_idx} out of range; \
         {} source(s) configured",
        source_frame_ids.len()
    );
    let fid = source_frame_ids[source_idx].inertial;
    assert_ne!(
        fid, root_frame_id,
        "set_source_state: cannot set state of the root (central body) source"
    );
    let node = frame_tree.get_mut(fid);
    node.state.trans.position = position;
    node.state.trans.velocity = velocity;
}
