//! Mapping from a gravity source to its frame-tree nodes.
//!
//! Lifted out of `jeod_runner::simulation::types` (issue #71) so both
//! `jeod_runner` and ECS adapters (the `bevy_jeod` root crate) can build
//! source frames against a shared structure. Lives at the orchestration
//! layer (`jeod_sim`) — `jeod_runner` is a peer consumer of `jeod_sim`,
//! not a layer above it (CLAUDE.md three-layer rule).

use jeod_frames::FrameId;

/// Maps a gravity source to its frame tree nodes.
///
/// Every gravity source has an inertial frame; sources with a planet rotation
/// model also have a planet-fixed (pfix) child frame. The orchestration helpers
/// in [`crate::frame_orchestration`] and [`crate::source_state`] take a
/// `&[SourceFrameIds]` slice keyed by the same source-index ordering the
/// caller uses for `GravityControls<usize>`.
#[derive(Debug, Clone, Copy)]
pub struct SourceFrameIds {
    /// Inertial frame for this source (e.g., "Earth.inertial").
    pub inertial: FrameId,
    /// Planet-fixed frame (e.g., "Earth.pfix"), if the source has a rotation
    /// model. `None` for sources with [`crate::RotationModel::None`].
    pub pfix: Option<FrameId>,
}
