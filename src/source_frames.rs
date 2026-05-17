//! Mapping from a gravity source to its frame-tree nodes.
//!
//! Lifted out of `astrodyn_runner::simulation::types` (issue #71) so both
//! `astrodyn_runner` and ECS adapters (the `astrodyn_bevy` root crate) can build
//! source frames against a shared structure. Lives at the orchestration
//! layer (`astrodyn`) — `astrodyn_runner` is a peer consumer of `astrodyn`,
//! not a layer above it (CLAUDE.md three-layer rule).

use astrodyn_frames::FrameId;

/// Maps a gravity source to its frame tree nodes.
///
/// Every gravity source has an inertial frame; sources with a planet rotation
/// model also have a planet-fixed (pfix) child frame. The orchestration helpers
/// in [`crate::frame_orchestration`] and [`crate::source_state`] take a
/// `&[SourceFrameIds]` slice keyed by the same source-index ordering the
/// caller uses for `GravityControls<usize>`.
///
/// Both `astrodyn_runner` and `astrodyn_bevy` always allocate a *distinct*
/// inertial frame node for every gravity source (matching JEOD's
/// `BasePlanet { inertial, pfix }` canonical structure — see
/// `models/environment/planet/include/base_planet.hh:109,121`). For sources
/// at the heliocentric origin (the central body), the inertial frame node
/// is a child of root with identity rotation and zero position — the
/// structural separation is preserved even when the inertial frame is
/// numerically equivalent to root. `JEOD_INV: RF.13` (PR #566) guarantees
/// that composition walks through such identity intermediate frames
/// preserve f64 bits, so the structural alignment between runner and Bevy
/// is bit-safe.
#[derive(Debug, Clone, Copy)]
pub struct SourceFrameIds {
    /// Inertial frame node for this source (e.g., `"Earth.inertial"`),
    /// always a distinct child of root.
    pub inertial: FrameId,
    /// Planet-fixed frame (e.g., "Earth.pfix"), if the source has a rotation
    /// model. `None` for sources with [`crate::RotationModel::None`].
    pub pfix: Option<FrameId>,
    /// Whether this source is the simulation's central body. Replaces the
    /// pre-#567 `inertial == root_frame_id` alias-based check — the central
    /// source's inertial frame is now structurally distinct from root, so
    /// callers needing "is this source the central body?" must consult
    /// this flag instead of comparing frame ids. Only one source per
    /// `astrodyn_runner::Simulation` may carry `central = true`. The Bevy
    /// adapter has no equivalent concept (all sources are non-central
    /// children of root in the ECS hierarchy) and leaves this `false`.
    pub central: bool,
}
