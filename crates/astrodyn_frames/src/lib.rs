//! Reference frames, frame tree, and Earth/Mars/Moon rotation models.
//!
//! Port of JEOD's `models/utils/ref_frames/` (the `RefFrame` tree that is the
//! backbone of every coordinate system in JEOD) and `models/environment/RNP/`
//! (precession, nutation, polar motion). The crate is pure Rust with zero
//! Bevy dependency.
//!
//! ## Frame tree
//!
//! - [`FrameTree`], [`FrameNode`], [`FrameId`] — an arena-based hierarchy
//!   that mirrors JEOD's `RefFrame` parent/child links. Each node stores
//!   state **relative to its parent** (translation, velocity, orientation,
//!   angular velocity); relative states between arbitrary frames are
//!   computed by walking to the common ancestor and composing/negating
//!   states. The arena is a flat `Vec` with parallel `Option<FrameId>`
//!   parent pointers, which keeps lookups cache-friendly and avoids the
//!   pointer chasing of JEOD's intrusive linked lists.
//! - [`RefFrameState`], [`RefFrameStateTyped`], [`RefFrameTrans`],
//!   [`RefFrameRot`], [`RefFrameKind`] — the per-node state structs
//!   (re-exported from [`ref_frame_state`]). Quaternions are JEOD-convention
//!   scalar-first, left-transformation, matching `astrodyn_math::JeodQuat`.
//!
//! ## Earth / Mars / Moon rotation
//!
//! - [`rotation_j2000`], [`precession_j2000`], [`nutation_j2000`],
//!   [`data_nutation_j2000`] — Earth precession + IAU-1980 nutation series
//!   (the `data_nutation_j2000` table is the 106-term series ported from
//!   JEOD's `RNP_J2000` data files).
//! - [`rotation_mars`], [`rotation_moon`] — body-fixed rotation models for
//!   the Mars and Moon target bodies used by JEOD verification sims.
//!
//! JEOD source: `models/utils/ref_frames/` and `models/environment/RNP/`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod data_nutation_j2000;
pub mod frame_storage;
pub mod frame_tree;
pub mod nutation_j2000;
pub mod orchestration;
pub mod precession_j2000;
pub mod ref_frame_state;
pub mod rotation_j2000;
pub mod rotation_mars;
pub mod rotation_moon;

pub use frame_storage::{
    common_ancestor, compose_to_ancestor, compute_relative_state, FrameStorage,
};
pub use frame_tree::{FrameId, FrameNode, FrameTree};
pub use orchestration::{
    compute_relative_state_typed, frame_origin, frame_origin_typed, sync_pfix_rotation,
};
pub use ref_frame_state::*;
