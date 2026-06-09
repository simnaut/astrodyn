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
//!   [`RefFrameRot`] — the per-node state structs (re-exported from
//!   [`ref_frame_state`]). Quaternions are JEOD-convention scalar-first,
//!   left-transformation, matching `astrodyn_math::JeodQuat`. Every node
//!   carries a required runtime identity
//!   (`astrodyn_quantities::frame_descriptor::FrameUid`) whose
//!   `FrameClass` is the runtime taxonomy (issue #664 removed the legacy
//!   3-variant `RefFrameKind`).
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
//!
//! ## Example
//!
//! Build a minimal frame tree with one root inertial frame and a
//! planet-fixed child — every node carries a minted identity:
//!
//! ```
//! use astrodyn_frames::{FrameTree, RefFrameState};
//! use astrodyn_quantities::frame::{Ecef, RootInertial};
//! use astrodyn_quantities::frame_descriptor::FrameUid;
//!
//! let mut tree = FrameTree::new();
//! let root = tree.add_root_typed::<RootInertial>("J2000".to_string());
//! let _ecef = tree.add_child_uid(
//!     root,
//!     FrameUid::of::<Ecef>(),
//!     "ECEF".to_string(),
//!     RefFrameState::default(),
//!     None,
//! );
//!
//! // Both frames live in the arena, addressable by identity.
//! assert_eq!(tree.len(), 2);
//! assert_eq!(tree.find(&FrameUid::of::<Ecef>()), Some(_ecef));
//! ```

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
pub mod topocentric_pose;

pub use frame_storage::{
    common_ancestor, compose_to_ancestor, compute_relative_state, FrameStorage,
};
pub use frame_tree::{FrameId, FrameNode, FrameTree, FrameTreeError};
pub use orchestration::{
    compute_relative_state_typed, frame_origin, frame_origin_typed, sync_pfix_rotation,
};
pub use ref_frame_state::*;
pub use topocentric_pose::topocentric_enu_state;
