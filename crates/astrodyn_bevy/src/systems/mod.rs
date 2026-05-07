//! Bevy `System`s that delegate per-body work to `astrodyn` per-body
//! orchestration functions. Each system queries the relevant components,
//! calls into `astrodyn`, and writes the result back. No physics
//! algorithms live here.
//!
//! Frame-tree state lives entirely on Bevy entities: every source /
//! body has a [`crate::components::FrameEntityC`] handle pointing at
//! its frame entity, which carries
//! [`crate::components::FrameTransC`] / [`crate::components::FrameRotC`] /
//! [`crate::components::FrameAngVelC`]. Cross-frame queries flow
//! through [`crate::frame_param::RelativeFrameState`] and
//! [`crate::frame_param::FrameOrigin`].
//!
//! # Submodule layout
//!
//! Per-stage modules host the [`AstrodynSet`](crate::AstrodynSet) pipeline
//! systems; the `glue` submodule hosts ECS plumbing systems
//! (frame-tree registration, joint kinematics, mass-point sync,
//! despawn observers, mass recompute) that wire alongside the
//! pipeline rather than living on a single set.
//!
//! - `time_update`
//! - `ephemeris_update`
//! - `environment`
//! - `interaction`
//! - `force_collection`
//! - `integration`
//! - `derived_state`
//! - `glue`
//!
//! Submodules are private to keep the per-stage names from colliding
//! with the same names in [`crate::components`] under
//! `pub use components::*` / `pub use systems::*` glob exports.
//! All public items from each submodule are re-exported here so
//! external `astrodyn_bevy::systems::<name>` paths continue to resolve.

mod derived_state;
mod environment;
mod ephemeris_update;
mod force_collection;
mod glue;
mod integration;
mod interaction;
mod time_update;
mod util;

pub use derived_state::*;
pub use environment::*;
pub use ephemeris_update::*;
pub use force_collection::*;
pub use glue::*;
pub use integration::*;
pub use interaction::*;
pub use time_update::*;
