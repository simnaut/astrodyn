//! Standalone simulation runner for JEOD physics.
//!
//! Provides a [`Simulation`] struct for batch propagation, scripting, and
//! Tier 3 cross-validation tests. Owns all state and runs the `astrodyn`
//! pipeline internally.
//!
//! ECS adapters should **not** depend on this crate — use the per-body
//! functions from `astrodyn` directly instead.
//!
//! # Example
//! ```
//! use astrodyn_runner::SimulationBuilderExt;
//! use astrodyn::recipes::Mission;
//!
//! let mut sim = Mission::iss_leo().into_builder().build().unwrap();
//! sim.step_n(10);
//! let output = sim.body(0);
//! assert!(output.trans.position.length() > 6_000_000.0);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod branded;
pub mod builder;
pub mod error;
pub mod prelude;
mod simulation;

pub use branded::{BodyIdx, BrandedSimulation, SourceIdx};
pub use error::StepError;

// Re-export astrodyn so downstream tests can access types through either path.
pub use astrodyn;
pub use astrodyn::RotationModel;

// Re-export the runner-side terminal-method extension trait from `builder`.
pub use builder::SimulationBuilderExt;

// (Phase-10 cleanup, issue #253) The 12 types relocated from `astrodyn_runner`
// to `astrodyn` in Phase 6 of #101 are no longer re-exported here. Consumers
// import them directly from `astrodyn`:
//   `use astrodyn::{VehicleConfig, GravitySourceEntry, SrpModel, ...};`
//
// `astrodyn_runner::astrodyn` is still re-exported above so any straggling
// `astrodyn_runner::astrodyn::VehicleConfig`-style paths continue to compile.

// Re-export FrameId for downstream API.
pub use astrodyn_frames::FrameId;

// `Simulation` and its supporting public types live in the `simulation`
// submodule (issue #253). Re-exported here for API stability.
pub use simulation::{
    ContactPairConfig, DetachedSubtreeState, FrameAttachState, GroundContactPairConfig,
    GroundFacet, Simulation, SphericalTerrain, Terrain, VehicleOutput,
};
