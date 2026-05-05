//! Standalone simulation runner for JEOD physics.
//!
//! Provides a [`Simulation`] struct for batch propagation, scripting, and
//! Tier 3 cross-validation tests. Owns all state and runs the `jeod_sim`
//! pipeline internally.
//!
//! ECS adapters should **not** depend on this crate — use the per-body
//! functions from `jeod_sim` directly instead.
//!
//! # Example
//! ```
//! use jeod_runner::SimulationBuilderExt;
//! use jeod_sim::recipes::Mission;
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
#[cfg(feature = "verification")]
pub mod run_verification;
mod simulation;

pub use branded::{BodyIdx, BrandedSimulation, SourceIdx};
pub use error::StepError;

// Re-export jeod_sim so downstream tests can access types through either path.
pub use jeod_sim;
pub use jeod_sim::RotationModel;

// Re-export the runner-side terminal-method extension trait from `builder`.
pub use builder::SimulationBuilderExt;

// Re-export the Phase-7 Tier 3 verification-case extension trait.
#[cfg(feature = "verification")]
pub use run_verification::VerificationCaseExt;

// (Phase-10 cleanup, issue #253) The 12 types relocated from `jeod_runner`
// to `jeod_sim` in Phase 6 of #101 are no longer re-exported here. Consumers
// import them directly from `jeod_sim`:
//   `use jeod_sim::{VehicleConfig, GravitySourceEntry, SrpModel, ...};`
//
// `jeod_runner::jeod_sim` is still re-exported above so any straggling
// `jeod_runner::jeod_sim::VehicleConfig`-style paths continue to compile.

// Re-export FrameId for downstream API.
pub use jeod_frames::FrameId;

// `Simulation` and its supporting public types live in the `simulation`
// submodule (issue #253). Re-exported here for API stability.
pub use simulation::{
    ContactPairConfig, DetachedSubtreeState, FrameAttachState, GroundContactPairConfig,
    GroundFacet, Simulation, SphericalTerrain, Terrain, VehicleOutput,
};
