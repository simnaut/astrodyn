//! Glob-import prelude for runner-side ergonomics.
//!
//! ```
//! use jeod_runner::prelude::*;
//! ```
//!
//! Brings into scope the runner-side extension traits:
//! - [`crate::SimulationBuilderExt`] — terminal `.build()` on
//!   [`jeod_sim::SimulationBuilder`].
//! - [`crate::VerificationCaseExt`] — terminal
//!   `.run_and_assert()` on
//!   [`jeod_sim::recipes::verification::VerificationCase`]. Available
//!   only when the default `verification` feature is enabled.

pub use crate::SimulationBuilderExt;
#[cfg(feature = "verification")]
pub use crate::VerificationCaseExt;
