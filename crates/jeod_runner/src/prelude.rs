//! Glob-import prelude for runner-side ergonomics.
//!
//! ```ignore
//! use jeod_runner::prelude::*;
//! ```
//!
//! Brings into scope the runner-side extension traits:
//! - [`SimulationBuilderExt`](crate::SimulationBuilderExt) — terminal
//!   `.build()` / `.build_unchecked()` on
//!   [`jeod_sim::SimulationBuilder`].
//! - [`VerificationCaseExt`](crate::VerificationCaseExt) — terminal
//!   `.run_and_assert()` on
//!   [`jeod_sim::recipes::verification::VerificationCase`].

pub use crate::SimulationBuilderExt;
pub use crate::VerificationCaseExt;
