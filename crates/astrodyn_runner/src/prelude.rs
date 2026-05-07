//! Glob-import prelude for runner-side ergonomics.
//!
//! ```
//! use astrodyn_runner::prelude::*;
//! ```
//!
//! Brings into scope the runner-side extension traits:
//! - [`crate::SimulationBuilderExt`] — terminal `.build()` on
//!   [`astrodyn::SimulationBuilder`].
//! - [`crate::VerificationCaseExt`] — terminal
//!   `.run_and_assert()` on
//!   [`astrodyn::recipes::verification::VerificationCase`]. Available
//!   only when the default `verification` feature is enabled.

pub use crate::SimulationBuilderExt;
#[cfg(feature = "verification")]
pub use crate::VerificationCaseExt;
