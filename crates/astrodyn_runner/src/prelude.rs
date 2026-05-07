//! Glob-import prelude for runner-side ergonomics.
//!
//! ```
//! use astrodyn_runner::prelude::*;
//! ```
//!
//! Brings into scope the runner-side extension traits:
//! - [`crate::SimulationBuilderExt`] — terminal `.build()` on
//!   [`astrodyn::SimulationBuilder`].

pub use crate::SimulationBuilderExt;
