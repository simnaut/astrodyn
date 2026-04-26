//! Named building blocks, scenarios, and verification cases.
//!
//! Phase 6 of #101 introduces this module as the user-facing entry
//! point for typed scenario construction. Recipes are organized in
//! three layers:
//!
//! - **Building blocks** ([`earth`], [`moon`], [`sun`], [`mars`],
//!   [`atmosphere`], [`epoch`], [`vehicle`], [`orbital_elements`],
//!   [`constants`]): named typed primitives that mission code combines
//!   freely.
//! - **Scenarios** ([`scenarios`]): pre-composed
//!   [`SimulationBuilder`](crate::SimulationBuilder)s for common
//!   reference setups (`scenarios::iss_leo()`,
//!   `scenarios::clementine_lunar()`, …). Each scenario is the
//!   smallest composition of building blocks plus vehicle config that
//!   matches a JEOD verification simulation.
//! - **Verification** ([`verification`]):
//!   [`VerificationCase`](verification::VerificationCase) bundles a
//!   scenario with reference-CSV data and tolerances. Phase 7/8 will
//!   populate the catalog of cases; Phase 6 ships the scaffold.
//!
//! Mission code consumes recipes through whichever adapter it targets:
//!
//! ```ignore
//! // Standalone runner
//! use jeod_runner::SimulationBuilderExt;          // .build() terminal
//! use jeod_sim::recipes::scenarios;
//! let sim = scenarios::iss_leo().build()?;
//!
//! // Bevy adapter (Phase 9)
//! use bevy_jeod::prelude::*;                      // .spawn() terminal
//! commands.spawn_scenario(scenarios::iss_leo());
//! ```
//!
//! Recipes that depend on JEOD source data (`$JEOD_HOME/...`) panic
//! at construction with the exact `cargo run` / Docker command if the
//! data is missing — see `feedback_no_graceful_skip.md`.

pub mod atmosphere;
pub mod constants;
pub mod earth;
pub mod epoch;
pub mod helpers;
pub mod mars;
pub mod moon;
pub mod orbital_elements;
pub mod scenarios;
pub mod sun;
pub mod vehicle;
pub mod verification;
