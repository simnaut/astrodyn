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
//! - **Missions** ([`Mission`]): first-class catalog of pre-composed
//!   reference scenarios. `Mission::iss_leo()`,
//!   `Mission::clementine_lunar()`, … each return a typed `Mission`
//!   that materializes into a
//!   [`SimulationBuilder`](crate::SimulationBuilder) via
//!   [`into_builder`](Mission::into_builder). Each scenario is the
//!   smallest composition of building blocks plus vehicle config that
//!   matches a JEOD verification simulation.
//! - **Verification** ([`verification`]):
//!   [`VerificationCase`](verification::VerificationCase) bundles a
//!   scenario with reference-CSV data and tolerances. Phase 7/8 will
//!   populate the catalog of cases; Phase 6 ships the scaffold.
//!
//! Mission code consumes recipes through whichever adapter it targets:
//!
//! // reason: doctest references downstream adapter crates (jeod_runner,
//! // bevy_jeod) that jeod_sim cannot depend on without a circular
//! // workspace dependency.
//! ```ignore
//! // Standalone runner
//! use jeod_runner::SimulationBuilderExt;          // .build() terminal
//! use jeod_sim::recipes::Mission;
//! let sim = Mission::iss_leo().into_builder().build()?;
//!
//! // Bevy adapter (Phase 9)
//! use bevy_jeod::prelude::*;                      // .spawn() terminal
//! commands.spawn_scenario(Mission::iss_leo().into_builder());
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
pub mod mission;
pub mod moon;
pub mod orbital_elements;
pub mod scenarios;
pub mod sun;
pub mod vehicle;
pub mod verification;

pub use mission::Mission;
