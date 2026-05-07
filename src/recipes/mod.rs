//! Named building blocks and pre-composed mission scenarios.
//!
//! Phase 6 of #101 introduces this module as the user-facing entry
//! point for typed scenario construction. Recipes are organized in
//! two layers:
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
//!   [`into_builder`](Mission::into_builder).
//!
//! Mission code consumes recipes through whichever adapter it targets:
//!
//! // reason: doctest references downstream adapter crates (astrodyn_runner, astrodyn_bevy) that astrodyn cannot depend on without a circular workspace dependency.
//! ```ignore
//! // Standalone runner
//! use astrodyn_runner::SimulationBuilderExt;          // .build() terminal
//! use astrodyn::recipes::Mission;
//! let sim = Mission::iss_leo().into_builder().build()?;
//!
//! // Bevy adapter (Phase 9)
//! use astrodyn_bevy::prelude::*;                      // .spawn() terminal
//! commands.spawn_scenario(Mission::iss_leo().into_builder());
//! ```

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

pub use mission::Mission;
