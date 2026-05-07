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

/// Tier 3 cross-validation scaffolding (workspace-internal).
///
/// Contains [`VerificationCase`](verification::VerificationCase) and
/// the SH-gravity recipes that back it
/// (`verification::reference_data::*`). Consumed by `astrodyn_runner`'s
/// `run_verification` rigs and the workspace-internal examples; these
/// types reach into committed test data under `test_data/` that ships
/// only with this repository, so downstream mission code should not
/// depend on this surface.
///
/// Hidden from rendered rustdoc so the published page surfaces only
/// the mission-facing recipes.
#[doc(hidden)] // allowed: workspace-internal Tier 3 scaffolding; see module docs and #249
pub mod verification;

pub use mission::Mission;
