//! Per-scenario `SimulationBuilder` constructors.
//!
//! These submodules host the per-scenario physics setup. The user-facing
//! catalog is [`crate::recipes::Mission`] — mission code should construct
//! scenarios as `Mission::iss_leo().into_builder()` rather than calling
//! into these submodules directly.
//!
//! // reason: doctest references the downstream `astrodyn_runner` adapter crate, which astrodyn cannot depend on without a circular workspace dependency.
//! ```ignore
//! use astrodyn_runner::SimulationBuilderExt;
//! use astrodyn::recipes::Mission;
//! let sim = Mission::iss_leo().into_builder().build()?;
//! ```
//!
//! Scenarios that mirror JEOD verification simulations (Tier 3
//! reference cases) need high-fidelity gravity / ephemeris / rotation
//! kernels. Those binaries ship with the workspace (and inside the
//! published `.crate`) via `include_bytes!` in `astrodyn_gravity` and
//! `astrodyn_ephemeris`, exposed as the mission-facing recipes
//! [`crate::recipes::earth::ggm05c`], [`crate::recipes::moon::lp150q`],
//! [`crate::recipes::mars::mro110b2`], and
//! [`crate::recipes::ephemeris::de421`]. Scenarios in this module
//! compose those recipes with the mission-side building blocks; no JEOD
//! checkout is required.

pub mod apollo;
pub mod clementine_lunar;
pub mod earth_moon;
pub mod geostationary;
pub mod iss_leo;
pub mod mars_orbit;
pub mod mercury;
