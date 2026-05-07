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
//! kernels. Phase 6 of #101 routed those data dependencies into
//! `astrodyn_verif_jeod::verification::reference_data`
//! so mission scenarios in this module function independently of any
//! JEOD checkout. Tier 3 scenarios (and the examples that mirror
//! them) compose `verification::reference_data::*` with the
//! mission-side building blocks.

pub mod apollo;
pub mod clementine_lunar;
pub mod earth_moon;
pub mod geostationary;
pub mod iss_leo;
pub mod mars_orbit;
pub mod mercury;
