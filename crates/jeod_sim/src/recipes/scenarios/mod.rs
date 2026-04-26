//! Pre-composed [`SimulationBuilder`](crate::SimulationBuilder)s for
//! common reference scenarios.
//!
//! Each scenario function returns a fully-configured builder. Mission
//! code adds an adapter-specific terminal step:
//!
//! ```ignore
//! use jeod_runner::SimulationBuilderExt;
//! use jeod_sim::recipes::scenarios;
//! let sim = scenarios::iss_leo().build()?;
//! ```
//!
//! Scenarios that mirror JEOD verification simulations (Tier 3
//! reference cases) need high-fidelity gravity / ephemeris / rotation
//! kernels. Phase 6 of #101 routed those data dependencies into
//! [`verification::reference_data`](super::verification::reference_data)
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

pub use apollo::apollo_translunar;
pub use clementine_lunar::clementine_lunar;
pub use earth_moon::earth_moon_translunar;
pub use geostationary::geo;
pub use iss_leo::{iss_leo, iss_leo_drag};
pub use mars_orbit::mars_orbit;
pub use mercury::mercury_relativistic;
