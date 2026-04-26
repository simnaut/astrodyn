//! Recipe catalogue re-export: `use bevy_jeod::recipes::*;`.
//!
//! Re-exports [`jeod_sim::recipes`] under the `bevy_jeod` crate root so
//! mission crates that depend only on `bevy_jeod` can compose scenarios
//! without an explicit `jeod_sim` dependency line in `Cargo.toml`.
//!
//! Submodules:
//!
//! - `constants` — typed physical constants (`mu_ggm05c()`, `r_eq_earth()`, …)
//! - `earth` / `moon` / `sun` / `mars` / `mercury` — gravity-source presets
//!   (`earth::point_mass()`, `earth::ggm05c_8x8()`, …)
//! - `orbital_elements` — reference orbits (`orbital_elements::iss()`,
//!   `orbital_elements::geo_inclined()`, …)
//! - `vehicle` — vehicle mass-property presets (`vehicle::iss_mass()`, …)
//! - `epoch` — time epochs (`epoch::j2000()`, …)
//! - `atmosphere` — atmospheric-model presets
//! - `scenarios` — fully-composed `SimulationBuilder` shortcuts
//! - `verification` — Tier 3 `VerificationCase` shape (cross-validation tests)
//! - `helpers` — propagation utilities for archetype-B Tier 3 tests
//!
//! See the corresponding modules in `jeod_sim::recipes` for full inventories.

pub use jeod_sim::recipes::*;
