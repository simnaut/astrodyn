#![cfg(feature = "verification")]

//! Tier 3: SIM_SolarBeta edge-case cross-validation via Simulation pipeline.
//!
//! - `RUN_incl_0` — equatorial orbit (i=0), point-mass gravity. Beta tracks
//!   Sun declination (~23.4°).
//! - `RUN_incl_23_4` — Earth-obliquity inclination (23.44°), 8×8 spherical
//!   harmonics gravity. Captures J2 RAAN drift that changes orbital-plane
//!   orientation vs Sun, directly affecting solar beta.
//!
//! Migrated from bespoke per-step propagation loops (~280 LoC of shared
//! helper) to recipe one-liners using `ExtrasComparator::SolarBeta` (#169).
//! Each test compares `body.solar_beta` against JEOD's logged
//! `SIM_SolarBeta` reference column at every CSV record.

use jeod_runner::run_verification::sim_solar_beta;
use jeod_runner::VerificationCaseExt;

/// Equatorial orbit; no J2 RAAN drift, so point-mass is sufficient.
#[test]
fn tier3_simulation_solar_beta_equ() {
    sim_solar_beta::solar_beta_equ().run_and_assert();
}

/// Inclined orbit; 8×8 SH gravity captures J2 RAAN drift that changes
/// orbital-plane orientation vs Sun, directly affecting solar beta.
#[test]
fn tier3_simulation_solar_beta_obliquity() {
    sim_solar_beta::solar_beta_obliquity().run_and_assert();
}
