//! Bevy ↔ runner parity for SIM_SolarBeta edge cases (equatorial
//! orbit with point-mass Earth and Earth-obliquity orbit with 8×8
//! spherical-harmonics Earth).
//!
//! Mirrors `tier3_sim_solar_beta_edge.rs` 1-to-1 so the two test files
//! stay visibly in sync — a new edge-case recipe added there has an
//! obvious parity sibling slot here. The recipes themselves
//! (`sim_solar_beta::solar_beta_equ`, `solar_beta_obliquity`) are
//! shared with the base `bevy_parity_solar_beta` wrapper; running them
//! again from this file is intentional: the topic-level coverage check
//! in `parity_coverage` is per-file, so `solar_beta_edge` needs its
//! own wrapper file to satisfy the `tier3_topics ⊂ bevy_parity_topics`
//! invariant.

use astrodyn_verif_jeod::run_verification::sim_solar_beta;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_solar_beta_edge_equ() {
    sim_solar_beta::solar_beta_equ().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_solar_beta_edge_obliquity() {
    sim_solar_beta::solar_beta_obliquity().run_and_assert_parity::<astrodyn::Earth>();
}
