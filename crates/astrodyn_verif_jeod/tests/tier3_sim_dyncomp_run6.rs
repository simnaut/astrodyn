#![cfg(feature = "verification")]

//! Tier 3: SIM_dyncomp RUN_6A/6B — Drag (constant density and MET atmosphere)
//!
//! All simulation parameters (epoch, step size, mu, mass) are loaded from JEOD
//! source files rather than hardcoded, per issue #44.
//!
//! Phase 7 of #101 collapsed the per-test setup into the
//! [`run_verification::sim_dyncomp`](astrodyn_verif_jeod::run_verification::sim_dyncomp)
//! recipe family.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_jeod::VerificationCaseExt;

#[test]
fn tier3_simulation_run6a_const_density_drag() {
    sim_dyncomp::run6a_const_density_drag().run_and_assert();
}

#[test]
fn tier3_simulation_run6b_drag() {
    sim_dyncomp::run6b_drag().run_and_assert();
}
