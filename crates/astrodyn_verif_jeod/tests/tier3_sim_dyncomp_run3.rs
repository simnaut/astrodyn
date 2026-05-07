//! Tier 3: SIM_dyncomp RUN_3A/3B — Spherical harmonics gravity (4x4 / 8x8 + RNP)
//!
//! All simulation parameters (epoch, step size, gravity degree/order) are loaded
//! from the JEOD source files rather than hardcoded, per issue #44.
//!
//! Phase 7 of #101 collapsed the per-test setup into the
//! [`run_verification::sim_dyncomp`](astrodyn_verif_jeod::run_verification::sim_dyncomp)
//! recipe family.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_jeod::VerificationCaseExt;

#[test]
fn tier3_simulation_run3a_sh4x4() {
    sim_dyncomp::run3a_sh4x4().run_and_assert();
}

#[test]
fn tier3_simulation_run3b_sh8x8() {
    sim_dyncomp::run3b_sh8x8().run_and_assert();
}
