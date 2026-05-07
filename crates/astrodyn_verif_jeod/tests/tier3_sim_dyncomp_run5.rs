//! Tier 3: SIM_dyncomp RUN_5B/5C — Elliptical orbit, 6-DOF (ISS mass)
//!
//! JEOD labels these "atmosphere comparison" runs, but drag is disabled
//! so the atmosphere model has no effect on the trajectory. These are
//! effectively point-mass 6-DOF tests with elliptical orbit ICs and ISS
//! mass/inertia, with the gravity gradient flag enabled.
//!
//! Phase 7 of #101 collapsed the per-test setup into the
//! [`run_verification::sim_dyncomp`](astrodyn_verif_jeod::run_verification::sim_dyncomp)
//! recipe family.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_jeod::VerificationCaseExt;

#[test]
fn tier3_simulation_run5b_atmosphere_mean() {
    sim_dyncomp::run5b_atmosphere_mean().run_and_assert();
}

#[test]
fn tier3_simulation_run5c_atmosphere_max() {
    sim_dyncomp::run5c_atmosphere_max().run_and_assert();
}
