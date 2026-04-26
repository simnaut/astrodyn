//! Tier 3: SIM_SolarBeta cross-validation (derived_state/verif/SIM_SolarBeta)
//!
//! Validates solar beta wiring via the RUN_2 point-mass trajectory (8h,
//! validated to point-mass tolerance) with DE421 ephemeris for Sun
//! direction. Sun has mu=0 because the reference comes from RUN_2
//! (Earth-only gravity); the Sun source is used solely for solar beta
//! direction, not gravitational perturbation. For 3rd-body gravity
//! validation, see `tier3_sim_dyncomp_run4`.
//!
//! Migrated from a 200-line bespoke per-step loop to a recipe one-liner
//! using the `pre_step` hook (#156, #162). The recipe constructor lives
//! at `jeod_runner::run_verification::sim_solar_beta::solar_beta_run2`;
//! the per-step DE421 update is its `pre_step` factory.

use jeod_runner::run_verification::sim_solar_beta;
use jeod_runner::VerificationCaseExt;

#[test]
fn tier3_simulation_solar_beta() {
    sim_solar_beta::solar_beta_run2().run_and_assert();
}
