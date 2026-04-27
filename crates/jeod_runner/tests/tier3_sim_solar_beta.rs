//! Tier 3: ISS LEO trajectory cross-validation against SIM_dyncomp RUN_2,
//! with the Sun source registered + DE421-driven each step so
//! `body.solar_beta` is computed throughout. Sun has mu=0 — the Sun
//! position update doesn't perturb the trajectory; the assertion only
//! checks position vs JEOD's RUN_2 reference at point-mass tolerance
//! (8 hours).
//!
//! `body.solar_beta` itself is *not* externally validated against
//! `SIM_SolarBeta`'s logged beta column in this case. That validation
//! requires `ExtrasComparator::SolarBeta` framework support and is
//! tracked as #169 (a focused follow-up that benefits this test plus
//! the 4 `tier3_sim_solar_beta_edge` cases). For 3rd-body gravity
//! validation, see `tier3_sim_dyncomp_run4`.
//!
//! Migrated from a 200-line bespoke per-step loop to this one-liner
//! using the `pre_step` hook (#156, #162). The recipe constructor lives
//! at `jeod_runner::run_verification::sim_solar_beta::solar_beta_run2`;
//! the per-step DE421 update is its `pre_step` factory.

use jeod_runner::run_verification::sim_solar_beta;
use jeod_runner::VerificationCaseExt;

#[test]
fn tier3_simulation_solar_beta() {
    sim_solar_beta::solar_beta_run2().run_and_assert();
}
