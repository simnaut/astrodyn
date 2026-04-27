//! Tier 3: SIM_tide_verif RUN_01 — solid body tides cross-validation.
//!
//! Migrated from a 290-line bespoke per-step ephemeris-update loop to
//! this one-liner using the `pre_step` hook +
//! [`ExtrasComparator::TideDc20`] (#162). The recipe lives in
//! `jeod_runner::run_verification::sim_tide_verif::run01`; the
//! per-step DE421 update + tidal-body refresh is its `pre_step`
//! factory.

use jeod_runner::run_verification::sim_tide_verif;
use jeod_runner::VerificationCaseExt;

#[test]
fn tier3_simulation_tide_run01() {
    sim_tide_verif::run01().run_and_assert();
}
