//! Tier 3: SIM_tide_verif RUN_01 — solid body tides cross-validation.
//!
//! Migrated from a 290-line bespoke per-step ephemeris-update loop to
//! this one-liner using the `pre_step` hook +
//! [`ExtrasComparator::TideDc20`] (#162). The recipe lives in
//! `astrodyn_verif_jeod::run_verification::sim_tide_verif::run01`; the
//! per-step DE421 update + tidal-body refresh is its `pre_step`
//! factory.

use astrodyn_verif_jeod::run_verification::sim_tide_verif;
use astrodyn_verif_jeod::VerificationCaseExt;

#[test]
fn tier3_simulation_tide_run01() {
    sim_tide_verif::run01().run_and_assert();
}

/// RUN_02: same scenario as RUN_01 but Earth gravity is split into a
/// spherical (point-mass) control + a perturbing-only spherical-harmonics
/// control (`perturbing_only`). The split reproduces the full field;
/// validated against JEOD's distinct `tide_run02` log.
#[test]
fn tier3_simulation_tide_run02() {
    sim_tide_verif::run02().run_and_assert();
}
