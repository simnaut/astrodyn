//! Tier 3: SIM_3_ORBIT_1st_ORDER cross-validation.
//!
//! Migrated from a 287-line bespoke per-step ephemeris-update loop to
//! this one-liner using the `pre_step` hook (#162). The recipe lives in
//! `astrodyn_verif_jeod::run_verification::sim_srp::srp_1st_order_trajectory`;
//! the per-step DE421 update is its `pre_step` factory.

use astrodyn_verif_jeod::run_verification::sim_srp;
use astrodyn_verif_jeod::VerificationCaseExt;

#[test]
fn tier3_srp_1st_order_trajectory() {
    sim_srp::srp_1st_order_trajectory().run_and_assert();
}
