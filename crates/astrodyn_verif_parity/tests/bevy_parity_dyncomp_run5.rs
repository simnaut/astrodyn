//! Bevy ↔ runner parity for SIM_dyncomp RUN_5b/c (mean / max
//! atmosphere-density variants). The MET-atmosphere variant
//! (`run5a_met`) lives in `bevy_parity_met.rs` to match its
//! tier3 sibling `tier3_sim_met.rs`. Wrappers land as part of #389.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_dyncomp_run5b_atmosphere_mean() {
    sim_dyncomp::run5b_atmosphere_mean().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_run5c_atmosphere_max() {
    sim_dyncomp::run5c_atmosphere_max().run_and_assert_parity::<astrodyn::Earth>();
}
