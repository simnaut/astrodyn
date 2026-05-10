//! Bevy ↔ runner parity for SIM_dyncomp RUN_5A (MET atmosphere + drag).
//! The recipe itself has `pre_step: None`, but the topic was tracked in
//! `KNOWN_PARITY_GAPS` until #395 because the parity infrastructure
//! couldn't drive a `pre_step` recipe at all; with the bridge in place
//! every dyncomp recipe is wrappable.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_met_dyncomp_run5a_met() {
    sim_dyncomp::run5a_met().run_and_assert_parity::<astrodyn::Earth>();
}
