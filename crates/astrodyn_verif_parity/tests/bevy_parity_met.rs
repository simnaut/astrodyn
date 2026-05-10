//! Bevy ↔ runner parity for SIM_dyncomp RUN_5a (MET atmosphere with
//! per-step Sun position injection), via the `VerificationCaseParityExt`
//! trait.
//!
//! Unblocked by issue #395's `BevySimContext`: the recipe's `pre_step`
//! drives `set_source_position` for the Sun source on both runtimes,
//! which the MET model reads to compute solar-flux-driven density.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_dyncomp_run5a_met() {
    sim_dyncomp::run5a_met().run_and_assert_parity::<astrodyn::Earth>();
}
