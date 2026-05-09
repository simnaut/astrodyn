//! Bevy ↔ runner parity for SRP basic single-plate scenarios
//! (default and varied-Cr). Mirrors the hand-rolled cases that used
//! to live in `bevy_parity_srp.rs`; the recipes are at
//! `sim_srp::srp_basic_{default,varied_cr}`.

use astrodyn_verif_jeod::run_verification::sim_srp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn tier3_bevy_srp_basic_default() {
    sim_srp::srp_basic_default().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_srp_basic_varied_cr() {
    sim_srp::srp_basic_varied_cr().run_and_assert_parity::<astrodyn::Earth>();
}
