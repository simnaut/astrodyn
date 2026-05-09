//! Bevy ↔ runner parity for the SIM_2A_SHADOW_CALC variants —
//! single-plate 6-DOF, ε=0.5 (annular) and ε=0.9 (cooling), Earth
//! shadow on. The recipes are at
//! `sim_srp::shadow_2a_{annular,cooling}`.

use astrodyn_verif_jeod::run_verification::sim_srp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn tier3_bevy_shadow_2a_annular() {
    sim_srp::shadow_2a_annular().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_shadow_2a_cooling() {
    sim_srp::shadow_2a_cooling().run_and_assert_parity::<astrodyn::Earth>();
}
