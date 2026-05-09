//! Bevy ↔ runner parity for SIM_LVLH (inclined / eccentric / equatorial
//! LVLH-frame derived-state recipes). Wrappers land as part of #389.

use astrodyn_verif_jeod::run_verification::sim_derived_state;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn tier3_bevy_lvlh_inc() {
    sim_derived_state::lvlh_inc().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_lvlh_ecc() {
    sim_derived_state::lvlh_ecc().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_lvlh_equ() {
    sim_derived_state::lvlh_equ().run_and_assert_parity::<astrodyn::Earth>();
}
