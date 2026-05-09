//! Bevy ↔ runner parity for SIM_Euler (Euler-angle derived state along
//! the SIM_dyncomp RUN_2 trajectory plus `_ecc` and `_equ` orbit
//! variants). Recipes live in `sim_derived_state`; wrappers land as
//! part of #389.

use astrodyn_verif_jeod::run_verification::sim_derived_state;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn tier3_bevy_euler_run2() {
    sim_derived_state::euler_run2().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_euler_ecc() {
    sim_derived_state::euler_ecc().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_euler_equ() {
    sim_derived_state::euler_equ().run_and_assert_parity::<astrodyn::Earth>();
}
