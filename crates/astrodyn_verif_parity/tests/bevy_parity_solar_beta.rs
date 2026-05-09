//! Bevy ↔ runner parity for the SIM_SolarBeta recipes. `solar_beta_run2`
//! drives the Sun source via a per-record `pre_step` (closed by #395);
//! `solar_beta_equ` and `solar_beta_obliquity` use no `pre_step` but
//! were gated on the same parity-trait infrastructure landing first.

use astrodyn_verif_jeod::run_verification::sim_solar_beta;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn tier3_bevy_solar_beta_run2() {
    sim_solar_beta::solar_beta_run2().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_solar_beta_equ() {
    sim_solar_beta::solar_beta_equ().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_solar_beta_obliquity() {
    sim_solar_beta::solar_beta_obliquity().run_and_assert_parity::<astrodyn::Earth>();
}
