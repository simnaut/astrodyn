//! Bevy ↔ runner parity for SIM_solar_beta (per-step Sun ephemeris
//! injection driving the body's solar-beta extra), via the
//! `VerificationCaseParityExt` trait.
//!
//! Unblocked by issue #395's `BevySimContext`: every variant's
//! `pre_step` drives `set_source_position` for the Sun source on both
//! runtimes, which `body.solar_beta` reads each step.

use astrodyn_verif_jeod::run_verification::sim_solar_beta;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_solar_beta_run2() {
    sim_solar_beta::solar_beta_run2().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_solar_beta_equ() {
    sim_solar_beta::solar_beta_equ().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_solar_beta_obliquity() {
    sim_solar_beta::solar_beta_obliquity().run_and_assert_parity::<astrodyn::Earth>();
}
