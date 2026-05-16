//! Bevy ↔ runner parity for SIM_Planetary derived-state regimes.
//!
//! Five orbit regimes (LEO inclined, polar, eccentric, equatorial, GEO)
//! exercise coordinate singularities (equatorial RAAN, polar LVLH).
//! All five share an Earth point-mass scenario, so each runs through
//! `populate_app::<Earth>` and asserts bit-identical state at every CSV
//! checkpoint.
//!
//! The runner-side counterpart is
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_planetary.rs`; transitivity
//! of the two assertions is the goal.

use astrodyn_verif_jeod::run_verification::sim_planetary;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_planetary_leo_inc() {
    sim_planetary::leo_inc().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_planetary_leo_polar() {
    sim_planetary::leo_polar().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_planetary_leo_ecc() {
    sim_planetary::leo_ecc().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_planetary_leo_equ() {
    sim_planetary::leo_equ().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_planetary_geo() {
    sim_planetary::geo().run_and_assert_parity::<astrodyn::Earth>();
}
