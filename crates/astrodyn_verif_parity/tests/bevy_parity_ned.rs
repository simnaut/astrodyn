//! Bevy ↔ runner parity for SIM_NED (geodetic + spherical NED on
//! ellipsoidal and spherical Earth, inclined and polar orbits).
//! Wrappers land as part of #389.

use astrodyn_verif_jeod::run_verification::sim_derived_state;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn tier3_bevy_ned_ell_inc() {
    sim_derived_state::ned_ell_inc().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_ned_ell_polar() {
    sim_derived_state::ned_ell_polar().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_ned_sph_inc() {
    sim_derived_state::ned_sph_inc().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_ned_sph_polar() {
    sim_derived_state::ned_sph_polar().run_and_assert_parity::<astrodyn::Earth>();
}
