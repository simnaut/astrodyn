//! Bevy ↔ runner parity for the analytical orbinit-round-trip
//! scenarios (`tier3_sim_orbinit_roundtrip`). Wrappers land as part of
//! #389.
//!
//! The matching tier3 file drives each scenario through the runner and
//! asserts a closed-form round-trip property (shape and orientation
//! orbital elements recover to the initial values after propagation
//! under point-mass gravity, or specific energy for the near-circular
//! case). This file pairs each recipe with the parity trait so the
//! same scenarios also run through the Bevy adapter and assert
//! `runner ↔ bevy` bit-identity at every synthetic record — the second
//! half of the `runner ↔ JEOD ≈ bevy` transitivity argument the
//! issue's matrix covers.

use astrodyn_verif_jeod::run_verification::sim_orbinit_roundtrip;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_orbinit_roundtrip_circular() {
    sim_orbinit_roundtrip::circular().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_roundtrip_eccentric() {
    sim_orbinit_roundtrip::eccentric().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_roundtrip_retrograde() {
    sim_orbinit_roundtrip::retrograde().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_roundtrip_equatorial() {
    sim_orbinit_roundtrip::equatorial().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_roundtrip_polar() {
    sim_orbinit_roundtrip::polar().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_roundtrip_molniya() {
    sim_orbinit_roundtrip::molniya().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_roundtrip_hyperbolic() {
    sim_orbinit_roundtrip::hyperbolic().run_and_assert_parity::<astrodyn::Earth>();
}
