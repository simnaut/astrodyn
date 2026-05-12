//! Bevy ↔ runner parity for the analytical extended relative-dynamics
//! scenarios (`tier3_sim_relative_extended`). Wrappers land as part of
//! #389.
//!
//! The matching tier3 file drives each scenario through the runner and
//! asserts a closed-form relative-state property (co-orbiting LVLH
//! separation bound, Hohmann-geometry separation oscillation, same-orbit
//! 90° chord length, cross-track amplitude r·sin(i),
//! r_AB = -r_BA symmetry). This file pairs each recipe with the parity
//! trait so the same scenarios also run through the Bevy adapter and
//! assert `runner ↔ bevy` bit-identity at every synthetic record — the
//! second half of the `runner ↔ JEOD ≈ bevy` transitivity argument the
//! issue's matrix covers.

use astrodyn_verif_jeod::run_verification::sim_relative_extended;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_relative_extended_two_coorbiting_vehicles() {
    sim_relative_extended::two_coorbiting_vehicles().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_relative_extended_hohmann_transfer_geometry() {
    sim_relative_extended::hohmann_transfer_geometry().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_relative_extended_same_orbit_phase_difference() {
    sim_relative_extended::same_orbit_phase_difference().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_relative_extended_different_inclinations() {
    sim_relative_extended::different_inclinations().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_relative_extended_round_trip_frames() {
    sim_relative_extended::round_trip_frames().run_and_assert_parity::<astrodyn::Earth>();
}
