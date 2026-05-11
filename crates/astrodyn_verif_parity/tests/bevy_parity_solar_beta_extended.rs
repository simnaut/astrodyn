//! Bevy ↔ runner parity for the analytical SIM_SolarBeta-extended
//! scenarios (`tier3_sim_solar_beta_extended`). Wrappers land as part of
//! #389.
//!
//! The matching tier3 file drives each scenario through the runner and
//! asserts a closed-form solar-beta property (β = 0 when ĥ ⊥ ŝ, |β| = π/2
//! when ĥ ∥ ŝ, β = π/2 − i when Sun is on the polar axis of an inclined
//! orbit, |β| ≤ π/2 over any propagation window). This file pairs each
//! recipe with the parity trait so the same scenarios also run through
//! the Bevy adapter and assert `runner ↔ bevy` bit-identity at every
//! synthetic record — the second half of the `runner ↔ JEOD ≈ bevy`
//! transitivity argument the issue's matrix covers.

use astrodyn_verif_jeod::run_verification::sim_solar_beta_extended;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_solar_beta_equatorial_at_equinox() {
    sim_solar_beta_extended::equatorial_at_equinox().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_solar_beta_polar_sun_x() {
    sim_solar_beta_extended::polar_sun_x().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_solar_beta_polar_sun_y() {
    sim_solar_beta_extended::polar_sun_y().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_solar_beta_polar_sun_z() {
    sim_solar_beta_extended::polar_sun_z().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_solar_beta_iss_sun_x() {
    sim_solar_beta_extended::iss_sun_x().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_solar_beta_iss_sun_z() {
    sim_solar_beta_extended::iss_sun_z().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_solar_beta_iss_sun_neg_y() {
    sim_solar_beta_extended::iss_sun_neg_y().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_solar_beta_sun_in_orbital_plane() {
    sim_solar_beta_extended::sun_in_orbital_plane().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_solar_beta_sun_perpendicular_to_plane() {
    sim_solar_beta_extended::sun_perpendicular_to_plane()
        .run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_solar_beta_bounded() {
    sim_solar_beta_extended::bounded().run_and_assert_parity::<astrodyn::Earth>();
}
