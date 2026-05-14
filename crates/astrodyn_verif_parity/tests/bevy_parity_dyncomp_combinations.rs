//! Bevy ↔ runner parity for the analytical SIM_dyncomp physics-
//! combinations family (`tier3_sim_dyncomp_combinations`).
//!
//! The matching tier3 file drives each scenario through the runner and
//! asserts a closed-form / conservation-law property (Keplerian energy
//! plus angular-momentum conservation, third-body torque, monotonic SMA
//! decay under drag, torque-free rigid-body inertial angular-momentum
//! conservation, constant-force impulse response, constant-torque
//! impulse response, major-axis spin stability). This file pairs each
//! recipe with the parity trait so the same scenarios also run through
//! the Bevy adapter and assert `runner ↔ bevy` bit-identity at every
//! synthetic record. Bit-identity here plus the analytical assertions
//! in the sibling tier3 file together imply the Bevy adapter satisfies
//! the same closed-form properties — the analytical analog of the
//! `runner ↔ bevy` (this file) plus `runner ↔ JEOD` (sibling tier3
//! assertions) ⇒ `bevy ↔ JEOD` transitivity argument the issue's
//! matrix covers for CSV-backed scenarios, within the runner's
//! tolerance.
//!
//! The force-impulse case ships *two* parity wrappers — one for the
//! forced sim and one for its no-force Kepler sibling. The tier3
//! assertion subtracts the no-force final velocity from the forced
//! final velocity to isolate the impulse contribution from gravity's
//! delta-v; both legs must therefore agree bit-for-bit between runner
//! and Bevy for the analytical-analog transitivity argument to carry
//! through to the Bevy adapter.

use astrodyn_verif_jeod::run_verification::sim_dyncomp_combinations;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_dyncomp_combinations_point_mass_3dof_conservation() {
    sim_dyncomp_combinations::point_mass_3dof_conservation()
        .run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_combinations_point_mass_plus_thirdbody_conservation() {
    sim_dyncomp_combinations::point_mass_plus_thirdbody_conservation()
        .run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_combinations_drag_point_mass_monotonic_decay() {
    sim_dyncomp_combinations::drag_point_mass_monotonic_decay()
        .run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_combinations_rigid_body_invariance_6dof() {
    sim_dyncomp_combinations::rigid_body_invariance_6dof()
        .run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_combinations_external_force_impulse_response() {
    sim_dyncomp_combinations::external_force_impulse_response()
        .run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_combinations_external_force_impulse_kepler_reference() {
    sim_dyncomp_combinations::external_force_impulse_kepler_reference()
        .run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_combinations_external_torque_impulse_response() {
    sim_dyncomp_combinations::external_torque_impulse_response()
        .run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_combinations_attitude_stability_major_axis() {
    sim_dyncomp_combinations::attitude_stability_major_axis()
        .run_and_assert_parity::<astrodyn::Earth>();
}
