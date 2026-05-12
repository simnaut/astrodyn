//! Bevy ↔ runner parity for the analytical SIM_force_torque scenarios
//! (`tier3_sim_force_torque_response`).
//!
//! The matching tier3 file drives each scenario through the runner and
//! asserts a closed-form analytical identity (F = m·a translation,
//! τ = I·α rotation, the decoupling of CoM force from rotation, and
//! the symmetric ±F impulse returning the body to rest). This file
//! pairs each recipe with the parity trait so the same scenarios also
//! run through the Bevy adapter and assert `runner ↔ bevy` bit-
//! identity at every synthetic record — the second half of the
//! `runner ↔ JEOD ≈ bevy` transitivity argument the issue's matrix
//! covers.
//!
//! The symmetric-impulse wrapper exercises the new
//! `SimContext::set_body_external_force` surface on both runtimes: the
//! recipe's `pre_step` closure flips the inertial-frame force sign at
//! the midpoint record, and the `BevySimContext` mirror writes the
//! same value into `ExternalForceC` so both runtimes integrate
//! bit-identical sub-steps through the rest of the propagation.

use astrodyn_verif_jeod::run_verification::sim_force_torque_response;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_force_torque_response_force_constant_acceleration() {
    sim_force_torque_response::force_constant_acceleration()
        .run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_force_torque_response_torque_constant_angular_acceleration() {
    sim_force_torque_response::torque_constant_angular_acceleration()
        .run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_force_torque_response_force_and_torque_decoupled_force() {
    sim_force_torque_response::force_and_torque_decoupled_force()
        .run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_force_torque_response_force_and_torque_decoupled_torque() {
    sim_force_torque_response::force_and_torque_decoupled_torque()
        .run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_force_torque_response_force_and_torque_decoupled_both() {
    sim_force_torque_response::force_and_torque_decoupled_both()
        .run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_force_torque_response_force_symmetric_impulse() {
    sim_force_torque_response::force_symmetric_impulse().run_and_assert_parity::<astrodyn::Earth>();
}
