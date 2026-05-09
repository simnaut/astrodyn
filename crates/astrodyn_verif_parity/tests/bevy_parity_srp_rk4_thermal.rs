//! Bevy ↔ runner parity for derivative-class thermal SRP scenarios:
//! first-order and RK4 thermal integration order, plus the rotated-
//! `t_struct_body` regression case for the structural↔body torque
//! rotation in the coupled RK4 stage closure.

use astrodyn_verif_jeod::run_verification::sim_srp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn tier3_bevy_srp_derivative_first_order() {
    sim_srp::srp_derivative_first_order().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_srp_derivative_rk4() {
    sim_srp::srp_derivative_rk4().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_srp_derivative_rk4_with_rotated_struct_frame() {
    sim_srp::srp_derivative_rk4_rotated_struct().run_and_assert_parity::<astrodyn::Earth>();
}
