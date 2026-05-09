//! Bevy ↔ runner parity for SIM_dyncomp RUN_10 family — gravity-
//! gradient torque (circular, elliptical, and elliptical with initial
//! body rate). All three are `pre_step: None`; wrappers land as part
//! of #389.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn tier3_bevy_dyncomp_run10a_gravity_torque() {
    sim_dyncomp::run10a_gravity_torque().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_dyncomp_run10c_gravity_torque_elliptical() {
    sim_dyncomp::run10c_gravity_torque_elliptical().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_dyncomp_run10d_gravity_torque_elliptical_rate() {
    sim_dyncomp::run10d_gravity_torque_elliptical_rate().run_and_assert_parity::<astrodyn::Earth>();
}
