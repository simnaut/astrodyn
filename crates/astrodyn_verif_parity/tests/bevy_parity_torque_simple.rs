//! Bevy ↔ runner parity for SIM_torque_compare_simple RUN_01–RUN_06.
//! Every variant shares `torque_simple_pre_step` (Sun/Moon position
//! refresh from DE421); closed by #395.

use astrodyn_verif_jeod::run_verification::sim_torque_simple;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_torque_simple_run01() {
    sim_torque_simple::run01().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_torque_simple_run02() {
    sim_torque_simple::run02().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_torque_simple_run03() {
    sim_torque_simple::run03().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_torque_simple_run04() {
    sim_torque_simple::run04().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_torque_simple_run05() {
    sim_torque_simple::run05().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_torque_simple_run06() {
    sim_torque_simple::run06().run_and_assert_parity::<astrodyn::Earth>();
}
