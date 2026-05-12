//! Bevy ↔ runner parity for SIM_dyncomp RUN_9A/9C/9D — time-scheduled
//! external force / torque pulses applied through the recipe's
//! `pre_step` closure on both runtimes.
//!
//! The recipes live in
//! [`astrodyn_verif_jeod::run_verification::sim_dyncomp::run9{a,c,d}_*`]
//! and share a single `pre_step` factory per family. The closure
//! observes only the [`astrodyn_verif_jeod::verification::SimContext`]
//! trait surface — `set_body_external_force` /
//! `set_body_external_torque` for the writes, plus
//! `body_q_inertial_body` for reading the current attitude that
//! rotates the body-frame force into inertial. Identical numeric
//! inputs flow into both `astrodyn_runner::Simulation` and the Bevy
//! [`astrodyn_bevy::App`], so `runner ↔ bevy` bit-identity holds at
//! every reference record and the matching `tier3_sim_dyncomp_run9`
//! tolerances carry transitively to the Bevy adapter against the
//! JEOD oracle.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_dyncomp_run9a_torque() {
    sim_dyncomp::run9a_torque().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_run9c_force_torque() {
    sim_dyncomp::run9c_force_torque().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_run9d_force_torque_rate() {
    sim_dyncomp::run9d_force_torque_rate().run_and_assert_parity::<astrodyn::Earth>();
}
