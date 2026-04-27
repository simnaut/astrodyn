//! Tier 3: SIM_torque_compare_simple — six runs of progressive
//! gravity + gravity-gradient complexity.
//!
//! Migrated from a 569-line bespoke parameterized loop to six recipe
//! one-liners (#162). Recipes live in
//! `jeod_runner::run_verification::sim_torque_simple::run0{1..6}`;
//! the per-step DE421 update is shared via the recipe's
//! `pre_step` factory.

use jeod_runner::run_verification::sim_torque_simple;
use jeod_runner::VerificationCaseExt;

#[test]
fn tier3_torque_simple_run01() {
    sim_torque_simple::run01().run_and_assert();
}

#[test]
fn tier3_torque_simple_run02() {
    sim_torque_simple::run02().run_and_assert();
}

#[test]
fn tier3_torque_simple_run03() {
    sim_torque_simple::run03().run_and_assert();
}

#[test]
fn tier3_torque_simple_run04() {
    sim_torque_simple::run04().run_and_assert();
}

#[test]
fn tier3_torque_simple_run05() {
    sim_torque_simple::run05().run_and_assert();
}

#[test]
fn tier3_torque_simple_run06() {
    sim_torque_simple::run06().run_and_assert();
}
