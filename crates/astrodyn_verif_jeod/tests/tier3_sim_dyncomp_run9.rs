//! Tier 3: SIM_dyncomp RUN_9A / RUN_9B / RUN_9C / RUN_9D.
//!
//! The recipe factories (`sim_dyncomp::run9{a,c,d}_*`) carry the JEOD
//! pulse-window schedule (`t ∈ [1000, 2000) s` with body-frame
//! `[10, 0, 0]` force / torque) via a per-tick `pre_step` closure and
//! cross-validate against `dyncomp_run9{a,c,d}_state.csv`. The
//! per-tick cadence matches JEOD's Trick scheduler (32 Hz) so the
//! force / torque direction tracks the integrator's body quaternion
//! through the pulse rather than freezing across the 60 s reference
//! cadence — that mismatch was the failure mode in the recipe's
//! first attempt at extraction.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_jeod::VerificationCaseExt;

#[test]
fn tier3_simulation_run9a_torque() {
    sim_dyncomp::run9a_torque().run_and_assert();
}

#[test]
fn tier3_simulation_run9b_torque_initial_rate() {
    sim_dyncomp::run9b_torque_initial_rate().run_and_assert();
}

#[test]
fn tier3_simulation_run9c_force_torque() {
    sim_dyncomp::run9c_force_torque().run_and_assert();
}

#[test]
fn tier3_simulation_run9d_force_torque_rate() {
    sim_dyncomp::run9d_force_torque_rate().run_and_assert();
}
