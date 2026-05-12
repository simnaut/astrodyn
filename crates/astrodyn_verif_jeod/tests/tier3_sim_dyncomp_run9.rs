//! Tier 3: SIM_dyncomp RUN_9A/9C/9D — External force / torque via the
//! recipe pipeline.
//!
//! Migrated from a hand-rolled per-tick `set_body_external_*` loop to
//! the recipe + `VerificationCaseExt::run_and_assert` shape. The
//! scenario builders, time-scheduled force / torque injection, and
//! per-component tolerances live in
//! [`astrodyn_verif_jeod::run_verification::sim_dyncomp::run9{a,c,d}_*`];
//! each test below is a one-liner that materializes the recipe into
//! a runtime `Simulation` and asserts the tolerances against the
//! committed reference CSVs.
//!
//! The pulse window `t ∈ [1000, 2000) s` and the body-frame loads
//! `[10, 0, 0] N` / `[10, 0, 0] N·m` come from
//! `JEOD_HOME/models/dynamics/dyn_body/verif/SIM_dyncomp/SET_test/RUN_9*/input.py`.
//! The recipe pre-step closure rotates the body-frame force to the
//! inertial frame via `SimContext::body_q_inertial_body`, the same
//! adapter-neutral path the matching `bevy_parity_dyncomp_run9` wrapper
//! uses to drive both runtimes from a single closure.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_jeod::VerificationCaseExt;

#[test]
fn tier3_simulation_run9a_torque() {
    sim_dyncomp::run9a_torque().run_and_assert();
}

#[test]
fn tier3_simulation_run9c_force_torque() {
    sim_dyncomp::run9c_force_torque().run_and_assert();
}

#[test]
fn tier3_simulation_run9d_force_torque_rate() {
    sim_dyncomp::run9d_force_torque_rate().run_and_assert();
}
