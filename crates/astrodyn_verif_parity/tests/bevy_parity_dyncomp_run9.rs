//! Bevy ↔ runner parity for SIM_dyncomp RUN_9A / RUN_9C / RUN_9D —
//! scheduled external force / torque pulses (`t ∈ [1000, 2000) s`) on
//! the 6-DOF ISS scenario, via the `VerificationCaseParityExt` trait
//! (issue #389).
//!
//! These wrappers exercise the per-tick `pre_step` cadence the parity
//! machinery added alongside the recipe: every integration tick (32 Hz)
//! invokes the closure on *both* runtimes with the same `t_end`
//! argument, and each side overwrites its external-force /
//! external-torque field with the same value before stepping. Bit-
//! identity is the contract — if a tick fires the pulse on the runner
//! but not on the Bevy side (or vice versa), the divergence accumulates
//! through subsequent ticks and `run_and_assert_parity` flags it at the
//! next reference-CSV checkpoint, naming the offending body, component,
//! and record time (not the originating tick — assertions sample at
//! the CSV cadence, not per integration step).
//!
//! Companion runner-vs-JEOD tests live in
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_dyncomp_run9.rs`.

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
