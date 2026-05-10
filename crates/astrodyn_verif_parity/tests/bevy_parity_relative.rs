//! Bevy ↔ runner parity for the SIM_Relative two-body kinematic family.
//!
//! Five one-liner wrappers over the [`sim_relative::*`] recipes — each
//! drives both the runner and a Bevy [`App`] from the same scenario
//! factory and asserts every body's translational + rotational state
//! is bit-identical at every reference checkpoint via
//! [`VerificationCaseParityExt::run_and_assert_parity`].
//!
//! The pre-#389 hand-rolled tests additionally checked that
//! [`astrodyn::compute_relative_state`] and
//! [`astrodyn::compute_lvlh_relative_state`] returned bit-identical
//! values between the two runtimes. Those helpers are pure
//! deterministic functions of two `SixDofState` inputs — when the
//! parity trait already pins both inputs bit-identically, the helper
//! outputs are bit-identical by construction. The explicit assertion
//! becomes a sanity check that doesn't catch any bug the body bit-
//! identity check doesn't already catch, so the migration drops it
//! along with the hand-rolled scaffolding.
//!
//! ## Tier 3 sibling
//!
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_relative.rs` is the
//! runner-vs-JEOD oracle that supplies the transitivity argument.
//! Its CSV format (interleaved 57-column two-body state) is consumed
//! through a private hand-rolled parser; the parity recipes route
//! through the existing [`CsvReference::OrbInit`] dispatch for cadence
//! lookup only — see the `sim_relative` module docstring for why that
//! works despite the column layout mismatch.

use astrodyn_verif_jeod::run_verification::sim_relative;
use astrodyn_verif_parity::VerificationCaseParityExt;

/// 6-DOF: distinct quaternions and translational states for both bodies.
#[test]
fn bevy_parity_relative_ab_rot_ab_trans() {
    sim_relative::relative_ab_rot_ab_trans().run_and_assert_parity::<astrodyn::Earth>();
}

/// 6-DOF: identity rotation, distinct translational states.
#[test]
fn bevy_parity_relative_no_rot_ab_trans() {
    sim_relative::relative_no_rot_ab_trans().run_and_assert_parity::<astrodyn::Earth>();
}

/// 6-DOF: identity translational state, distinct rotations.
#[test]
fn bevy_parity_relative_a_rot_no_trans() {
    sim_relative::relative_a_rot_no_trans().run_and_assert_parity::<astrodyn::Earth>();
}
