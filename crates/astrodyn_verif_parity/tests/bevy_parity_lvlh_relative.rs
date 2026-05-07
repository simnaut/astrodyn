//! Bevy ↔ runner parity for the SIM_lvlh_relative two-body
//! kinematic-with-LVLH-derivation family.
//!
//! Two one-liner wrappers over the [`sim_relative::lvlhrel_*`] recipes
//! (which sit alongside the SIM_Relative recipes since the scenario
//! shape is the same — two free-flying bodies, no gravity — and only
//! the post-step derived-state computation differs). Each drives both
//! the runner and a Bevy [`App`] from the same scenario factory and
//! asserts every body's translational state is bit-identical at every
//! reference checkpoint.
//!
//! The pre-#389 hand-rolled tests additionally checked that
//! [`astrodyn::compute_lvlh_relative_state`] returned bit-identical
//! values between the two runtimes. That helper is a deterministic
//! function of two [`astrodyn::TranslationalState`] inputs — when the
//! parity trait already pins both inputs bit-identically, the helper
//! output is bit-identical by construction.

use astrodyn_verif_jeod::run_verification::sim_relative;
use astrodyn_verif_parity::VerificationCaseParityExt;

/// 3-DOF LVLH-relative: lateral offset.
#[test]
fn tier3_bevy_lvlhrel_test0() {
    sim_relative::lvlhrel_test0().run_and_assert_parity::<astrodyn::Earth>();
}

/// 3-DOF LVLH-relative: coplanar along-track separation.
#[test]
fn tier3_bevy_lvlhrel_test1() {
    sim_relative::lvlhrel_test1().run_and_assert_parity::<astrodyn::Earth>();
}
