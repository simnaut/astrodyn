//! Bevy ↔ runner parity for the analytical SIM_LVLH-extended scenarios
//! (`tier3_sim_lvlh_extended`). Wrappers land as part of #389.
//!
//! The matching tier3 file drives each scenario through the runner and
//! asserts a closed-form LVLH property (Y-axis sign flip on retrograde,
//! `|ω| = |h|/r²` at radius extrema, return-to-orientation after one
//! period). This file pairs each recipe with the parity trait so the
//! same scenarios also run through the Bevy adapter and assert
//! `runner ↔ bevy` bit-identity at every record — the second half of the
//! `runner ↔ JEOD ≈ bevy` transitivity argument the issue's matrix
//! covers.

use astrodyn_verif_jeod::run_verification::sim_lvlh_extended;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_lvlh_prograde_circular() {
    sim_lvlh_extended::prograde_circular().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_lvlh_retrograde_circular() {
    sim_lvlh_extended::retrograde_circular().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_lvlh_eccentric() {
    sim_lvlh_extended::eccentric().run_and_assert_parity::<astrodyn::Earth>();
}

// `periodicity` deliberately uses `dt = period / 560`, which is
// irrational in seconds (≈9.917 s). The runner integrates with the f64
// value directly; the Bevy adapter now reads `dt` from
// `IntegrationDtR` (bit-exact f64) rather than
// `Time<Fixed>::delta_secs_f64()` (rounded through
// `Duration::from_secs_f64` to integer nanoseconds), so the two paths
// share the same `dt` bit pattern at every tick and parity holds
// bit-identical. The runner-side `tier3_lvlh_periodicity` continues to
// exercise the recipe under the analytical assertion.
#[test]
fn bevy_parity_lvlh_periodicity() {
    sim_lvlh_extended::periodicity().run_and_assert_parity::<astrodyn::Earth>();
}
