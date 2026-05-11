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
// value directly; the Bevy adapter routes time through
// `Time<Fixed>::advance_by(Duration::from_secs_f64(dt))`, which rounds
// to integer nanoseconds. The two paths therefore diverge in the LSBs
// of position after the first few ticks even though the underlying
// `astrodyn_*` math is identical. Re-enabling this wrapper requires a
// Bevy-side time-advance path that preserves full f64 dt precision —
// tracked alongside other parity-trait-infrastructure follow-ups to
// #389. The runner-side `tier3_lvlh_periodicity` continues to exercise
// the recipe under the analytical assertion.
#[test]
#[ignore = "parity-gap: irrational dt loses precision through Time<Fixed>'s \
            Duration round-trip; needs Bevy-side f64 time advance"]
fn bevy_parity_lvlh_periodicity() {
    sim_lvlh_extended::periodicity().run_and_assert_parity::<astrodyn::Earth>();
}
