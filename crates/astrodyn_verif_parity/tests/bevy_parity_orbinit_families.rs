//! Bevy ↔ runner parity for the analytical orbinit-families
//! conservation scans (`tier3_sim_orbinit_families`). Wrappers land as
//! part of #389.
//!
//! The matching tier3 file drives each scenario through the runner and
//! asserts conservation invariants (specific orbital energy and
//! angular momentum under point-mass gravity, plus per-family
//! geometric checks). This file pairs each recipe with the parity
//! trait so the same scenarios also run through the Bevy adapter and
//! assert `runner ↔ bevy` bit-identity at every synthetic record —
//! the second half of the `runner ↔ JEOD ≈ bevy` transitivity
//! argument the issue's matrix covers.

use astrodyn_verif_jeod::run_verification::sim_orbinit_families;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_orbinit_families_circular_leo() {
    sim_orbinit_families::circular_leo().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_families_eccentric() {
    sim_orbinit_families::eccentric().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_families_highly_eccentric() {
    sim_orbinit_families::highly_eccentric().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_families_retrograde() {
    sim_orbinit_families::retrograde().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_families_equatorial() {
    sim_orbinit_families::equatorial().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_families_polar() {
    sim_orbinit_families::polar().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_families_hyperbolic() {
    sim_orbinit_families::hyperbolic().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_families_near_parabolic() {
    sim_orbinit_families::near_parabolic().run_and_assert_parity::<astrodyn::Earth>();
}
