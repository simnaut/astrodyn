//! Bevy ↔ runner parity for the SIM_OrbElem comprehensive sweep
//! (`tier3_sim_orbelem_comprehensive`). Wrappers land as part of #389.
//!
//! The matching tier3 file builds each scenario through the runner,
//! propagates a single tiny-dt step, and asserts per-orbital-element
//! bounds on the resulting state against the JEOD-logged t=0 row. This
//! file pairs each recipe with the parity trait so the same scenarios
//! also run through the Bevy adapter and assert `runner ↔ bevy`
//! bit-identity at the synthetic checkpoint — the second half of the
//! `runner ↔ JEOD ≈ bevy` transitivity argument the issue's matrix
//! covers.

use astrodyn_verif_jeod::run_verification::sim_orbelem_comprehensive;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_orbelem_comprehensive_t01() {
    sim_orbelem_comprehensive::t01().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbelem_comprehensive_t10() {
    sim_orbelem_comprehensive::t10().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbelem_comprehensive_t20() {
    sim_orbelem_comprehensive::t20().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbelem_comprehensive_t30() {
    sim_orbelem_comprehensive::t30().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbelem_comprehensive_t40() {
    sim_orbelem_comprehensive::t40().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbelem_comprehensive_t50() {
    sim_orbelem_comprehensive::t50().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbelem_comprehensive_t55() {
    sim_orbelem_comprehensive::t55().run_and_assert_parity::<astrodyn::Earth>();
}
