//! Bevy ↔ runner parity for the SIM_orbinit Docker scenarios
//! (`tier3_sim_orbinit_docker`). Wrappers land as part of #389.
//!
//! The matching tier3 file builds each scenario through the runner,
//! compares the recipe's computed initial state against the JEOD-logged
//! t=0 row, and propagates a single step so the integrator and
//! frame-propagation stages run end-to-end. This file pairs each recipe
//! with the parity trait so the same scenarios also run through the
//! Bevy adapter and assert `runner ↔ bevy` bit-identity at the
//! synthetic checkpoint — the second half of the
//! `runner ↔ JEOD ≈ bevy` transitivity argument the issue's matrix
//! covers. Because the orbital-element-to-Cartesian conversion (and the
//! optional pfix→inertial rotation for the pfix variants) lives inside
//! the recipe's scenario factory, bit-identity here also implies
//! `init_from_mean_anomaly` and the RNP rotation produced the same f64
//! bit-pattern on both sides.

use astrodyn_verif_jeod::run_verification::sim_orbinit_docker;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_orbinit_docker_run_0001() {
    sim_orbinit_docker::run_0001().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0101() {
    sim_orbinit_docker::run_0101().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0201() {
    sim_orbinit_docker::run_0201().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0301() {
    sim_orbinit_docker::run_0301().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0401() {
    sim_orbinit_docker::run_0401().run_and_assert_parity::<astrodyn::Earth>();
}
