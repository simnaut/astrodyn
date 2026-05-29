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
fn bevy_parity_orbinit_docker_run_0002() {
    sim_orbinit_docker::run_0002().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0102() {
    sim_orbinit_docker::run_0102().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0003() {
    sim_orbinit_docker::run_0003().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0103() {
    sim_orbinit_docker::run_0103().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0004() {
    sim_orbinit_docker::run_0004().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0104() {
    sim_orbinit_docker::run_0104().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0005() {
    sim_orbinit_docker::run_0005().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0105() {
    sim_orbinit_docker::run_0105().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0006() {
    sim_orbinit_docker::run_0006().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0106() {
    sim_orbinit_docker::run_0106().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0010() {
    sim_orbinit_docker::run_0010().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0110() {
    sim_orbinit_docker::run_0110().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0011() {
    sim_orbinit_docker::run_0011().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0111() {
    sim_orbinit_docker::run_0111().run_and_assert_parity::<astrodyn::Earth>();
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
fn bevy_parity_orbinit_docker_run_0202() {
    sim_orbinit_docker::run_0202().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0302() {
    sim_orbinit_docker::run_0302().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0203() {
    sim_orbinit_docker::run_0203().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0303() {
    sim_orbinit_docker::run_0303().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0204() {
    sim_orbinit_docker::run_0204().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0304() {
    sim_orbinit_docker::run_0304().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0205() {
    sim_orbinit_docker::run_0205().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0305() {
    sim_orbinit_docker::run_0305().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0206() {
    sim_orbinit_docker::run_0206().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0306() {
    sim_orbinit_docker::run_0306().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0210() {
    sim_orbinit_docker::run_0210().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0310() {
    sim_orbinit_docker::run_0310().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0211() {
    sim_orbinit_docker::run_0211().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0311() {
    sim_orbinit_docker::run_0311().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_docker_run_0401() {
    sim_orbinit_docker::run_0401().run_and_assert_parity::<astrodyn::Earth>();
}
