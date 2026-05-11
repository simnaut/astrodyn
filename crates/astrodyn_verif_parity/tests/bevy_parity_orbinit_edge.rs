//! Bevy ↔ runner parity for the SIM_orbinit edge scenarios
//! (`tier3_sim_orbinit_edge`). Wrappers land as part of #389.
//!
//! The matching tier3 file builds each scenario through the runner,
//! propagates a single step, and asserts a range + cross-RUN
//! consistency property on the JEOD-source initial states. This file
//! pairs each recipe with the parity trait so the same scenarios also
//! run through the Bevy adapter and assert `runner ↔ bevy` bit-identity
//! at the synthetic checkpoint — the second half of the
//! `runner ↔ JEOD ≈ bevy` transitivity argument the issue's matrix
//! covers.

use astrodyn_verif_jeod::run_verification::sim_orbinit_edge;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_orbinit_edge_run_0101() {
    sim_orbinit_edge::run_0101().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_edge_run_0201() {
    sim_orbinit_edge::run_0201().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_edge_run_0301() {
    sim_orbinit_edge::run_0301().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_orbinit_edge_run_0401() {
    sim_orbinit_edge::run_0401().run_and_assert_parity::<astrodyn::Earth>();
}
