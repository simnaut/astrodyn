//! Bevy ↔ runner parity for SIM_NED edge cases (polar orbit on the
//! ellipsoidal Earth and inclined/polar orbits on the spherical Earth).
//!
//! Mirrors `tier3_sim_ned_edge.rs` 1-to-1 so the two test files stay
//! visibly in sync — a new edge-case recipe added there has an obvious
//! parity sibling slot here. The recipes themselves
//! (`sim_derived_state::ned_ell_polar`, `ned_sph_inc`, `ned_sph_polar`)
//! are shared with the base `bevy_parity_ned` wrapper; running them
//! again from this file is intentional: the topic-level coverage check
//! in `parity_coverage` is per-file, so `ned_edge` needs its own
//! wrapper file to satisfy the `tier3_topics ⊂ bevy_parity_topics`
//! invariant.

use astrodyn_verif_jeod::run_verification::sim_derived_state;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_ned_edge_ell_polar() {
    sim_derived_state::ned_ell_polar().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_ned_edge_sph_inc() {
    sim_derived_state::ned_sph_inc().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_ned_edge_sph_polar() {
    sim_derived_state::ned_sph_polar().run_and_assert_parity::<astrodyn::Earth>();
}
