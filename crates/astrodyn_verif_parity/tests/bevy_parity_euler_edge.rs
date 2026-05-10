//! Bevy ↔ runner parity for SIM_Euler edge cases (eccentric and
//! equatorial orbit variants of the Euler-angle derived state).
//!
//! Mirrors `tier3_sim_euler_edge.rs` 1-to-1 so the two test files stay
//! visibly in sync — a new edge-case recipe added there has an obvious
//! parity sibling slot here. The recipes themselves
//! (`sim_derived_state::euler_ecc`, `euler_equ`) are shared with the
//! base `bevy_parity_euler` wrapper; running them again from this file
//! is intentional: the topic-level coverage check in `parity_coverage`
//! is per-file, so `euler_edge` needs its own wrapper file to satisfy
//! the `tier3_topics ⊂ bevy_parity_topics` invariant.

use astrodyn_verif_jeod::run_verification::sim_derived_state;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_euler_edge_ecc() {
    sim_derived_state::euler_ecc().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_euler_edge_equ() {
    sim_derived_state::euler_equ().run_and_assert_parity::<astrodyn::Earth>();
}
