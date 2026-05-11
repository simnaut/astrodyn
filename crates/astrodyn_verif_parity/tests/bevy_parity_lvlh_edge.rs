//! Bevy ↔ runner parity for SIM_LVLH edge cases (eccentric and
//! equatorial orbit variants of the LVLH-frame derived state).
//!
//! Mirrors `tier3_sim_lvlh_edge.rs` 1-to-1 so the two test files stay
//! visibly in sync — a new edge-case recipe added there has an obvious
//! parity sibling slot here. The recipes themselves
//! (`sim_derived_state::lvlh_ecc`, `lvlh_equ`) are shared with the
//! base `bevy_parity_lvlh` wrapper; running them again from this file
//! is intentional: the topic-level coverage check in `parity_coverage`
//! is per-file, so `lvlh_edge` needs its own wrapper file to satisfy
//! the `tier3_topics ⊂ bevy_parity_topics` invariant.

use astrodyn_verif_jeod::run_verification::sim_derived_state;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_lvlh_edge_ecc() {
    sim_derived_state::lvlh_ecc().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_lvlh_edge_equ() {
    sim_derived_state::lvlh_equ().run_and_assert_parity::<astrodyn::Earth>();
}
