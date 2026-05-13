//! Bevy ↔ runner parity for the 6-DOF drag analytical scenarios
//! (`tier3_sim_drag_6dof`).
//!
//! The matching tier3 file drives each scenario through the runner and
//! asserts a closed-form drag property (monotonic specific-orbital-energy
//! loss under ballistic drag with rotation; attitude-invariance of the
//! ballistic-drag translational trajectory). This file pairs each recipe
//! with the parity trait so the same scenarios also run through the
//! Bevy adapter and assert `runner ↔ bevy` bit-identity at every
//! synthetic record. Bit-identity here plus the analytical assertions in
//! the sibling tier3 file together imply the Bevy adapter satisfies the
//! same closed-form drag properties — the analytical analog of the
//! `runner ↔ bevy` (this file) + `runner ↔ JEOD` (sibling tier3
//! assertions) ⇒ `bevy ↔ JEOD` transitivity argument the issue's matrix
//! covers for CSV-backed scenarios, within the runner's tolerance.

use astrodyn_verif_jeod::run_verification::sim_drag_6dof;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_drag_6dof_drag_with_rotation_energy_loss() {
    sim_drag_6dof::drag_with_rotation_energy_loss().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_drag_6dof_drag_attitude_invariance_identity() {
    sim_drag_6dof::drag_attitude_invariance_identity().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_drag_6dof_drag_attitude_invariance_rotated() {
    sim_drag_6dof::drag_attitude_invariance_rotated().run_and_assert_parity::<astrodyn::Earth>();
}
