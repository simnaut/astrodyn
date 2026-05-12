//! Bevy ↔ runner parity for the 6-DOF drag analytical scenarios
//! (`tier3_sim_drag_6dof`). Wrappers land as part of #389.
//!
//! The matching tier3 file drives each scenario through the runner and
//! asserts a closed-form drag property (monotonic specific-orbital-energy
//! loss under ballistic drag with rotation; attitude-invariance of the
//! ballistic-drag translational trajectory). This file pairs each recipe
//! with the parity trait so the same scenarios also run through the
//! Bevy adapter and assert `runner ↔ bevy` bit-identity at every
//! synthetic record — the second half of the
//! `runner ↔ JEOD ≈ bevy` transitivity argument the issue's matrix
//! covers.

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
