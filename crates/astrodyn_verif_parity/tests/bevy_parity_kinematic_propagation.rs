//! Bevy ↔ runner parity for kinematic-state propagation, via
//! [`VerificationCaseParityExt::run_and_assert_parity`] (#395 sub-task A).
//!
//! Pre-#395 this file hand-rolled the parent + kinematic-child
//! topology in both runtimes (~515 lines), driving the runtime attach
//! surfaces directly and asserting per-tick bit-identity. With the
//! `BevySimContext::attach` / `mark_kinematic_only` plumbing in place,
//! the same scenario collapses to a one-liner over the
//! `sim_kinematic_propagation::simple_chain` recipe.
//!
//! The recipe's `pre_step` closure schedules the attach at record 1
//! (`t = DT = 0.1 s`) and the `mark_kinematic_only` transition at
//! record 2 (`t = 0.2 s`), preserving the hand-rolled test's
//! tick-1 / tick-2 separation that avoids the
//! `composite_mass_system → staging_system` race documented in the
//! pre-#395 file-level docstring.

use astrodyn_verif_jeod::run_verification::sim_kinematic_propagation;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_kinematic_propagation_simple_chain() {
    sim_kinematic_propagation::simple_chain().run_and_assert_parity::<astrodyn::Earth>();
}
