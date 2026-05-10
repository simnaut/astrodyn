//! Bevy ↔ runner parity for the SIM_simple_attach_detach 3-body
//! trajectory, via [`VerificationCaseParityExt::run_and_assert_parity`]
//! (#395 sub-task A).
//!
//! Pre-#395 this file hand-rolled the 3-vehicle topology in both
//! runtimes (~518 lines), firing the runtime attach surfaces directly
//! and asserting per-tick bit-identity. With
//! `BevySimContext::attach` / `mark_kinematic_only` in place, the
//! scenario collapses to a one-liner over the
//! `sim_attach_detach_trajectory::simple` recipe.
//!
//! The recipe's `pre_step` schedules `attach(v1, v2)` at the record
//! advancing to `t = ATTACH_TIME = 10 s` and `mark_kinematic_only(v1)`
//! at `t = ATTACH_TIME + DT`, then runs strictly before
//! `t = DETACH_TIME = 20 s` (the dual-write fence #308 the
//! pre-#395 file-level docstring documents).

use astrodyn_verif_jeod::run_verification::sim_attach_detach_trajectory;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_attach_detach_trajectory_simple() {
    sim_attach_detach_trajectory::simple().run_and_assert_parity::<astrodyn::Earth>();
}
