//! Tier 3: SIM_Relative — relative state between two vehicles via Simulation pipeline.
//!
//! Validates `compute_relative_state()` against JEOD SIM_Relative reference.
//! The sim is purely kinematic (no gravity) — two bodies are propagated
//! force-free through `Simulation::step()`, and the per-step relative state
//! is asserted via [`ExtrasComparator::Relative`] against JEOD's logged
//! `rel_pos` / `rel_vel` columns. Recipes, ICs, the 57-column CSV layout,
//! and the `(rel_pos < 3.8e-5 m, rel_vel < 3.0e-6 m/s)` tolerances all
//! live in the `sim_relative` module; each test below collapses to a
//! one-liner over `run_and_assert`.

use astrodyn_verif_jeod::run_verification::sim_relative;
use astrodyn_verif_jeod::VerificationCaseExt;

#[test]
fn tier3_simulation_relative_ab_rot_ab_trans() {
    sim_relative::relative_ab_rot_ab_trans().run_and_assert();
}

#[test]
fn tier3_simulation_relative_no_rot_ab_trans() {
    sim_relative::relative_no_rot_ab_trans().run_and_assert();
}

#[test]
fn tier3_simulation_relative_a_rot_no_trans() {
    sim_relative::relative_a_rot_no_trans().run_and_assert();
}
