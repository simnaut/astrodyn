//! Bevy ↔ runner parity for SIM_dyncomp RUN_2 (3-DOF baseline ISS, point-mass
//! Earth, Rk4, 8-hour propagation), via the `VerificationCaseParityExt`
//! trait introduced for issue #389.
//!
//! This is the Phase 4 pilot wrapper: the simplest recipe-based Tier 3
//! scenario in the workspace, no rotational state, no pre_step, no
//! ephemeris. If `populate_app::<Earth>` plus the parity trait can't
//! reproduce this scenario bit-for-bit against
//! `astrodyn_runner::Simulation::step_until`, the bridge is broken at
//! its simplest configuration and nothing more elaborate will hold.
//!
//! The corresponding runner-vs-JEOD test is
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_dyncomp_run2.rs::tier3_simulation_run2_3dof`;
//! transitivity of the two assertions is the issue's stated goal.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_dyncomp_run2_3dof() {
    sim_dyncomp::run2_3dof().run_and_assert_parity::<astrodyn::Earth>();
}
