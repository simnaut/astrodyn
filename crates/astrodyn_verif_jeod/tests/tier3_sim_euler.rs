#![cfg(feature = "verification")]

//! Tier 3: SIM_Euler cross-validation (derived_state/verif/SIM_Euler)
//!
//! Uses the RUN_2 point-mass 6-DOF trajectory (which has quaternion data)
//! to validate Euler angle computation through the Simulation pipeline.

use astrodyn_verif_jeod::run_verification::sim_derived_state;
use astrodyn_verif_jeod::VerificationCaseExt;

#[test]
fn tier3_simulation_euler() {
    sim_derived_state::euler_run2().run_and_assert();
}
