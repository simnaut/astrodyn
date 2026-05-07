#![cfg(feature = "verification")]

//! Tier 3: SIM_dyncomp RUN_5A — MET atmosphere via Simulation pipeline
//!
//! Propagates an ISS-like elliptical orbit with point-mass gravity and MET
//! atmosphere through `Simulation::step()`, comparing atmosphere density
//! against JEOD reference trajectory at each checkpoint.
//!
//! RUN_5A: minimum solar activity (F10.7=70, Ap=0). Drag is disabled in JEOD's
//! RUN_5A config, so the atmosphere has no effect on the trajectory. We still
//! configure it to validate that our MET density computation matches JEOD's.

use astrodyn_verif_jeod::VerificationCaseExt;
use astrodyn_verif_jeod::run_verification::sim_dyncomp;

#[test]
fn tier3_simulation_met_run5a() {
    sim_dyncomp::run5a_met().run_and_assert();
}
