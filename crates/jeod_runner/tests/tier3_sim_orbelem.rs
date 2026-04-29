#![cfg(feature = "verification")]

//! Tier 3: SIM_OrbElem cross-validation (derived_state/verif/SIM_OrbElem)
//!
//! Point-mass gravity, eccentric orbit (e=0.36), 24h, dt=0.03125s.
//! The Simulation integrates the orbit and computes orbital elements each step.

use jeod_runner::prelude::*;
use jeod_runner::run_verification::sim_derived_state;

#[test]
fn tier3_simulation_orbelem() {
    sim_derived_state::orbelem_ecc().run_and_assert();
}
