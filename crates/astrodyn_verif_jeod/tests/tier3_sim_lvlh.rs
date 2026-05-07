#![cfg(feature = "verification")]

//! Tier 3: SIM_LVLH cross-validation (derived_state/verif/SIM_LVLH)
//!
//! Point-mass gravity, 400 km circular LEO (i=45 deg), 24h.
//! The Simulation integrates and computes LVLH frame each step.

use astrodyn_runner::prelude::*;
use astrodyn_runner::run_verification::sim_derived_state;

#[test]
fn tier3_simulation_lvlh() {
    sim_derived_state::lvlh_inc().run_and_assert();
}
