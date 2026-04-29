#![cfg(feature = "verification")]

//! Tier 3: SIM_dyncomp RUN_6B — aerodynamic drag via Simulation pipeline
//!
//! Propagates a 1 kg sphere in elliptical orbit with point-mass gravity, MET
//! atmosphere (mean solar activity), and Cd-based drag through
//! `Simulation::step()`. Compares trajectory against JEOD's logged
//! aero-trajectory CSV (position+velocity only — `aero_force` is not
//! exposed on `VehicleOutput`).

use jeod_runner::prelude::*;
use jeod_runner::run_verification::sim_dyncomp;

// non-recipe: SIM_dyncomp RUN_6B — 1 kg sphere, Cd=0.02, MET atmosphere on
// elliptical orbit; reproduces JEOD's verification fixture exactly.
#[test]
fn tier3_simulation_drag_run6b() {
    sim_dyncomp::run6b_drag_aero_traj().run_and_assert();
}
