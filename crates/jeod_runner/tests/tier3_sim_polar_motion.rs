//! Tier 3: Polar motion regression check (point-mass gravity).
//!
//! Validates that enabling `Simulation::polar_motion` does not break
//! point-mass propagation. With point-mass gravity (`t_inertial_pfix: None`),
//! the planet-fixed rotation is never used, so polar motion has zero
//! trajectory effect — errors should match RUN_2 exactly.

use jeod_runner::prelude::*;
use jeod_runner::run_verification::sim_polar_motion;

#[test]
fn tier3_simulation_run2p_polar_motion() {
    sim_polar_motion::run2p_polar_motion().run_and_assert();
}
