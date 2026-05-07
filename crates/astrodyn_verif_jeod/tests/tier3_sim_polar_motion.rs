#![cfg(feature = "verification")]

//! Tier 3: Polar motion regression check (point-mass gravity).
//!
//! Validates that enabling `Simulation::polar_motion` does not break
//! point-mass propagation. With point-mass gravity (`t_inertial_pfix: None`),
//! the planet-fixed rotation is never used, so polar motion has zero
//! trajectory effect — errors should match RUN_2 exactly.

use astrodyn_verif_jeod::VerificationCaseExt;
use astrodyn_verif_jeod::run_verification::sim_polar_motion;

#[test]
fn tier3_simulation_run2p_polar_motion() {
    sim_polar_motion::run2p_polar_motion().run_and_assert();
}
