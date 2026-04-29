#![cfg(feature = "verification")]

//! Tier 3: SIM_dyncomp RUN_6B with non-identity structural-to-body rotation
//!
//! Same scenario as `tier3_sim_drag_verif` (1 kg sphere, elliptical orbit,
//! point-mass gravity, MET atmosphere, Cd=0.02) but with a 15-degree eigen
//! rotation about [1,1,1] (normalized) applied to the structural-to-body
//! transform. For ballistic drag on a sphere, the inertial-frame force is
//! mathematically invariant under structural rotation, so the trajectory
//! should match the identity case. Any divergence indicates a frame-transform
//! bug in the force/torque collection pipeline.
//!
//! Issue: #14

use jeod_runner::prelude::*;
use jeod_runner::run_verification::sim_dyncomp;

// non-recipe: SIM_dyncomp RUN_6B with structural-to-body rotation; the test
// content is the rotation-frame invariance, not a recipe vehicle.
#[test]
fn tier3_simulation_drag_run6b_rotated() {
    sim_dyncomp::run6b_drag_rotated_struct().run_and_assert();
}
