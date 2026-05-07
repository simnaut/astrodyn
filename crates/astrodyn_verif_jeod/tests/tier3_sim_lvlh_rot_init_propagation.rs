//! Tier 3: SIM_dyncomp RUN_2 with `BodyAction::InitLvlhRot` post-init
//! propagation.
//!
//! Closes the Tier 3 gap noted on the predecessor LVLH-rot-init port.
//! The sibling [`sim_dyncomp::run2_6dof`] reads JEOD's t=0 quaternion
//! straight off `dyncomp_run2_state.csv`, so the rotational
//! initializer never runs in production tests. This case instead
//! computes the t=0 attitude by feeding JEOD's
//! `Modified_data/state.py` Yaw-Pitch-Roll Euler triple +
//! LVLH-relative angular velocity through
//! `BodyAction::InitLvlhRot.apply_rotational()` (which delegates to
//! `astrodyn_dynamics::body_init::init_rot_from_lvlh`), then propagates
//! 8 hours under point-mass gravity and compares the trajectory
//! against the existing JEOD reference CSV.
//!
//! Per CLAUDE.md "Computational Independence": JEOD source files are
//! permitted as initial-condition inputs (`Modified_data/state.py`);
//! JEOD's CSV output is consumed only as the reference for
//! comparison, never fed back into our integration.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_jeod::VerificationCaseExt;

#[test]
fn tier3_simulation_run2_lvlh_rot_init_propagation() {
    sim_dyncomp::run2_lvlh_rot_init_propagation().run_and_assert();
}
