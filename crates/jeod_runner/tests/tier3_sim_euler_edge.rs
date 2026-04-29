#![cfg(feature = "verification")]

//! Tier 3: SIM_Euler edge-case cross-validation
//!
//! RUN_ecc: Eccentric orbit (400 km x 8000 km altitude) — varying orbital rate
//!          exercises Euler angle computation at different angular velocities.
//! RUN_equ: Equatorial orbit (i=0) — exercises gimbal-lock-adjacent sequences.
//!
//! Both use point-mass Earth gravity, RK4 at the SIM_Euler S_define step size, 24h duration.
//! Euler angles are validated against JEOD's logged quaternion data.

use jeod_runner::prelude::*;
use jeod_runner::run_verification::sim_derived_state;

#[test]
fn tier3_simulation_euler_ecc() {
    sim_derived_state::euler_ecc().run_and_assert();
}

#[test]
fn tier3_simulation_euler_equ() {
    sim_derived_state::euler_equ().run_and_assert();
}
