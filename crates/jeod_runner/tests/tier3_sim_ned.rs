#![cfg(feature = "verification")]

//! Tier 3: SIM_NED cross-validation (derived_state/verif/SIM_NED)
//!
//! Matches the JEOD SIM_NED configuration:
//!   - Epoch: 1991-01-01 00:00:00 UTC (TAI-UTC=26s, UT1-TAI=-25.3812215s)
//!   - Gravity: point-mass (JEOD veh_config.py sets spherical=1)
//!   - RNP: precession + nutation + GAST (polar motion disabled)
//!   - Integration: RK4 at 1.0s step
//!
//! Validates the full Simulation pipeline: orbit integration -> RNP rotation
//! -> geodetic coordinate conversion, compared against JEOD CSV values.

use jeod_runner::prelude::*;
use jeod_runner::run_verification::sim_derived_state;

#[test]
fn tier3_simulation_geodetic() {
    sim_derived_state::ned_ell_inc().run_and_assert();
}
