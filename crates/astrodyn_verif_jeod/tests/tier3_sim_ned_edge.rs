//! Tier 3: SIM_NED edge-case cross-validation
//!
//! RUN_ell_polar: Polar orbit on ellipsoidal Earth — geodetic singularity at poles
//! RUN_sph_inc:   Inclined orbit on spherical Earth — validates spherical geodetic path
//! RUN_sph_polar: Polar orbit on spherical Earth — combines both edge cases
//!
//! All use point-mass gravity, RNP rotation (polar motion disabled),
//! RK4 at NED_DT=1.0s, 24h.
//! Epoch: 1991-01-01 00:00:00 UTC (same as existing SIM_NED RUN_ell_inc).

use astrodyn_verif_jeod::run_verification::sim_derived_state;
use astrodyn_verif_jeod::VerificationCaseExt;

#[test]
fn tier3_simulation_ned_polar() {
    sim_derived_state::ned_ell_polar().run_and_assert();
}

#[test]
fn tier3_simulation_ned_sph_inc() {
    sim_derived_state::ned_sph_inc().run_and_assert();
}

#[test]
fn tier3_simulation_ned_sph_polar() {
    sim_derived_state::ned_sph_polar().run_and_assert();
}
