//! Tier 3: SIM_LVLH edge-case cross-validation
//!
//! RUN_ecc: Eccentric orbit (400 km x 8000 km) — varying orbital rate
//!          exercises LVLH frame computation at different velocities.
//! RUN_equ: Equatorial orbit (i=0) — near-singular LVLH at zero inclination.
//!
//! Point-mass Earth gravity, RK4 at the SIM_LVLH S_define step size, 24h.

use astrodyn_verif_jeod::run_verification::sim_derived_state;
use astrodyn_verif_jeod::VerificationCaseExt;

#[test]
fn tier3_simulation_lvlh_ecc() {
    sim_derived_state::lvlh_ecc().run_and_assert();
}

#[test]
fn tier3_simulation_lvlh_equ() {
    sim_derived_state::lvlh_equ().run_and_assert();
}
