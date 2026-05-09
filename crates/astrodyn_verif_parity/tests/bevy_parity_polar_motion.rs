//! Bevy ↔ runner parity for SIM_polar_motion RUN_2P (polar-motion
//! corrections applied to Earth rotation). Wrapper lands as part of
//! #389.

use astrodyn_verif_jeod::run_verification::sim_polar_motion;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn tier3_bevy_polar_motion_run2p() {
    sim_polar_motion::run2p_polar_motion().run_and_assert_parity::<astrodyn::Earth>();
}
