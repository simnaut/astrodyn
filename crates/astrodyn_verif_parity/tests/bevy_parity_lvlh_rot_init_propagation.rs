//! Bevy ↔ runner parity for SIM_dyncomp RUN_2 with LVLH-frame
//! rotational initial conditions. Validates the LVLH attitude-init
//! path is bit-identical between the two runtimes; wrapper lands as
//! part of #389.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_lvlh_rot_init_propagation() {
    sim_dyncomp::run2_lvlh_rot_init_propagation().run_and_assert_parity::<astrodyn::Earth>();
}
