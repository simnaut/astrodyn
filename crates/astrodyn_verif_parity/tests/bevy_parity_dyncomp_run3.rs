//! Bevy ↔ runner parity for SIM_dyncomp RUN_3 (4×4 and 8×8 spherical
//! harmonics gravity, 3-DOF ISS, 8 hours). Both variants are
//! `pre_step: None` recipes already in `sim_dyncomp`; the wrapper
//! lands as part of issue #389.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_dyncomp_run3a_sh4x4() {
    sim_dyncomp::run3a_sh4x4().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_run3b_sh8x8() {
    sim_dyncomp::run3b_sh8x8().run_and_assert_parity::<astrodyn::Earth>();
}
