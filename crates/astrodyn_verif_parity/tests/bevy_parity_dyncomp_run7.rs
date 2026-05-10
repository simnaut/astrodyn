//! Bevy ↔ runner parity for SIM_dyncomp RUN_7A–RUN_7D (4×4 / 8×8
//! spherical-harmonic Earth + DE421 Sun/Moon third-body, ± MET drag).
//! All four variants share `run7_pre_step` — closed by #395 once the
//! Bevy `AppSimContext::set_source_position` bridge landed.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_dyncomp_run7a_sh4x4_3rd_body() {
    sim_dyncomp::run7a_sh4x4_3rd_body().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_run7b_sh8x8_3rd_body() {
    sim_dyncomp::run7b_sh8x8_3rd_body().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_run7c_sh4x4_3rd_body_drag() {
    sim_dyncomp::run7c_sh4x4_3rd_body_drag().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_dyncomp_run7d_sh8x8_3rd_body_drag() {
    sim_dyncomp::run7d_sh8x8_3rd_body_drag().run_and_assert_parity::<astrodyn::Earth>();
}
