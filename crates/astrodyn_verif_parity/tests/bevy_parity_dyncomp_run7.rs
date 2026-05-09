//! Bevy ↔ runner parity for SIM_dyncomp RUN_7a–d (spherical-harmonics
//! gravity + 3rd-body ephemeris updates, plus drag for c/d variants),
//! via the `VerificationCaseParityExt` trait.
//!
//! Unblocked by issue #395's `BevySimContext`: each recipe's `pre_step`
//! drives `set_source_position` for Sun + Moon at each CSV record on
//! both runtimes; bit-identity follows from identical numeric inputs.

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
