//! Tier 3: SIM_dyncomp RUN_7A–7D — Spherical harmonics + Sun/Moon
//! third-body (± drag).
//!
//! Migrated from a 394-line bespoke parameterized loop to four recipe
//! one-liners (#162). Recipes live in
//! `astrodyn_verif_jeod::run_verification::sim_dyncomp::run7{a,b,c,d}_*`; the
//! per-step DE421 update is shared via `run7_pre_step`.

use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_verif_jeod::VerificationCaseExt;

#[test]
fn tier3_simulation_run7a_sh4x4_3rd_body() {
    sim_dyncomp::run7a_sh4x4_3rd_body().run_and_assert();
}

#[test]
fn tier3_simulation_run7b_sh8x8_3rd_body() {
    sim_dyncomp::run7b_sh8x8_3rd_body().run_and_assert();
}

#[test]
fn tier3_simulation_run7c_sh4x4_3rd_body_drag() {
    sim_dyncomp::run7c_sh4x4_3rd_body_drag().run_and_assert();
}

#[test]
fn tier3_simulation_run7d_sh8x8_3rd_body_drag() {
    sim_dyncomp::run7d_sh8x8_3rd_body_drag().run_and_assert();
}
