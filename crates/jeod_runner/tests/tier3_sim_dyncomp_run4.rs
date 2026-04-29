#![cfg(feature = "verification")]

//! Tier 3: SIM_dyncomp RUN_4 — Spherical Earth + Sun/Moon third-body.
//!
//! Migrated from a 280-line bespoke per-step ephemeris-update loop to
//! this one-liner using the `pre_step` hook (#162). The recipe lives in
//! `jeod_runner::run_verification::sim_dyncomp::run4_3rd_body`; the
//! per-step DE421 update is its `pre_step` factory.

use jeod_runner::run_verification::sim_dyncomp;
use jeod_runner::VerificationCaseExt;

#[test]
fn tier3_simulation_run4_3rd_body() {
    sim_dyncomp::run4_3rd_body().run_and_assert();
}
