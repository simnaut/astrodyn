//! Tier 3: SIM_GJ_test — Gauss-Jackson cross-validation against JEOD.
//!
//! Cross-validates our Gauss-Jackson (Störmer-Cowell) integrator
//! against JEOD's implementation on a circular orbit. Scenario,
//! initial conditions, μ, and per-variant tolerances live in the
//! `sim_gj` recipes; each test below collapses to a one-liner over
//! `run_and_assert`.

use astrodyn_verif_jeod::run_verification::sim_gj;
use astrodyn_verif_jeod::VerificationCaseExt;

#[test]
fn tier3_simulation_gj_order8() {
    sim_gj::gj_order8().run_and_assert();
}

#[test]
fn tier3_simulation_gj_order4() {
    sim_gj::gj_order4().run_and_assert();
}

#[test]
fn tier3_simulation_gj_order12() {
    sim_gj::gj_order12().run_and_assert();
}

#[test]
fn tier3_simulation_gj_dt10() {
    sim_gj::gj_dt10().run_and_assert();
}
