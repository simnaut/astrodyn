//! Tier 3: SIM_GJ_test — Gauss-Jackson cross-validation against JEOD.
//!
//! Cross-validates our Gauss-Jackson (Störmer-Cowell) integrator against
//! JEOD's implementation on a circular orbit using an artificial μ
//! (5.76e14 — see [`sim_gj::MU_GJ_TEST`]).
//!
//! Each `#[test]` is a one-liner over a recipe in
//! [`crate::run_verification::sim_gj`], driven through
//! [`VerificationCaseExt::run_and_assert`]. The recipe carries the
//! scenario constructor, CSV reference, and per-component tolerances
//! verbatim; this file's only job is to dispatch the variant. Both
//! the runner-vs-JEOD oracle (this file) and the runner-vs-bevy parity
//! sibling (`bevy_parity_gj.rs`) drive identical scenarios from the same
//! source-of-truth recipes.
//!
//! Variants:
//! - [`tier3_simulation_gj_order8`] — baseline GJ order 8, dt=1 s
//! - [`tier3_simulation_gj_order4`] — GJ order 4, dt=1 s
//! - [`tier3_simulation_gj_order12`] — GJ order 12, dt=1 s
//! - [`tier3_simulation_gj_dt10`] — GJ order 8 with `time_scale_factor=10`

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
