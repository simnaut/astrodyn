// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Bevy ↔ runner parity for the SRP family (flat-plate, shadow,
//! cannonball, derivative-class thermal).
//!
//! Each `#[test]` is a one-liner over a `sim_srp::*` recipe; the
//! recipes encode the synthetic-Sun scenarios that used to live in
//! this file as hand-rolled `App` + `Simulation` setups. The bit-
//! identity contract (per-component `f64::to_bits()` equality at every
//! reference checkpoint) replaces the bespoke `assert_sixdof_eq` /
//! `assert_trans_eq` helpers.
//!
//! The pre-migration `tier3_bevy_srp_derivative_rk4_with_rotated_struct_frame`
//! test additionally asserted that the offset-plate scenario produces a
//! detectable angular-velocity change. That sanity check is a
//! fixture-validity guard, not a math check — bit-identity between the
//! two runtimes is the contract this file exists to assert. The recipe
//! still uses the same offset-plate geometry, so the underlying
//! frame-conversion code path is exercised on both runtimes; if one
//! drifted it would land here as a per-component bit mismatch.

use astrodyn_verif_jeod::run_verification::sim_srp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn tier3_bevy_full_stack_sixdof() {
    sim_srp::full_stack_sixdof().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_flat_plate_srp_with_shadow() {
    sim_srp::flat_plate_with_shadow().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_shadow_2a_annular() {
    sim_srp::shadow_2a_annular().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_shadow_2a_cooling() {
    sim_srp::shadow_2a_cooling().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_srp_basic_default() {
    sim_srp::srp_basic_default().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_srp_basic_varied_cr() {
    sim_srp::srp_basic_varied_cr().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_srp_derivative_first_order() {
    sim_srp::srp_derivative_first_order().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_srp_derivative_rk4() {
    sim_srp::srp_derivative_rk4().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn tier3_bevy_srp_derivative_rk4_with_rotated_struct_frame() {
    sim_srp::srp_derivative_rk4_rotated_struct().run_and_assert_parity::<astrodyn::Earth>();
}
