// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Bevy ↔ runner parity for the nine Bevy-mechanism SRP stress
//! scenarios, via [`VerificationCaseParityExt::run_and_assert_parity`]
//! (issue #395 sub-task B).
//!
//! Each scenario is captured as a `sim_srp::*` recipe in
//! `crates/astrodyn_verif_jeod/src/run_verification/sim_srp.rs`. The
//! recipes share an in-memory 100-step / 10 s cadence
//! (`CsvReference::SyntheticTimes`) — these scenarios deliberately
//! do not pair with a JEOD reference trajectory; only runner ↔ bevy
//! bit-identity is asserted.
//!
//! Pre-#395 this file hand-rolled the same nine scenarios by directly
//! mutating `astrodyn_runner::Simulation` and a Bevy `App` in
//! parallel (~880 lines). The migration depends on three fixes
//! landed alongside the recipe additions:
//!
//! - `GravitySourceEntry::marker_only` so `populate_app` spawns the
//!   mu=0 Sun as a `SunMarker`-only entity matching the hand-rolled
//!   shape.
//! - `VehicleConfigBevyExt::spawn_bevy` lowering of `cfg.drag`,
//!   `cfg.srp` (FlatPlate / Cannonball), and `cfg.shadow_body` onto
//!   the corresponding Bevy components (`DragConfigC`,
//!   `FlatPlateConfigC` / `CannonballSrpC`, `ShadowBodyC` on the
//!   referenced source). Without this, recipe-driven scenarios
//!   silently lost drag / SRP / shadow on the Bevy side and parity
//!   diverged by ~68 bits per tick (the SRP-acceleration ULP gap).
//! - `CsvReference::SyntheticTimes` so the recipes don't need a
//!   committed CSV fixture under `test_data/`.

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
