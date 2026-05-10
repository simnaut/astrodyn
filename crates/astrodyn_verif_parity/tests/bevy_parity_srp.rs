// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Bevy ↔ runner parity for the headline SRP scenarios — full-stack
//! drag + 1 SRP plate + gravity-torque, and the 6-plate flat-plate SRP
//! with shadow.
//!
//! The full SRP-family parity coverage is split across several files so
//! the `parity_coverage` meta-test's prefix-match rule treats each
//! tier3 sibling as covered:
//!
//! - `bevy_parity_srp.rs` — `srp` (this file)
//! - `bevy_parity_srp_basic.rs` — `srp_basic`
//! - `bevy_parity_srp_rk4_thermal.rs` — `srp_rk4_thermal`
//! - `bevy_parity_srp_1st_order.rs` — `srp_1st_order` (uses `pre_step`)
//! - `bevy_parity_shadow_2a.rs` — `shadow_2a`
//!
//! The recipes themselves all live in `sim_srp`; the file split is
//! organizational, driven by the coverage meta-test's name-matching.
//! See the `KNOWN_PARITY_GAPS` constant in `parity_coverage.rs` for
//! historical context.

use astrodyn_verif_jeod::run_verification::sim_srp;
use astrodyn_verif_parity::VerificationCaseParityExt;

#[test]
fn bevy_parity_srp_full_stack_sixdof() {
    sim_srp::full_stack_sixdof().run_and_assert_parity::<astrodyn::Earth>();
}

#[test]
fn bevy_parity_srp_flat_plate_srp_with_shadow() {
    sim_srp::flat_plate_with_shadow().run_and_assert_parity::<astrodyn::Earth>();
}
