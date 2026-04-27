//! Tier 3: SIM_3_ORBIT — flat-plate SRP + conical Earth shadow, GEO
//! orbit, ~23 days.
//!
//! Migrated from a 347-line bespoke `SunTable` interpolation loop to
//! this one-liner using the simulation's auto-ephemeris path (#162).
//! The recipe lives in
//! `jeod_runner::run_verification::sim_srp::srp_orbit_trajectory`;
//! the Sun source is wired to DE421 via `set_source_ephemeris` so the
//! simulation refreshes Sun position every internal step (matching
//! JEOD's 1 s cadence) without needing a `pre_step` hook.

use jeod_runner::run_verification::sim_srp;
use jeod_runner::VerificationCaseExt;

#[test]
fn tier3_simulation_srp_flat_plate() {
    sim_srp::srp_orbit_trajectory().run_and_assert();
}
