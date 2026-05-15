//! Negative tests for `Simulation::set_earth_rnp_refresh_cadence` —
//! the runtime entry point that materializes PF.06 (RNP refresh
//! cadence configuration). Pins the `assert!(cadence_s.is_finite() &&
//! cadence_s >= 0.0)` guard so a non-finite or negative cadence fails
//! loudly at the API boundary instead of silently corrupting the
//! `earth_rnp_cache` reuse-vs-recompute decision in
//! `Simulation::refresh_earth_rnp`.
//!
//! A NaN cadence would make every `(elapsed - last_refresh) >=
//! cadence` comparison false (NaN comparisons are always false), so
//! the cached matrix would be served forever; a negative cadence
//! would invert the reuse window. Both cases produce wrong RNP
//! transforms with no visible numeric NaN at the per-step ephemeris
//! site, which is the silent-corruption failure mode the catalog
//! row PF.06 guards against.

use astrodyn::SimulationTime;
use astrodyn_runner::Simulation;

/// Pins the `cadence_s.is_finite()` half of the PF.06 guard. NaN
/// would make the `(elapsed - last) >= cadence` comparison in
/// `refresh_earth_rnp` always false, freezing the cached matrix.
// JEOD_INV: PF.06 — negative test: non-finite RNP cadence rejected
#[test]
#[should_panic(expected = "cadence must be finite and >= 0")]
fn pf_06_panics_on_non_finite_rnp_cadence() {
    // JEOD_INV: PF.06 — NaN cadence at `set_earth_rnp_refresh_cadence` entry.
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, 60.0);
    sim.set_earth_rnp_refresh_cadence(f64::NAN);
}

/// Pins the `cadence_s >= 0.0` half of the PF.06 guard. A negative
/// cadence is always a caller error — the reuse window only makes
/// physical sense as a non-negative simulated-seconds interval.
// JEOD_INV: PF.06 — negative test: negative RNP cadence rejected
#[test]
#[should_panic(expected = "cadence must be finite and >= 0")]
fn pf_06_panics_on_negative_rnp_cadence() {
    // JEOD_INV: PF.06 — negative cadence at `set_earth_rnp_refresh_cadence` entry.
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, 60.0);
    sim.set_earth_rnp_refresh_cadence(-1.0);
}
