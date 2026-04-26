//! Mission epoch recipes.
//!
//! Each function returns a [`SimulationTime`] anchored at a named
//! reference epoch.
//!
//! ```
//! use jeod_sim::recipes::epoch;
//! let t = epoch::j2000();
//! assert!(t.tai_tjt > 10_000.0);
//! ```

use crate::SimulationTime;
use jeod_time::leap_second::default_leap_second_table;

/// J2000 reference epoch: 2000-01-01 11:58:55.816 TAI (12:00:00 TT).
pub fn j2000() -> SimulationTime {
    SimulationTime::at_j2000(default_leap_second_table())
}

/// Epoch from a TAI truncated Julian time (days since 1969-12-24 00:00:00).
///
/// Used by JEOD verification simulations to anchor reference state at
/// run-specific UTC dates.
pub fn at_tai_tjt(tai_tjt: f64) -> SimulationTime {
    SimulationTime::new(tai_tjt, default_leap_second_table())
}

/// Clementine lunar mission epoch: 1994-02-19 00:00:00 UTC.
///
/// Anchors `crates/jeod_runner/examples/earth_moon.rs` and the
/// `tier3_sim_earth_moon` Tier 3 case.
pub fn clementine_1994() -> SimulationTime {
    // TAI-UTC = 28s in 1994. tai_tjt encodes the offset internally.
    at_tai_tjt(8_815.000_324_074_073)
}

/// Dawn-at-Mars epoch: 2009-02-17 23:00:00 UTC (TAI-UTC = 34s).
///
/// Anchors `crates/jeod_runner/examples/mars_orbit.rs`.
pub fn dawn_mars_2009() -> SimulationTime {
    at_tai_tjt(14_879.958_727)
}
