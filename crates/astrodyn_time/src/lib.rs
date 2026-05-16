//! Time scales, leap seconds, calendar dates, and the time manager.
//!
//! Pure-Rust port of JEOD's `models/environment/time/`. The crate models the
//! standard astronomical and engineering time scales (TAI, UTC, UT1, TT,
//! TDB, GPS, GMST), the leap-second table that ties UTC to TAI, and the
//! per-simulation user-defined epoch (UDE) and mission-elapsed time (MET)
//! that mission code typically logs against.
//!
//! ## Public surface
//!
//! - [`TimeManager`] and [`TimeScaleId`] — the orchestrator that owns the
//!   currently registered time scales and answers conversions between
//!   them. Mirrors JEOD's `TimeManager`.
//! - [`SimulationTime`] — the resource the integration loop advances each
//!   step; downstream consumers (gravity, ephemeris, atmosphere) read from
//!   it rather than tracking their own clocks.
//! - [`DynamicTime`] — the dynamics-frame time state passed through the
//!   integrator.
//! - [`LeapSecondTable`] — parser/lookup for JEOD's
//!   `models/environment/time/data/Leap_Second.dat`. Required for any
//!   simulation that hits a UTC↔TAI conversion.
//! - **Calendar / epoch types**: [`CalendarDate`] and [`UTC_EPOCH_TAI_TJT`]
//!   from [`time_utc`], [`UserDefinedEpoch`] from [`time_ude`],
//!   [`MissionElapsedTime`] from [`time_met`], [`GpsTimeComponents`] and
//!   [`TAI_GPS_OFFSET`] from [`time_gps`], plus the [`epoch`] module.
//! - **Per-pair time converters**: [`time_converter_tai_tdb`],
//!   [`time_converter_tai_tt`], [`time_converter_tai_ut1`], and
//!   [`time_converter_ut1_gmst`] each port one of JEOD's
//!   `TimeConverter_*` classes. GMST in particular drives Earth's
//!   body-fixed rotation in `astrodyn_frames`. The TAI↔UT1 converter
//!   is backed by an [`EopTable`] (IERS EOP 14 C04 series, daily
//!   linear interpolation) reachable via [`default_eop_table`].
//!
//! JEOD source: `models/environment/time/` (and the
//! `models/environment/time/data/` subdirectory for `Leap_Second.dat`). Pure
//! Rust, zero Bevy dependency.
//!
//! ## Example
//!
//! Build a calendar date and round-trip it through the truncated Julian
//! representation that the per-scale converters consume internally:
//!
//! ```
//! use astrodyn_time::time_utc::{calendar_to_tjt, tjt_to_calendar, CalendarDate};
//!
//! let cal = CalendarDate::new(2025, 1, 1, 0, 0, 0.0);
//! let tjt = calendar_to_tjt(&cal);
//! let back = tjt_to_calendar(tjt);
//! assert_eq!(back.year, cal.year);
//! assert_eq!(back.month, cal.month);
//! assert_eq!(back.day, cal.day);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub use astrodyn_quantities::prelude::*;

pub mod epoch;
pub mod leap_second;
pub mod simulation_time;
pub mod time_converter_tai_tdb;
pub mod time_converter_tai_tt;
pub mod time_converter_tai_ut1;
pub mod time_converter_ut1_gmst;
pub mod time_dyn;
pub mod time_gps;
pub mod time_manager;
pub mod time_met;
pub mod time_ude;
pub mod time_utc;

pub use epoch::*;
pub use leap_second::LeapSecondTable;
pub use simulation_time::SimulationTime;
pub use time_converter_tai_ut1::{default_eop_table, EopLoadError, EopTable};
pub use time_dyn::DynamicTime;
pub use time_gps::{GpsTimeComponents, TAI_GPS_OFFSET};
pub use time_manager::{TimeManager, TimeScaleId};
pub use time_met::MissionElapsedTime;
pub use time_ude::UserDefinedEpoch;
pub use time_utc::{CalendarDate, UTC_EPOCH_TAI_TJT};
