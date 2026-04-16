//! Time Manager — dependency-ordered conversion between time scales.
//!
//! Ported from JEOD `time_manager.cc`.
//!
//! This Rust implementation does not maintain a dynamic converter graph.
//! Instead, `TimeManager` stores the supported time scales in a flat struct
//! and recomputes derived values in a fixed dependency order whenever TAI
//! advances, keeping the registered scales in sync.

use crate::epoch::{J2000_NOON_TJT, J2000_TAI_TJT, SECONDS_PER_DAY, TAI_TT_OFFSET};
use crate::leap_second::LeapSecondTable;
use crate::time_converter_tai_tdb;
use crate::time_converter_tai_tt;
use crate::time_converter_ut1_gmst;
use crate::time_dyn::DynamicTime;
use crate::time_gps;
use crate::time_met::MissionElapsedTime;
use crate::time_ude::UserDefinedEpoch;

/// Identifies a time scale in the converter graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeScaleId {
    /// International Atomic Time — the master clock.
    TAI,
    /// Terrestrial Time: TAI + 32.184s.
    TT,
    /// Barycentric Dynamical Time: TT + periodic correction.
    TDB,
    /// Coordinated Universal Time: TAI - leap_seconds.
    UTC,
    /// Universal Time 1: TAI + UT1-TAI offset (IERS data).
    UT1,
    /// Greenwich Mean Sidereal Time.
    GMST,
    /// GPS Time: TAI - 19s.
    GPS,
    /// Mission Elapsed Time (user-defined epoch, optional hold).
    MET,
    /// Dynamic Time (integration clock).
    DYN,
    /// User-Defined Epoch time.
    UDE,
}

// JEOD_INV: TM.03 — time types updated in dependency order (TAI -> TT -> TDB -> UTC -> UT1 -> GMST -> GPS -> DYN -> MET -> UDE)
// JEOD_INV: TM.04 — all time types reachable from initializer (all scales updated from TAI in update_all)
// JEOD_INV: TM.06 — no duplicate converters (each scale has exactly one conversion path)
/// Time manager: maintains all time scale values and propagates updates.
///
/// Mirrors JEOD `TimeManager` but uses a flat struct instead of a dynamic
/// converter graph, matching the existing `SimulationTime` pattern.
/// All conversions are hardcoded in dependency order; optional scales
/// (MET, UDE) are stored as `Option`.
#[derive(Debug, Clone)]
pub struct TimeManager {
    // --- Core state (always present) ---
    /// TAI seconds elapsed since simulation start.
    pub tai_seconds: f64,
    /// TAI truncated Julian time (absolute calendar reference).
    pub tai_tjt: f64,
    /// TAI TJT at simulation start (set once, never changes).
    pub tai_tjt_at_epoch: f64,

    // --- Standard derived scales ---
    /// TT seconds: TAI + 32.184.
    pub tt_seconds: f64,
    /// TDB seconds: TT + periodic offset.
    pub tdb_seconds: f64,
    /// UTC seconds elapsed since simulation start.
    pub utc_seconds: f64,
    /// UT1 seconds: TAI + ut1_tai_offset.
    pub ut1_seconds: f64,
    /// GMST in radians (0 to 2pi).
    pub gmst_radians: f64,
    /// GMST in accumulated sidereal seconds since J2000.
    pub gmst_seconds: f64,
    /// GPS seconds: TAI - 19.
    pub gps_seconds: f64,

    // --- Infrastructure ---
    /// Leap second table for TAI-UTC conversion.
    pub leap_second_table: LeapSecondTable,
    /// UT1-TAI offset in seconds (from IERS data).
    pub ut1_tai_offset: f64,

    // --- Dynamic time ---
    /// Dynamic time state (integration clock). Private to ensure
    /// `DynamicTime` invariants are maintained — use `set_scale_factor()`
    /// to change the integration rate.
    dyn_time: DynamicTime,
    /// Raw simulation time (always advances forward).
    pub simtime: f64,

    // --- Optional scales ---
    /// Mission elapsed time (optional).
    pub met: Option<MissionElapsedTime>,
    /// User-defined epoch times (optional, can have multiple).
    pub ude: Vec<UserDefinedEpoch>,
}

impl TimeManager {
    // JEOD_INV: TM.07 — JEOD uses -1.0 sentinel; we call update_all() at construction instead
    /// Create a new TimeManager starting at the given TAI TJT.
    pub fn new(tai_tjt_at_epoch: f64, leap_table: LeapSecondTable) -> Self {
        let tai_utc_s = leap_table.tai_utc_at_tai_tjt(tai_tjt_at_epoch);
        let ut1_tai_offset = -tai_utc_s;

        let mut mgr = Self {
            tai_seconds: 0.0,
            tai_tjt: tai_tjt_at_epoch,
            tai_tjt_at_epoch,
            tt_seconds: 0.0,
            tdb_seconds: 0.0,
            utc_seconds: 0.0,
            ut1_seconds: 0.0,
            gmst_radians: 0.0,
            gmst_seconds: 0.0,
            gps_seconds: 0.0,
            leap_second_table: leap_table,
            ut1_tai_offset,
            dyn_time: DynamicTime::new(),
            simtime: 0.0,
            met: None,
            ude: Vec::new(),
        };
        mgr.update_all();
        mgr
    }

    /// Create a TimeManager starting at J2000.0 TT epoch.
    pub fn at_j2000(leap_table: LeapSecondTable) -> Self {
        Self::new(J2000_TAI_TJT, leap_table)
    }

    /// Set the UT1-TAI offset in seconds (from IERS bulletin data).
    pub fn set_ut1_tai_offset(&mut self, offset_seconds: f64) {
        self.ut1_tai_offset = offset_seconds;
        self.update_all();
    }

    /// Set the dynamic time scale factor.
    ///
    /// 1.0 = real-time, >1.0 = fast-forward, <0 = time reversal.
    /// The offset is adjusted automatically on the next `advance()` to
    /// maintain continuity.
    pub fn set_scale_factor(&mut self, factor: f64) {
        self.dyn_time.scale_factor = factor;
    }

    /// Get the current dynamic time scale factor.
    pub fn scale_factor(&self) -> f64 {
        self.dyn_time.scale_factor
    }

    /// Add a Mission Elapsed Time scale with epoch at the given TAI seconds.
    pub fn add_met(&mut self, tai_seconds_at_epoch: f64) {
        let mut met = MissionElapsedTime::new(tai_seconds_at_epoch);
        met.update(self.tai_seconds);
        self.met = Some(met);
    }

    /// Add a User-Defined Epoch time scale.
    pub fn add_ude(&mut self, epoch_in_parent: f64) -> usize {
        let idx = self.ude.len();
        let mut ude = UserDefinedEpoch::new(epoch_in_parent);
        ude.update(self.tai_seconds);
        self.ude.push(ude);
        idx
    }

    /// Advance the simulation by `sim_dt` seconds.
    ///
    /// Dynamic time (TAI, TDB, etc.) advances by `sim_dt * scale_factor`,
    /// while `simtime` always advances by raw `sim_dt`. When
    /// `dyn_time.scale_factor = -1.0`, TAI runs backward while simtime
    /// runs forward, matching `SimulationTime::advance()` behavior.
    ///
    /// # Panics
    /// Panics if `sim_dt` is not finite or is negative.
    pub fn advance(&mut self, sim_dt: f64) {
        assert!(
            sim_dt.is_finite() && sim_dt >= 0.0,
            "sim_dt must be finite and >= 0, got {sim_dt}"
        );

        // Apply any pending scale-factor change so DynamicTime's offset stays
        // consistent, then advance simtime and let DYN compute its own seconds.
        self.dyn_time.update_offset(self.simtime);
        self.simtime += sim_dt;
        self.dyn_time.update(self.simtime);

        self.tai_seconds = self.dyn_time.seconds;
        self.tai_tjt = self.tai_tjt_at_epoch + self.tai_seconds / SECONDS_PER_DAY;

        self.update_all();
    }

    /// Retrieve the value of a specific time scale in seconds.
    ///
    /// For UDE scales, this returns the first UDE. Use [`get_ude_seconds`]
    /// to access a specific UDE by index.
    pub fn get_seconds(&self, scale: TimeScaleId) -> f64 {
        match scale {
            TimeScaleId::TAI => self.tai_seconds,
            TimeScaleId::TT => self.tt_seconds,
            TimeScaleId::TDB => self.tdb_seconds,
            TimeScaleId::UTC => self.utc_seconds,
            TimeScaleId::UT1 => self.ut1_seconds,
            TimeScaleId::GMST => self.gmst_seconds,
            TimeScaleId::GPS => self.gps_seconds,
            TimeScaleId::DYN => self.dyn_time.seconds,
            TimeScaleId::MET => self.met.as_ref().map_or(0.0, |m| m.seconds),
            TimeScaleId::UDE => self.ude.first().map_or(0.0, |u| u.seconds),
        }
    }

    /// Retrieve UDE seconds by index.
    ///
    /// # Panics
    /// Panics if `idx` is out of range.
    pub fn get_ude_seconds(&self, idx: usize) -> f64 {
        self.ude[idx].seconds
    }

    /// Get a reference to a UDE by index.
    ///
    /// # Panics
    /// Panics if `idx` is out of range.
    pub fn get_ude(&self, idx: usize) -> &UserDefinedEpoch {
        &self.ude[idx]
    }

    /// TDB Julian Date (for ephemeris queries).
    pub fn tdb_julian_date(&self) -> f64 {
        let tdb_tai_offset_s = self.tdb_seconds - self.tai_seconds;
        let tdb_tjt = self.tai_tjt + tdb_tai_offset_s / SECONDS_PER_DAY;
        tdb_tjt + 40_000.0 + 2_400_000.5
    }

    /// TT truncated Julian time.
    pub fn tt_tjt(&self) -> f64 {
        self.tai_tjt + TAI_TT_OFFSET / SECONDS_PER_DAY
    }

    /// TT Julian Date.
    pub fn tt_julian_date(&self) -> f64 {
        self.tt_tjt() + 40_000.0 + 2_400_000.5
    }

    // JEOD_INV: TM.03 — time types updated in dependency order (TAI -> TT -> TDB -> UTC -> UT1 -> GMST -> GPS -> MET -> UDE)
    /// Recompute all derived time scales from current TAI state.
    fn update_all(&mut self) {
        // TT = TAI + 32.184
        self.tt_seconds = time_converter_tai_tt::tai_to_tt(self.tai_seconds);

        // TDB = TT + periodic offset
        self.tdb_seconds = time_converter_tai_tdb::tai_to_tdb(self.tai_seconds, self.tai_tjt);

        // UTC via leap second table
        let utc_tjt = self.leap_second_table.tai_to_utc_tjt(self.tai_tjt);
        let utc_tjt_at_epoch = self.leap_second_table.tai_to_utc_tjt(self.tai_tjt_at_epoch);
        self.utc_seconds = (utc_tjt - utc_tjt_at_epoch) * SECONDS_PER_DAY;

        // GPS = TAI - 19s
        self.gps_seconds = time_gps::tai_to_gps(self.tai_seconds);

        // UT1 = TAI + ut1_tai_offset
        self.ut1_seconds = self.tai_seconds + self.ut1_tai_offset;

        // GMST from UT1
        let ut1_tjt = self.tai_tjt + self.ut1_tai_offset / SECONDS_PER_DAY;
        let du = ut1_tjt - J2000_NOON_TJT;
        self.gmst_seconds = time_converter_ut1_gmst::ut1_to_gmst_seconds(du);
        self.gmst_radians = time_converter_ut1_gmst::ut1_to_gmst_radians(du);

        // MET (optional)
        if let Some(ref mut met) = self.met {
            met.update(self.tai_seconds);
        }

        // UDE (optional, multiple)
        for ude in &mut self.ude {
            ude.update(self.tai_seconds);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leap_second::default_leap_second_table;
    use std::f64::consts::PI;

    #[test]
    fn time_manager_at_j2000() {
        let mgr = TimeManager::at_j2000(default_leap_second_table());
        assert_eq!(mgr.tai_seconds, 0.0);
        assert_eq!(mgr.simtime, 0.0);
        assert!((mgr.tt_seconds - 32.184).abs() < 1e-10);
        assert!((mgr.gps_seconds - (-19.0)).abs() < 1e-15);
    }

    #[test]
    fn time_manager_advance_updates_all() {
        let mut mgr = TimeManager::at_j2000(default_leap_second_table());
        mgr.advance(3600.0);

        assert!((mgr.tai_seconds - 3600.0).abs() < 1e-15);
        assert!((mgr.tt_seconds - (3600.0 + 32.184)).abs() < 1e-10);
        assert!((mgr.gps_seconds - (3600.0 - 19.0)).abs() < 1e-15);
        assert!((mgr.simtime - 3600.0).abs() < 1e-15);
        assert!((mgr.get_seconds(TimeScaleId::DYN) - 3600.0).abs() < 1e-15);
    }

    #[test]
    fn time_manager_get_seconds() {
        let mut mgr = TimeManager::at_j2000(default_leap_second_table());
        mgr.advance(100.0);

        assert!((mgr.get_seconds(TimeScaleId::TAI) - 100.0).abs() < 1e-15);
        assert!((mgr.get_seconds(TimeScaleId::TT) - 132.184).abs() < 1e-10);
        assert!((mgr.get_seconds(TimeScaleId::GPS) - 81.0).abs() < 1e-15);
        assert!((mgr.get_seconds(TimeScaleId::DYN) - 100.0).abs() < 1e-15);
    }

    #[test]
    fn time_manager_with_met() {
        let mut mgr = TimeManager::at_j2000(default_leap_second_table());
        mgr.add_met(0.0); // MET epoch at TAI=0
        mgr.advance(500.0);
        assert!(
            (mgr.met.as_ref().unwrap().seconds - 500.0).abs() < 1e-15,
            "MET should be 500s"
        );
    }

    #[test]
    fn time_manager_with_ude() {
        let mut mgr = TimeManager::at_j2000(default_leap_second_table());
        let idx = mgr.add_ude(1000.0); // UDE epoch at TAI=1000s
        mgr.advance(1500.0);
        assert!(
            (mgr.ude[idx].seconds - 500.0).abs() < 1e-15,
            "UDE should be 500s"
        );
    }

    #[test]
    fn time_manager_gmst_at_j2000() {
        let mgr = TimeManager::at_j2000(default_leap_second_table());
        let gmst_deg = mgr.gmst_radians * 180.0 / PI;
        assert!(
            (gmst_deg - 280.19).abs() < 0.05,
            "GMST at J2000: {:.4} degrees, expected ~280.19",
            gmst_deg
        );
    }

    #[test]
    fn time_manager_tdb_julian_date() {
        let mgr = TimeManager::at_j2000(default_leap_second_table());
        let jd = mgr.tdb_julian_date();
        assert!((jd - 2_451_545.0).abs() < 0.001, "TDB JD at J2000: {}", jd);
    }

    #[test]
    fn time_manager_scale_factor() {
        let mut mgr = TimeManager::at_j2000(default_leap_second_table());
        mgr.advance(100.0);
        let tai_100 = mgr.tai_seconds;

        // Reverse time
        mgr.set_scale_factor(-1.0);
        mgr.advance(100.0);

        assert!(
            mgr.tai_seconds.abs() < 1e-15,
            "TAI should return to 0 after reversal, got {}",
            mgr.tai_seconds
        );
        let _ = tai_100;
    }

    #[test]
    fn time_manager_multiple_udes() {
        let mut mgr = TimeManager::at_j2000(default_leap_second_table());
        let idx0 = mgr.add_ude(0.0);
        let idx1 = mgr.add_ude(500.0);
        mgr.advance(1000.0);
        assert!((mgr.ude[idx0].seconds - 1000.0).abs() < 1e-15);
        assert!((mgr.ude[idx1].seconds - 500.0).abs() < 1e-15);
    }
}
