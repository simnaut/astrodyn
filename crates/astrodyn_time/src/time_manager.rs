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
use crate::time_converter_tai_ut1::EopTable;
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
    // UDE is intentionally excluded — use get_ude_seconds(idx) for indexed access.
}

// JEOD_INV: TM.03 — time types updated in dependency order:
// TAI drives TT, UTC, GPS, MET, and UDE; TDB depends on TT; UT1 depends on TAI
// plus ut1_tai_offset; GMST depends on UT1.
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
    /// UT1-TAI offset in seconds (from IERS data). Recomputed from
    /// the installed `EopTable` on every `advance()` when one is set
    /// (via [`Self::with_eop_table`]); otherwise held constant at the
    /// value installed by [`Self::set_ut1_tai_offset`] (or the default
    /// `-tai_utc_s` computed at construction).
    pub ut1_tai_offset: f64,
    /// IERS EOP-driven UT1-TAI lookup. When `Some`, every `advance()`
    /// re-evaluates `ut1_tai_offset` at the current `tai_tjt` via
    /// [`EopTable::ut1_minus_tai_seconds`] (linear interpolation
    /// between adjacent daily samples), matching JEOD
    /// `time_converter_tai_ut1::convert_a_to_b`. `None` mirrors
    /// JEOD's `override_data_table=true` mode where a caller-supplied
    /// constant offset stays put across the run.
    eop_table: Option<EopTable>,
    /// Cached UTC TJT at simulation epoch (constant, avoids repeated leap-second lookup).
    utc_tjt_at_epoch: f64,

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
    ///
    /// `ut1_tai_offset` is initialised to `-(TAI - UTC)` (i.e.
    /// UT1 ≈ UTC at the epoch). For sub-second-correct UT1 across a
    /// run, install the IERS EOP table via
    /// [`Self::with_eop_table`](Self::with_eop_table); otherwise the
    /// offset stays constant across `advance()` calls.
    pub fn new(tai_tjt_at_epoch: f64, leap_table: LeapSecondTable) -> Self {
        let tai_utc_s = leap_table.tai_utc_at_tai_tjt(tai_tjt_at_epoch);
        let ut1_tai_offset = -tai_utc_s;
        let utc_tjt_at_epoch = leap_table.tai_to_utc_tjt(tai_tjt_at_epoch);

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
            eop_table: None,
            utc_tjt_at_epoch,
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

    /// Install an IERS EOP table so `ut1_tai_offset` is interpolated
    /// from the table on every `advance()` (and recomputed at install
    /// time for the current `tai_tjt`). Mirrors JEOD's default
    /// behaviour where `override_data_table=false` and the converter
    /// reads from `when_vec`/`val_vec` per call.
    ///
    /// Pair with [`crate::default_eop_table`] for the bundled IERS EOP
    /// 14 C04 fixture.
    // JEOD_INV: TM.42 — UT1-TAI is interpolated from the IERS EOP table
    // (not held constant); fail-loud out of range — see EopTable::with_clamp_out_of_range.
    pub fn with_eop_table(mut self, eop: EopTable) -> Self {
        self.ut1_tai_offset = eop.ut1_minus_tai_seconds(self.tai_tjt);
        self.eop_table = Some(eop);
        self.update_all();
        self
    }

    /// Whether a per-step IERS EOP interpolation is active. `false`
    /// means `ut1_tai_offset` is held constant (JEOD
    /// `override_data_table=true` mode).
    pub fn has_eop_table(&self) -> bool {
        self.eop_table.is_some()
    }

    /// Set the UT1-TAI offset in seconds (from IERS bulletin data).
    ///
    /// Mirrors JEOD's `override_data_table=true` mode: the explicit
    /// override **disables** any installed EOP table so the constant
    /// stays put across the run. Re-install the table via
    /// [`Self::with_eop_table`] to resume interpolation.
    pub fn set_ut1_tai_offset(&mut self, offset_seconds: f64) {
        self.ut1_tai_offset = offset_seconds;
        self.eop_table = None;
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

        // JEOD_INV: TM.42 — re-interpolate UT1-TAI from the IERS EOP
        // table at the new TAI before propagating derived scales.
        // Mirrors `time_converter_tai_ut1::convert_a_to_b` running
        // every step. Held constant when `eop_table` is None.
        if let Some(ref eop) = self.eop_table {
            self.ut1_tai_offset = eop.ut1_minus_tai_seconds(self.tai_tjt);
        }

        self.update_all();
    }

    /// Retrieve the value of a specific time scale in seconds.
    ///
    /// For MET, panics if MET has not been registered. Use
    /// [`Self::get_met_seconds`] for an `Option`-returning variant.
    /// For UDE scales, use [`Self::get_ude_seconds`] with an explicit index.
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
            TimeScaleId::MET => {
                self.met
                    .as_ref()
                    .expect("MET scale not registered; call add_met() first")
                    .seconds
            }
        }
    }

    /// Retrieve MET seconds, or `None` if MET is not registered.
    pub fn get_met_seconds(&self) -> Option<f64> {
        self.met.as_ref().map(|m| m.seconds)
    }

    /// Retrieve UDE seconds by index, or `None` if the index is out of range.
    pub fn get_ude_seconds(&self, idx: usize) -> Option<f64> {
        self.ude.get(idx).map(|u| u.seconds)
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

    // JEOD_INV: TM.03 — time types updated in dependency order
    // (TAI -> TT -> TDB -> UTC -> GPS -> UT1 -> GMST -> MET -> UDE);
    // GPS depends only on TAI, so it is computed before UT1/GMST.
    /// Recompute all derived time scales from current TAI state.
    fn update_all(&mut self) {
        // TT = TAI + 32.184
        self.tt_seconds = time_converter_tai_tt::tai_to_tt(self.tai_seconds);

        // TDB = TT + periodic offset
        self.tdb_seconds = time_converter_tai_tdb::tai_to_tdb(self.tai_seconds, self.tai_tjt);

        // UTC via leap second table (epoch value cached at construction)
        let utc_tjt = self.leap_second_table.tai_to_utc_tjt(self.tai_tjt);
        self.utc_seconds = (utc_tjt - self.utc_tjt_at_epoch) * SECONDS_PER_DAY;

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
#[allow(
    clippy::float_cmp,
    reason = "initial-state tests assert bit-exact zero / literal field values at known epochs"
)]
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

    /// EOP wiring: when an `EopTable` is installed,
    /// `ut1_tai_offset` updates per-step from the table rather than
    /// staying constant. Sample at TAI TJT 11178 (1998-12-31; JEOD's
    /// `val_vec[13513] = -31.2824458`) and one day later (TJT 11179;
    /// `val_vec[13514] = -31.2835239` per the JEOD source). The
    /// difference (~1.08 ms/day) is exactly the per-day drift the
    /// constant-offset path missed.
    #[test]
    fn time_manager_with_eop_interpolates_per_step() {
        use crate::default_eop_table;
        // 1998-12-31 TAI TJT (matches SIM_4_common_usage epoch).
        let init_tai_tjt = 11_178.0;
        let mgr0 = TimeManager::new(init_tai_tjt, default_leap_second_table())
            .with_eop_table(default_eop_table());
        let off0 = mgr0.ut1_tai_offset;
        assert!(
            (off0 - (-31.2824458)).abs() < 1e-12,
            "EOP at TJT 11178: got {off0}"
        );

        // Advance one full day; EOP should re-interpolate to entry 13514.
        let mut mgr = mgr0.clone();
        mgr.advance(SECONDS_PER_DAY);
        let off1 = mgr.ut1_tai_offset;
        assert!(
            (off1 - (-31.2835239)).abs() < 1e-12,
            "EOP at TJT 11179 (after one-day advance): got {off1}"
        );
        assert_ne!(off0, off1, "EOP-driven offset must change across the day");
    }

    /// `set_ut1_tai_offset` must disable the EOP table (mirrors JEOD's
    /// `override_data_table=true` mode); subsequent advances hold the
    /// constant value.
    #[test]
    fn time_manager_set_ut1_tai_offset_disables_eop() {
        use crate::default_eop_table;
        let init_tai_tjt = 11_178.0;
        let mut mgr = TimeManager::new(init_tai_tjt, default_leap_second_table())
            .with_eop_table(default_eop_table());
        assert!(mgr.has_eop_table());

        mgr.set_ut1_tai_offset(-30.0);
        assert!(!mgr.has_eop_table(), "explicit override disables EOP");
        assert_eq!(mgr.ut1_tai_offset, -30.0);

        mgr.advance(SECONDS_PER_DAY);
        assert_eq!(
            mgr.ut1_tai_offset, -30.0,
            "constant offset must hold after explicit override"
        );
    }
}
