//! [`SimulationTime`] — the per-step time-state resource that the
//! integration loop advances each step and that downstream consumers
//! (gravity, ephemeris, atmosphere) read from.
//!
//! Mirrors JEOD's
//! [`TimeManager`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/time/src/time_manager.cc)
//! aggregate from JEOD v5.4.0. All time scales live in a single struct
//! mutated through `&mut self`, so any reader holding a shared borrow
//! observes a consistent snapshot — the exclusive borrow on the update
//! path (which calls `recompute_derived` before returning) prevents
//! mid-step partial reads.
//!
//! The 7 standard scales (TAI/TT/TDB/UTC/UT1/GMST/GPS) plus `simtime`
//! are always present. The optional [`Option<EopTable>`](EopTable)
//! drives per-step IERS UT1 interpolation; [`MissionElapsedTime`] and
//! [`UserDefinedEpoch`] are opt-in via [`SimulationTime::add_met`] and
//! [`SimulationTime::add_ude`]. The integration clock is governed by a
//! private [`DynamicTime`] reachable through
//! [`SimulationTime::set_scale_factor`].

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

/// Identifies a time scale for use with [`SimulationTime::get_seconds`].
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

// JEOD_INV: TM.04 — all time types reachable from initializer (all scales hardcoded in single struct)
// JEOD_INV: TM.05 — all time types reachable from TimeDyn (all scales updated in recompute_derived)
// JEOD_INV: TM.06 — no duplicate converters (each scale has exactly one conversion path)
/// Complete simulation time state across all time scales.
///
/// All seconds fields are measured from the TAI epoch (which coincides with
/// the simulation start). The TAI TJT (truncated Julian time) anchors
/// everything to an absolute calendar reference.
///
/// Mirrors JEOD's `TimeManager` aggregate.
#[derive(Debug, Clone)]
pub struct SimulationTime {
    /// TAI seconds elapsed since simulation start (0.0 at construction).
    pub tai_seconds: f64,
    /// TAI truncated Julian time (absolute calendar reference, advances with tai_seconds).
    pub tai_tjt: f64,
    /// TAI TJT at simulation start (set once at construction, never changes).
    pub tai_tjt_at_epoch: f64,
    /// UTC seconds elapsed since simulation start. Differs from tai_seconds by
    /// accumulated leap seconds over the span.
    pub utc_seconds: f64,
    /// UT1 seconds: `tai_seconds + ut1_tai_offset`. At construction with the
    /// default offset (−TAI_UTC), this equals approximately 0; with a custom
    /// IERS offset it may be nonzero at t=0.
    pub ut1_seconds: f64,
    /// TT seconds: `tai_seconds + 32.184`. Nonzero at t=0 (starts at 32.184).
    pub tt_seconds: f64,
    /// TDB seconds: `tai_seconds + 32.184 + periodic`. Differs from TT by a
    /// periodic correction with amplitude ~1.7 ms.
    pub tdb_seconds: f64,
    /// Greenwich Mean Sidereal Time (radians, 0 to 2π).
    pub gmst_radians: f64,
    /// GMST in accumulated sidereal seconds since J2000.
    /// Matches JEOD's `TimeGMST::seconds`.
    pub gmst_seconds: f64,
    /// GPS seconds: TAI seconds minus 19 s (constant offset per the GPS epoch
    /// 1980-01-06 definition). Starts at −19 when `tai_seconds` is 0.
    pub gps_seconds: f64,
    /// Elapsed simulation time (always advances forward, regardless of scale factor).
    pub simtime: f64,
    /// Leap second table for TAI↔UTC conversion.
    pub leap_second_table: LeapSecondTable,
    /// UT1-TAI offset in seconds (from IERS data). Recomputed from the
    /// installed `EopTable` on every `advance()` when one is set (via
    /// [`Self::with_eop_table`]); otherwise held constant at the value
    /// installed by [`Self::set_ut1_tai_offset`] (or the default
    /// `-tai_utc_s` computed at construction).
    pub ut1_tai_offset: f64,
    /// IERS EOP-driven UT1-TAI lookup. When `Some`, every `advance()`
    /// re-evaluates `ut1_tai_offset` at the current `tai_tjt` via
    /// [`EopTable::ut1_minus_tai_seconds`] (linear interpolation between
    /// adjacent daily samples), matching JEOD
    /// `time_converter_tai_ut1::convert_a_to_b`. `None` mirrors JEOD's
    /// `override_data_table=true` mode where a caller-supplied constant
    /// offset stays put across the run.
    eop_table: Option<EopTable>,
    /// Cached UTC TJT at simulation epoch (constant, avoids repeated leap-second lookup).
    utc_tjt_at_epoch: f64,
    /// Dynamic time state (integration clock). Private to ensure
    /// `DynamicTime` invariants are maintained — use
    /// [`Self::set_scale_factor`] to change the integration rate.
    dyn_time: DynamicTime,
    /// Mission elapsed time (optional). Registered via [`Self::add_met`].
    pub met: Option<MissionElapsedTime>,
    /// User-defined epoch times (optional, can have multiple). Registered via [`Self::add_ude`].
    pub ude: Vec<UserDefinedEpoch>,
}

impl SimulationTime {
    // JEOD_INV: TM.07 — JEOD uses -1.0 sentinel; we call recompute_derived() at construction instead
    // JEOD_INV: TM.21 — TAI↔UTC requires a leap-second table; `leap_table` is a mandatory arg (no override path).
    // JEOD_INV: TM.31 — sim-start is specified by the mandatory `tai_tjt_at_epoch` parameter;
    // calendar/decimal ambiguity does not exist in our API.
    /// Create a new SimulationTime starting at the given TAI TJT.
    ///
    /// `ut1_tai_offset` is initialised to `-(TAI - UTC)` (i.e. UT1 ≈ UTC at
    /// the epoch). For sub-second-correct UT1 across a run, install the IERS
    /// EOP table via [`Self::with_eop_table`]; otherwise the offset stays
    /// constant across `advance()` calls.
    ///
    /// # Arguments
    /// * `tai_tjt_at_epoch` - TAI truncated Julian time at simulation start.
    ///   Use `J2000_TAI_TJT` (≈ 11544.4996275) for a J2000 epoch start.
    /// * `leap_table` - Leap second table for TAI↔UTC conversion.
    pub fn new(tai_tjt_at_epoch: f64, leap_table: LeapSecondTable) -> Self {
        let tai_utc_s = leap_table.tai_utc_at_tai_tjt(tai_tjt_at_epoch);
        let ut1_tai_offset = -tai_utc_s;
        let utc_tjt_at_epoch = leap_table.tai_to_utc_tjt(tai_tjt_at_epoch);

        let mut sim = Self {
            tai_seconds: 0.0,
            tai_tjt: tai_tjt_at_epoch,
            tai_tjt_at_epoch,
            utc_seconds: 0.0,
            ut1_seconds: 0.0,
            tt_seconds: 0.0,
            tdb_seconds: 0.0,
            gps_seconds: 0.0,
            gmst_radians: 0.0,
            gmst_seconds: 0.0,
            simtime: 0.0,
            leap_second_table: leap_table,
            ut1_tai_offset,
            eop_table: None,
            utc_tjt_at_epoch,
            dyn_time: DynamicTime::new(),
            met: None,
            ude: Vec::new(),
        };
        sim.recompute_derived();
        sim
    }

    /// Create a SimulationTime starting at J2000.0 TT epoch.
    pub fn at_j2000(leap_table: LeapSecondTable) -> Self {
        Self::new(J2000_TAI_TJT, leap_table)
    }

    // JEOD_INV: TM.42 — UT1-TAI is interpolated from the IERS EOP table
    // (not held constant); fail-loud out of range — see EopTable::with_clamp_out_of_range.
    /// Install an IERS EOP table so `ut1_tai_offset` is interpolated from the
    /// table on every `advance()` (and recomputed at install time for the
    /// current `tai_tjt`). Mirrors JEOD's default behaviour where
    /// `override_data_table=false` and the converter reads from
    /// `when_vec`/`val_vec` per call.
    ///
    /// Pair with [`crate::default_eop_table`] for the bundled IERS EOP 14
    /// C04 fixture.
    pub fn with_eop_table(mut self, eop: EopTable) -> Self {
        self.ut1_tai_offset = eop.ut1_minus_tai_seconds(self.tai_tjt);
        self.eop_table = Some(eop);
        self.recompute_derived();
        self
    }

    /// Whether a per-step IERS EOP interpolation is active. `false` means
    /// `ut1_tai_offset` is held constant (JEOD `override_data_table=true` mode).
    pub fn has_eop_table(&self) -> bool {
        self.eop_table.is_some()
    }

    /// Set the UT1-TAI offset in seconds (from IERS bulletin data).
    ///
    /// UT1-TAI = (UT1-UTC) - (TAI-UTC). For example, if UT1-UTC = 0.3s and
    /// TAI-UTC = 32s, then UT1-TAI = 0.3 - 32 = -31.7s.
    ///
    /// Mirrors JEOD's `override_data_table=true` mode: the explicit override
    /// **disables** any installed EOP table so the constant stays put across
    /// the run. Re-install the table via [`Self::with_eop_table`] to resume
    /// interpolation.
    pub fn set_ut1_tai_offset(&mut self, offset_seconds: f64) {
        self.ut1_tai_offset = offset_seconds;
        self.eop_table = None;
        self.recompute_derived();
    }

    /// Set the dynamic time scale factor.
    ///
    /// 1.0 = real-time, >1.0 = fast-forward, <0 = time reversal. The offset
    /// is adjusted automatically on the next `advance()` to maintain continuity.
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

    // JEOD_INV: TM.03 — time types updated in dependency order via recompute_derived()
    /// Advance the simulation by `sim_dt` seconds (must be non-negative).
    ///
    /// Dynamic time (TAI, TDB, etc.) advances by `sim_dt * scale_factor`,
    /// while `simtime` always advances by raw `sim_dt`. When
    /// `scale_factor() = -1.0`, TAI runs backward while simtime runs forward,
    /// matching JEOD's `TimeDyn::scale_factor` behavior.
    ///
    /// # Panics
    /// Panics if `sim_dt` is not finite or is negative.
    pub fn advance(&mut self, sim_dt: f64) {
        // JEOD_INV: TM.40 — time advance inputs must be finite; JEOD relies on sim-input
        // validity, we assert defensively so bad `dt` values fail loudly rather than poison the tree.
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
        // table at the new TAI before propagating derived scales. Mirrors
        // `time_converter_tai_ut1::convert_a_to_b` running every step.
        // Held constant when `eop_table` is None.
        if let Some(ref eop) = self.eop_table {
            self.ut1_tai_offset = eop.ut1_minus_tai_seconds(self.tai_tjt);
        }

        self.recompute_derived();
    }

    /// Retrieve the value of a specific time scale in seconds.
    ///
    /// For MET, panics if MET has not been registered. Use
    /// [`Self::get_met_seconds`] for an `Option`-returning variant. For UDE
    /// scales, use [`Self::get_ude_seconds`] with an explicit index.
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
        // TDB TJT = TAI TJT + (TDB-TAI)/86400
        let tdb_tai_offset_s = self.tdb_seconds - self.tai_seconds;
        let tdb_tjt = self.tai_tjt + tdb_tai_offset_s / SECONDS_PER_DAY;
        // TJT -> JD: JD = TJT + 40000 + 2400000.5
        tdb_tjt + 40_000.0 + 2_400_000.5
    }

    /// TT truncated Julian time.
    ///
    /// `TT TJT = TAI TJT + TAI_TT_OFFSET / SECONDS_PER_DAY`
    ///
    /// Used by Earth rotation (RNP) computations that need the TT epoch.
    pub fn tt_tjt(&self) -> f64 {
        self.tai_tjt + TAI_TT_OFFSET / SECONDS_PER_DAY
    }

    /// TT Julian Date.
    pub fn tt_julian_date(&self) -> f64 {
        self.tt_tjt() + 40_000.0 + 2_400_000.5
    }

    // --- Typed accessors (Phase 1 #103) ------------------------------------
    //
    // Additive typed getters returning `SecondsSince<S>` for each time scale
    // present as a pub f64 field. These delegate to the existing f64 fields
    // (no recomputation) so behavior is bit-identical to the f64 API.
    //
    // Note: no typed getter for GMST is provided as `SecondsSince<GMST>` —
    // the primary representation of GMST is an angle (see `gmst_angle()`).

    /// TAI seconds since simulation start, as a typed quantity.
    #[inline]
    pub fn tai(&self) -> crate::SecondsSince<crate::TAI> {
        crate::SecondsSince::from_seconds(self.tai_seconds)
    }

    /// UTC seconds since simulation start, as a typed quantity.
    #[inline]
    pub fn utc(&self) -> crate::SecondsSince<crate::UTC> {
        crate::SecondsSince::from_seconds(self.utc_seconds)
    }

    /// UT1 seconds since simulation start, as a typed quantity.
    #[inline]
    pub fn ut1(&self) -> crate::SecondsSince<crate::UT1> {
        crate::SecondsSince::from_seconds(self.ut1_seconds)
    }

    /// TT seconds since simulation start, as a typed quantity.
    #[inline]
    pub fn tt(&self) -> crate::SecondsSince<crate::TT> {
        crate::SecondsSince::from_seconds(self.tt_seconds)
    }

    /// TDB seconds since simulation start, as a typed quantity.
    #[inline]
    pub fn tdb(&self) -> crate::SecondsSince<crate::TDB> {
        crate::SecondsSince::from_seconds(self.tdb_seconds)
    }

    /// GPS seconds since simulation start, as a typed quantity.
    #[inline]
    pub fn gps(&self) -> crate::SecondsSince<crate::GPS> {
        crate::SecondsSince::from_seconds(self.gps_seconds)
    }

    /// Greenwich Mean Sidereal Time as a typed angle.
    ///
    /// GMST is fundamentally an angle; the primary representation is radians
    /// wrapped to `[0, 2π)`. Use this instead of a `SecondsSince<GMST>` getter.
    #[inline]
    pub fn gmst_angle(&self) -> uom::si::f64::Angle {
        uom::si::f64::Angle::new::<uom::si::angle::radian>(self.gmst_radians)
    }

    // JEOD_INV: TM.03 — time types updated in dependency order
    // (TAI -> TT -> TDB -> UTC -> GPS -> UT1 -> GMST -> MET -> UDE);
    // GPS depends only on TAI, so it is computed before UT1/GMST.
    /// Recompute all derived time scales from current TAI state.
    fn recompute_derived(&mut self) {
        // TT = TAI + 32.184
        self.tt_seconds = time_converter_tai_tt::tai_to_tt(self.tai_seconds);

        // TDB = TT + periodic offset
        self.tdb_seconds = time_converter_tai_tdb::tai_to_tdb(self.tai_seconds, self.tai_tjt);

        // UTC via leap second table (epoch value cached at construction)
        let utc_tjt = self.leap_second_table.tai_to_utc_tjt(self.tai_tjt);
        self.utc_seconds = (utc_tjt - self.utc_tjt_at_epoch) * SECONDS_PER_DAY;

        // GPS = TAI − 19s (constant offset, defined at GPS epoch 1980-01-06).
        // GPS time measures elapsed seconds since the GPS epoch, offset from
        // TAI by exactly 19 leap seconds that had accumulated by 1980.
        self.gps_seconds = time_gps::tai_to_gps(self.tai_seconds);

        // UT1 = TAI + ut1_tai_offset (offset is approximately constant over
        // short spans; updated from IERS bulletins via set_ut1_tai_offset
        // or per-tick by the installed EopTable).
        self.ut1_seconds = self.tai_seconds + self.ut1_tai_offset;

        // GMST from UT1 days since noon 2000-01-01 (Astronomical Almanac convention).
        // UT1 TJT = TAI TJT + ut1_tai_offset / 86400
        let ut1_tjt = self.tai_tjt + self.ut1_tai_offset / SECONDS_PER_DAY;
        // d_u = JD(UT1) - 2451545.0 = ut1_tjt - 11544.5
        // Matches JEOD: ut1_ptr->trunc_julian_time - 11544.5
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
    fn initial_state_at_j2000() {
        let sim = SimulationTime::at_j2000(default_leap_second_table());
        assert_eq!(sim.tai_seconds, 0.0);
        assert_eq!(sim.simtime, 0.0);
        // TT = TAI + 32.184
        assert!((sim.tt_seconds - 32.184).abs() < 1e-10);
        // GMST at TAI J2000 epoch. The TAI epoch is ~64s before noon UT1
        // (32.184s TAI-TT offset + 32s TAI-UTC with default UT1-UTC≈0),
        // so du ≈ −0.000743 days, giving GMST ≈ 280.46° − 0.27° ≈ 280.19°.
        let gmst_deg = sim.gmst_radians * 180.0 / PI;
        assert!(
            (gmst_deg - 280.19).abs() < 0.05,
            "GMST at J2000: {:.4} degrees, expected ~280.19",
            gmst_deg
        );
    }

    #[test]
    fn advance_increases_all_scales() {
        let mut sim = SimulationTime::at_j2000(default_leap_second_table());
        let dt = 3600.0; // 1 hour
        sim.advance(dt);
        assert!((sim.tai_seconds - dt).abs() < 1e-15);
        assert!((sim.tt_seconds - (dt + 32.184)).abs() < 1e-10);
        assert!((sim.simtime - dt).abs() < 1e-15);
    }

    #[test]
    fn tdb_julian_date_at_j2000() {
        let sim = SimulationTime::at_j2000(default_leap_second_table());
        let jd = sim.tdb_julian_date();
        // Should be close to 2451545.0 (J2000 TT JD)
        assert!((jd - 2_451_545.0).abs() < 0.001, "TDB JD at J2000: {}", jd);
    }

    #[test]
    fn tt_julian_date_at_j2000() {
        let sim = SimulationTime::at_j2000(default_leap_second_table());
        let jd = sim.tt_julian_date();
        assert!(
            (jd - 2_451_545.0).abs() < 1e-8,
            "TT JD at J2000: {}, expected 2451545.0",
            jd
        );
    }

    #[test]
    fn advance_one_day() {
        let mut sim = SimulationTime::at_j2000(default_leap_second_table());
        let one_day = 86400.0;
        sim.advance(one_day);
        // TAI TJT should advance by 1 day
        assert!(
            (sim.tai_tjt - sim.tai_tjt_at_epoch - 1.0).abs() < 1e-12,
            "TAI TJT should advance by 1 day"
        );
        // GMST should have advanced by about 1 sidereal day (~1.00274 solar days)
        let gmst_advance_days = sim.gmst_seconds / 86400.0
            - SimulationTime::at_j2000(default_leap_second_table()).gmst_seconds / 86400.0;
        assert!(
            (gmst_advance_days - 1.00274).abs() < 0.001,
            "GMST advance over 1 solar day: {} sidereal days (expected ~1.00274)",
            gmst_advance_days
        );
    }

    #[test]
    #[should_panic(expected = "must be finite")]
    fn advance_nan_panics() {
        let mut sim = SimulationTime::at_j2000(default_leap_second_table());
        sim.advance(f64::NAN);
    }

    #[test]
    #[should_panic(expected = "must be finite")]
    fn advance_inf_panics() {
        let mut sim = SimulationTime::at_j2000(default_leap_second_table());
        sim.advance(f64::INFINITY);
    }

    /// Pins the `assert!(sim_dt.is_finite() && sim_dt >= 0.0)` guard at the
    /// entry of `SimulationTime::advance`. A NaN `sim_dt` would propagate
    /// through `tai_seconds += dyn_dt`, contaminating every derived time
    /// scale (TT, TDB, UTC, UT1, GMST) and the ephemeris/RNP downstream
    /// without surfacing a numeric NaN at the right call site. JEOD's
    /// integration loop assumes a valid sim-input `dt`; we assert
    /// defensively so a bad upstream value fails loudly at the time-advance
    /// boundary.
    // JEOD_INV: TM.40 — negative test: non-finite advance dt rejected
    #[test]
    #[should_panic(expected = "sim_dt must be finite and >= 0")]
    fn tm_40_panics_on_non_finite_advance_dt() {
        // JEOD_INV: TM.40 — NaN sim_dt at SimulationTime::advance entry.
        let mut sim = SimulationTime::at_j2000(default_leap_second_table());
        sim.advance(f64::NAN);
    }

    /// Pins the `sim_dt >= 0.0` half of the TM.40 guard. JEOD treats a
    /// reverse-time advance by passing `dt > 0` with `time_scale_factor < 0`
    /// (see `advance_time_scale_factor_reversal`); a negative `sim_dt` is
    /// always a caller error because `simtime` must run forward.
    // JEOD_INV: TM.40 — negative test: negative advance dt rejected
    #[test]
    #[should_panic(expected = "sim_dt must be finite and >= 0")]
    fn tm_40_panics_on_negative_advance_dt() {
        // JEOD_INV: TM.40 — sign violation at SimulationTime::advance entry.
        let mut sim = SimulationTime::at_j2000(default_leap_second_table());
        sim.advance(-1.0);
    }

    #[test]
    fn advance_time_scale_factor_reversal() {
        let mut sim = SimulationTime::at_j2000(default_leap_second_table());
        let dt = 3600.0;
        sim.advance(dt);
        let tai_after_forward = sim.tai_seconds;
        let gmst_after_forward = sim.gmst_seconds;

        // Reverse dynamic time via set_scale_factor(-1.0)
        sim.set_scale_factor(-1.0);
        sim.advance(dt);
        assert!(
            sim.tai_seconds.abs() < 1e-15,
            "tai_seconds should return to 0, got {}",
            sim.tai_seconds
        );
        assert!(
            (sim.gmst_seconds - SimulationTime::at_j2000(default_leap_second_table()).gmst_seconds)
                .abs()
                < 1e-10,
            "GMST should return to initial value"
        );
        // Forward-backward round-trip for GPS
        assert!(
            (sim.gps_seconds - (-19.0)).abs() < 1e-15,
            "GPS should return to initial value (-19.0), got {}",
            sim.gps_seconds
        );
        let _ = (tai_after_forward, gmst_after_forward); // suppress unused
    }

    #[test]
    fn gps_offset() {
        let sim = SimulationTime::at_j2000(default_leap_second_table());
        // GPS = TAI - 19s. At t=0, tai_seconds = 0, so gps_seconds = -19.
        assert!(
            (sim.gps_seconds - (-19.0)).abs() < 1e-15,
            "GPS at t=0: expected -19.0, got {}",
            sim.gps_seconds
        );
    }

    #[test]
    fn gps_advances_with_tai() {
        let mut sim = SimulationTime::at_j2000(default_leap_second_table());
        sim.advance(100.0);
        // GPS = TAI - 19 = 100 - 19 = 81
        assert!(
            (sim.gps_seconds - 81.0).abs() < 1e-15,
            "GPS after 100s: expected 81.0, got {}",
            sim.gps_seconds
        );
    }

    // --- Typed-accessor tests (Phase 1 #103) -------------------------------

    #[test]
    fn typed_getters_match_f64_fields() {
        // Advance by a non-trivial duration to exercise each scale.
        let mut sim = SimulationTime::at_j2000(default_leap_second_table());
        sim.advance(1_000_000.0);

        // Each typed getter round-trips back to the matching f64 field.
        assert_eq!(sim.tai().as_seconds(), sim.tai_seconds);
        assert_eq!(sim.utc().as_seconds(), sim.utc_seconds);
        assert_eq!(sim.ut1().as_seconds(), sim.ut1_seconds);
        assert_eq!(sim.tt().as_seconds(), sim.tt_seconds);
        assert_eq!(sim.tdb().as_seconds(), sim.tdb_seconds);
        assert_eq!(sim.gps().as_seconds(), sim.gps_seconds);

        // `gmst_angle()` returns uom Angle; extracting radians must match.
        use uom::si::angle::radian;
        let gmst_rad = sim.gmst_angle().get::<radian>();
        assert_eq!(gmst_rad, sim.gmst_radians);
    }

    #[test]
    fn typed_tai_tt_roundtrip_via_simulation_time() {
        // Read TAI out via the typed getter, convert to TT via the typed
        // converter, convert back, and verify bit-identical (tolerance 1e-14 s)
        // agreement with the original reading.
        use crate::time_converter_tai_tt::{tai_to_tt_typed, tt_to_tai_typed};

        let mut sim = SimulationTime::at_j2000(default_leap_second_table());
        sim.advance(1_000_000.0);

        let tai = sim.tai();
        let tt = tai_to_tt_typed(tai);
        let back = tt_to_tai_typed(tt);
        let err = (back.as_seconds() - tai.as_seconds()).abs();
        assert!(
            err < 1e-14,
            "TAI->TT->TAI typed roundtrip err={} (tolerance 1e-14 s)",
            err
        );
    }

    // --- Tests ported from TimeManager (consolidated under SimulationTime) -

    #[test]
    fn get_seconds_returns_all_scales() {
        let mut sim = SimulationTime::at_j2000(default_leap_second_table());
        sim.advance(100.0);

        assert!((sim.get_seconds(TimeScaleId::TAI) - 100.0).abs() < 1e-15);
        assert!((sim.get_seconds(TimeScaleId::TT) - 132.184).abs() < 1e-10);
        assert!((sim.get_seconds(TimeScaleId::GPS) - 81.0).abs() < 1e-15);
        assert!((sim.get_seconds(TimeScaleId::DYN) - 100.0).abs() < 1e-15);
    }

    #[test]
    fn with_met_tracks_tai() {
        let mut sim = SimulationTime::at_j2000(default_leap_second_table());
        sim.add_met(0.0); // MET epoch at TAI=0
        sim.advance(500.0);
        assert!(
            (sim.met.as_ref().unwrap().seconds - 500.0).abs() < 1e-15,
            "MET should be 500s"
        );
    }

    #[test]
    fn with_ude_tracks_tai() {
        let mut sim = SimulationTime::at_j2000(default_leap_second_table());
        let idx = sim.add_ude(1000.0); // UDE epoch at TAI=1000s
        sim.advance(1500.0);
        assert!(
            (sim.ude[idx].seconds - 500.0).abs() < 1e-15,
            "UDE should be 500s"
        );
    }

    #[test]
    fn set_scale_factor_reverses_tai() {
        let mut sim = SimulationTime::at_j2000(default_leap_second_table());
        sim.advance(100.0);
        let tai_100 = sim.tai_seconds;

        // Reverse time
        sim.set_scale_factor(-1.0);
        sim.advance(100.0);

        assert!(
            sim.tai_seconds.abs() < 1e-15,
            "TAI should return to 0 after reversal, got {}",
            sim.tai_seconds
        );
        let _ = tai_100;
    }

    #[test]
    fn multiple_udes() {
        let mut sim = SimulationTime::at_j2000(default_leap_second_table());
        let idx0 = sim.add_ude(0.0);
        let idx1 = sim.add_ude(500.0);
        sim.advance(1000.0);
        assert!((sim.ude[idx0].seconds - 1000.0).abs() < 1e-15);
        assert!((sim.ude[idx1].seconds - 500.0).abs() < 1e-15);
    }

    /// EOP wiring: when an `EopTable` is installed, `ut1_tai_offset` updates
    /// per-step from the table rather than staying constant. Sample at TAI
    /// TJT 11178 (1998-12-31; JEOD's `val_vec[13513] = -31.2824458`) and one
    /// day later (TJT 11179; `val_vec[13514] = -31.2835239` per the JEOD
    /// source). The difference (~1.08 ms/day) is exactly the per-day drift
    /// the constant-offset path missed.
    #[test]
    fn with_eop_interpolates_per_step() {
        use crate::default_eop_table;
        // 1998-12-31 TAI TJT (matches SIM_4_common_usage epoch).
        let init_tai_tjt = 11_178.0;
        let sim0 = SimulationTime::new(init_tai_tjt, default_leap_second_table())
            .with_eop_table(default_eop_table());
        let off0 = sim0.ut1_tai_offset;
        assert!(
            (off0 - (-31.2824458)).abs() < 1e-12,
            "EOP at TJT 11178: got {off0}"
        );

        // Advance one full day; EOP should re-interpolate to entry 13514.
        let mut sim = sim0.clone();
        sim.advance(SECONDS_PER_DAY);
        let off1 = sim.ut1_tai_offset;
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
    fn set_ut1_tai_offset_disables_eop() {
        use crate::default_eop_table;
        let init_tai_tjt = 11_178.0;
        let mut sim = SimulationTime::new(init_tai_tjt, default_leap_second_table())
            .with_eop_table(default_eop_table());
        assert!(sim.has_eop_table());

        sim.set_ut1_tai_offset(-30.0);
        assert!(!sim.has_eop_table(), "explicit override disables EOP");
        assert_eq!(sim.ut1_tai_offset, -30.0);

        sim.advance(SECONDS_PER_DAY);
        assert_eq!(
            sim.ut1_tai_offset, -30.0,
            "constant offset must hold after explicit override"
        );
    }
}
