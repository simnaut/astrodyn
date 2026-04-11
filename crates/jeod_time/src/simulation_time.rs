use crate::epoch::{J2000_NOON_TJT, J2000_TAI_TJT, SECONDS_PER_DAY, TAI_TT_OFFSET};
use crate::leap_second::LeapSecondTable;
use crate::time_converter_tai_tdb;
use crate::time_converter_tai_tt;
use crate::time_converter_ut1_gmst;

// JEOD_INV: TM.04 — all time types reachable from initializer (all scales hardcoded in single struct)
// JEOD_INV: TM.05 — all time types reachable from TimeDyn (all scales updated in recompute_derived)
// JEOD_INV: TM.06 — no duplicate converters (each scale has exactly one conversion path)
/// Complete simulation time state across all time scales.
///
/// All seconds fields are measured from the TAI epoch (which coincides with
/// the simulation start). The TAI TJT (truncated Julian time) anchors
/// everything to an absolute calendar reference.
///
/// Mirrors JEOD's `TimeManager` + individual time type state.
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
    /// Elapsed simulation time (same as tai_seconds for forward sim).
    pub simtime: f64,
    /// Leap second table for TAI↔UTC conversion.
    pub leap_second_table: LeapSecondTable,
    /// UT1-TAI offset in seconds (from IERS data; approximately constant
    /// over short simulation spans). Default: computed assuming UT1-UTC ≈ 0,
    /// so `ut1_tai_offset ≈ -(TAI-UTC)`.
    ///
    /// For historical or far-future epochs, UT1-UTC can reach ±0.9 s, which
    /// affects GMST and Earth rotation calculations. Call
    /// [`set_ut1_tai_offset`](Self::set_ut1_tai_offset) with a value from
    /// IERS Bulletin A/B when sub-second accuracy in Earth orientation is needed.
    pub ut1_tai_offset: f64,
}

impl SimulationTime {
    // JEOD_INV: TM.07 — JEOD uses -1.0 sentinel; we call recompute_derived() at construction instead
    /// Create a new SimulationTime starting at the given TAI TJT.
    ///
    /// # Arguments
    /// * `tai_tjt_at_epoch` - TAI truncated Julian time at simulation start.
    ///   Use `J2000_TAI_TJT` (≈ 11544.4996275) for a J2000 epoch start.
    /// * `leap_table` - Leap second table for TAI↔UTC conversion.
    pub fn new(tai_tjt_at_epoch: f64, leap_table: LeapSecondTable) -> Self {
        // Compute initial TAI-UTC offset
        let tai_utc_s = leap_table.tai_utc_at_tai_tjt(tai_tjt_at_epoch);
        // Default UT1-TAI ≈ -(TAI-UTC) (assumes UT1-UTC ≈ 0)
        let ut1_tai_offset = -tai_utc_s;

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
        };
        sim.recompute_derived();
        sim
    }

    /// Create a SimulationTime starting at J2000.0 TT epoch.
    pub fn at_j2000(leap_table: LeapSecondTable) -> Self {
        Self::new(J2000_TAI_TJT, leap_table)
    }

    /// Set the UT1-TAI offset in seconds (from IERS bulletin data).
    ///
    /// UT1-TAI = (UT1-UTC) - (TAI-UTC). For example, if UT1-UTC = 0.3s
    /// and TAI-UTC = 32s, then UT1-TAI = 0.3 - 32 = -31.7s.
    pub fn set_ut1_tai_offset(&mut self, offset_seconds: f64) {
        self.ut1_tai_offset = offset_seconds;
        self.recompute_derived();
    }

    // JEOD_INV: TM.03 — time types updated in dependency order via recompute_derived()
    /// Advance the simulation by `dt` seconds.
    ///
    /// Negative `dt` is permitted for time-reversed propagation (JEOD's
    /// `scale_factor = -1` mode). All derived time scales (TT, TDB, UTC,
    /// UT1, GMST, GPS) are recomputed correctly for both forward and
    /// backward time.
    ///
    /// # Panics
    /// Panics if `dt` is NaN or infinite.
    pub fn advance(&mut self, dt: f64) {
        assert!(dt.is_finite(), "dt must be finite, got {dt}");
        self.tai_seconds += dt;
        self.tai_tjt = self.tai_tjt_at_epoch + self.tai_seconds / SECONDS_PER_DAY;
        self.simtime += dt;
        self.recompute_derived();
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

    // JEOD_INV: TM.03 — time types updated in dependency order (TAI -> TT -> TDB -> UTC -> UT1 -> GMST)
    /// Recompute all derived time scales from TAI.
    fn recompute_derived(&mut self) {
        // TT = TAI + 32.184
        self.tt_seconds = time_converter_tai_tt::tai_to_tt(self.tai_seconds);

        // TDB = TT + periodic offset
        self.tdb_seconds = time_converter_tai_tdb::tai_to_tdb(self.tai_seconds, self.tai_tjt);

        // UTC via leap second table
        let utc_tjt = self.leap_second_table.tai_to_utc_tjt(self.tai_tjt);
        let utc_tjt_at_epoch = self.leap_second_table.tai_to_utc_tjt(self.tai_tjt_at_epoch);
        self.utc_seconds = (utc_tjt - utc_tjt_at_epoch) * SECONDS_PER_DAY;

        // GPS = TAI − 19s (constant offset, defined at GPS epoch 1980-01-06).
        // GPS time measures elapsed seconds since the GPS epoch, offset from
        // TAI by exactly 19 leap seconds that had accumulated by 1980.
        self.gps_seconds = self.tai_seconds - 19.0;

        // UT1 = TAI + ut1_tai_offset (offset is approximately constant over
        // short spans; updated from IERS bulletins via set_ut1_tai_offset).
        self.ut1_seconds = self.tai_seconds + self.ut1_tai_offset;

        // GMST from UT1 days since noon 2000-01-01 (Astronomical Almanac convention).
        // UT1 TJT = TAI TJT + ut1_tai_offset / 86400
        let ut1_tjt = self.tai_tjt + self.ut1_tai_offset / SECONDS_PER_DAY;
        // d_u = JD(UT1) - 2451545.0 = ut1_tjt - 11544.5
        // Matches JEOD: ut1_ptr->trunc_julian_time - 11544.5
        let du = ut1_tjt - J2000_NOON_TJT;
        self.gmst_seconds = time_converter_ut1_gmst::ut1_to_gmst_seconds(du);
        self.gmst_radians = time_converter_ut1_gmst::ut1_to_gmst_radians(du);
    }
}

#[cfg(test)]
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

    #[test]
    fn advance_negative_for_time_reversal() {
        let mut sim = SimulationTime::at_j2000(default_leap_second_table());
        let dt = 3600.0;
        sim.advance(dt);
        let tai_after_forward = sim.tai_seconds;
        let gmst_after_forward = sim.gmst_seconds;

        // Reverse time by the same amount — should return to initial state
        sim.advance(-dt);
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
}
