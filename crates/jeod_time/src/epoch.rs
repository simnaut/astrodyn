/// J2000.0 epoch as Julian Date: 2000-01-01 12:00:00 TT.
pub const J2000_TT_JD: f64 = 2_451_545.0;

/// J2000.0 epoch as Modified Julian Date (JD - 2400000.5).
pub const J2000_TT_MJD: f64 = 51_544.5;

/// JEOD Truncated Julian Time offset: TJT = MJD - 40000.
pub const TJT_OFFSET: f64 = 40_000.0;

/// J2000.0 as TJT: MJD 51544.5 - 40000 = 11544.5
pub const J2000_TT_TJT: f64 = 11_544.5;

/// TAI TJT at J2000.0 TT epoch.
/// TT = TAI + 32.184s, so TAI is 32.184s behind TT.
/// 32.184s = 32.184/86400 days = 0.000372500 days
/// TAI TJT = 11544.5 - 0.000372500 = 11544.499627500
pub const J2000_TAI_TJT: f64 = 11_544.499_627_5;

/// TT - TAI offset in seconds (exact by definition).
pub const TAI_TT_OFFSET: f64 = 32.184;

/// Seconds per day.
pub const SECONDS_PER_DAY: f64 = 86_400.0;

/// Convert MJD to TJT (truncated Julian time).
pub fn mjd_to_tjt(mjd: f64) -> f64 {
    mjd - TJT_OFFSET
}

/// Convert TJT to MJD.
pub fn tjt_to_mjd(tjt: f64) -> f64 {
    tjt + TJT_OFFSET
}

/// Convert TJT to Julian Date.
pub fn tjt_to_jd(tjt: f64) -> f64 {
    tjt + TJT_OFFSET + 2_400_000.5
}

/// Convert Julian Date to TJT.
pub fn jd_to_tjt(jd: f64) -> f64 {
    jd - 2_400_000.5 - TJT_OFFSET
}
