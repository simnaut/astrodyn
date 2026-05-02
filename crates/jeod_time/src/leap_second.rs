//! [`LeapSecondTable`] — TAI ↔ UTC leap-second lookup.
//!
//! Ports
//! [`models/environment/time/src/tai_to_utc.cc`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/time/src/tai_to_utc.cc)
//! and the
//! [`Leap_Second.dat`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/time/data/Leap_Second.dat)
//! parser from JEOD v5.4.0. Required for any simulation that hits a
//! UTC ↔ TAI conversion.

use crate::epoch::{mjd_to_tjt, SECONDS_PER_DAY};
use log::warn;
use std::sync::atomic::{AtomicBool, Ordering};

/// Leap second lookup table for TAI↔UTC conversion.
///
/// Each entry maps a truncated Julian time (TJT) boundary to the cumulative
/// TAI-UTC offset in seconds at that boundary. Entries are sorted by TJT.
///
/// Mirrors JEOD's `when_vec` / `val_vec` arrays from `tai_to_utc.cc`.
#[derive(Debug, Clone)]
pub struct LeapSecondTable {
    /// (tjt_boundary, tai_minus_utc_seconds) pairs, sorted by TJT.
    entries: Vec<(f64, f64)>,
}

impl LeapSecondTable {
    /// Create a leap second table from (TJT, TAI-UTC seconds) pairs.
    /// Entries must be sorted by TJT.
    pub fn from_entries(entries: Vec<(f64, f64)>) -> Self {
        // JEOD_INV: TM.39 — leap-second table must be non-empty and monotonic in TJT.
        // JEOD's `when_vec` is assumed monotonic in `time_converter_tai_utc.cc`; we enforce at construction.
        assert!(!entries.is_empty(), "Leap second table must not be empty");
        assert!(
            entries.windows(2).all(|w| w[0].0 <= w[1].0),
            "Leap second entries must be sorted by TJT"
        );
        Self { entries }
    }

    /// Create from (MJD, TAI-UTC seconds) pairs (as found in Leap_Second.dat).
    pub fn from_mjd_entries(mjd_entries: &[(f64, f64)]) -> Self {
        let entries: Vec<(f64, f64)> = mjd_entries
            .iter()
            .map(|&(mjd, tai_utc)| (mjd_to_tjt(mjd), tai_utc))
            .collect();
        Self::from_entries(entries)
    }

    /// Look up the TAI-UTC offset (in seconds) at a given TAI TJT.
    ///
    /// Returns the cumulative number of leap seconds (TAI - UTC) in seconds.
    pub fn tai_utc_at_tai_tjt(&self, tai_tjt: f64) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        // The TAI TJT includes the leap second offset itself, so we need to
        // find the entry where the UTC boundary (in TJT) matches. JEOD converts
        // the offset to days and applies: utc_tjt = tai_tjt - tai_utc/86400.
        // We iterate to find the correct bracket.
        let index = self.find_index_for_tai(tai_tjt);
        self.entries[index].1
    }

    /// Look up the TAI-UTC offset (in seconds) at a given UTC TJT.
    pub fn tai_utc_at_utc_tjt(&self, utc_tjt: f64) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let index = self.find_index_for_utc(utc_tjt);
        self.entries[index].1
    }

    /// Convert TAI TJT to UTC TJT.
    pub fn tai_to_utc_tjt(&self, tai_tjt: f64) -> f64 {
        let tai_utc = self.tai_utc_at_tai_tjt(tai_tjt);
        tai_tjt - tai_utc / SECONDS_PER_DAY
    }

    /// Convert UTC TJT to TAI TJT.
    pub fn utc_to_tai_tjt(&self, utc_tjt: f64) -> f64 {
        let tai_utc = self.tai_utc_at_utc_tjt(utc_tjt);
        utc_tjt + tai_utc / SECONDS_PER_DAY
    }

    /// Number of entries in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find the index for a TAI TJT value.
    ///
    /// `entries[i] = (when_vec[i], val_vec[i])` where `when_vec[i]` is a UTC
    /// boundary in TJT and `val_vec[i]` is TAI-UTC (in seconds) applied AT and
    /// AFTER that boundary. The TAI instant at which the i-th transition
    /// occurs (UTC first reaches `when_vec[i]` using the previous offset
    /// `val_vec[i-1]`) is `tai_boundary[i] = when_vec[i] + val_vec[i-1]/86400`.
    ///
    /// JEOD's `convert_a_to_b` updates the offset stepwise: it holds index
    /// `i-1` until UTC (computed with that offset) reaches `when_vec[i]`,
    /// then advances to index `i`. Equivalently, for a given TAI the offset
    /// is `val_vec[i]` where `i` is the largest index with
    /// `tai_tjt >= when_vec[i] + val_vec[i-1]/86400`. This matches JEOD's
    /// "true_utc" convention where the leap second is counted as a real UTC
    /// second: within the leap second, UTC rewinds by 1 s so the interval
    /// `23:59:59..24:00:00` is traversed twice (once labelled 23:59:59, once
    /// labelled 23:59:60).
    ///
    /// O(n) over the leap-second table (28 entries today, growing by 0–1 per
    /// year). Fine for occasional UTC↔TAI conversions; do not call in tight
    /// integrator inner loops — cache the result if you need it per step.
    fn find_index_for_tai(&self, tai_tjt: f64) -> usize {
        let last = self.entries.len() - 1;
        // Before the first entry: JEOD uses val_vec[0], so return index 0.
        // Boundary at i=0 in the "TAI frame" is `when_vec[0] + val_vec[0]/86400`
        // (no prior entry, so we use val_vec[0] itself). Mirror JEOD's
        // operational one-time WARN when TAI precedes the leap-second table.
        let first_tai_boundary = self.entries[0].0 + self.entries[0].1 / SECONDS_PER_DAY;
        if tai_tjt < first_tai_boundary {
            static WARNED_BEFORE_TAI: AtomicBool = AtomicBool::new(false);
            if !WARNED_BEFORE_TAI.swap(true, Ordering::Relaxed) {
                warn!(
                    "TAI time precedes first leap second table boundary; \
                     using first value ({} s)",
                    self.entries[0].1
                );
            }
            return 0;
        }
        // Mirror JEOD's one-time WARN when TAI follows the last boundary.
        // The last regime extends forward indefinitely, so we only emit the
        // warning; the actual index still comes from the search below.
        let last_tai_boundary = if last == 0 {
            self.entries[0].0 + self.entries[0].1 / SECONDS_PER_DAY
        } else {
            self.entries[last].0 + self.entries[last - 1].1 / SECONDS_PER_DAY
        };
        if tai_tjt >= last_tai_boundary {
            static WARNED_AFTER_TAI: AtomicBool = AtomicBool::new(false);
            if !WARNED_AFTER_TAI.swap(true, Ordering::Relaxed) {
                warn!(
                    "TAI time follows last leap second table boundary; \
                     using last value ({} s)",
                    self.entries[last].1
                );
            }
        }
        // Search from the top down for the largest i with
        //   tai_tjt >= when_vec[i] + val_vec[i-1]/86400.
        for i in (1..=last).rev() {
            let tai_boundary_i = self.entries[i].0 + self.entries[i - 1].1 / SECONDS_PER_DAY;
            if tai_tjt >= tai_boundary_i {
                return i;
            }
        }
        0
    }

    /// Find the index for a UTC TJT value using binary-style search.
    fn find_index_for_utc(&self, utc_tjt: f64) -> usize {
        let last = self.entries.len() - 1;
        // JEOD time_converter_tai_utc.cc:158-233: log INFORM when time is
        // outside the leap second table range.
        if utc_tjt < self.entries[0].0 {
            static WARNED_BEFORE: AtomicBool = AtomicBool::new(false);
            if !WARNED_BEFORE.swap(true, Ordering::Relaxed) {
                warn!(
                    "Time precedes first leap second table entry; \
                     using first value ({} s)",
                    self.entries[0].1
                );
            }
            return 0;
        }
        if utc_tjt >= self.entries[last].0 {
            static WARNED_AFTER: AtomicBool = AtomicBool::new(false);
            if !WARNED_AFTER.swap(true, Ordering::Relaxed) {
                warn!(
                    "Time follows last leap second table entry; \
                     using last value ({} s)",
                    self.entries[last].1
                );
            }
            return last;
        }
        // Linear search (28 entries, fast enough; matches JEOD's loop)
        for i in 0..last {
            if utc_tjt >= self.entries[i].0 && utc_tjt < self.entries[i + 1].0 {
                return i;
            }
        }
        last
    }
}

/// Create the default leap second table from hardcoded JEOD data.
///
/// Data from JEOD `Leap_Second.dat` (28 entries, 1972-2017).
/// MJD values from the file, TAI-UTC in seconds.
pub fn default_leap_second_table() -> LeapSecondTable {
    // (MJD, TAI-UTC seconds) from JEOD Leap_Second.dat
    let mjd_entries: &[(f64, f64)] = &[
        (41317.0, 10.0), // 1972-01-01
        (41499.0, 11.0), // 1972-07-01
        (41683.0, 12.0), // 1973-01-01
        (42048.0, 13.0), // 1974-01-01
        (42413.0, 14.0), // 1975-01-01
        (42778.0, 15.0), // 1976-01-01
        (43144.0, 16.0), // 1977-01-01
        (43509.0, 17.0), // 1978-01-01
        (43874.0, 18.0), // 1979-01-01
        (44239.0, 19.0), // 1980-01-01
        (44786.0, 20.0), // 1981-07-01
        (45151.0, 21.0), // 1982-07-01
        (45516.0, 22.0), // 1983-07-01
        (46247.0, 23.0), // 1985-07-01
        (47161.0, 24.0), // 1988-01-01
        (47892.0, 25.0), // 1990-01-01
        (48257.0, 26.0), // 1991-01-01
        (48804.0, 27.0), // 1992-07-01
        (49169.0, 28.0), // 1993-07-01
        (49534.0, 29.0), // 1994-07-01
        (50083.0, 30.0), // 1996-01-01
        (50630.0, 31.0), // 1997-07-01
        (51179.0, 32.0), // 1999-01-01
        (53736.0, 33.0), // 2006-01-01
        (54832.0, 34.0), // 2009-01-01
        (56109.0, 35.0), // 2012-07-01
        (57204.0, 36.0), // 2015-07-01
        (57754.0, 37.0), // 2017-01-01
    ];
    LeapSecondTable::from_mjd_entries(mjd_entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "must not be empty")]
    fn empty_table_panics() {
        LeapSecondTable::from_entries(vec![]);
    }

    #[test]
    fn default_table_has_28_entries() {
        let table = default_leap_second_table();
        assert_eq!(table.len(), 28);
    }

    #[test]
    fn tai_utc_at_j2000() {
        // At J2000 (2000-01-01 12:00 TT), TAI-UTC = 32s
        // J2000 TAI TJT ≈ 11544.4996275
        let table = default_leap_second_table();
        let tai_utc = table.tai_utc_at_tai_tjt(11544.4996);
        assert_eq!(tai_utc, 32.0, "TAI-UTC at J2000 should be 32s");
    }

    #[test]
    fn tai_utc_round_trip() {
        let table = default_leap_second_table();
        // Test a TJT in 2010 (between 2009 and 2012 leap seconds)
        // MJD 55000 = TJT 15000.0, TAI-UTC should be 34
        let utc_tjt = 15000.0;
        let tai_tjt = table.utc_to_tai_tjt(utc_tjt);
        let utc_back = table.tai_to_utc_tjt(tai_tjt);
        assert!(
            (utc_back - utc_tjt).abs() < 1e-12,
            "Round trip: {} -> {} -> {}",
            utc_tjt,
            tai_tjt,
            utc_back
        );
    }

    #[test]
    fn first_and_last_entries() {
        let table = default_leap_second_table();
        // Before first entry: should use first entry's offset
        let tjt_before = mjd_to_tjt(41000.0);
        assert_eq!(table.tai_utc_at_utc_tjt(tjt_before), 10.0);

        // After last entry
        let tjt_after = mjd_to_tjt(58000.0);
        assert_eq!(table.tai_utc_at_utc_tjt(tjt_after), 37.0);
    }

    #[test]
    fn tai_utc_correct_at_all_28_boundaries() {
        let table = default_leap_second_table();
        // (MJD, expected TAI-UTC) for all 28 entries from JEOD Leap_Second.dat
        let boundaries: &[(f64, f64)] = &[
            (41317.0, 10.0),
            (41499.0, 11.0),
            (41683.0, 12.0),
            (42048.0, 13.0),
            (42413.0, 14.0),
            (42778.0, 15.0),
            (43144.0, 16.0),
            (43509.0, 17.0),
            (43874.0, 18.0),
            (44239.0, 19.0),
            (44786.0, 20.0),
            (45151.0, 21.0),
            (45516.0, 22.0),
            (46247.0, 23.0),
            (47161.0, 24.0),
            (47892.0, 25.0),
            (48257.0, 26.0),
            (48804.0, 27.0),
            (49169.0, 28.0),
            (49534.0, 29.0),
            (50083.0, 30.0),
            (50630.0, 31.0),
            (51179.0, 32.0),
            (53736.0, 33.0),
            (54832.0, 34.0),
            (56109.0, 35.0),
            (57204.0, 36.0),
            (57754.0, 37.0),
        ];
        for &(mjd, expected_tai_utc) in boundaries {
            let utc_tjt = mjd_to_tjt(mjd);
            let tai_utc = table.tai_utc_at_utc_tjt(utc_tjt);
            assert_eq!(
                tai_utc, expected_tai_utc,
                "At MJD {}: expected TAI-UTC={}s, got {}s",
                mjd, expected_tai_utc, tai_utc
            );
        }
    }

    #[test]
    fn tai_utc_round_trip_all_boundaries() {
        let table = default_leap_second_table();
        let mjds: &[f64] = &[
            41317.0, 41499.0, 42048.0, 43144.0, 44786.0, 47161.0, 49169.0, 50083.0, 51179.0,
            53736.0, 56109.0, 57754.0,
        ];
        for &mjd in mjds {
            // Test slightly after the boundary (to avoid edge ambiguity)
            let utc_tjt = mjd_to_tjt(mjd) + 0.5; // half a day after
            let tai_tjt = table.utc_to_tai_tjt(utc_tjt);
            let utc_back = table.tai_to_utc_tjt(tai_tjt);
            assert!(
                (utc_back - utc_tjt).abs() < 1e-12,
                "Round trip failed at MJD {}: {} -> {} -> {}, err={}",
                mjd,
                utc_tjt,
                tai_tjt,
                utc_back,
                (utc_back - utc_tjt).abs()
            );
        }
    }

    /// At the instant TAI reaches a leap-second boundary in TAI time
    /// (`tai_tjt = when_vec[i] + val_vec[i-1]/86400`), JEOD's
    /// `convert_a_to_b` updates the offset to `val_vec[i]`. Verify our
    /// stateless lookup reproduces this (matches JEOD time_converter_tai_utc.cc
    /// line 271 `while(utc >= next_when)`).
    #[test]
    fn tai_utc_at_tai_boundary_uses_post_leap_offset() {
        let table = default_leap_second_table();
        // 1999-01-01 UTC boundary: when_vec[22] = MJD 51179 = TJT 11179.
        // TAI-UTC transitions 31 → 32 at this UTC boundary.
        let when_i = 11179.0_f64;
        let prev_val: f64 = 31.0;
        // At TAI exactly when UTC reaches boundary with old offset applied:
        let tai_at_boundary = when_i + prev_val / SECONDS_PER_DAY;
        // JEOD updates offset here, so TAI-UTC at this TAI = 32 (post-leap).
        assert_eq!(
            table.tai_utc_at_tai_tjt(tai_at_boundary),
            32.0,
            "At TAI reaching a leap boundary, should use the post-leap offset"
        );
        // And just before (in TAI), should still be 31.
        let before_boundary = tai_at_boundary - 1e-9;
        assert_eq!(
            table.tai_utc_at_tai_tjt(before_boundary),
            31.0,
            "Just before the TAI boundary, should still use pre-leap offset"
        );
    }

    /// Exact reproduction of the SIM_4 RUN_JEOD2x row at t=86400 in
    /// `tier3_sim_time_docker.rs`: TAI TJT = 11179.0003587963, JEOD UTC TJT =
    /// 11178.99998842593. The boundary is crossed exactly at this TAI — JEOD
    /// updates its internal state on crossing and reports the post-leap offset.
    #[test]
    fn tai_utc_matches_jeod_sim4_at_leap_crossing() {
        let table = default_leap_second_table();
        let tai_tjt = 11179.0003587963_f64;
        let expected_utc_tjt = 11178.99998842593_f64;
        let got = table.tai_to_utc_tjt(tai_tjt);
        let err_sec = (got - expected_utc_tjt).abs() * SECONDS_PER_DAY;
        assert!(
            err_sec < 1e-6,
            "SIM_4 leap crossing: got UTC TJT {got}, expected {expected_utc_tjt} \
             (diff {err_sec:.4e} s)"
        );
    }
}
