//! [`EopTable`] — IERS EOP-driven UT1-TAI lookup with daily linear
//! interpolation.
//!
//! Ports
//! [`models/environment/time/src/time_converter_tai_ut1.cc`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/time/src/time_converter_tai_ut1.cc)
//! and the data table generated under
//! [`models/environment/time/data/src/tai_to_ut1.cc`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/time/data/src/tai_to_ut1.cc)
//! from JEOD v5.4.0. The table source is the IERS EOP 14 C04 series
//! (`eopc04_14_IAU2000.62-now`); each daily row carries
//! `UT1-UTC` and the parser converts it to `UT1-TAI` by subtracting
//! that day's leap-second value.
//!
//! Units in the table:
//! - `when_tjt[i]` — TAI Truncated Julian Time in days (`MJD - 40000`).
//! - `val_seconds[i]` — `UT1 - TAI` in seconds at that day boundary.
//!
//! Linear interpolation between adjacent daily entries follows JEOD's
//! `convert_a_to_b`: `offset_s = prev_value + (tai_tjt - prev_when) * gradient`
//! where `gradient = (next_value - prev_value) / (next_when - prev_when)`
//! has units `s/day` because consecutive entries are spaced 1 day apart.
//!
//! Out-of-range epochs panic by default (Fail-Loudly): silently extending
//! a constant boundary value through a multi-day propagation accumulates
//! ~1 ms/day of wrong-physics drift in GMST and Earth rotation. Opt in
//! to JEOD-faithful clamp-and-inform via [`EopTable::with_clamp_out_of_range`].
//!
//! The default IERS table is bundled at compile time and reachable through
//! [`default_eop_table`]; refresh the binary via the `extract_eop_table`
//! regen binary (`crates/astrodyn_time/src/bin/extract_eop_table.rs`).

use log::warn;
use std::sync::atomic::{AtomicBool, Ordering};

/// On-disk magic identifying an EOP `.bin` fixture.
const EOP_BINARY_MAGIC: &[u8; 4] = b"EOPT";

/// On-disk format version.
const EOP_BINARY_VERSION: u32 = 1;

/// Bundled IERS EOP fixture (from JEOD v5.4.0; refresh via
/// `extract_eop_table`).
const DEFAULT_EOP_BIN: &[u8] = include_bytes!("../test_data/eop/iers_eop_c04.bin");

/// IERS Earth Orientation Parameters lookup driving UT1-TAI offset.
///
/// Each entry is `(tai_tjt_day, ut1_minus_tai_seconds)` with daily
/// spacing (1.0-day step in `tai_tjt_day`). Consecutive samples are
/// linearly interpolated to produce the running UT1-TAI offset.
///
/// # Out-of-range policy
///
/// By default, a TAI epoch outside `[when_tjt[0], when_tjt[last]]` makes
/// [`Self::ut1_minus_tai_seconds`] **panic** with a diagnostic naming the
/// requested epoch and the table's covered range. JEOD's
/// `time_converter_tai_ut1.cc` issues an `inform` and clamps to the
/// boundary value; that mode is reachable by chaining
/// [`Self::with_clamp_out_of_range(true)`](Self::with_clamp_out_of_range).
/// See `JEOD_invariants.md` row `TM.42` for the rationale (we pick the
/// fail-loud default for the same reason as `LeapSecondTable`: a
/// silently-extended boundary value accumulates wrong-physics drift in
/// GMST that downstream consumers cannot detect).
#[derive(Debug, Clone)]
pub struct EopTable {
    /// TAI TJT (days) of each daily sample, monotonically increasing
    /// with 1-day spacing.
    when_tjt: Vec<f64>,
    /// UT1 - TAI (seconds) at each `when_tjt` entry.
    val_seconds: Vec<f64>,
    /// When `false` (the default), out-of-range lookups panic. When
    /// `true`, the table emits a one-time `log::warn!` and returns the
    /// nearest boundary value (matches JEOD's `inform`-and-continue).
    clamp_out_of_range: bool,
}

/// Errors returned when loading an EOP binary fixture.
#[derive(Debug, thiserror::Error)]
pub enum EopLoadError {
    /// Bytes could not be read from disk.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Buffer is shorter than the format requires.
    #[error("EOP binary truncated at offset {offset} (need {needed} bytes, have {have})")]
    Truncated {
        /// Offset at which decoding ran out of bytes.
        offset: usize,
        /// Number of bytes the field needs.
        needed: usize,
        /// Number of bytes remaining in the buffer.
        have: usize,
    },
    /// Magic header bytes do not match `EOPT`.
    #[error("EOP binary magic mismatch: expected `EOPT`, got {0:?}")]
    BadMagic([u8; 4]),
    /// Version byte does not match the current parser.
    #[error("EOP binary version {found} unsupported (this build expects {expected})")]
    BadVersion {
        /// Version present on disk.
        found: u32,
        /// Version this build understands.
        expected: u32,
    },
    /// Table failed an invariant check at construction.
    #[error("EOP table invariant violated: {0}")]
    InvalidTable(String),
}

impl EopTable {
    /// Build an EOP table from `(tai_tjt_day, ut1_minus_tai_seconds)`
    /// pairs.
    ///
    /// # Panics
    ///
    /// Panics if the table is empty or `when_tjt` is not strictly
    /// monotonic. JEOD's converter assumes both invariants and we
    /// enforce at construction.
    pub fn from_entries(entries: Vec<(f64, f64)>) -> Self {
        // JEOD_INV: TM.42 — EOP table must be non-empty and strictly
        // monotonic in TAI TJT for the linear-interpolation bracket
        // search to be well-defined.
        assert!(!entries.is_empty(), "EOP table must not be empty");
        assert!(
            entries.windows(2).all(|w| w[0].0 < w[1].0),
            "EOP entries must be strictly increasing in TAI TJT"
        );
        let (when_tjt, val_seconds) = entries.into_iter().unzip();
        Self {
            when_tjt,
            val_seconds,
            clamp_out_of_range: false,
        }
    }

    /// Opt in to JEOD-faithful clamp-and-inform behavior on
    /// out-of-range lookups. When `true`, a `log::warn!` is emitted
    /// once per OOR direction (before-start / after-end) and the
    /// nearest boundary value is returned. Default `false` (panic).
    pub fn with_clamp_out_of_range(mut self, clamp: bool) -> Self {
        self.clamp_out_of_range = clamp;
        self
    }

    /// Number of daily entries in the table.
    pub fn len(&self) -> usize {
        self.when_tjt.len()
    }

    /// Whether the table is empty. Always `false` for a constructed
    /// table (`from_entries` panics on empty input).
    pub fn is_empty(&self) -> bool {
        self.when_tjt.is_empty()
    }

    /// First TAI TJT (days) covered by the table.
    pub fn first_tai_tjt(&self) -> f64 {
        self.when_tjt[0]
    }

    /// Last TAI TJT (days) covered by the table.
    pub fn last_tai_tjt(&self) -> f64 {
        self.when_tjt[self.when_tjt.len() - 1]
    }

    /// Return `UT1 - TAI` in seconds at `tai_tjt`.
    ///
    /// Linearly interpolates between the two adjacent daily samples,
    /// matching JEOD `time_converter_tai_ut1.cc::convert_a_to_b`.
    ///
    /// # Panics
    ///
    /// Panics when `tai_tjt` falls outside `[first_tai_tjt(),
    /// last_tai_tjt()]` and the table was not opted into clamp mode.
    pub fn ut1_minus_tai_seconds(&self, tai_tjt: f64) -> f64 {
        assert!(
            tai_tjt.is_finite(),
            "EopTable lookup requires a finite TAI TJT, got {tai_tjt}"
        );
        let last = self.when_tjt.len() - 1;
        let first_when = self.when_tjt[0];
        let last_when = self.when_tjt[last];

        if tai_tjt < first_when {
            // JEOD_INV: TM.42 — out-of-range EOP lookup before table start.
            assert!(
                self.clamp_out_of_range,
                "TAI TJT {tai_tjt} precedes first EOP table entry ({first_when}). \
                 Refusing to silently extend the boundary UT1-TAI value ({} s) \
                 across an uncovered epoch — that would accumulate ~1 ms/day of \
                 wrong-physics drift in GMST. Use a covered epoch (table covers \
                 TJT {first_when}..{last_when}), regenerate the EOP fixture for \
                 your epoch via `extract_eop_table`, or call \
                 `.with_clamp_out_of_range(true)` to opt into JEOD-faithful \
                 boundary clamping.",
                self.val_seconds[0]
            );
            // JEOD_INV: TM.42 — JEOD-faithful warn on opt-in clamp path.
            static WARNED_BEFORE: AtomicBool = AtomicBool::new(false);
            if !WARNED_BEFORE.swap(true, Ordering::Relaxed) {
                warn!(
                    "TAI TJT precedes first EOP table entry; \
                     using first value ({} s)",
                    self.val_seconds[0]
                );
            }
            return self.val_seconds[0];
        }
        if tai_tjt > last_when {
            // JEOD_INV: TM.42 — out-of-range EOP lookup past table end.
            assert!(
                self.clamp_out_of_range,
                "TAI TJT {tai_tjt} follows last EOP table entry ({last_when}). \
                 Refusing to silently extend the boundary UT1-TAI value ({} s) \
                 across an uncovered epoch — that would accumulate ~1 ms/day of \
                 wrong-physics drift in GMST. Refresh the EOP fixture via \
                 `extract_eop_table` (table covers TJT {first_when}..{last_when}), \
                 or call `.with_clamp_out_of_range(true)` to opt into \
                 JEOD-faithful boundary clamping.",
                self.val_seconds[last]
            );
            // JEOD_INV: TM.42 — JEOD-faithful warn on opt-in clamp path.
            static WARNED_AFTER: AtomicBool = AtomicBool::new(false);
            if !WARNED_AFTER.swap(true, Ordering::Relaxed) {
                warn!(
                    "TAI TJT follows last EOP table entry; \
                     using last value ({} s)",
                    self.val_seconds[last]
                );
            }
            return self.val_seconds[last];
        }

        // In-range: bracket-search for the largest `i` with
        // `when_tjt[i] <= tai_tjt < when_tjt[i+1]`. The table is daily
        // and uniformly spaced, so a direct index lookup from the
        // distance to the first sample is exact.
        let idx = self.bracket_index(tai_tjt);
        if idx == last {
            return self.val_seconds[last];
        }
        let prev_when = self.when_tjt[idx];
        let next_when = self.when_tjt[idx + 1];
        let prev_value = self.val_seconds[idx];
        let next_value = self.val_seconds[idx + 1];
        let gradient = (next_value - prev_value) / (next_when - prev_when);
        prev_value + (tai_tjt - prev_when) * gradient
    }

    /// Find the index `i` with `when_tjt[i] <= tai_tjt < when_tjt[i+1]`
    /// (or `last` when `tai_tjt == when_tjt[last]`).
    ///
    /// Uses a binary search; works even if the table's daily spacing
    /// drifts (e.g., a future regen produces a non-uniform table).
    fn bracket_index(&self, tai_tjt: f64) -> usize {
        // `partition_point` returns the first `i` with
        // `when_tjt[i] > tai_tjt`; the bracket starts one earlier. The
        // caller has already validated `tai_tjt >= when_tjt[0]`.
        let upper = self.when_tjt.partition_point(|&w| w <= tai_tjt);
        upper.saturating_sub(1)
    }

    /// Serialize the table to a compact little-endian binary blob.
    ///
    /// Layout (all little-endian):
    /// ```text
    /// magic    : 4 bytes ("EOPT")
    /// version  : u32     (1)
    /// count    : u64     (number of entries)
    /// when_tjt : count * f64
    /// val_s    : count * f64
    /// ```
    pub fn save_binary(&self, path: &std::path::Path) -> std::io::Result<()> {
        let bytes = self.to_binary_bytes();
        std::fs::write(path, bytes)
    }

    /// Encode the table as the binary blob `save_binary` writes to disk.
    pub fn to_binary_bytes(&self) -> Vec<u8> {
        let count = self.when_tjt.len();
        let mut buf = Vec::with_capacity(4 + 4 + 8 + 16 * count);
        buf.extend_from_slice(EOP_BINARY_MAGIC);
        buf.extend_from_slice(&EOP_BINARY_VERSION.to_le_bytes());
        buf.extend_from_slice(&(count as u64).to_le_bytes());
        for w in &self.when_tjt {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        for v in &self.val_seconds {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    /// Load an EOP table from a `.bin` file produced by [`Self::save_binary`].
    pub fn load_binary(path: &std::path::Path) -> Result<Self, EopLoadError> {
        let buf = std::fs::read(path)?;
        Self::load_binary_from_bytes(&buf)
    }

    /// Load an EOP table from a byte slice produced by [`Self::save_binary`]
    /// (or [`Self::to_binary_bytes`]).
    pub fn load_binary_from_bytes(buf: &[u8]) -> Result<Self, EopLoadError> {
        let mut cursor = 0usize;
        let mut take = |n: usize| -> Result<&[u8], EopLoadError> {
            if cursor + n > buf.len() {
                return Err(EopLoadError::Truncated {
                    offset: cursor,
                    needed: n,
                    have: buf.len() - cursor,
                });
            }
            let s = &buf[cursor..cursor + n];
            cursor += n;
            Ok(s)
        };

        let magic_slice = take(4)?;
        let mut magic = [0u8; 4];
        magic.copy_from_slice(magic_slice);
        if magic != *EOP_BINARY_MAGIC {
            return Err(EopLoadError::BadMagic(magic));
        }
        let mut v_buf = [0u8; 4];
        v_buf.copy_from_slice(take(4)?);
        let version = u32::from_le_bytes(v_buf);
        if version != EOP_BINARY_VERSION {
            return Err(EopLoadError::BadVersion {
                found: version,
                expected: EOP_BINARY_VERSION,
            });
        }
        let mut c_buf = [0u8; 8];
        c_buf.copy_from_slice(take(8)?);
        let count = u64::from_le_bytes(c_buf);
        // The header `count` bounds two parallel f64 arrays. A bogus
        // value would either truncate (handled by `take`) or make us
        // allocate gigabytes; cap at a reasonable upper bound that
        // also fits a 32-bit `usize`.
        if count > 10_000_000 {
            return Err(EopLoadError::InvalidTable(format!(
                "implausible entry count {count} (max 10,000,000)"
            )));
        }
        let count_usize = usize::try_from(count).map_err(|_| {
            EopLoadError::InvalidTable(format!(
                "entry count {count} does not fit in a usize on this target"
            ))
        })?;
        let mut when_tjt = Vec::with_capacity(count_usize);
        for _ in 0..count_usize {
            let mut b = [0u8; 8];
            b.copy_from_slice(take(8)?);
            when_tjt.push(f64::from_le_bytes(b));
        }
        let mut val_seconds = Vec::with_capacity(count_usize);
        for _ in 0..count_usize {
            let mut b = [0u8; 8];
            b.copy_from_slice(take(8)?);
            val_seconds.push(f64::from_le_bytes(b));
        }
        if when_tjt.is_empty() {
            return Err(EopLoadError::InvalidTable("empty table".to_string()));
        }
        if !when_tjt.windows(2).all(|w| w[0] < w[1]) {
            return Err(EopLoadError::InvalidTable(
                "when_tjt is not strictly monotonic".to_string(),
            ));
        }
        let entries: Vec<(f64, f64)> = when_tjt.into_iter().zip(val_seconds).collect();
        Ok(Self::from_entries(entries))
    }

    /// Parse a JEOD-generated `tai_to_ut1.cc` source file into an
    /// `EopTable`. Used by the `extract_eop_table` regen binary.
    ///
    /// JEOD writes one pair of lines per entry:
    /// ```text
    ///    TimeConverter_TAI_UT1_ptr->when_vec[<i>] = <tjt_days>; /* yyyy m d */
    ///    TimeConverter_TAI_UT1_ptr->val_vec[<i>] = <ut1_minus_tai_seconds>;
    /// ```
    /// We scan for the `when_vec[...]` and `val_vec[...]` assignments,
    /// build a sparse map keyed by index, and emit a contiguous table.
    pub fn parse_jeod_cc(src: &str) -> Result<Self, EopLoadError> {
        let mut whens: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
        let mut vals: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
        for line in src.lines() {
            let l = line.trim();
            if let Some((idx, val)) = parse_indexed_assignment(l, "when_vec") {
                whens.insert(idx, val);
            } else if let Some((idx, val)) = parse_indexed_assignment(l, "val_vec") {
                vals.insert(idx, val);
            }
        }
        if whens.is_empty() || vals.is_empty() {
            return Err(EopLoadError::InvalidTable(
                "no when_vec/val_vec assignments found in JEOD source".to_string(),
            ));
        }
        if whens.len() != vals.len() {
            return Err(EopLoadError::InvalidTable(format!(
                "when_vec / val_vec length mismatch ({} vs {})",
                whens.len(),
                vals.len()
            )));
        }
        let n = whens.len();
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let when = whens.get(&i).ok_or_else(|| {
                EopLoadError::InvalidTable(format!("missing when_vec[{i}] in JEOD source"))
            })?;
            let val = vals.get(&i).ok_or_else(|| {
                EopLoadError::InvalidTable(format!("missing val_vec[{i}] in JEOD source"))
            })?;
            entries.push((*when, *val));
        }
        Ok(Self::from_entries(entries))
    }
}

/// Extract `<idx>` and `<value>` from a line of the form
/// `<lhs>->key[<idx>] = <value>;` (any token before `->`). Returns
/// `None` if the line does not match.
fn parse_indexed_assignment(line: &str, key: &str) -> Option<(usize, f64)> {
    let needle = format!("->{key}[");
    let p = line.find(&needle)?;
    let after = &line[p + needle.len()..];
    let close = after.find(']')?;
    let idx_str = &after[..close];
    let rest = after.get(close + 1..)?.trim_start();
    let rest = rest.strip_prefix('=')?.trim_start();
    // Value runs until the trailing `;` (and optional inline `/* ... */`).
    let semi = rest.find(';')?;
    let val_str = rest[..semi].trim();
    let idx: usize = idx_str.parse().ok()?;
    let val: f64 = val_str.parse().ok()?;
    Some((idx, val))
}

/// Construct the bundled IERS EOP table (JEOD v5.4 source generation,
/// IERS EOP 14 C04 series) for production use.
///
/// The returned table panics on out-of-range lookups by default
/// (`TM.42`); callers needing JEOD-faithful clamp behavior should
/// chain [`EopTable::with_clamp_out_of_range(true)`](EopTable::with_clamp_out_of_range)
/// after this factory.
///
/// # Panics
///
/// Panics if the bundled binary fails to decode — this would indicate
/// a corrupt build artifact and is unrecoverable.
pub fn default_eop_table() -> EopTable {
    EopTable::load_binary_from_bytes(DEFAULT_EOP_BIN)
        .expect("bundled IERS EOP fixture: regenerate with `extract_eop_table`")
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "interpolation tests assert bit-exact equality at table sample points"
)]
mod tests {
    use super::*;

    fn synthetic_table() -> EopTable {
        // 5 daily samples; values rise by 0.001 s/day so interpolation
        // tests can assert exact midpoints.
        EopTable::from_entries(vec![
            (1000.0, 0.100),
            (1001.0, 0.101),
            (1002.0, 0.102),
            (1003.0, 0.103),
            (1004.0, 0.104),
        ])
    }

    #[test]
    fn exact_at_sample_points() {
        let t = synthetic_table();
        // Five fixed daily samples; integer days are exactly representable in `f64`.
        let pts: [(f64, f64); 5] = [
            (1000.0, 0.100),
            (1001.0, 0.101),
            (1002.0, 0.102),
            (1003.0, 0.103),
            (1004.0, 0.104),
        ];
        for (when, expected) in pts {
            assert_eq!(t.ut1_minus_tai_seconds(when), expected);
        }
    }

    #[test]
    fn linear_interpolation_midpoint() {
        let t = synthetic_table();
        // Midway between day 1001 and 1002 -> (0.101 + 0.102) / 2 = 0.1015.
        let v = t.ut1_minus_tai_seconds(1001.5);
        assert!((v - 0.101_5).abs() < 1e-15, "got {v}");
    }

    #[test]
    fn linear_interpolation_quarter_point() {
        let t = synthetic_table();
        // Quarter-day past 1002 -> 0.102 + 0.25 * (0.103 - 0.102) = 0.10225.
        let v = t.ut1_minus_tai_seconds(1002.25);
        assert!((v - 0.102_25).abs() < 1e-15, "got {v}");
    }

    // JEOD_INV: TM.42 — out-of-range before table start panics with
    // a diagnostic naming the requested epoch and the covered range.
    #[test]
    #[should_panic(expected = "precedes first EOP table entry")]
    fn out_of_range_before_panics_by_default() {
        let t = synthetic_table();
        let _ = t.ut1_minus_tai_seconds(999.999);
    }

    // JEOD_INV: TM.42 — out-of-range past table end panics with the
    // same diagnostic shape; this also covers the negative-test
    // requirement for the new TM row.
    #[test]
    #[should_panic(expected = "follows last EOP table entry")]
    fn out_of_range_after_panics_by_default() {
        let t = synthetic_table();
        let _ = t.ut1_minus_tai_seconds(1004.5);
    }

    #[test]
    fn clamp_opt_in_returns_boundary_value() {
        let t = synthetic_table().with_clamp_out_of_range(true);
        // Before start clamps to first value.
        assert_eq!(t.ut1_minus_tai_seconds(900.0), 0.100);
        // Past end clamps to last value.
        assert_eq!(t.ut1_minus_tai_seconds(2000.0), 0.104);
    }

    #[test]
    fn binary_round_trip() {
        let t = synthetic_table();
        let bytes = t.to_binary_bytes();
        let back = EopTable::load_binary_from_bytes(&bytes).expect("round-trip decode");
        assert_eq!(back.len(), t.len());
        for i in 0..t.len() {
            assert_eq!(back.when_tjt[i], t.when_tjt[i]);
            assert_eq!(back.val_seconds[i], t.val_seconds[i]);
        }
    }

    #[test]
    fn binary_rejects_bad_magic() {
        let mut bytes = synthetic_table().to_binary_bytes();
        bytes[0] = b'X';
        let err = EopTable::load_binary_from_bytes(&bytes).expect_err("bad magic must reject");
        assert!(matches!(err, EopLoadError::BadMagic(_)));
    }

    #[test]
    fn binary_rejects_bad_version() {
        let mut bytes = synthetic_table().to_binary_bytes();
        // Version is at offset 4..8, little-endian u32.
        bytes[4] = 99;
        let err = EopTable::load_binary_from_bytes(&bytes).expect_err("bad version must reject");
        assert!(matches!(err, EopLoadError::BadVersion { .. }));
    }

    #[test]
    fn parse_jeod_cc_minimal() {
        let src = "
            TimeConverter_TAI_UT1_ptr->last_index = 2;
            TimeConverter_TAI_UT1_ptr->when_vec[0] = -2335.0; /* 1962 1 1 */
            TimeConverter_TAI_UT1_ptr->val_vec[0] = -9.9673662;
            TimeConverter_TAI_UT1_ptr->when_vec[1] = -2334.0; /* 1962 1 2 */
            TimeConverter_TAI_UT1_ptr->val_vec[1] = -9.9679453;
            TimeConverter_TAI_UT1_ptr->when_vec[2] = -2333.0; /* 1962 1 3 */
            TimeConverter_TAI_UT1_ptr->val_vec[2] = -9.9684474;
        ";
        let t = EopTable::parse_jeod_cc(src).expect("parse minimal JEOD source");
        assert_eq!(t.len(), 3);
        assert_eq!(t.first_tai_tjt(), -2335.0);
        assert_eq!(t.last_tai_tjt(), -2333.0);
        assert_eq!(t.ut1_minus_tai_seconds(-2334.0), -9.9679453);
    }

    #[test]
    fn default_eop_table_loads() {
        let t = default_eop_table();
        // The JEOD v5.4 table starts on 1962-01-01 (TAI TJT = -2335)
        // and runs daily through late 2025 (TAI TJT 21032).
        assert!(
            t.len() > 20_000,
            "expected daily IERS EOP table, got {} entries",
            t.len()
        );
        assert_eq!(t.first_tai_tjt(), -2335.0);
        assert!(
            t.last_tai_tjt() > 21_000.0,
            "table should cover late-2020s epochs, got last TJT = {}",
            t.last_tai_tjt()
        );
    }

    #[test]
    fn default_eop_table_matches_jeod_source_value() {
        // TAI TJT 11178 is 1998-12-31 (MJD 51178). JEOD's generated
        // tai_to_ut1.cc records val_vec[13513] = -31.2824458 at that
        // entry (UT1-UTC + 0.7176 minus the 32-s TAI-UTC offset that
        // applied just after the 1999-01-01 leap second). Pin
        // bit-exact to the source value.
        let t = default_eop_table();
        let v = t.ut1_minus_tai_seconds(11_178.0);
        assert_eq!(v, -31.2824458, "EOP at TAI TJT 11178");
    }
}
