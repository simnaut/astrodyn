//! Sync check: hardcoded `default_leap_second_table()` matches the
//! committed `test_data/Leap_Second.dat` byte for byte.

#![allow(
    clippy::float_cmp,
    reason = "TAI-UTC offsets are integer-valued; byte-for-byte sync requires bit-exact equality"
)]
//!
//! `default_leap_second_table()` is a 28-entry hardcoded snapshot of
//! NASA/USNO leap-second data, mirrored verbatim from JEOD's
//! `models/environment/time/data/Leap_Second.dat`. The hardcoded
//! values are what production code uses at runtime (no startup file
//! IO); the committed `.dat` file is the canonical source of truth.
//!
//! This test guards against silent drift — if a future leap second is
//! added to the `.dat` (refreshed via
//! `cargo run -p astrodyn_verif_jeod --bin extract_jeod_validation`)
//! but the hardcoded vector isn't updated, this test fails with a
//! diff naming the missing rows, instead of production code silently
//! computing TAI-UTC offsets that disagree with NASA's authoritative
//! values.
//!
//! Refresh procedure when a new leap second lands:
//!   1. `cargo run -p astrodyn_verif_jeod --bin extract_jeod_validation`
//!      (or `cp $JEOD_HOME/models/environment/time/data/Leap_Second.dat
//!      crates/astrodyn_time/test_data/Leap_Second.dat`)
//!   2. Add the new `(mjd, tai_utc)` row to `default_leap_second_table()`
//!      in `crates/astrodyn_time/src/leap_second.rs`.
//!   3. Re-run this test.

use astrodyn_time::epoch::mjd_to_tjt;
use astrodyn_time::leap_second::default_leap_second_table;
use std::path::PathBuf;

/// One non-comment line from `Leap_Second.dat`:
///   `MJD  day month year  TAI-UTC`
#[derive(Debug, PartialEq)]
struct DatEntry {
    mjd: f64,
    tai_utc: f64,
}

/// Parse `Leap_Second.dat` content into entries, skipping comment and
/// blank lines. Format: each data line has 5 whitespace-separated
/// fields — `MJD day month year TAI-UTC`. We only need MJD and the
/// TAI-UTC offset for the sync check; the calendar columns are
/// redundant with the MJD.
fn parse_dat(content: &str) -> Vec<DatEntry> {
    let mut out = Vec::new();
    for (i, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(
            fields.len(),
            5,
            "Leap_Second.dat:{}: expected 5 fields (mjd day month year tai_utc), got {}: {:?}",
            i + 1,
            fields.len(),
            line,
        );
        let mjd: f64 = fields[0]
            .parse()
            .unwrap_or_else(|_| panic!("Leap_Second.dat:{}: bad MJD `{}`", i + 1, fields[0]));
        let tai_utc: f64 = fields[4]
            .parse()
            .unwrap_or_else(|_| panic!("Leap_Second.dat:{}: bad TAI-UTC `{}`", i + 1, fields[4]));
        out.push(DatEntry { mjd, tai_utc });
    }
    out
}

fn dat_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data/Leap_Second.dat")
}

#[test]
fn tier2_leap_second_sync() {
    let path = dat_path();
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", path.display()));
    let parsed = parse_dat(&content);
    let table = default_leap_second_table();

    assert_eq!(
        parsed.len(),
        table.len(),
        "Entry-count mismatch:\n  Leap_Second.dat: {} entries\n  default_leap_second_table(): {} entries\n  \
         Refresh the hardcoded table in crates/astrodyn_time/src/leap_second.rs.",
        parsed.len(),
        table.len(),
    );

    // Probe each parsed boundary against the hardcoded table. We use
    // `tai_utc_at_utc_tjt` because it returns the offset that becomes
    // active **at** the boundary's UTC TJT — exactly the value the
    // .dat encodes for that MJD row.
    let mut diffs = Vec::new();
    for entry in &parsed {
        let utc_tjt = mjd_to_tjt(entry.mjd);
        let actual = table.tai_utc_at_utc_tjt(utc_tjt);
        if actual != entry.tai_utc {
            diffs.push(format!(
                "  MJD {}: .dat says {} s, default_leap_second_table() says {} s",
                entry.mjd, entry.tai_utc, actual,
            ));
        }
    }
    assert!(
        diffs.is_empty(),
        "Hardcoded leap-second table drifted from Leap_Second.dat:\n{}\n\
         Update the (mjd, tai_utc) array in default_leap_second_table().",
        diffs.join("\n"),
    );
}
