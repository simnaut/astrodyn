//! Parser for JEOD's `Leap_Second.dat` table.
//!
//! Reads
//! [`models/environment/time/data/Leap_Second.dat`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/time/data/Leap_Second.dat)
//! from JEOD v5.4.0 (mirrored at `test_data/time/Leap_Second.dat`)
//! into a `Vec<LeapSecondEntry>` consumed by `jeod_time::LeapSecondTable`.

/// A single entry from JEOD's `Leap_Second.dat`.
///
/// Each non-comment line: `MJD  day month year  TAI-UTC`
#[derive(Debug, Clone)]
pub struct LeapSecondEntry {
    /// Modified Julian Date of the leap-second boundary.
    pub mjd: f64,
    /// Calendar day of month (1–31).
    pub day: u32,
    /// Calendar month (1–12).
    pub month: u32,
    /// Calendar year (e.g. `2017`).
    pub year: u32,
    /// TAI minus UTC in seconds (cumulative offset).
    pub tai_utc: f64,
}

/// Load the leap second table from the committed
/// `test_data/time/Leap_Second.dat` fixture.
///
/// The fixture is a verbatim copy of JEOD's
/// `models/environment/time/data/Leap_Second.dat`. Refresh after a JEOD
/// upgrade via `cargo run -p jeod_test_data --bin extract_jeod_validation`.
///
/// # Panics
/// Panics if the fixture is missing or malformed; the message includes
/// the regen command.
pub fn load_leap_second_table() -> Vec<LeapSecondEntry> {
    let path = crate::tier3_csv::test_data_path("time/Leap_Second.dat");
    let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "Cannot read {}: {e}. Regenerate with: cargo run -p jeod_test_data \
             --bin extract_jeod_validation",
            path.display(),
        )
    });
    parse_leap_second_table(&content, &path)
}

/// Parse the line-oriented `Leap_Second.dat` content into entries.
///
/// Used by [`load_leap_second_table`] (committed fixture); kept public
/// in case the regen binary needs to verify a fresh JEOD copy.
pub fn parse_leap_second_table(content: &str, source: &std::path::Path) -> Vec<LeapSecondEntry> {
    let mut entries = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 5 {
            panic!(
                "{}:{}: expected at least 5 fields, got {}: {:?}",
                source.display(),
                line_num + 1,
                fields.len(),
                line
            );
        }
        entries.push(LeapSecondEntry {
            mjd: fields[0].parse().unwrap(),
            day: fields[1].parse().unwrap(),
            month: fields[2].parse().unwrap(),
            year: fields[3].parse().unwrap(),
            tai_utc: fields[4].parse().unwrap(),
        });
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leap_second_parser_spot_check() {
        let entries = load_leap_second_table();
        assert_eq!(entries.len(), 28, "Expected 28 leap second entries");

        // First entry: 1972-01-01, MJD 41317, TAI-UTC = 10
        assert_eq!(entries[0].mjd, 41317.0);
        assert_eq!(entries[0].day, 1);
        assert_eq!(entries[0].month, 1);
        assert_eq!(entries[0].year, 1972);
        assert_eq!(entries[0].tai_utc, 10.0);

        // Last entry: 2017-01-01, MJD 57754, TAI-UTC = 37
        let last = entries.last().unwrap();
        assert_eq!(last.mjd, 57754.0);
        assert_eq!(last.year, 2017);
        assert_eq!(last.tai_utc, 37.0);
    }
}
