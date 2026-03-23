/// A single entry from JEOD's `Leap_Second.dat`.
///
/// Each non-comment line: `MJD  day month year  TAI-UTC`
#[derive(Debug, Clone)]
pub struct LeapSecondEntry {
    /// Modified Julian Date of the leap second boundary.
    pub mjd: f64,
    pub day: u32,
    pub month: u32,
    pub year: u32,
    /// TAI minus UTC in seconds (cumulative offset).
    pub tai_utc: f64,
}

/// Load the leap second table from JEOD's `Leap_Second.dat`.
///
/// File location: `models/environment/time/data/Leap_Second.dat`
///
/// # Panics
/// Panics if the file cannot be read or contains malformed data.
pub fn load_leap_second_table(jeod_root: &std::path::Path) -> Vec<LeapSecondEntry> {
    let path = jeod_root.join("models/environment/time/data/Leap_Second.dat");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));

    let mut entries = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 5 {
            continue;
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
    use crate::jeod_path;

    #[test]
    fn leap_second_parser_spot_check() {
        let root = jeod_path();
        if !root.exists() {
            panic!(
                "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
                root.display()
            );
        }
        let entries = load_leap_second_table(&root);
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
