//! Parser for JEOD Trick time-initializer input files.
//!
//! Reads `Modified_data/time.py` / `Modified_data/date_and_time.py`
//! files like
//! [`verif/SIM_dyncomp/Modified_data/date_n_time/date_and_time.py`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/verif/SIM_dyncomp/Modified_data/date_n_time/date_and_time.py)
//! and produces the `(scale, calendar-date)` pair the Tier 3 harness
//! feeds into `jeod_time::SimulationTime` at sim startup.

use regex::Regex;
use std::path::Path;

/// Which time scale the `set_date_and_time` call specifies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInitializer {
    /// Date is UTC (most common: `time_utc.set_date_and_time(...)`)
    Utc,
    /// Date is TAI (e.g., `time_tai.set_date_and_time(...)`)
    Tai,
}

/// Time configuration parsed from a JEOD Trick time input file.
///
/// Parses files like `Modified_data/time.py` or `Modified_data/date_and_time.py`
/// containing:
/// - `initializer = "UTC"` or `initializer = "TAI"`
/// - `set_date_and_time(year, month, day, hour, minute, second)`
/// - `leap_sec_override_val = <float>` (TAI - UTC override)
/// - `tai_to_ut1_override_val = <float>` (UT1 - TAI override)
#[derive(Debug, Clone)]
pub struct TimeConfig {
    /// Which time scale the parsed date refers to.
    pub initializer: TimeInitializer,
    /// Calendar year.
    pub utc_year: i32,
    /// Calendar month (1–12).
    pub utc_month: u32,
    /// Calendar day of month (1–31).
    pub utc_day: u32,
    /// Hour of day (0–23).
    pub utc_hour: u32,
    /// Minute of hour (0–59).
    pub utc_minute: u32,
    /// Seconds of minute, in `[0.0, 61.0)`. The standard range is
    /// `[0.0, 60.0)`; values in `[60.0, 61.0)` are reserved for the
    /// positive UTC leap second (`23:59:60` and any fractional second
    /// within it).
    pub utc_second: f64,
    /// Override `TAI − UTC` in seconds, when JEOD's input forces a value.
    pub tai_utc_override: Option<f64>,
    /// Override `UT1 − TAI` in seconds, when JEOD's input forces a value.
    pub tai_to_ut1_override: Option<f64>,
}

impl TimeConfig {
    /// Compute UTC truncated Julian time (TJT = MJD - 40000) from the parsed date.
    pub fn utc_tjt(&self) -> f64 {
        let mjd = calendar_to_mjd(
            self.utc_year,
            self.utc_month,
            self.utc_day,
            self.utc_hour,
            self.utc_minute,
            self.utc_second,
        );
        mjd - 40_000.0
    }

    /// Compute TAI TJT from UTC TJT + TAI-UTC offset.
    ///
    /// Panics if `tai_utc_override` is None (caller must ensure the time file
    /// contains a leap second override, or use the leap second table directly).
    pub fn tai_tjt(&self) -> f64 {
        match self.initializer {
            TimeInitializer::Tai => {
                // The date IS TAI — the TJT from the calendar date is TAI TJT directly.
                self.utc_tjt() // Misleadingly named, but calendar_to_mjd just does date math
            }
            TimeInitializer::Utc => {
                let tai_utc = self
                    .tai_utc_override
                    .expect("tai_utc_override not set in time config; cannot compute TAI TJT");
                self.utc_tjt() + tai_utc / 86_400.0
            }
        }
    }

    /// Compute TAI TJT using an external TAI-UTC offset (from a leap second table).
    ///
    /// Use this when the time file does not contain a `leap_sec_override_val`
    /// but the epoch date is UTC and the caller can supply the TAI-UTC offset.
    pub fn tai_tjt_with_offset(&self, tai_utc_s: f64) -> f64 {
        assert_eq!(
            self.initializer,
            TimeInitializer::Utc,
            "tai_tjt_with_offset only valid for UTC-initialized epochs"
        );
        self.utc_tjt() + tai_utc_s / 86_400.0
    }

    /// Return the UT1-TAI offset in seconds, if overridden in the time file.
    pub fn ut1_tai_offset(&self) -> Option<f64> {
        self.tai_to_ut1_override
    }
}

/// Parse time configuration from a JEOD Trick Python file.
///
/// Scans the file for:
/// - `set_date_and_time(year, month, day, hour, minute, second)`
/// - `leap_sec_override_val = <float>`
/// - `tai_to_ut1_override_val = <float>` or expressions like `0.0115221 - 4.2`
///
/// # Panics
/// Panics if the file cannot be read or does not contain a `set_date_and_time` call.
pub fn load_time_config(py_path: &Path) -> TimeConfig {
    let content = std::fs::read_to_string(py_path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", py_path.display(), e));

    parse_time_config_str(&content, py_path)
}

/// Parse time config from string content (for testing).
fn parse_time_config_str(content: &str, source: &Path) -> TimeConfig {
    // Match: set_date_and_time(year, month, day, hour, minute, second)
    let date_re = Regex::new(
        r"set_date_and_time\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)",
    )
    .unwrap();

    // Match: leap_sec_override_val = <float>
    let leap_re = Regex::new(r"leap_sec_override_val\s*=\s*([-\d.eE+]+)").unwrap();

    // Match: tai_to_ut1_override_val = <expression>
    // Handles both simple values and expressions like "0.0115221 - 4.2"
    let ut1_re = Regex::new(r"tai_to_ut1_override_val\s*=\s*(.+)").unwrap();

    // Match: initializer = "UTC" or "TAI"
    let init_re = Regex::new(r#"initializer\s*=\s*"(UTC|TAI)""#).unwrap();

    let mut initializer = None;
    let mut utc_year = None;
    let mut utc_month = None;
    let mut utc_day = None;
    let mut utc_hour = None;
    let mut utc_minute = None;
    let mut utc_second = None;
    let mut tai_utc_override = None;
    let mut tai_to_ut1_override = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            continue;
        }

        if let Some(cap) = init_re.captures(trimmed) {
            initializer = Some(match &cap[1] {
                "TAI" => TimeInitializer::Tai,
                "UTC" => TimeInitializer::Utc,
                other => panic!(
                    "Unknown time initializer '{}' in {}",
                    other,
                    source.display()
                ),
            });
        }

        if let Some(cap) = date_re.captures(trimmed) {
            utc_year = Some(cap[1].parse::<i32>().unwrap());
            utc_month = Some(cap[2].parse::<u32>().unwrap());
            utc_day = Some(cap[3].parse::<u32>().unwrap());
            utc_hour = Some(cap[4].parse::<u32>().unwrap());
            utc_minute = Some(cap[5].parse::<u32>().unwrap());
            utc_second = Some(cap[6].parse::<f64>().unwrap());
        }

        if let Some(cap) = leap_re.captures(trimmed) {
            tai_utc_override = Some(cap[1].parse::<f64>().unwrap());
        }

        if let Some(cap) = ut1_re.captures(trimmed) {
            tai_to_ut1_override = Some(eval_simple_sum(cap[1].trim()));
        }
    }

    // All date fields are captured by a single regex — if year is missing, none are set.
    if utc_year.is_none() {
        panic!("No set_date_and_time() found in {}", source.display());
    }
    TimeConfig {
        initializer: initializer.unwrap_or(TimeInitializer::Utc),
        utc_year: utc_year.unwrap(),
        utc_month: utc_month.unwrap(),
        utc_day: utc_day.unwrap(),
        utc_hour: utc_hour.unwrap(),
        utc_minute: utc_minute.unwrap(),
        utc_second: utc_second.unwrap(),
        tai_utc_override,
        tai_to_ut1_override,
    }
}

/// Evaluate a simple arithmetic expression with addition and subtraction only.
///
/// Handles patterns like: `32`, `-32.469`, `0.0115221 - 4.2`
fn eval_simple_sum(expr: &str) -> f64 {
    let trimmed = expr.trim();

    // Try direct parse first (handles simple values and negative numbers)
    if let Ok(val) = trimmed.parse::<f64>() {
        return val;
    }

    // Tokenize: split into signed numeric tokens.
    // Walk characters, splitting on +/- that are NOT part of scientific notation.
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();

    for ch in trimmed.chars() {
        if (ch == '+' || ch == '-') && !current.trim().is_empty() {
            // Check if this sign is part of scientific notation (e.g., 1e-5)
            let trimmed_current = current.trim_end();
            if trimmed_current.ends_with('e') || trimmed_current.ends_with('E') {
                current.push(ch);
                continue;
            }
            tokens.push(current.trim().to_string());
            current.clear();
            if ch == '-' {
                current.push('-');
            }
        } else {
            current.push(ch);
        }
    }
    if !current.trim().is_empty() {
        tokens.push(current.trim().to_string());
    }

    tokens
        .iter()
        .map(|t| {
            // Remove internal whitespace (e.g., "- 4.2" → "-4.2")
            let cleaned: String = t.chars().filter(|c| !c.is_whitespace()).collect();
            cleaned
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("Cannot parse '{}' as f64: {}", cleaned, e))
        })
        .sum()
}

/// Convert a Gregorian calendar date to Modified Julian Date (MJD).
///
/// Uses the standard algorithm valid for dates after 1582-10-15 (Gregorian calendar).
fn calendar_to_mjd(year: i32, month: u32, day: u32, hour: u32, minute: u32, second: f64) -> f64 {
    // Algorithm: Meeus, Astronomical Algorithms, Ch. 7
    let (y, m) = if month <= 2 {
        (year - 1, month + 12)
    } else {
        (year, month)
    };

    let a = y / 100;
    let b = 2 - a + a / 4;

    let jd = (365.25 * (y + 4716) as f64).floor()
        + (30.6001 * (m + 1) as f64).floor()
        + day as f64
        + b as f64
        - 1524.5;

    let day_fraction = (hour as f64 + minute as f64 / 60.0 + second / 3600.0) / 24.0;

    // MJD = JD - 2400000.5
    jd + day_fraction - 2_400_000.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_calendar_to_mjd_j2000() {
        // J2000.0 = 2000-01-01 12:00:00 TT → MJD 51544.5
        let mjd = calendar_to_mjd(2000, 1, 1, 12, 0, 0.0);
        assert!(
            (mjd - 51544.5).abs() < 1e-10,
            "J2000 MJD: expected 51544.5, got {}",
            mjd
        );
    }

    #[test]
    fn test_calendar_to_mjd_2007_nov_20() {
        // 2007-11-20 00:00:00 UTC → MJD 54424.0 → TJT 14424.0
        let mjd = calendar_to_mjd(2007, 11, 20, 0, 0, 0.0);
        let tjt = mjd - 40_000.0;
        assert!(
            (tjt - 14424.0).abs() < 1e-10,
            "SIM_dyncomp epoch TJT: expected 14424.0, got {}",
            tjt
        );
    }

    #[test]
    fn test_calendar_to_mjd_1991_jan_01() {
        // 1991-01-01 00:00:00 → MJD 48257.0 → TJT 8257.0
        let mjd = calendar_to_mjd(1991, 1, 1, 0, 0, 0.0);
        let tjt = mjd - 40_000.0;
        assert!(
            (tjt - 8257.0).abs() < 1e-10,
            "1991-01-01 TJT: expected 8257.0, got {}",
            tjt
        );
    }

    #[test]
    fn test_eval_simple_sum() {
        assert!((eval_simple_sum("32") - 32.0).abs() < 1e-15);
        assert!((eval_simple_sum("-32.469") - (-32.469)).abs() < 1e-15);
        assert!((eval_simple_sum("0.0115221 - 4.2") - (0.0115221 - 4.2)).abs() < 1e-15);
        assert!((eval_simple_sum("1.5 + 2.3") - 3.8).abs() < 1e-15);
        assert!((eval_simple_sum("1e2 + 3") - 103.0).abs() < 1e-15);
    }

    #[test]
    fn test_parse_dyncomp_time() {
        let content = r#"
jeod_time.time_manager_init.initializer = "UTC"
jeod_time.time_manager_init.sim_start_format = trick.TimeEnum.calendar
jeod_time.time_utc.set_date_and_time(2007, 11, 20, 0, 0, 0.0)

jeod_time.time_tai.initialize_from_name = "UTC"
jeod_time.time_ut1.initialize_from_name = "TAI"

jeod_time.time_converter_tai_utc.override_data_table = True
jeod_time.time_converter_tai_utc.leap_sec_override_val = 32

jeod_time.time_converter_tai_ut1.override_data_table = True
jeod_time.time_converter_tai_ut1.tai_to_ut1_override_val = -32.469
"#;
        let cfg = parse_time_config_str(content, &PathBuf::from("test"));

        assert_eq!(cfg.initializer, TimeInitializer::Utc);
        assert_eq!(cfg.utc_year, 2007);
        assert_eq!(cfg.utc_month, 11);
        assert_eq!(cfg.utc_day, 20);
        assert_eq!(cfg.utc_hour, 0);
        assert_eq!(cfg.utc_minute, 0);
        assert!((cfg.utc_second - 0.0).abs() < 1e-15);
        assert!((cfg.tai_utc_override.unwrap() - 32.0).abs() < 1e-15);
        assert!((cfg.tai_to_ut1_override.unwrap() - (-32.469)).abs() < 1e-15);

        // TJT should be 14424.0
        assert!(
            (cfg.utc_tjt() - 14424.0).abs() < 1e-10,
            "UTC TJT: {}",
            cfg.utc_tjt()
        );

        // TAI TJT = 14424.0 + 32/86400
        let expected_tai_tjt = 14424.0 + 32.0 / 86400.0;
        assert!(
            (cfg.tai_tjt() - expected_tai_tjt).abs() < 1e-12,
            "TAI TJT: {}",
            cfg.tai_tjt()
        );
    }

    #[test]
    fn test_parse_apollo_time_with_expression() {
        let content = r#"
jeod_time.time_utc.set_date_and_time(1969, 7, 16, 13, 44, 0.0)
jeod_time.time_converter_tai_utc.override_data_table = True
jeod_time.time_converter_tai_utc.leap_sec_override_val = 4.2
jeod_time.time_converter_tai_ut1.override_data_table = True
jeod_time.time_converter_tai_ut1.tai_to_ut1_override_val = 0.0115221 - 4.2
"#;
        let cfg = parse_time_config_str(content, &PathBuf::from("test"));

        assert_eq!(cfg.utc_year, 1969);
        assert_eq!(cfg.utc_month, 7);
        assert!((cfg.tai_utc_override.unwrap() - 4.2).abs() < 1e-15);
        assert!(
            (cfg.tai_to_ut1_override.unwrap() - (0.0115221 - 4.2)).abs() < 1e-15,
            "UT1-TAI: {}",
            cfg.tai_to_ut1_override.unwrap()
        );
    }

    #[test]
    fn test_parse_tai_initialized_time() {
        // SRP SIM date_and_time.py: TAI initializer, date is TAI not UTC.
        let content = r#"
jeod_time.time_manager_init.initializer = "TAI"
jeod_time.time_manager_init.sim_start_format = trick.TimeEnum.calendar
jeod_time.time_tai.set_date_and_time(1998, 12, 1, 0, 0, 31.0)
jeod_time.time_tai.update_from_name = "Dyn"
jeod_time.time_tt.initialize_from_name = "TAI"
jeod_time.time_tt.update_from_name = "TAI"
"#;
        let cfg = parse_time_config_str(content, &PathBuf::from("test"));

        assert_eq!(cfg.initializer, TimeInitializer::Tai);
        assert_eq!(cfg.utc_year, 1998);
        assert_eq!(cfg.utc_month, 12);
        assert_eq!(cfg.utc_day, 1);
        assert!((cfg.utc_second - 31.0).abs() < 1e-15);
        assert!(cfg.tai_utc_override.is_none());

        // TAI TJT: 1998-12-01 00:00:31 TAI → MJD 51148.0 + 31/86400 → TJT 11148.0 + 31/86400
        // Note: calendar_to_mjd converts seconds via hour/minute/second path, so the
        // result differs from the exact 31.0/86400.0 by ~8e-14 due to floating-point
        // rounding in the intermediate 31/3600/24 computation.
        let expected_tai_tjt = 11148.0 + 31.0 / 86400.0;
        assert!(
            (cfg.tai_tjt() - expected_tai_tjt).abs() < 1e-10,
            "TAI TJT: expected {}, got {}",
            expected_tai_tjt,
            cfg.tai_tjt()
        );
    }

    #[test]
    fn test_parse_utc_no_overrides() {
        // NED SIM date_and_time.py: UTC initializer, no leap sec override.
        let content = r#"
jeod_time.time_manager_init.initializer = "UTC"
jeod_time.time_manager_init.sim_start_format = trick.TimeEnum.calendar
jeod_time.time_utc.set_date_and_time(1991, 1, 1, 0, 0, 0.0)
jeod_time.time_tai.initialize_from_name = "UTC"
jeod_time.time_ut1.initialize_from_name = "TAI"
"#;
        let cfg = parse_time_config_str(content, &PathBuf::from("test"));

        assert_eq!(cfg.initializer, TimeInitializer::Utc);
        assert_eq!(cfg.utc_year, 1991);
        assert!(cfg.tai_utc_override.is_none());
        assert!(cfg.tai_to_ut1_override.is_none());

        // UTC TJT: 1991-01-01 → MJD 48257.0 → TJT 8257.0
        assert!(
            (cfg.utc_tjt() - 8257.0).abs() < 1e-10,
            "UTC TJT: {}",
            cfg.utc_tjt()
        );

        // tai_tjt_with_offset with external TAI-UTC=26s
        let expected_tai_tjt = 8257.0 + 26.0 / 86400.0;
        assert!(
            (cfg.tai_tjt_with_offset(26.0) - expected_tai_tjt).abs() < 1e-12,
            "TAI TJT with offset: {}",
            cfg.tai_tjt_with_offset(26.0)
        );
    }
}
