//! Loader for the JEOD `SIM_MET` atmosphere reference CSVs.
//!
//! The CSVs are produced by Trick (via `xtask regenerate-tier3`) running
//! `models/environment/atmosphere/MET/verif/SIM_MET/` against the inputs in
//! `SET_test/RUN_T0{1,2,3}_*`. Each CSV row holds the geodetic position
//! that JEOD's `MET_atmosphere::update_atmosphere` was called with plus the
//! density and temperature it returned.
//!
//! Header (columns in order):
//!   `sys.exec.out.time {s}`,
//!   `vehicle.atmos_state.density {kg/m3}`,
//!   `vehicle.atmos_state.temperature {K}`,
//!   `vehicle.pos.ellip_coords.altitude {m}`,
//!   `vehicle.pos.ellip_coords.latitude {rad}`,
//!   `vehicle.pos.ellip_coords.longitude {rad}`
//!
//! Solar-activity and time inputs are constant across every row of every
//! RUN; they live in `SET_test/input_core.py` and are encoded in the
//! consuming Tier 2 test rather than parsed from the CSV.
//!
//! The shape mirrors `gravity_verif::load_gravity_test_cases`: a struct
//! per row, a single loader function that panics with the file path on any
//! parse failure.

use std::path::Path;

/// One sampled output row from a `SIM_MET` Trick CSV.
#[derive(Debug, Clone, Copy)]
pub struct MetSample {
    /// Sample index (Trick `sys.exec.out.time` in seconds; the sim freezes
    /// `jeod_time` at t=0 so this is just an ordinal counter).
    pub sample_index: f64,
    /// Atmospheric density in kg/m^3 reported by JEOD's MET model.
    pub density: f64,
    /// Atmospheric temperature in K reported by JEOD's MET model.
    pub temperature: f64,
    /// Geodetic altitude in metres of the evaluation point.
    pub altitude_m: f64,
    /// Geodetic latitude in radians of the evaluation point.
    pub latitude_rad: f64,
    /// Geodetic longitude in radians of the evaluation point.
    pub longitude_rad: f64,
}

/// Load every row of a `SIM_MET` reference CSV.
///
/// `path` should point at a file under `test_data/met_t0*_*.csv`. The first
/// line is the header — it is skipped. Every subsequent non-blank line must
/// have at least six comma-separated numeric fields in the order documented
/// in [`MetSample`].
///
/// # Panics
/// Panics if the file cannot be read, if the header is missing, if any row
/// has fewer than six fields, or if any field fails to parse as `f64`.
pub fn load_met_run_csv(path: &Path) -> Vec<MetSample> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));

    let mut lines = content.lines();
    let header = lines
        .next()
        .unwrap_or_else(|| panic!("{}: empty file (no header)", path.display()));
    assert!(
        header.contains("density") && header.contains("temperature"),
        "{}: unexpected header (missing density/temperature columns): {header:?}",
        path.display()
    );

    let mut samples = Vec::new();
    for (idx, line) in lines.enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        assert!(
            fields.len() >= 6,
            "{}: row {} has {} fields, expected >= 6: {:?}",
            path.display(),
            idx + 2, // +1 for header, +1 for 1-indexed
            fields.len(),
            line
        );

        let parse = |s: &str, col: &str| -> f64 {
            s.parse().unwrap_or_else(|e| {
                panic!(
                    "{}: row {} column {} ({:?}) failed to parse as f64: {}",
                    path.display(),
                    idx + 2,
                    col,
                    s,
                    e
                )
            })
        };

        samples.push(MetSample {
            sample_index: parse(fields[0], "time"),
            density: parse(fields[1], "density"),
            temperature: parse(fields[2], "temperature"),
            altitude_m: parse(fields[3], "altitude"),
            latitude_rad: parse(fields[4], "latitude"),
            longitude_rad: parse(fields[5], "longitude"),
        });
    }

    assert!(
        !samples.is_empty(),
        "{}: no data rows after the header",
        path.display()
    );
    samples
}
