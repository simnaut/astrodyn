//! Tier 3: SIM_MET — MET atmosphere density/temperature parity.
//!
//! Static cross-validation: each CSV row has altitude/latitude/longitude and
//! the corresponding MET atmosphere density and temperature. We call our MET
//! model at the same coordinates and compare outputs.

mod sim_test_helpers;
use sim_test_helpers::*;

use jeod_atmosphere::met;

/// A MET reference data point from the CSV.
struct MetRefPoint {
    density: f64,
    temperature: f64,
    altitude: f64,  // metres
    latitude: f64,  // radians
    longitude: f64, // radians
}

fn load_met_csv(path: &std::path::Path) -> Vec<MetRefPoint> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_MET CSV from {}: {e}\n\
             Generate with Docker (see CLAUDE.md).",
            path.display()
        )
    });
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 6,
            "line {}: expected >=6 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(MetRefPoint {
            density: p(1),
            temperature: p(2),
            altitude: p(3),
            latitude: p(4),
            longitude: p(5),
        });
    }
    records
}

fn run_met_scenario(label: &str, csv_name: &str) {
    // Default epoch: 2000-01-01 01:31:48 UTC (from input_core.py)
    // MJD = 51544.0 + 5508/86400 = 51544.06375, TJT = MJD - 40000
    run_met_scenario_with_tjt(label, csv_name, 11544.06375);
}

fn run_met_scenario_with_tjt(label: &str, csv_name: &str, tjt: f64) {
    let csv_path = test_data_path(csv_name);
    let ref_points = load_met_csv(&csv_path);
    assert!(!ref_points.is_empty(), "{label}: no reference data");

    // Match JEOD SIM_MET input: F10=230, F10B=230, AP=20.30
    let atmos = met::MetAtmosphere {
        f10: 230.0,
        f10b: 230.0,
        geo_index: 20.3,
        geo_index_type: met::GeoIndexType::Ap,
    };

    let mut max_density_rel_err = 0.0_f64;
    let mut max_temp_rel_err = 0.0_f64;
    let mut count = 0;

    for (i, pt) in ref_points.iter().enumerate() {
        if pt.density <= 0.0 || pt.altitude < 90_000.0 {
            continue; // Skip below MET valid range (~90 km)
        }

        let alt_km = pt.altitude / 1000.0; // Convert m → km
        let state = atmos.density(alt_km, pt.latitude, pt.longitude, tjt);

        let density_rel_err = if pt.density > 0.0 {
            ((state.density - pt.density) / pt.density).abs()
        } else {
            0.0
        };
        let temp_rel_err = if pt.temperature > 0.0 {
            ((state.temperature - pt.temperature) / pt.temperature).abs()
        } else {
            0.0
        };

        max_density_rel_err = max_density_rel_err.max(density_rel_err);
        max_temp_rel_err = max_temp_rel_err.max(temp_rel_err);
        count += 1;

        // With matching F10.7/AP inputs and correct TJT, density matches to
        // machine precision. Use 1e-12 tolerance to allow for platform-level
        // floating-point variation.
        assert!(
            density_rel_err < 1e-12,
            "{label} point {i}: density rel error {density_rel_err:.4e} exceeds 1e-12 \
             (ours={:.4e}, JEOD={:.4e}, alt={:.1} km)",
            state.density,
            pt.density,
            alt_km
        );
        assert!(
            temp_rel_err < 1e-12,
            "{label} point {i}: temperature rel error {temp_rel_err:.4e} exceeds 1e-12 \
             (ours={:.4}, JEOD={:.4}, alt={:.1} km)",
            state.temperature,
            pt.temperature,
            alt_km
        );
    }

    assert!(count > 0, "{label}: no valid points evaluated");
    println!(
        "  {label}: {count} points, max density rel err = {max_density_rel_err:.4e}, \
         max temp rel err = {max_temp_rel_err:.4e}"
    );
}

#[test]
fn tier3_simulation_met_t01() {
    // RUN_T01: epoch 1995-01-01 00:00:01 UTC
    // MJD = 49718.0 + 1/86400, TJT = MJD - 40000 = 9718.000012
    run_met_scenario_with_tjt("met_t01", "met_t01_met.csv", 9_718.000_011_574_077);
}

#[test]
fn tier3_simulation_met_t02() {
    run_met_scenario("met_t02", "met_t02_met.csv");
}

#[test]
fn tier3_simulation_met_t03_gram() {
    run_met_scenario("met_t03_gram", "met_t03_gram_met.csv");
}
