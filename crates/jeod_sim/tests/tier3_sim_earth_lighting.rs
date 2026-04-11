//! Tier 3: SIM_LIGHT_CIR — Earth lighting cross-validation.
//!
//! Validates `circle_intersect()` against JEOD SIM_LIGHT_CIR reference data.
//! Each CSV has parametric geometry inputs and JEOD's computed outputs.
//!
//! JEOD's logged `area` field is always 0 in these CSVs due to Trick scheduling
//! lag (area is computed from the previous step's inputs). We validate our
//! `circle_intersect` with geometric bounds: area must be non-negative, at most
//! the smaller circle's area, and consistent with the overlap/containment
//! geometry.

mod sim_test_helpers;
use sim_test_helpers::*;

use jeod_interactions::earth_lighting::circle_intersect;

struct LightingRecord {
    time: f64,
    r_bottom: f64,
    r_top: f64,
    d_centers: f64,
}

fn load_lighting_csv(path: &std::path::Path) -> Vec<LightingRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_LIGHT_CIR CSV from {}: {e}\n\
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
            f.len() >= 15,
            "line {}: expected >=15 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(LightingRecord {
            time: p(0),
            r_bottom: p(1),
            r_top: p(2),
            d_centers: p(3),
        });
    }
    records
}

fn run_lighting_scenario(label: &str, csv_name: &str) {
    let csv_path = test_data_path(csv_name);
    let records = load_lighting_csv(&csv_path);
    assert!(!records.is_empty(), "{label}: no reference data");

    let pi = std::f64::consts::PI;
    for (i, rec) in records.iter().enumerate() {
        if rec.r_bottom <= 0.0 || rec.r_top <= 0.0 {
            continue;
        }

        let (intersects, area) = circle_intersect(rec.r_bottom, rec.r_top, rec.d_centers);
        let min_r = rec.r_bottom.min(rec.r_top);
        let max_circle = pi * min_r * min_r;

        assert!(
            area >= 0.0,
            "{label}[{i}] t={}: area must be non-negative, got {area}",
            rec.time
        );
        assert!(
            area <= max_circle + 1e-12,
            "{label}[{i}] t={}: area {area} exceeds smaller circle area {max_circle}",
            rec.time
        );

        // Separated circles: no intersection (strict >, matching circle_intersect)
        if rec.d_centers > rec.r_bottom + rec.r_top {
            assert!(
                !intersects,
                "{label}[{i}]: separated circles should not intersect"
            );
            assert!(
                area.abs() < 1e-15,
                "{label}[{i}]: separated circles area should be 0"
            );
        }
        // One circle contains the other
        if rec.d_centers + min_r <= rec.r_bottom.max(rec.r_top) + 1e-12 {
            assert!(
                (area - max_circle).abs() < 1e-10,
                "{label}[{i}] t={}: contained circle area should be {max_circle}, got {area}",
                rec.time
            );
        }
    }
    println!("  {label}: validated {} records", records.len());
}

#[test]
fn tier3_simulation_lighting_t01() {
    run_lighting_scenario("lighting_t01", "lighting_t01_lighting.csv");
}
#[test]
fn tier3_simulation_lighting_t02() {
    run_lighting_scenario("lighting_t02", "lighting_t02_lighting.csv");
}
#[test]
fn tier3_simulation_lighting_t03() {
    run_lighting_scenario("lighting_t03", "lighting_t03_lighting.csv");
}
#[test]
fn tier3_simulation_lighting_t04() {
    run_lighting_scenario("lighting_t04", "lighting_t04_lighting.csv");
}
#[test]
fn tier3_simulation_lighting_t05() {
    run_lighting_scenario("lighting_t05", "lighting_t05_lighting.csv");
}
#[test]
fn tier3_simulation_lighting_t06() {
    run_lighting_scenario("lighting_t06", "lighting_t06_lighting.csv");
}
#[test]
fn tier3_simulation_lighting_t07() {
    run_lighting_scenario("lighting_t07", "lighting_t07_lighting.csv");
}
#[test]
fn tier3_simulation_lighting_t08() {
    run_lighting_scenario("lighting_t08", "lighting_t08_lighting.csv");
}
#[test]
fn tier3_simulation_lighting_t09() {
    run_lighting_scenario("lighting_t09", "lighting_t09_lighting.csv");
}
#[test]
fn tier3_simulation_lighting_t10() {
    run_lighting_scenario("lighting_t10", "lighting_t10_lighting.csv");
}
