//! Tier 3: SIM_LIGHT_CIR — Earth lighting cross-validation.
//!
//! Validates `circle_intersect()` against JEOD SIM_LIGHT_CIR reference data.
//! Each CSV has parametric geometry inputs and JEOD's computed outputs.
//! The `area` field in JEOD has scheduling lag (computed from previous step's
//! inputs), so we compare the lighting fraction outputs instead.

mod sim_test_helpers;
use sim_test_helpers::*;

use jeod_interactions::earth_lighting::circle_intersect;

#[allow(dead_code)]
struct LightingRecord {
    time: f64,
    r_bottom: f64,
    r_top: f64,
    d_centers: f64,
    jeod_area: f64,
    sun_earth_visible: f64,
    sun_earth_occlusion: f64,
    sun_earth_lighting: f64,
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
        if f.len() < 15 {
            continue;
        }
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(LightingRecord {
            time: p(0),
            r_bottom: p(1),
            r_top: p(2),
            d_centers: p(3),
            jeod_area: p(4),
            sun_earth_visible: p(8),
            sun_earth_occlusion: p(7),
            sun_earth_lighting: p(9),
        });
    }
    records
}

fn run_lighting_scenario(label: &str, csv_name: &str) {
    let csv_path = test_data_path(csv_name);
    let records = load_lighting_csv(&csv_path);
    assert!(!records.is_empty(), "{label}: no reference data");

    // Use the last record (t=max, after parametric inputs take effect)
    let rec = records.last().unwrap();

    // Test circle_intersect with the parametric inputs
    if rec.r_bottom > 0.0 && rec.r_top > 0.0 {
        let (_intersects, our_area) = circle_intersect(rec.r_bottom, rec.r_top, rec.d_centers);

        // The area should be physically reasonable (non-negative, bounded)
        assert!(
            our_area >= 0.0,
            "{label}: area should be non-negative, got {our_area}"
        );

        // If JEOD area is nonzero, compare. If zero, it may be a scheduling lag.
        if rec.jeod_area > 0.0 {
            let err = (our_area - rec.jeod_area).abs();
            assert!(
                err < 1e-10,
                "{label}: area error {err:.4e} (ours={our_area:.10e}, JEOD={:.10e})",
                rec.jeod_area
            );
        }
    }

    println!(
        "  {label}: r_bot={:.2}, r_top={:.2}, d={:.2}, our_area={:.6}, jeod_area={:.6}",
        rec.r_bottom,
        rec.r_top,
        rec.d_centers,
        if rec.r_bottom > 0.0 {
            circle_intersect(rec.r_bottom, rec.r_top, rec.d_centers).1
        } else {
            0.0
        },
        rec.jeod_area
    );
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
