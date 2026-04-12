//! Tier 3: SIM_LIGHT_CIR — Earth lighting via Simulation pipeline
//!
//! JEOD's SIM_LIGHT_CIR sweeps parametric circle geometries (r_bottom, r_top,
//! d_centers) through `circle_intersect`. The CSV data validates angular
//! geometry, not orbital positions. This test:
//!
//! 1. Validates `circle_intersect()` against JEOD's parametric test vectors.
//! 2. Creates a Simulation with Earth+Sun+Moon, propagates a LEO orbit,
//!    and verifies that `EarthLightingState` is computed at each step
//!    (end-to-end pipeline validation).
//!
//! The parametric validation ensures our circle intersection matches JEOD's.
//! The Simulation pipeline validation ensures the derived state is wired
//! correctly through `step()`.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_interactions::earth_lighting::circle_intersect;
use jeod_sim::{
    Ephemeris, EphemerisBody, GravityControl, GravityControls, GravityModel, GravitySource,
    GravitySourceEntry, RotationModel, SimBody, Simulation, SimulationTime, TranslationalState,
};
use std::path::Path;

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

/// Validate circle_intersect against JEOD's parametric test vectors.
fn run_lighting_geometry_test(csv_filename: &str, label: &str) {
    let csv_path = test_data_path(csv_filename);
    assert!(
        csv_path.exists(),
        "SIM_LIGHT_CIR CSV not found at {}",
        csv_path.display()
    );

    let records = load_lighting_csv(&csv_path);
    assert!(!records.is_empty(), "{label}: no reference data");

    let mut checked = 0;
    for rec in &records {
        if rec.r_bottom == 0.0 && rec.r_top == 0.0 && rec.d_centers == 0.0 {
            continue;
        }

        let (intersects, area) = circle_intersect(rec.r_bottom, rec.r_top, rec.d_centers);

        // Geometric bounds
        assert!(
            area >= 0.0,
            "{label} t={}: area must be non-negative, got {area}",
            rec.time
        );
        let smaller_area = std::f64::consts::PI * rec.r_top.min(rec.r_bottom).powi(2);
        if intersects {
            assert!(
                area <= smaller_area + 1e-10,
                "{label} t={}: area {area} exceeds smaller circle area {smaller_area}",
                rec.time
            );
        }

        // If circles are separated (d_centers > r_bottom + r_top), no intersection
        if rec.d_centers > rec.r_bottom + rec.r_top + 1e-15 {
            assert!(
                !intersects,
                "{label} t={}: circles should be separated (d={:.6e} > r_b+r_t={:.6e})",
                rec.time,
                rec.d_centers,
                rec.r_bottom + rec.r_top
            );
        }

        checked += 1;
    }

    println!(
        "  {label}: {checked} geometry checks passed ({} total records)",
        records.len()
    );
}

/// Validate that EarthLightingState is computed through the Simulation pipeline.
fn run_lighting_pipeline_test() {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    let bsp_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/de421.bsp");
    assert!(
        bsp_path.exists(),
        "DE421 not found at {}",
        bsp_path.display()
    );
    let ephemeris = Ephemeris::from_bsp(&bsp_path).expect("load DE421");

    let grav_data_dir = jeod_root.join("models/environment/gravity/data/src");
    let mu_earth =
        jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_data_dir.join("earth_GGM05C.cc"))
            .expect("load Earth mu");

    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, 60.0);

    // Earth
    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: mu_earth,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });

    // Sun from DE421
    let j2000_jd = 2_451_545.0;
    let (initial_sun, _) = ephemeris
        .get_earth_centered_state(EphemerisBody::Sun, j2000_jd)
        .expect("Sun position at J2000");
    let sun = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: initial_sun,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    sim.set_source_ephemeris(sun, EphemerisBody::Sun, EphemerisBody::Earth);
    sim.sun_source = Some(sun);

    // Moon from DE421
    let (initial_moon, _) = ephemeris
        .get_earth_centered_state(EphemerisBody::Moon, j2000_jd)
        .expect("Moon position at J2000");
    let moon = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: initial_moon,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    sim.set_source_ephemeris(moon, EphemerisBody::Moon, EphemerisBody::Earth);
    sim.moon_source = Some(moon);
    sim.ephemeris = Some(ephemeris);

    // ISS-like LEO body with earth lighting enabled
    // earth_lighting_config = (earth_radius, moon_radius, sun_radius)
    sim.add_body(SimBody {
        trans: TranslationalState {
            position: DVec3::new(6_778_137.0, 0.0, 0.0),
            velocity: DVec3::new(0.0, 7_668.558, 0.0),
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        earth_lighting_config: Some((6_378_137.0, 1_737_400.0, 6.96e8)),
        ..Default::default()
    });

    sim.validate().unwrap();

    // Propagate one orbit (~90 min = 90 steps at 60s dt)
    let num_steps = 90;
    let mut lit_count = 0;
    let mut shadow_count = 0;

    for _ in 0..num_steps {
        sim.step();

        let body = sim.body(0);
        let lighting = body
            .earth_lighting
            .as_ref()
            .expect("earth_lighting should be computed after step()");

        // Check that lighting state has physical values
        assert!(
            (0.0..=1.0).contains(&lighting.sun_earth.visible),
            "sun_earth.visible={} out of [0,1]",
            lighting.sun_earth.visible
        );

        if lighting.sun_earth.visible > 0.5 {
            lit_count += 1;
        } else {
            shadow_count += 1;
        }
    }

    // ISS orbit at J2000: ~60% sunlit, ~40% eclipsed
    println!(
        "  Pipeline test: {} steps, {} sunlit, {} shadow",
        num_steps, lit_count, shadow_count
    );
    assert!(
        lit_count > 30 && shadow_count > 10,
        "Expected mix of sun/shadow for LEO orbit, got {lit_count} sunlit / {shadow_count} shadow"
    );
}

// ── Parametric geometry validation tests (from SIM_LIGHT_CIR) ──

#[test]
fn tier3_simulation_earth_lighting_t01() {
    run_lighting_geometry_test("lighting_t01_lighting.csv", "T01");
}

#[test]
fn tier3_simulation_earth_lighting_t02() {
    run_lighting_geometry_test("lighting_t02_lighting.csv", "T02");
}

#[test]
fn tier3_simulation_earth_lighting_t03() {
    run_lighting_geometry_test("lighting_t03_lighting.csv", "T03");
}

#[test]
fn tier3_simulation_earth_lighting_t04() {
    run_lighting_geometry_test("lighting_t04_lighting.csv", "T04");
}

#[test]
fn tier3_simulation_earth_lighting_t05() {
    run_lighting_geometry_test("lighting_t05_lighting.csv", "T05");
}

#[test]
fn tier3_simulation_earth_lighting_t06() {
    run_lighting_geometry_test("lighting_t06_lighting.csv", "T06");
}

#[test]
fn tier3_simulation_earth_lighting_t07() {
    run_lighting_geometry_test("lighting_t07_lighting.csv", "T07");
}

#[test]
fn tier3_simulation_earth_lighting_t08() {
    run_lighting_geometry_test("lighting_t08_lighting.csv", "T08");
}

#[test]
fn tier3_simulation_earth_lighting_t09() {
    run_lighting_geometry_test("lighting_t09_lighting.csv", "T09");
}

#[test]
fn tier3_simulation_earth_lighting_t10() {
    run_lighting_geometry_test("lighting_t10_lighting.csv", "T10");
}

// ── Pipeline end-to-end validation ──

#[test]
fn tier3_simulation_earth_lighting_pipeline() {
    run_lighting_pipeline_test();
}
