//! Tier 3: SIM_2A_SHADOW_CALC — shadow geometry via Simulation pipeline
//!
//! JEOD's SIM_2A_SHADOW_CALC uses prescribed (non-integrated) motion to sweep
//! a vehicle through shadow geometries. Integration is explicitly disabled in
//! the S_define. This test matches JEOD's approach: creates a Simulation,
//! advances time (for ephemeris Sun position updates), sets body position from
//! the reference data at each checkpoint, and compares shadow fraction.
//!
//! The full trajectory SRP+shadow test is `tier3_sim_srp.rs` (SIM_3_ORBIT).
//! This test validates the shadow geometry computation specifically.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_runner::{
    GravitySourceEntry, RotationModel, ShadowBody, Simulation, SrpModel, VehicleConfig,
};
use jeod_sim::{
    compute_shadow_fraction, solar_flux_at_distance, Ephemeris, EphemerisBody, GravityModel,
    GravitySource, SimulationTime, TranslationalState, SOLAR_RADIUS,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use std::path::Path;

/// SIM_2A_SHADOW_CALC directory relative to JEOD root.
const SIM_2A: &str = "models/interactions/radiation_pressure/verif/SIM_2A_SHADOW_CALC";

/// Earth equatorial radius for shadow geometry (from JEOD planet/data/src/earth.cc).
const R_EARTH: f64 = jeod_sim::EARTH.shadow_radius;

fn run_shadow_comparison(csv_filename: &str, label: &str, test_name: &str, frac_tol: f64) {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    let csv_path = test_data_path(csv_filename);
    assert!(
        csv_path.exists(),
        "SIM_2A_SHADOW_CALC CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let bsp_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/de421.bsp");
    assert!(
        bsp_path.exists(),
        "DE421 not found at {}",
        bsp_path.display()
    );
    let ephemeris = Ephemeris::from_bsp(&bsp_path).expect("load DE421");

    // Load epoch from JEOD time config
    let sim_dir = jeod_root.join(SIM_2A);
    let time_cfg = jeod_test_data::time_config::load_time_config(
        &sim_dir.join("Modified_data/date_and_time.py"),
    );
    let epoch_tjt = time_cfg.tai_tjt();

    let records = load_shadow_calc_csv(&csv_path);
    assert!(
        records.len() >= 2,
        "{label}: expected at least 2 records, got {}",
        records.len()
    );

    // Create Simulation at the SIM_2A epoch with Earth + Sun (ephemeris-driven)
    let time = SimulationTime::new(epoch_tjt, jeod_sim::default_leap_second_table());
    let dt = 1.0; // SIM_2A logs at 1s intervals
    let mut sim = Simulation::new(time, dt);

    // Earth at origin
    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });

    // Sun from DE421 (query in TDB JD, not TAI JD)
    let tdb_jd = sim.time.tdb_julian_date();
    let (initial_sun, _) = ephemeris
        .get_earth_centered_state(EphemerisBody::Sun, tdb_jd)
        .expect("Sun position at epoch");
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
    sim.ephemeris = Some(ephemeris);

    // Vehicle body — SIM_2A uses prescribed motion (no integration).
    // Position is set from reference data at each timestep to match JEOD's
    // test configuration, which explicitly disables integration.
    let init = &records[0];
    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.position,
            velocity: DVec3::ZERO,
        },
        mass: Some(jeod_sim::MassProperties::new(1.0)),
        shadow_body: Some(ShadowBody {
            source_idx: earth,
            radius: R_EARTH,
        }),
        srp: Some(SrpModel::Cannonball {
            cx_area: 1.0,
            albedo: 0.0,
            diffuse: 0.5,
        }),
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): SIM_2A_SHADOW_CALC {label}, {} points",
        records.len()
    );

    let mut max_frac_err = 0.0_f64;
    let mut shadow_state_mismatches = 0;
    let our_states: Vec<StateLog> = records
        .iter()
        .map(|r| StateLog {
            time: r.time,
            ..Default::default()
        })
        .collect();
    let ref_states = our_states.clone();

    for (i, record) in records.iter().enumerate() {
        // Advance time + ephemeris before comparing (skip for t=0)
        if i > 0 {
            sim.step();
        }

        // Set prescribed position (matching JEOD's non-integrated motion)
        sim.set_body_position(0, record.position);

        // Compute our shadow fraction at the current Sun position
        let sun_pos = sim.sources[sun].position;
        let our_frac =
            compute_shadow_fraction(record.position, sun_pos, DVec3::ZERO, R_EARTH, SOLAR_RADIUS);

        // Derive JEOD's shadow fraction from flux ratio
        let sun_dist = (sun_pos - record.position).length();
        let full_sun_flux = solar_flux_at_distance(sun_dist);
        let jeod_frac = if full_sun_flux > 1.0 {
            (record.flux_mag / full_sun_flux).min(1.0)
        } else {
            0.0
        };

        let frac_err = (our_frac - jeod_frac).abs();
        max_frac_err = max_frac_err.max(frac_err);

        let our_state = if our_frac < 0.001 {
            "shadow"
        } else if our_frac > 0.999 {
            "sun"
        } else {
            "penumbra"
        };
        let jeod_state = if jeod_frac < 0.001 {
            "shadow"
        } else if jeod_frac > 0.999 {
            "sun"
        } else {
            "penumbra"
        };
        if our_state != jeod_state {
            shadow_state_mismatches += 1;
            if i < 10 {
                println!(
                    "  MISMATCH t={:5.0}s: our={:.6} jeod={:.6} err={:.3e} [{}/{}]",
                    record.time, our_frac, jeod_frac, frac_err, our_state, jeod_state,
                );
            }
        }
    }

    println!("  Max shadow fraction error:  {:.6e}", max_frac_err);
    println!("  Shadow state mismatches:    {shadow_state_mismatches}");

    let mut report = CrossvalReport::compute(test_name, &our_states, &ref_states);
    report.add_extra("shadow_fraction", max_frac_err, "");
    report.add_extra("shadow_mismatches", shadow_state_mismatches as f64, "");
    report.write();

    assert!(
        max_frac_err < frac_tol,
        "{label}: shadow fraction error {max_frac_err:.3e} exceeds {frac_tol:.3e}"
    );
    assert_eq!(
        shadow_state_mismatches, 0,
        "{label}: {shadow_state_mismatches} shadow state disagreements (expected 0)"
    );
}

#[test]
fn tier3_simulation_shadow_2a_annular() {
    run_shadow_comparison(
        "shadow_2a_annular_shadow_calc.csv",
        "RUN_annular_eclipse",
        "tier3_simulation_shadow_2a_annular",
        5.71e-3,
    );
}

#[test]
fn tier3_simulation_shadow_2a_cooling() {
    run_shadow_comparison(
        "shadow_2a_cooling_shadow_calc.csv",
        "RUN_shadow_cooling",
        "tier3_simulation_shadow_2a_cooling",
        1e-10,
    );
}
