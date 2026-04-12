//! Tier 3: SIM_2A_SHADOW_CALC cross-validation
//!
//! Validates our conical shadow model against JEOD's SIM_2A_SHADOW_CALC.
//! Computes shadow fraction from each CSV position + Sun position from DE421,
//! then compares against JEOD's logged flux to verify shadow state agreement.
//!
//!   RUN_annular_eclipse: Vehicle at varying distances, exercises shadow transitions.
//!   RUN_shadow_cooling:  Vehicle near Earth surface, persistent shadow.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_interactions::{compute_shadow_fraction, solar_flux_at_distance};
use jeod_sim::{Ephemeris, EphemerisBody};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use std::path::Path;

/// SIM_2A_SHADOW_CALC directory relative to JEOD root.
const SIM_2A: &str = "models/interactions/radiation_pressure/verif/SIM_2A_SHADOW_CALC";

/// Sun radius (m).
const R_SUN: f64 = 6.96e8;
/// Earth equatorial radius (m).
const R_EARTH: f64 = 6_378_137.0;

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

    // Load epoch from JEOD time config. SIM_2A uses TAI initializer.
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

    println!(
        "Tier 3 (Simulation): SIM_2A_SHADOW_CALC {label}, {} points",
        records.len()
    );

    let base_jd = epoch_tjt + 40000.0 + 2_400_000.5;

    let mut max_frac_err = 0.0_f64;
    let mut shadow_state_mismatches = 0;

    // These tests don't propagate state; use empty state logs and report via extras.
    let our_states: Vec<StateLog> = records
        .iter()
        .map(|r| StateLog {
            time: r.time,
            ..Default::default()
        })
        .collect();
    let ref_states: Vec<StateLog> = our_states.clone();

    for record in &records {
        let tdb_jd = base_jd + record.time / 86400.0;
        let (sun_pos, _) = ephemeris
            .get_earth_centered_state(EphemerisBody::Sun, tdb_jd)
            .expect("Sun position");

        // Our shadow fraction from geometry
        let our_frac =
            compute_shadow_fraction(record.position, sun_pos, DVec3::ZERO, R_EARTH, R_SUN);

        // Derive JEOD's shadow fraction: compute what full-sun flux would be at
        // this vehicle's distance from Sun, then ratio with actual logged flux.
        // Both flux_mag (from JEOD CSV) and solar_flux_at_distance() are in W/m2.
        // (The CSV header labels flux as `{N/m2}` — Trick's default unit label
        // for radiation flux — but the physical quantity is irradiance in W/m2.)
        let sun_dist = (sun_pos - record.position).length();
        let full_sun_flux = solar_flux_at_distance(sun_dist);
        let jeod_frac = if full_sun_flux > 1.0 {
            (record.flux_mag / full_sun_flux).min(1.0)
        } else {
            0.0
        };

        let frac_err = (our_frac - jeod_frac).abs();
        max_frac_err = max_frac_err.max(frac_err);

        // Check shadow state agreement: both in shadow, both in sun, or both in penumbra
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
            println!(
                "  MISMATCH t={:5.0}s: our={:.6} jeod={:.6} err={:.3e} [{}/{}]",
                record.time, our_frac, jeod_frac, frac_err, our_state, jeod_state,
            );
        }
    }

    println!("  Max shadow fraction error:  {:.6e}", max_frac_err);
    println!("  Shadow state mismatches:    {shadow_state_mismatches}");

    let mut report = CrossvalReport::compute(test_name, &our_states, &ref_states);
    report.add_extra("shadow_fraction", max_frac_err, "");
    assert!(max_frac_err < frac_tol, "shadow_fraction");
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
fn tier3_shadow_2a_annular() {
    run_shadow_comparison(
        "shadow_2a_annular_shadow_calc.csv",
        "RUN_annular_eclipse",
        "tier3_shadow_2a_annular",
        5.71e-3,
    );
}

#[test]
fn tier3_shadow_2a_cooling() {
    run_shadow_comparison(
        "shadow_2a_cooling_shadow_calc.csv",
        "RUN_shadow_cooling",
        "tier3_shadow_2a_cooling",
        1e-10,
    );
}
