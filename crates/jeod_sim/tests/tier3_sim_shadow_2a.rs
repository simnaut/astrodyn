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
use std::path::Path;

/// Sun radius (m).
const R_SUN: f64 = 6.96e8;
/// Earth equatorial radius (m).
const R_EARTH: f64 = 6_378_137.0;
/// SIM_2A epoch: 1998-12-01 00:00:31 TAI.
const EPOCH_TJT: f64 = 11148.0 + 31.0 / 86400.0;

fn run_shadow_comparison(csv_filename: &str, label: &str) {
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

    let base_jd = EPOCH_TJT + 40000.0 + 2_400_000.5;

    let mut max_frac_err = 0.0_f64;
    let mut shadow_state_mismatches = 0;

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
        // Both flux_mag (from JEOD CSV) and solar_flux_at_distance() are in W/m².
        // (The CSV header labels flux as `{N/m2}` — Trick's default unit label
        // for radiation flux — but the physical quantity is irradiance in W/m².)
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

    // Shadow fraction agreement: measured 5.4e-3 max (at the umbra-antumbra
    // transition at ~1.4 Gm). The residual is dominated by the ~10 arcsecond
    // DE421 Sun position offset (#27) which shifts the shadow cone boundary
    // by ~70 m at 1.4 Gm distance. Error decreases with distance (1.2e-4
    // at 10 Gm, 3.5e-5 at 100 Gm). Shadow state (umbra/penumbra/sun) always
    // agrees — the discrepancy is in the fractional illumination during
    // penumbra transitions.
    assert!(
        max_frac_err < 0.01,
        "{label}: shadow fraction error {max_frac_err:.3e} exceeds 0.01"
    );
    assert_eq!(
        shadow_state_mismatches, 0,
        "{label}: {shadow_state_mismatches} shadow state disagreements (expected 0)"
    );
}

#[test]
fn tier3_shadow_2a_annular() {
    run_shadow_comparison("shadow_2a_annular_shadow_calc.csv", "RUN_annular_eclipse");
}

#[test]
fn tier3_shadow_2a_cooling() {
    run_shadow_comparison("shadow_2a_cooling_shadow_calc.csv", "RUN_shadow_cooling");
}
