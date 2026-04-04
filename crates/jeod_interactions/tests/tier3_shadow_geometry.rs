//! Tier 3: Shadow geometry cross-validation against JEOD SIM_2_SHADOW_CALC.
//!
//! Tests:
//!   - RUN_annular_eclipse: vehicle moves radially away from Earth along
//!     the Sun-Earth line, transitioning through umbra → antumbra → full sun.
//!   - RUN_transverse_shadow: vehicle moves transversely across the shadow
//!     boundary at fixed radial distance, transitioning through penumbra.
//!
//! These are KINEMATIC tests — vehicle positions are set explicitly (no
//! dynamics integration). We compare our `compute_shadow_fraction()` ×
//! `solar_flux_at_distance()` against JEOD's logged `flux_mag`.

use glam::DVec3;
use jeod_ephemeris::{Ephemeris, EphemerisBody};
use jeod_interactions::{compute_shadow_fraction, solar_flux_at_distance, SOLAR_RADIUS};
use jeod_test_data::crossval::crossval_report;
use std::path::Path;

const R_EARTH: f64 = 6_378_137.0; // WGS84 equatorial radius

/// Epoch used for SIM_2_SHADOW_CALC / SIM_3_ORBIT in truncated Julian time (TJT).
/// TJT = MJD - 40000; here EPOCH_TJT = 11148.0 => JD = 2_451_148.5 (~ 1998-12-01 00:00 TDB).
const EPOCH_TJT: f64 = 11148.0;

struct ShadowRecord {
    position: DVec3,
    flux_mag: f64,
}

fn load_shadow_csv(path: &Path) -> Vec<ShadowRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read shadow CSV from {}: {e}\n\
             Generate with: docker run --rm -v $(pwd)/test_data:/output \
             -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
            path.display()
        )
    });
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(f.len() >= 5, "line {}: expected >=5 columns", i + 1);
        let p = |s: &str| -> f64 { s.trim().parse().unwrap() };
        records.push(ShadowRecord {
            position: DVec3::new(p(f[1]), p(f[2]), p(f[3])),
            flux_mag: p(f[4]),
        });
    }
    records
}

fn test_data_path(filename: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(filename)
}

/// Get Sun position from DE421 ephemeris at the SIM_2_SHADOW_CALC epoch.
fn sun_position_at_epoch() -> DVec3 {
    let bsp_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/de421.bsp");
    assert!(
        bsp_path.exists(),
        "DE421 ephemeris not found at {}",
        bsp_path.display()
    );
    let ephemeris = Ephemeris::from_bsp(&bsp_path).unwrap();
    // TJT is truncated Julian time: MJD - 40000.
    // JD = MJD + 2400000.5, MJD = TJT + 40000
    // JD = TJT + 40000 + 2400000.5 = TJT + 2440000.5
    let tdb_jd = EPOCH_TJT + 2_440_000.5;
    let (pos, _vel) = ephemeris
        .get_earth_centered_state(EphemerisBody::Sun, tdb_jd)
        .expect("Failed to query Sun position from DE421");
    pos
}

#[test]
fn tier3_shadow_annular_eclipse() {
    let csv_path = test_data_path("shadow_annular_eclipse_shadow_calc.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_shadow_csv(&csv_path);
    assert!(!records.is_empty());

    let sun_pos = sun_position_at_epoch();
    let earth_pos = DVec3::ZERO; // Earth-centered inertial frame

    println!("=== Shadow: Annular Eclipse ({} points) ===", records.len());
    println!(
        "  Sun position: [{:.0}, {:.0}, {:.0}] m",
        sun_pos.x, sun_pos.y, sun_pos.z
    );

    let mut max_flux_err = 0.0_f64;

    for (i, rec) in records.iter().enumerate() {
        let our_shadow =
            compute_shadow_fraction(rec.position, sun_pos, earth_pos, R_EARTH, SOLAR_RADIUS);

        let sun_dist = (rec.position - sun_pos).length();
        let our_flux = solar_flux_at_distance(sun_dist) * our_shadow;

        let flux_err = (our_flux - rec.flux_mag).abs();
        max_flux_err = max_flux_err.max(flux_err);

        // Classify shadow state by comparing actual flux to expected
        // full-illumination flux at the same distance from the Sun.
        let full_flux = solar_flux_at_distance(sun_dist);
        let jeod_frac = if full_flux > 1e-10 {
            rec.flux_mag / full_flux
        } else {
            1.0
        };
        let jeod_state = if jeod_frac < 1e-6 {
            "shadow"
        } else if jeod_frac > 0.999 {
            "full_sun"
        } else {
            "partial"
        };
        let our_state = if our_shadow < 1e-6 {
            "shadow"
        } else if our_shadow > 0.999 {
            "full_sun"
        } else {
            "partial"
        };

        println!(
            "  [{:2}] r={:.3e} m  JEOD={:.2} W/m² ({})  ours={:.2} W/m² ({})  err={:.2}",
            i,
            rec.position.length(),
            rec.flux_mag,
            jeod_state,
            our_flux,
            our_state,
            flux_err
        );

        // Shadow state must agree (both shadow, both partial, or both full sun)
        assert_eq!(
            jeod_state, our_state,
            "Shadow state mismatch at point {i}: JEOD={jeod_state}, ours={our_state}"
        );
    }

    println!("  Max flux error: {:.6e} W/m²", max_flux_err);

    crossval_report(
        "tier3_shadow_annular_eclipse",
        &[("flux", max_flux_err, f64::INFINITY, "W/m²")],
    );

    // Flux magnitude should match within 1% for illuminated points
    for (i, rec) in records.iter().enumerate() {
        if rec.flux_mag > 1.0 {
            let our_shadow =
                compute_shadow_fraction(rec.position, sun_pos, earth_pos, R_EARTH, SOLAR_RADIUS);
            let sun_dist = (rec.position - sun_pos).length();
            let our_flux = solar_flux_at_distance(sun_dist) * our_shadow;
            let rel_err = (our_flux - rec.flux_mag).abs() / rec.flux_mag;
            assert!(
                rel_err < 0.01,
                "Flux relative error {rel_err:.4} exceeds 1% at point {i}"
            );
        }
    }
}

#[test]
fn tier3_shadow_transverse_shadow() {
    let csv_path = test_data_path("shadow_transverse_shadow_shadow_calc.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_shadow_csv(&csv_path);
    assert!(!records.is_empty());

    let sun_pos = sun_position_at_epoch();
    let earth_pos = DVec3::ZERO;

    println!(
        "=== Shadow: Transverse Shadow ({} points) ===",
        records.len()
    );

    let mut shadow_state_mismatches = 0;
    let mut max_rel_flux_err = 0.0_f64;

    for (i, rec) in records.iter().enumerate() {
        let our_shadow =
            compute_shadow_fraction(rec.position, sun_pos, earth_pos, R_EARTH, SOLAR_RADIUS);

        let sun_dist = (rec.position - sun_pos).length();
        let our_flux = solar_flux_at_distance(sun_dist) * our_shadow;

        let jeod_in_shadow = rec.flux_mag < 1e-10;
        let our_in_shadow = our_shadow < 1e-10;

        if jeod_in_shadow != our_in_shadow {
            shadow_state_mismatches += 1;
            println!(
                "  [{:2}] MISMATCH: JEOD flux={:.2}, our shadow={:.4}",
                i, rec.flux_mag, our_shadow
            );
        }

        if rec.flux_mag > 1.0 && our_flux > 1.0 {
            let rel_err = (our_flux - rec.flux_mag).abs() / rec.flux_mag;
            max_rel_flux_err = max_rel_flux_err.max(rel_err);
        }

        if i % 5 == 0 || jeod_in_shadow != our_in_shadow {
            println!(
                "  [{:2}] JEOD_flux={:10.2}  our_flux={:10.2}  shadow={:.4}",
                i, rec.flux_mag, our_flux, our_shadow
            );
        }
    }

    println!("  Shadow state mismatches: {shadow_state_mismatches}");
    println!("  Max relative flux error: {max_rel_flux_err:.6}");

    // Allow at most 2 mismatches (transition boundary timing)
    assert!(
        shadow_state_mismatches <= 2,
        "Shadow state mismatches: {shadow_state_mismatches} (expected <= 2)"
    );

    // Flux should match within 1% for illuminated points
    assert!(
        max_rel_flux_err < 0.01,
        "Max relative flux error {max_rel_flux_err:.4} exceeds 1%"
    );
}
