//! Tier 3: Cross-validate geodetic coordinate conversion against JEOD SIM_NED RUN_ell_inc.
//!
//! At each timestep, reads planet-fixed Cartesian coordinates (cart_coords)
//! from the CSV and computes `cartesian_to_geodetic()`. Compares against
//! JEOD's logged ellipsoidal coordinates (altitude, latitude, longitude).
//!
//! Requires Docker-generated CSV (see test_data/README.md).

use glam::DVec3;
use jeod_math::cartesian_to_geodetic;
use jeod_test_data::crossval::crossval_report;
use std::path::Path;

/// WGS84 equatorial radius (m).
const EARTH_R_EQ: f64 = 6_378_137.0;
/// WGS84 polar radius (m), matching JEOD's runtime derivation: r_eq * (1 - 1/flat_inv).
const EARTH_R_POL: f64 = EARTH_R_EQ * (1.0 - 1.0 / 298.257_223_563);

/// Parsed record from the SIM_NED CSV.
#[derive(Debug)]
#[allow(dead_code)]
struct NedRecord {
    time: f64,
    /// Cartesian coordinates in PCPF frame (m).
    cart_coords: DVec3,
    /// JEOD ellipsoidal altitude (m).
    ellip_altitude: f64,
    /// JEOD ellipsoidal latitude (rad).
    ellip_latitude: f64,
    /// JEOD ellipsoidal longitude (rad).
    ellip_longitude: f64,
    /// JEOD spherical altitude (m).
    sphere_altitude: f64,
    /// JEOD spherical latitude (rad).
    sphere_latitude: f64,
    /// JEOD spherical longitude (rad).
    sphere_longitude: f64,
    /// Inertial position (m).
    position: DVec3,
    /// Inertial velocity (m/s).
    velocity: DVec3,
}

fn load_ned_csv(path: &Path) -> Vec<NedRecord> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read SIM_NED CSV from {}: {e}", path.display()));

    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // skip header
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 16 {
            continue;
        }

        let line_no = i + 1;
        let parse = |col: usize| -> f64 {
            fields[col].trim().parse::<f64>().unwrap_or_else(|e| {
                panic!(
                    "Failed to parse NED CSV at line {line_no}, col {col}: {:?} ({e})",
                    fields[col]
                )
            })
        };

        // CSV columns:
        // 0: time
        // 1-3: cart_coords[0,1,2]
        // 4: ellip_coords.altitude, 5: sphere_coords.altitude
        // 6: ellip_coords.latitude, 7: sphere_coords.latitude
        // 8: ellip_coords.longitude, 9: sphere_coords.longitude
        // 10: position[0], 11: velocity[0]
        // 12: position[1], 13: velocity[1]
        // 14: position[2], 15: velocity[2]
        records.push(NedRecord {
            time: parse(0),
            cart_coords: DVec3::new(parse(1), parse(2), parse(3)),
            ellip_altitude: parse(4),
            ellip_latitude: parse(6),
            ellip_longitude: parse(8),
            sphere_altitude: parse(5),
            sphere_latitude: parse(7),
            sphere_longitude: parse(9),
            position: DVec3::new(parse(10), parse(12), parse(14)),
            velocity: DVec3::new(parse(11), parse(13), parse(15)),
        });
    }
    records
}

#[test]
fn tier3_geodetic_vs_jeod_sim_ned() {
    let csv_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/ned_ell_inc_ned.csv");

    assert!(
        csv_path.exists(),
        "SIM_NED RUN_ell_inc CSV not found at {}.\n\
         Generate with:\n  \
         docker build -f trick/Dockerfile -t jeod-trick ..\n  \
         docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let records = load_ned_csv(&csv_path);
    assert!(
        records.len() > 10,
        "Expected more than 10 records in NED CSV, got {}",
        records.len()
    );

    eprintln!(
        "Tier 3: SIM_NED RUN_ell_inc cross-validation ({} timesteps)",
        records.len()
    );

    let mut max_alt_err = 0.0_f64;
    let mut max_lat_err = 0.0_f64;
    let mut max_lon_err = 0.0_f64;

    for (idx, rec) in records.iter().enumerate() {
        // The cart_coords from the CSV are already in PCPF (planet-fixed frame).
        // Use these directly for geodetic conversion.
        let geo = cartesian_to_geodetic(rec.cart_coords, EARTH_R_EQ, EARTH_R_POL);

        let alt_err = (geo.altitude - rec.ellip_altitude).abs();
        let lat_err = (geo.latitude - rec.ellip_latitude).abs();
        let lon_err = (geo.longitude - rec.ellip_longitude).abs();

        max_alt_err = max_alt_err.max(alt_err);
        max_lat_err = max_lat_err.max(lat_err);
        max_lon_err = max_lon_err.max(lon_err);

        assert!(
            alt_err < 1e-7,
            "t={:.1}s: altitude error {alt_err:.6e} m exceeds 1e-7 m \
             (ours={:.15e}, JEOD={:.15e})",
            rec.time,
            geo.altitude,
            rec.ellip_altitude
        );
        assert!(
            lat_err < 1e-14,
            "t={:.1}s: latitude error {lat_err:.6e} rad exceeds 1e-14 rad \
             (ours={:.15e}, JEOD={:.15e})",
            rec.time,
            geo.latitude,
            rec.ellip_latitude
        );
        assert!(
            lon_err < 1e-14,
            "t={:.1}s: longitude error {lon_err:.6e} rad exceeds 1e-14 rad \
             (ours={:.15e}, JEOD={:.15e})",
            rec.time,
            geo.longitude,
            rec.ellip_longitude
        );

        // Log every 10th record
        if idx % 10 == 0 {
            eprintln!(
                "  t={:>8.1}s: alt_err={:.3e} m, lat_err={:.3e} rad, lon_err={:.3e} rad",
                rec.time, alt_err, lat_err, lon_err
            );
        }
    }

    eprintln!("\n  === Max errors across {} timesteps ===", records.len());
    eprintln!("  altitude:  {max_alt_err:.6e} m");
    eprintln!("  latitude:  {max_lat_err:.6e} rad");
    eprintln!("  longitude: {max_lon_err:.6e} rad");

    crossval_report(
        "tier3_geodetic_vs_jeod_sim_ned",
        &[
            ("altitude", max_alt_err, "m"),
            ("latitude", max_lat_err, "rad"),
            ("longitude", max_lon_err, "rad"),
        ],
    );
}
