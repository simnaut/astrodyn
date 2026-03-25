//! Tier 3: Cross-validate solar beta angle against JEOD SIM_SolarBeta RUN_incl_51_6.
//!
//! At each timestep, reads position and velocity from the JEOD CSV, computes
//! angular momentum h = r x v, queries the DE421 ephemeris for the Sun direction,
//! and computes `solar_beta_angle(h, sun_direction)`.
//!
//! The sim uses epoch 1991-01-01 00:00:00 UTC with TAI-UTC = 26s, 8x8 gravity,
//! Sun + Moon third-body perturbations, running for 10 days at 5400s log cycle.
//! Position in CSV is in the structure/inertial frame.
//!
//! Requires Docker-generated CSV and de421.bsp (see test_data/README.md).

use glam::DVec3;
use jeod_ephemeris::{Ephemeris, EphemerisBody};
use jeod_math::solar_beta_angle;
use std::path::Path;

/// Parsed record from the SIM_SolarBeta CSV.
#[derive(Debug)]
struct SolarBetaRecord {
    time: f64,
    solar_beta: f64,
    position: DVec3,
    velocity: DVec3,
}

fn load_solarbeta_csv(path: &Path) -> Vec<SolarBetaRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_SolarBeta CSV from {}: {e}",
            path.display()
        )
    });

    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // skip header
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 8 {
            continue;
        }

        let line_no = i + 1;
        let parse = |col: usize| -> f64 {
            fields[col].trim().parse::<f64>().unwrap_or_else(|e| {
                panic!(
                    "Failed to parse SolarBeta CSV at line {line_no}, col {col}: {:?} ({e})",
                    fields[col]
                )
            })
        };

        // CSV columns (interleaved position/velocity):
        // 0: time
        // 1: solar_beta
        // 2: position[0], 3: velocity[0]
        // 4: position[1], 5: velocity[1]
        // 6: position[2], 7: velocity[2]
        records.push(SolarBetaRecord {
            time: parse(0),
            solar_beta: parse(1),
            position: DVec3::new(parse(2), parse(4), parse(6)),
            velocity: DVec3::new(parse(3), parse(5), parse(7)),
        });
    }
    records
}

/// Convert sim elapsed time (seconds) to TDB Julian Date.
///
/// Epoch: 1991-01-01 00:00:00 UTC.
/// - TAI-UTC = 26s at this date
/// - TAI = UTC + 26s
/// - TT = TAI + 32.184s = UTC + 58.184s
/// - TDB ~ TT + periodic term (< 1.7ms, negligible for Sun direction)
///
/// 1991-01-01 00:00:00 UTC as JD:
///   JD(UTC) = 2448257.5
///   JD(TDB) ~ 2448257.5 + 58.184 / 86400
fn sim_time_to_tdb_jd(elapsed_s: f64) -> f64 {
    // Epoch: 1991-01-01 00:00:00 UTC = JD 2448257.5 (UTC)
    // TAI = UTC + 26s => TAI epoch = JD 2448257.5 + 26/86400
    // TT = TAI + 32.184s => TT epoch = JD 2448257.5 + 58.184/86400
    // TDB ~ TT for our purposes (error < 2ms, negligible for Sun direction)
    let epoch_tdb_jd = 2_448_257.5 + 58.184 / 86_400.0;
    epoch_tdb_jd + elapsed_s / 86_400.0
}

#[test]
fn tier3_solar_beta_vs_jeod_sim_solarbeta() {
    let csv_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/solarbeta_incl_51_6_solarbeta.csv");

    assert!(
        csv_path.exists(),
        "SIM_SolarBeta RUN_incl_51_6 CSV not found at {}.\n\
         Generate with:\n  \
         docker build -f trick/Dockerfile -t jeod-trick ..\n  \
         docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let bsp_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/de421.bsp");
    assert!(
        bsp_path.exists(),
        "DE421 ephemeris not found at {}.\n\
         Place de421.bsp in the test_data/ directory.",
        bsp_path.display()
    );

    let records = load_solarbeta_csv(&csv_path);
    assert!(
        records.len() > 10,
        "Expected more than 10 records in SolarBeta CSV, got {}",
        records.len()
    );

    let ephem = Ephemeris::from_bsp(&bsp_path)
        .expect("Failed to load DE421 ephemeris");

    eprintln!(
        "Tier 3: SIM_SolarBeta RUN_incl_51_6 cross-validation ({} timesteps over {:.1} days)",
        records.len(),
        records.last().map_or(0.0, |r| r.time) / 86_400.0
    );

    let mut max_beta_err = 0.0_f64;

    for (_idx, rec) in records.iter().enumerate() {
        // Compute angular momentum from position and velocity
        let h = rec.position.cross(rec.velocity);

        // Get Sun position relative to Earth at this epoch
        let tdb_jd = sim_time_to_tdb_jd(rec.time);
        let (sun_pos, _sun_vel) = ephem
            .get_earth_centered_state(EphemerisBody::Sun, tdb_jd)
            .unwrap_or_else(|e| {
                panic!(
                    "Ephemeris query failed at t={:.1}s (TDB JD={:.6}): {e}",
                    rec.time, tdb_jd
                )
            });

        // Compute solar beta angle
        let beta = solar_beta_angle(h, sun_pos);
        let beta_err = (beta - rec.solar_beta).abs();

        max_beta_err = max_beta_err.max(beta_err);

        assert!(
            beta_err < 1e-4,
            "t={:.1}s: solar_beta error {beta_err:.6e} rad exceeds 1e-4 rad \
             (ours={:.8} rad, JEOD={:.8} rad, diff={:.6} deg)",
            rec.time,
            beta,
            rec.solar_beta,
            beta_err.to_degrees()
        );

        // Log each record (relatively few data points with 5400s cycle)
        eprintln!(
            "  t={:>10.1}s ({:>5.2} days): beta_err={:.6e} rad ({:.4e} deg), \
             ours={:.6} deg, JEOD={:.6} deg",
            rec.time,
            rec.time / 86_400.0,
            beta_err,
            beta_err.to_degrees(),
            beta.to_degrees(),
            rec.solar_beta.to_degrees()
        );
    }

    eprintln!("\n  === Max errors across {} timesteps ===", records.len());
    eprintln!(
        "  solar_beta: {max_beta_err:.6e} rad ({:.6e} deg)",
        max_beta_err.to_degrees()
    );
}
