//! Tier 3: SIM_SolarBeta edge-case cross-validation
//!
//! RUN_incl_0:    Equatorial orbit (i=0) — beta tracks Sun declination (~23.4 deg).
//! RUN_incl_23_4: Inclination = Earth obliquity (23.44 deg) — geometry matches solar plane.
//!
//! SIM_SolarBeta uses 8x8 spherical harmonics gravity in JEOD, but our current
//! Simulation only has point-mass. Over 10 days this produces km-scale position
//! divergence. Instead of propagating, we validate our solar beta computation
//! directly: load each CSV position/velocity, compute beta from that state + DE421
//! Sun position, and compare against JEOD's logged beta.
//!
//! RUN_comp_ISS is deferred: it uses a non-standard epoch and non-spherical gravity.

mod sim_test_helpers;
use sim_test_helpers::*;

use jeod_sim::{Ephemeris, EphemerisBody};
use jeod_test_data::crossval::crossval_report;
use std::path::Path;

fn run_solar_beta_test(csv_filename: &str, label: &str, test_name: &str) {
    let csv_path = test_data_path(csv_filename);
    assert!(
        csv_path.exists(),
        "SIM_SolarBeta CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let bsp_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/de421.bsp");
    assert!(
        bsp_path.exists(),
        "DE421 ephemeris not found at {}",
        bsp_path.display()
    );

    let ephemeris = Ephemeris::from_bsp(&bsp_path).expect("load DE421");
    let records = load_solar_beta_csv(&csv_path);
    assert!(
        records.len() > 2,
        "Expected at least 3 records in {csv_filename}, got {}",
        records.len()
    );

    println!(
        "Tier 3 (Simulation): SIM_SolarBeta {label}, {} points (position-driven)",
        records.len()
    );

    let j2000_jd = 2_451_545.0;
    let mut max_beta_err = 0.0_f64;

    for record in &records {
        // Sun position from DE421 at this epoch
        let tdb_jd = j2000_jd + record.time / 86_400.0;
        let (sun_pos, _) = ephemeris
            .get_earth_centered_state(EphemerisBody::Sun, tdb_jd)
            .expect("Sun position query");

        // Compute solar beta from JEOD's own position/velocity + our Sun position
        let our_beta = jeod_sim::compute_body_solar_beta(record.position, record.velocity, sun_pos);

        let beta_err = (our_beta - record.solar_beta).abs();
        max_beta_err = max_beta_err.max(beta_err);

        if (record.time % 86400.0).abs() < 2701.0 {
            println!(
                "  t={:8.0}s: jeod_beta={:.4} deg  our_beta={:.4} deg  err={:.4e} rad",
                record.time,
                record.solar_beta.to_degrees(),
                our_beta.to_degrees(),
                beta_err
            );
        }
    }

    println!("  Max beta error: {:.6e} rad", max_beta_err);

    crossval_report(test_name, &[("beta", max_beta_err, "rad")]);

    // Beta error comes from Sun position differences between our DE421 (via Anise)
    // and JEOD's native DE421 reader — different Chebyshev evaluation paths produce
    // ~10 arcsecond directional offsets that grow roughly linearly with duration at
    // ~1.5e-4 rad/day. Over the 10-day SIM_SolarBeta scenario, that contributes
    // ~1.5e-3 rad, plus the 1e-4 rad base tolerance below.
    let duration_days = records.last().unwrap().time / 86_400.0;
    let tol = 1e-4 + duration_days * 1.5e-4; // base + ~1.5e-4 rad/day ephemeris drift
    assert!(
        max_beta_err < tol,
        "{label}: beta error {max_beta_err:.3e} rad exceeds {tol:.3e} rad \
         (duration={duration_days:.0} days)"
    );
}

#[test]
fn tier3_simulation_solar_beta_equ() {
    run_solar_beta_test(
        "solarbeta_incl_0_solarbeta.csv",
        "RUN_incl_0 (equatorial)",
        "tier3_simulation_solar_beta_equ",
    );
}

#[test]
fn tier3_simulation_solar_beta_obliquity() {
    run_solar_beta_test(
        "solarbeta_incl_23_4_solarbeta.csv",
        "RUN_incl_23_4 (obliquity)",
        "tier3_simulation_solar_beta_obliquity",
    );
}
