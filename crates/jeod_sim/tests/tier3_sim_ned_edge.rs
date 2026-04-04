//! Tier 3: SIM_NED edge-case cross-validation
//!
//! RUN_ell_polar: Polar orbit on ellipsoidal Earth — geodetic singularity at poles
//! RUN_sph_inc:   Inclined orbit on spherical Earth — validates spherical geodetic path
//! RUN_sph_polar: Polar orbit on spherical Earth — combines both edge cases
//!
//! All use point-mass gravity, RNP rotation (polar motion disabled),
//! RK4 at NED_DT=1.0s, 24h.
//! Epoch: 1991-01-01 00:00:00 UTC (same as existing SIM_NED RUN_ell_inc).

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, GravitySourceEntry, SimBody,
    Simulation, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::crossval_report;

const GEO_R_EQ: f64 = 6_378_137.0;
const GEO_R_POL: f64 = GEO_R_EQ * (1.0 - 1.0 / 298.257_223_563);
/// Spherical Earth radius (JEOD uses r_eq for spherical model).
const GEO_R_SPH: f64 = GEO_R_EQ;

/// SIM_NED epoch: 1991-01-01 00:00:00 UTC.
const NED_EPOCH_UTC_TJT: f64 = 8257.0;
const NED_TAI_UTC_S: f64 = 26.0;
const NED_UT1_TAI_S: f64 = -25.381_221_5;
const NED_DT: f64 = 1.0;

fn run_ned_test(
    csv_filename: &str,
    label: &str,
    use_spherical_earth: bool,
    tol_alt: f64,
    tol_lat: f64,
    tol_lon: f64,
    test_name: &str,
) {
    let csv_path = test_data_path(csv_filename);
    assert!(
        csv_path.exists(),
        "SIM_NED CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_ned_csv(&csv_path);
    assert!(records.len() > 100);
    let init = &records[0];

    let epoch_tai_tjt = NED_EPOCH_UTC_TJT + NED_TAI_UTC_S / 86400.0;
    let mut time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    time.set_ut1_tai_offset(NED_UT1_TAI_S);

    let mut sim = Simulation::new(time, NED_DT);

    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY),
    });

    let (r_eq, r_pol) = if use_spherical_earth {
        (GEO_R_SPH, GEO_R_SPH) // Spherical: r_eq = r_pol
    } else {
        (GEO_R_EQ, GEO_R_POL) // Ellipsoidal (WGS84)
    };

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        geodetic_planet: Some((earth, r_eq, r_pol)),
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): SIM_NED {label}, {} points",
        records.len()
    );

    let mut max_alt_err = 0.0_f64;
    let mut max_lat_err = 0.0_f64;
    let mut max_lon_err = 0.0_f64;
    let mut max_pos_err = 0.0_f64;

    // For spherical Earth runs, compare against sphere_* columns instead of ellip_*
    for record in &records[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);
        let pos_err = (body.trans.position - record.position).length();
        max_pos_err = max_pos_err.max(pos_err);

        let geo = body.geodetic_state.as_ref().unwrap_or_else(|| {
            panic!(
                "Simulation did not compute geodetic state at t={}",
                record.time
            )
        });

        let (ref_alt, ref_lat, ref_lon) = if use_spherical_earth {
            (
                record.sphere_altitude,
                record.sphere_latitude,
                record.sphere_longitude,
            )
        } else {
            (
                record.ellip_altitude,
                record.ellip_latitude,
                record.ellip_longitude,
            )
        };

        let alt_err = (geo.altitude - ref_alt).abs();
        let lat_err = (geo.latitude - ref_lat).abs();
        let lon_err = angle_diff(geo.longitude, ref_lon);

        max_alt_err = max_alt_err.max(alt_err);
        max_lat_err = max_lat_err.max(lat_err);
        max_lon_err = max_lon_err.max(lon_err);

        if (record.time % 7200.0).abs() < 6.1 {
            println!(
                "  t={:6.0}s: pos={:.3e} m  alt={:.3e} m  lat={:.3e} rad  lon={:.3e} rad",
                record.time, pos_err, alt_err, lat_err, lon_err
            );
        }
    }

    println!("  Max position error:  {:.6e} m", max_pos_err);
    println!("  Max altitude error:  {:.6e} m", max_alt_err);
    println!("  Max latitude error:  {:.6e} rad", max_lat_err);
    println!("  Max longitude error: {:.6e} rad", max_lon_err);

    crossval_report(
        test_name,
        &[
            ("position", max_pos_err, 0.5, "m"),
            ("altitude", max_alt_err, f64::INFINITY, "m"),
            ("latitude", max_lat_err, f64::INFINITY, "rad"),
            ("longitude", max_lon_err, f64::INFINITY, "rad"),
        ],
    );

    assert!(
        max_pos_err < 0.5,
        "{label}: position error {max_pos_err:.2} m exceeds 0.5 m"
    );
    assert!(
        max_alt_err < tol_alt,
        "{label}: altitude error {max_alt_err:.3e} m exceeds {tol_alt:.0e} m"
    );
    assert!(
        max_lat_err < tol_lat,
        "{label}: latitude error {max_lat_err:.3e} rad exceeds {tol_lat:.0e} rad"
    );
    assert!(
        max_lon_err < tol_lon,
        "{label}: longitude error {max_lon_err:.3e} rad exceeds {tol_lon:.0e} rad"
    );
}

#[test]
fn tier3_simulation_ned_polar() {
    // Polar orbit on ellipsoidal Earth. Position drift ~20 μm over 24h causes:
    //   altitude: 2e-4 m (comparable to existing ell_inc's 8.5e-4 m)
    //   latitude: 1e-8 rad (well-behaved even at poles)
    //   longitude: 3e-5 rad (geometrically ill-defined at poles — all meridians
    //     converge, so atan2(y,x) is hypersensitive to sub-mm position errors)
    run_ned_test(
        "ned_ell_polar_ned.csv",
        "RUN_ell_polar (ellipsoidal + polar)",
        false,
        1.0,  // altitude: same as existing ell_inc test
        1e-6, // latitude: same as existing ell_inc test
        0.1,  // longitude: pole singularity (actual: 3e-5 rad)
        "tier3_simulation_ned_polar",
    );
}

#[test]
fn tier3_simulation_ned_sph_inc() {
    // Inclined orbit on spherical Earth. All errors < 1e-6, same regime as ell_inc.
    run_ned_test(
        "ned_sph_inc_ned.csv",
        "RUN_sph_inc (spherical + inclined)",
        true,
        1.0,  // altitude: same as existing ell_inc test
        1e-6, // latitude: same as existing ell_inc test
        1e-6, // longitude: same as existing ell_inc test
        "tier3_simulation_ned_sph_inc",
    );
}

#[test]
fn tier3_simulation_ned_sph_polar() {
    // Polar orbit on spherical Earth. Same pole singularity as ell_polar.
    run_ned_test(
        "ned_sph_polar_ned.csv",
        "RUN_sph_polar (spherical + polar)",
        true,
        1.0,  // altitude: same as existing ell_inc test
        1e-6, // latitude: same as existing ell_inc test
        0.1,  // longitude: pole singularity (actual: 3e-5 rad)
        "tier3_simulation_ned_sph_polar",
    );
}
