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
use jeod_runner::{
    DerivedStateConfig, GeodeticConfig, GravitySourceEntry, RotationModel, Simulation,
    VehicleConfig,
};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, SimulationTime,
    TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};

use jeod_sim::LeapSecondTable;

const GEO_R_EQ: f64 = 6_378_137.0;
const GEO_R_POL: f64 = GEO_R_EQ * (1.0 - 1.0 / 298.257_223_563);
/// Spherical Earth radius (JEOD uses r_eq for spherical model).
const GEO_R_SPH: f64 = GEO_R_EQ;

/// Derived-state verif directory (shared Modified_data/ lives here, not in SIM_NED/).
const DERIVED_STATE_VERIF: &str = "models/dynamics/derived_state/verif";
/// SIM_NED directory relative to JEOD root.
const SIM_NED: &str = "models/dynamics/derived_state/verif/SIM_NED";

/// UT1-TAI from JEOD tai_to_ut1.cc at 1991-01-01 (index 10592).
/// This comes from JEOD's internal UT1 data table, not a sim config file.
const NED_UT1_TAI_S: f64 = -25.381_221_5;

#[allow(clippy::too_many_arguments)]
fn run_ned_test(
    csv_filename: &str,
    label: &str,
    use_spherical_earth: bool,
    pos_tol: [f64; 3],
    tol_alt: f64,
    tol_lat: f64,
    tol_lon: f64,
    test_name: &str,
    epoch_tai_tjt: f64,
    leap_table: LeapSecondTable,
    ned_dt: f64,
    mu_earth: f64,
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

    let mut time = SimulationTime::new(epoch_tai_tjt, leap_table);
    time.set_ut1_tai_offset(NED_UT1_TAI_S);

    let mut sim = Simulation::new(time, ned_dt);

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_earth,
                model: GravityModel::PointMass,
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: Some(DMat3::IDENTITY),
            delta_c20: 0.0,
            rotation_model: RotationModel::EarthRNP,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    let (r_eq, r_pol) = if use_spherical_earth {
        (GEO_R_SPH, GEO_R_SPH) // Spherical: r_eq = r_pol
    } else {
        (GEO_R_EQ, GEO_R_POL) // Ellipsoidal (WGS84)
    };

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        derived: DerivedStateConfig {
            geodetic: Some(GeodeticConfig {
                source_idx: earth,
                r_eq,
                r_pol,
            }),
            ..Default::default()
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): SIM_NED {label}, {} points",
        records.len()
    );

    let mut our_states = Vec::with_capacity(records.len() - 1);
    let mut ref_states = Vec::with_capacity(records.len() - 1);
    let mut max_alt_err = 0.0_f64;
    let mut max_lat_err = 0.0_f64;
    let mut max_lon_err = 0.0_f64;

    // For spherical Earth runs, compare against sphere_* columns instead of ellip_*
    for record in &records[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);

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

        our_states.push(StateLog {
            time: record.time,
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            ..Default::default()
        });
        ref_states.push(StateLog {
            time: record.time,
            position: Some(record.position),
            velocity: Some(record.velocity),
            ..Default::default()
        });

        if (record.time % 7200.0).abs() < 6.1 {
            let pos_err = (body.trans.position - record.position).length();
            println!(
                "  t={:6.0}s: pos={:.3e} m  alt={:.3e} m  lat={:.3e} rad  lon={:.3e} rad",
                record.time, pos_err, alt_err, lat_err, lon_err
            );
        }
    }

    let max_pos_err = our_states
        .iter()
        .zip(ref_states.iter())
        .map(|(a, b)| (a.position.unwrap() - b.position.unwrap()).length())
        .fold(0.0_f64, f64::max);

    println!("  Max position error:  {:.6e} m", max_pos_err);
    println!("  Max altitude error:  {:.6e} m", max_alt_err);
    println!("  Max latitude error:  {:.6e} rad", max_lat_err);
    println!("  Max longitude error: {:.6e} rad", max_lon_err);

    let mut report = CrossvalReport::compute(test_name, &our_states, &ref_states);
    report.add_extra("altitude", max_alt_err, "m");
    assert!(max_alt_err < tol_alt, "altitude");
    report.add_extra("latitude", max_lat_err, "rad");
    assert!(max_lat_err < tol_lat, "latitude");
    report.add_extra("longitude", max_lon_err, "rad");
    assert!(max_lon_err < tol_lon, "longitude");
    report.write();

    report.assert_position(pos_tol);
}

/// Load SIM_NED parameters from JEOD source files.
fn load_ned_params() -> (f64, LeapSecondTable, f64, f64) {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    let sim_dir = jeod_root.join(SIM_NED);
    let verif_dir = jeod_root.join(DERIVED_STATE_VERIF);
    let grav_data_dir = jeod_root.join("models/environment/gravity/data/src");

    // Load epoch from JEOD time config. The derived-state SIMs share
    // Modified_data/ at the verif/ level (not inside each SIM directory).
    let time_cfg = jeod_test_data::time_config::load_time_config(
        &verif_dir.join("Modified_data/date_and_time.py"),
    );
    let leap_table = jeod_sim::default_leap_second_table();
    let tai_utc_s = leap_table.tai_utc_at_utc_tjt(time_cfg.utc_tjt());
    let epoch_tai_tjt = time_cfg.tai_tjt_with_offset(tai_utc_s);

    // Load integration step size from S_define
    let ned_dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));

    // Load Earth mu from JEOD gravity data
    let mu_earth =
        jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_data_dir.join("earth_GGM05C.cc"))
            .expect("load Earth mu");

    (epoch_tai_tjt, leap_table, ned_dt, mu_earth)
}

#[test]
fn tier3_simulation_ned_polar() {
    let (epoch_tai_tjt, leap_table, ned_dt, mu_earth) = load_ned_params();
    // Polar orbit on ellipsoidal Earth. Position drift ~20 um over 24h causes:
    //   altitude: 2e-4 m (comparable to existing ell_inc's 8.5e-4 m)
    //   latitude: 1e-8 rad (well-behaved even at poles)
    //   longitude: 3e-5 rad (geometrically ill-defined at poles — all meridians
    //     converge, so atan2(y,x) is hypersensitive to sub-mm position errors)
    run_ned_test(
        "ned_ell_polar_ned.csv",
        "RUN_ell_polar (ellipsoidal + polar)",
        false,
        [3.464e-6, 1.911e-5, 1.967e-5],
        2.123e-4,
        1.089e-8,
        3.349e-5,
        "tier3_simulation_ned_polar",
        epoch_tai_tjt,
        leap_table,
        ned_dt,
        mu_earth,
    );
}

#[test]
fn tier3_simulation_ned_sph_inc() {
    let (epoch_tai_tjt, leap_table, ned_dt, mu_earth) = load_ned_params();
    // Inclined orbit on spherical Earth. All errors < 1e-6, same regime as ell_inc.
    run_ned_test(
        "ned_sph_inc_ned.csv",
        "RUN_sph_inc (spherical + inclined)",
        true,
        [3.78e-6, 5.155e-6, 3.717e-6],
        4.02e-7,
        4.181e-8,
        6.493e-8,
        "tier3_simulation_ned_sph_inc",
        epoch_tai_tjt,
        leap_table,
        ned_dt,
        mu_earth,
    );
}

#[test]
fn tier3_simulation_ned_sph_polar() {
    let (epoch_tai_tjt, leap_table, ned_dt, mu_earth) = load_ned_params();
    // Polar orbit on spherical Earth. Same pole singularity as ell_polar.
    run_ned_test(
        "ned_sph_polar_ned.csv",
        "RUN_sph_polar (spherical + polar)",
        true,
        [3.464e-6, 1.911e-5, 1.967e-5],
        3.984e-7,
        1.083e-8,
        3.349e-5,
        "tier3_simulation_ned_sph_polar",
        epoch_tai_tjt,
        leap_table,
        ned_dt,
        mu_earth,
    );
}
