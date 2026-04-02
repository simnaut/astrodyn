//! Tier 3: SIM_NED cross-validation (derived_state/verif/SIM_NED)
//!
//! Matches the JEOD SIM_NED configuration:
//!   - Epoch: 1991-01-01 00:00:00 UTC (TAI-UTC=26s, UT1-TAI=-25.3812215s)
//!   - Gravity: point-mass (JEOD veh_config.py sets spherical=1)
//!   - RNP: precession + nutation + GAST (polar motion disabled)
//!   - Integration: RK4 at 1.0s step
//!
//! Validates the full Simulation pipeline: orbit integration -> RNP rotation
//! -> geodetic coordinate conversion, compared against JEOD CSV values.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, GravitySourceEntry, SimBody,
    Simulation, SimulationTime, TranslationalState,
};

const GEO_R_EQ: f64 = 6_378_137.0;
const GEO_R_POL: f64 = GEO_R_EQ * (1.0 - 1.0 / 298.257_223_563);

/// SIM_NED epoch: 1991-01-01 00:00:00 UTC.
/// MJD = 48257.0, TJT = MJD - 40000 = 8257.0.
const NED_EPOCH_UTC_TJT: f64 = 8257.0;
const NED_TAI_UTC_S: f64 = 26.0;
/// UT1-TAI from JEOD tai_to_ut1.cc at 1991-01-01 (index 10592).
const NED_UT1_TAI_S: f64 = -25.381_221_5;
/// Integration step: 1.0s (matches JEOD SIM_NED DYNAMICS rate).
const NED_DT: f64 = 1.0;

#[test]
fn tier3_simulation_geodetic() {
    let csv_path = test_data_path("ned_ell_inc_ned.csv");
    assert!(
        csv_path.exists(),
        "SIM_NED CSV not found at {}.\n\
         Generate with: docker run --rm -e FORCE=1 -v $(pwd)/test_data:/output -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_ned_csv(&csv_path);
    assert!(records.len() > 100);
    let init = &records[0];

    // Initialize at 1991-01-01 00:00:00 UTC
    let epoch_tai_tjt = NED_EPOCH_UTC_TJT + NED_TAI_UTC_S / 86400.0;
    let mut time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    time.set_ut1_tai_offset(NED_UT1_TAI_S);

    let mut sim = Simulation::new(time, NED_DT);

    // Earth: point-mass gravity (JEOD SIM_NED uses spherical=1) with RNP rotation
    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY), // triggers RNP update for geodetic
    });

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        geodetic_planet: Some((earth, GEO_R_EQ, GEO_R_POL)),
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): SIM_NED geodetic (point-mass + RNP), {} points",
        records.len()
    );

    let mut max_alt_err = 0.0_f64;
    let mut max_lat_err = 0.0_f64;
    let mut max_lon_err = 0.0_f64;
    let mut max_pos_err = 0.0_f64;

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

        let alt_err = (geo.altitude - record.ellip_altitude).abs();
        let lat_err = (geo.latitude - record.ellip_latitude).abs();
        // Longitude wraps at +/-pi -- use angle_diff for correct comparison
        let lon_err = angle_diff(geo.longitude, record.ellip_longitude);

        max_alt_err = max_alt_err.max(alt_err);
        max_lat_err = max_lat_err.max(lat_err);
        max_lon_err = max_lon_err.max(lon_err);

        if (record.time % 3600.0).abs() < 6.1 {
            println!(
                "  t={:6.0}s: pos_err={:.4} m  alt_err={:.3e} m  lat_err={:.3e} rad  lon_err={:.3e} rad",
                record.time, pos_err, alt_err, lat_err, lon_err
            );
        }
    }

    println!("  Max position error:  {:.4} m", max_pos_err);
    println!("  Max altitude error:  {:.6e} m", max_alt_err);
    println!("  Max latitude error:  {:.6e} rad", max_lat_err);
    println!("  Max longitude error: {:.6e} rad", max_lon_err);

    // Point-mass gravity, 24h. Position should match JEOD to < 0.5m.
    assert!(
        max_pos_err < 0.5,
        "Position error {max_pos_err:.2} m exceeds 0.5 m"
    );
    // Geodetic tolerances: altitude sensitive to position error (~0.5m),
    // lat/lon from position error at ~6800 km radius (~7e-8 rad).
    assert!(
        max_alt_err < 1.0,
        "Altitude error {max_alt_err:.3e} m exceeds 1.0 m"
    );
    assert!(
        max_lat_err < 1e-6,
        "Latitude error {max_lat_err:.3e} rad exceeds 1e-6 rad"
    );
    assert!(
        max_lon_err < 1e-6,
        "Longitude error {max_lon_err:.3e} rad exceeds 1e-6 rad"
    );
}
