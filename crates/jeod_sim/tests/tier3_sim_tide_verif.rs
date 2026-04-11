//! Tier 3: SIM_tide_verif RUN_01 — solid body tides cross-validation
//!
//! Validates the tidal ΔC20 computation against JEOD's SIM_tide_verif.
//! RUN_01: GGM05C 8x8 + solid body tides + Sun/Moon 3rd-body, ISS highly
//! elliptical orbit, 8h at 60s logging. The CSV includes the computed
//! earth.sb_tide.dC20 value at each timestep.
//!
//! We compare:
//! 1. Our propagated trajectory against JEOD's position/velocity
//! 2. Our computed ΔC20 against JEOD's logged dC20

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_gravity::tides::{TidalBody, TidalConfig, EARTH_K2};
use jeod_sim::{
    DynamicsConfig, Ephemeris, EphemerisBody, GravityControl, GravityControls, GravityModel,
    GravitySource, GravitySourceEntry, JeodQuat, MassProperties, RotationModel, RotationalState,
    SimBody, Simulation, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use std::path::Path;

const MU_SUN: f64 = 1.327_124_40e20;
const MU_MOON: f64 = 4902.79980693169e9;

/// Epoch: Nov 20, 2007 00:00 UTC (same as SIM_dyncomp)
const EPOCH_UTC_TJT: f64 = 14424.0;
const TAI_UTC_S: f64 = 32.0;
const TAI_TO_UT1_S: f64 = -32.469;

fn earth_centered_position(body: EphemerisBody, tdb_jd: f64, ephemeris: &Ephemeris) -> DVec3 {
    let (pos, _) = ephemeris
        .get_earth_centered_state(body, tdb_jd)
        .expect("ephemeris query failed");
    pos
}

/// Parse tide CSV: time, pos[3], vel[3], dC20
fn load_tide_csv(path: &std::path::Path) -> Vec<(f64, DVec3, DVec3, f64)> {
    let content = std::fs::read_to_string(path).expect("read CSV");
    let mut records = Vec::new();
    for line in content.lines().skip(1) {
        let vals: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse::<f64>().unwrap())
            .collect();
        if vals.len() >= 8 {
            records.push((
                vals[0],
                DVec3::new(vals[1], vals[2], vals[3]),
                DVec3::new(vals[4], vals[5], vals[6]),
                vals[7],
            ));
        }
    }
    records
}

#[test]
fn tier3_simulation_tide_run01() {
    let csv_path = test_data_path("tide_run01_tide.csv");
    assert!(
        csv_path.exists(),
        "Tide reference not found at {}",
        csv_path.display()
    );

    let bsp_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/de421.bsp");
    assert!(
        bsp_path.exists(),
        "DE421 not found at {}",
        bsp_path.display()
    );
    let ephemeris = Ephemeris::from_bsp(&bsp_path).expect("load DE421");

    let jeod_root = jeod_test_data::jeod_path();
    let ggm05c_path = jeod_root.join("models/environment/gravity/data/src/earth_GGM05C.cc");
    let sh_data = jeod_sim::coefficients::load_from_jeod_cc(&ggm05c_path).expect("load GGM05C");
    let earth_mu = sh_data.mu;
    let earth_radius = sh_data.radius;

    let records = load_tide_csv(&csv_path);
    assert!(records.len() > 100);

    let (_, init_pos, init_vel, _) = &records[0];

    // Initialize at same epoch as SIM_dyncomp/SIM_tide_verif
    let epoch_tai_tjt = EPOCH_UTC_TJT + TAI_UTC_S / 86400.0;
    let mut time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    time.set_ut1_tai_offset(TAI_TO_UT1_S);

    let mut sim = Simulation::new(time, DT);

    // Earth source with SH gravity, RNP rotation, and tidal configuration
    let tdb_jd = sim.time.tdb_julian_date();
    let initial_sun = earth_centered_position(EphemerisBody::Sun, tdb_jd, &ephemeris);
    let initial_moon = earth_centered_position(EphemerisBody::Moon, tdb_jd, &ephemeris);

    let tidal_config = TidalConfig {
        k2: EARTH_K2,
        mu_primary: earth_mu,
        radius_primary: earth_radius,
        tidal_bodies: vec![
            TidalBody {
                mu: MU_MOON,
                position_inertial: initial_moon,
            },
            TidalBody {
                mu: MU_SUN,
                position_inertial: initial_sun,
            },
        ],
    };

    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: earth_mu,
            model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
        },
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY),
        rotation_model: RotationModel::EarthRNP,
        delta_c20: 0.0,
        tidal_config: Some(tidal_config),
    });

    // Sun: 3rd-body differential
    let sun = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_SUN,
            model: GravityModel::PointMass,
        },
        position: initial_sun,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });

    // Moon: 3rd-body differential
    let moon = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_MOON,
            model: GravityModel::PointMass,
        },
        position: initial_moon,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });

    // ISS mass (from Modified_data/mass.py — same as torque_simple)
    let inertia = DMat3::from_cols(
        DVec3::new(1.02e8, -6.96e6, -5.48e6),
        DVec3::new(-6.96e6, 0.91e8, 5.90e5),
        DVec3::new(-5.48e6, 5.90e5, 1.64e8),
    );
    let mass_props = MassProperties::with_inertia(400_000.0, inertia, DVec3::new(-3.0, -1.5, 4.0));

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: *init_pos,
            velocity: *init_vel,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        }),
        mass: Some(mass_props),
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: true,
            three_dof: false,
        },
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_nonspherical(earth, 8, 8, true),
                GravityControl::new_third_body(sun),
                GravityControl::new_third_body(moon),
            ],
        },
        compute_gravity_torque: true,
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): SIM_tide_verif RUN_01, {} points",
        records.len()
    );

    let mut our_states = Vec::with_capacity(records.len() - 1);
    let mut max_dc20_err = 0.0_f64;

    for &(time_s, _ref_pos, _ref_vel, ref_dc20) in &records[1..] {
        // Update Sun/Moon positions from ephemeris
        let target_tdb_jd = tdb_jd + time_s / 86400.0;
        let sun_pos = earth_centered_position(EphemerisBody::Sun, target_tdb_jd, &ephemeris);
        let moon_pos = earth_centered_position(EphemerisBody::Moon, target_tdb_jd, &ephemeris);

        sim.sources[sun].position = sun_pos;
        sim.sources[moon].position = moon_pos;

        // Update tidal body positions (Moon=0, Sun=1 in tidal_config)
        if let Some(ref mut tc) = sim.sources[earth].tidal_config {
            tc.tidal_bodies[0].position_inertial = moon_pos;
            tc.tidal_bodies[1].position_inertial = sun_pos;
        }

        sim.step_until(time_s);
        let body = sim.body(0);

        // Compare dC20
        let our_dc20 = sim.sources[earth].delta_c20;
        let dc20_err = (our_dc20 - ref_dc20).abs();
        max_dc20_err = max_dc20_err.max(dc20_err);

        our_states.push(StateLog {
            time: time_s,
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            ..Default::default()
        });
    }

    let ref_states: Vec<StateLog> = records[1..]
        .iter()
        .map(|&(t, pos, vel, _)| StateLog {
            time: t,
            position: Some(pos),
            velocity: Some(vel),
            ..Default::default()
        })
        .collect();

    let report = CrossvalReport::compute("tier3_simulation_tide_run01", &our_states, &ref_states);
    report.write();

    let max_pos = report.max_position_component();
    println!("  Max position error: {max_pos:.6e} m");
    println!("  Max dC20 error:     {max_dc20_err:.6e}");

    // Position error ~2 m dominated by DE421 interpolation difference
    // (same as other 3rd-body tests). ΔC20 matches at machine precision.
    report.assert_position([2.117, 1.786, 0.582]);
    report.assert_velocity([2.452e-3, 2.001e-3, 6.305e-4]);

    // dC20 matches at machine precision — validates the tidal formula
    assert!(
        max_dc20_err < 1e-14,
        "dC20 error {max_dc20_err:.2e} exceeds 1e-14"
    );
}
