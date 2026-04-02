//! Tier 3: SIM_3_ORBIT cross-validation (radiation_pressure/verif/SIM_3_ORBIT)
//!
//! Flat-plate SRP + conical Earth shadow, GEO orbit, ~23 days.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_sim::{
    Ephemeris, EphemerisBody, FlatPlate, FlatPlateParams, FlatPlateState, FlatPlateThermal,
    GravityControl, GravityControls, GravityModel, GravitySource, GravitySourceEntry,
    MassProperties, SimBody, Simulation, SimulationTime, TranslationalState,
};
use std::path::Path;

const SRP_MU_EARTH: f64 = 3.986_004_415e14;
const SRP_R_EARTH: f64 = 6_378_137.0;
const SRP_MASS: f64 = 300.0;
const SRP_DT: f64 = 1.0;
const SRP_EPOCH_TJT: f64 = 11148.0; // 1998-12-01 UTC

fn srp_plates() -> Vec<(FlatPlate, FlatPlateParams, FlatPlateThermal)> {
    let params = FlatPlateParams {
        albedo: 0.5,
        diffuse: 0.5,
    };
    let thermal = FlatPlateThermal {
        emissivity: 0.5,
        heat_capacity_per_area: 50.0,
    };
    vec![
        (
            FlatPlate {
                area: 60.0,
                normal: DVec3::X,
                position: DVec3::new(2.0, 0.0, 0.0),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 60.0,
                normal: -DVec3::Y,
                position: DVec3::new(0.0, -2.0, 0.0),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 60.0,
                normal: -DVec3::X,
                position: DVec3::new(-2.0, 0.0, 0.0),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 60.0,
                normal: DVec3::Y,
                position: DVec3::new(0.0, 2.0, 0.0),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 16.0,
                normal: DVec3::Z,
                position: DVec3::new(0.0, 0.0, 7.5),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 16.0,
                normal: -DVec3::Z,
                position: DVec3::new(0.0, 0.0, -7.5),
            },
            params,
            thermal,
        ),
    ]
}

fn srp_sun_position(sim_time: f64, ephemeris: &Ephemeris) -> DVec3 {
    let sim_days = sim_time / 86400.0;
    let tdb_jd = (SRP_EPOCH_TJT + sim_days) + 40000.0 + 2_400_000.5;
    let (sun_pos, _) = ephemeris
        .get_earth_centered_state(EphemerisBody::Sun, tdb_jd)
        .expect("Sun position query failed");
    sun_pos
}

#[test]
fn tier3_simulation_srp_flat_plate() {
    let csv_path = test_data_path("srp_orbit_radiation_srp_orbit.csv");
    assert!(
        csv_path.exists(),
        "SRP reference not found at {}",
        csv_path.display()
    );

    let bsp_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/de421.bsp");
    assert!(
        bsp_path.exists(),
        "DE421 ephemeris not found at {}",
        bsp_path.display()
    );
    let ephemeris = Ephemeris::from_bsp(&bsp_path).expect("load DE421");

    let trajectory = load_srp_trajectory(&csv_path);
    assert!(trajectory.len() > 100);
    let init = &trajectory[0];

    let plates = srp_plates();
    let num_plates = plates.len();
    let init_temp = 270.0_f64;

    // Epoch: 1998-12-01 UTC. TAI-UTC=31s at this date.
    let epoch_tai_tjt = SRP_EPOCH_TJT + 31.0 / 86400.0;
    let time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, SRP_DT);

    // Earth at origin (gravity source + shadow body)
    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: SRP_MU_EARTH,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        t_inertial_pfix: None,
    });

    // Sun (position updated each logging interval from ephemeris)
    let initial_sun = srp_sun_position(0.0, &ephemeris);
    let sun = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: initial_sun,
        t_inertial_pfix: None,
    });
    sim.sun_source = Some(sun);

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        mass: Some(MassProperties::with_inertia(
            SRP_MASS,
            DMat3::from_diagonal(DVec3::splat(1.0)),
            DVec3::ZERO,
        )),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        flat_plate_state: Some(FlatPlateState {
            plates,
            temperatures: vec![init_temp; num_plates],
            t_pow4_cached: vec![init_temp.powi(4); num_plates],
        }),
        shadow_body: Some((earth, SRP_R_EARTH)),
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): SRP flat-plate + shadow, {} points over {:.0} days",
        trajectory.len(),
        trajectory.last().unwrap().time / 86400.0
    );

    let mut max_pos_error = 0.0_f64;

    for record in &trajectory[1..] {
        // Update Sun position from ephemeris before stepping
        sim.sources[sun].position = srp_sun_position(record.time, &ephemeris);

        sim.step_until(record.time);

        let body = sim.body(0);
        let pos_error = (body.trans.position - record.position).length();
        max_pos_error = max_pos_error.max(pos_error);

        if (record.time % 86400.0).abs() < 500.1 {
            println!(
                "  t={:8.0}s ({:5.1}d): pos_err={:10.2} m",
                record.time,
                record.time / 86400.0,
                pos_error
            );
        }
    }

    println!("  Max position error: {:.2} m", max_pos_error);

    // Tolerance matches existing tier3_srp_trajectory test
    assert!(
        max_pos_error < 50.0,
        "Position error {max_pos_error:.2} m exceeds 50 m over ~23 days"
    );
}
