//! Tier 3: SIM_3_ORBIT_1st_ORDER cross-validation
//!
//! Full trajectory comparison using the same flat-plate SRP model as the
//! existing SIM_3_ORBIT test. The "1st-order" in the JEOD sim name refers
//! to the thermal ODE integrator order, not the SRP physics — the radiation
//! model is identical (6 flat plates, conical Earth shadow, thermal emission).
//!
//! Differences from SIM_3_ORBIT: the thermal state (plate temperatures) may
//! diverge due to integrator order differences, causing small force deviations
//! that accumulate over the ~23-day trajectory.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_sim::{
    Ephemeris, EphemerisBody, FlatPlate, FlatPlateParams, FlatPlateState, FlatPlateThermal,
    GravityControl, GravityControls, GravityModel, GravitySource, GravitySourceEntry,
    MassProperties, SimBody, Simulation, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
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
fn tier3_srp_1st_order_trajectory() {
    let csv_path = test_data_path("srp_1st_order_radiation_srp_orbit.csv");
    assert!(
        csv_path.exists(),
        "SIM_3_ORBIT_1st_ORDER CSV not found at {}.\n\
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

    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: SRP_MU_EARTH,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        t_inertial_pfix: None,
    });

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
        "Tier 3 (Simulation): SRP 1st-order trajectory, {} points over {:.0} days",
        trajectory.len(),
        trajectory.last().unwrap().time / 86400.0
    );

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    let mut ref_states = Vec::with_capacity(trajectory.len() - 1);

    for record in &trajectory[1..] {
        sim.sources[sun].position = srp_sun_position(record.time, &ephemeris);
        sim.step_until(record.time);

        let body = sim.body(0);

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

        if (record.time % 86400.0).abs() < 500.1 {
            let pos_error = (body.trans.position - record.position).length();
            println!(
                "  t={:8.0}s ({:5.1}d): pos_err={:10.2} m",
                record.time,
                record.time / 86400.0,
                pos_error
            );
        }
    }

    let report =
        CrossvalReport::compute("tier3_srp_1st_order_trajectory", &our_states, &ref_states);
    report.write();

    let max_pos_error = report.max_position_component();
    println!("  Max position error: {:.6e} m", max_pos_error);

    report.assert_position([8.296e1, 8.491e1, 3.686e1]);
}
