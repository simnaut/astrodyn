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
use jeod_runner::{
    GravitySourceEntry, RotationModel, ShadowBody, Simulation, SrpModel, VehicleConfig,
};
use jeod_sim::{
    Ephemeris, EphemerisBody, FlatPlate, FlatPlateParams, FlatPlateState, FlatPlateThermal,
    GravityControl, GravityControls, GravityModel, GravitySource, MassProperties, SimulationTime,
    ThermalIntegrationOrder, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use std::path::Path;

/// SIM_3_ORBIT_1st_ORDER directory relative to JEOD root.
const SIM_3_ORBIT_1ST: &str = "models/interactions/radiation_pressure/verif/SIM_3_ORBIT_1st_ORDER";

const SRP_MASS: f64 = 300.0;

fn srp_plates() -> Vec<(FlatPlate, FlatPlateParams, FlatPlateThermal)> {
    let params = FlatPlateParams {
        albedo: 0.5,
        diffuse: 0.5,
    };
    let thermal = FlatPlateThermal {
        emissivity: 0.5,
        heat_capacity_per_area: 50.0,
        thermal_power_dump: 0.0,
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

fn srp_sun_position(sim_time: f64, epoch_tai_tjt: f64, ephemeris: &Ephemeris) -> DVec3 {
    let sim_days = sim_time / 86400.0;
    let tdb_jd = (epoch_tai_tjt + sim_days) + 40000.0 + 2_400_000.5;
    // Phase 1 (#103): DVec3 accessor is deprecated; migration is Phase 3+ work.
    #[allow(deprecated)]
    let (sun_pos, _) = ephemeris
        .get_earth_centered_state(EphemerisBody::Sun, tdb_jd)
        .expect("Sun position query failed");
    sun_pos
}

#[test]
fn tier3_srp_1st_order_trajectory() {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

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

    let sim_dir = jeod_root.join(SIM_3_ORBIT_1ST);
    let grav_data_dir = jeod_root.join("models/environment/gravity/data/src");

    // Load epoch from JEOD time config. SIM_3_ORBIT_1st_ORDER uses TAI initializer.
    let time_cfg = jeod_test_data::time_config::load_time_config(
        &sim_dir.join("Modified_data/date_and_time.py"),
    );
    let epoch_tai_tjt = time_cfg.tai_tjt();

    // Load integration step size from S_define
    let srp_dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));

    // Load Earth mu from JEOD gravity data
    let srp_mu_earth =
        jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_data_dir.join("earth_GGM05C.cc"))
            .expect("load Earth mu");

    let trajectory = load_srp_trajectory(&csv_path);
    assert!(trajectory.len() > 100);
    let init = &trajectory[0];

    let plates = srp_plates();
    let num_plates = plates.len();
    let init_temp = 270.0_f64;

    let time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, srp_dt);

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: srp_mu_earth,
                model: GravityModel::PointMass,
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    // Sun: mu=0 because the JEOD SIM_3_ORBIT_1st_ORDER reference sim uses Sun
    // only for SRP direction, not gravitational perturbation.
    let initial_sun = srp_sun_position(0.0, epoch_tai_tjt, &ephemeris);
    let sun = sim.add_source(
        "Sun",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: initial_sun,
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
        },
    );
    sim.sun_source = Some(sun);

    sim.add_body(VehicleConfig {
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
        srp: Some(SrpModel::FlatPlate(FlatPlateState {
            plates,
            temperatures: vec![init_temp; num_plates],
            t_pow4_cached: vec![init_temp.powi(4); num_plates],
            // JEOD SIM_3_ORBIT_1st_ORDER wires `rad_pressure.update` as a
            // derivative-class job (per RK4 stage) with ER7_Utils
            // first-order integrator on temperature. Match that here.
            integration_order: ThermalIntegrationOrder::DerivativeFirstOrder,
            ..Default::default()
        })),
        shadow_body: Some(ShadowBody {
            source_idx: earth,
            radius: jeod_sim::EARTH.shadow_radius,
        }),
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
        sim.set_source_position(
            sun,
            srp_sun_position(record.time, epoch_tai_tjt, &ephemeris),
        );
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

    report.assert_position([7.709e1, 8.021e1, 3.481e1]);
}
