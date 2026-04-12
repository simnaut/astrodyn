//! Tier 3: SIM_1_BASIC — flat-plate SRP verification via Simulation pipeline
//!
//! SIM_1_BASIC places a 6-plate vehicle at ~1 AU from the Sun with zero velocity
//! and no gravity. The test creates a Simulation matching JEOD's configuration,
//! steps to exercise the SRP pipeline, and compares force/torque against
//! JEOD's reference CSV at each timestep.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_sim::{
    FlatPlate, FlatPlateParams, FlatPlateState, FlatPlateThermal, GravityModel, GravitySource,
    GravitySourceEntry, RotationModel, SimBody, Simulation, SimulationTime, TranslationalState,
};

/// Build the 6-plate surface matching SIM_1_BASIC's Modified_data/radiation_surface.py.
/// All plates: area=2.0 m², albedo=0.0, diffuse=0.5, emissivity=1.0,
/// heat_capacity_per_area=600.0, initial temperature=270.0 K.
fn sim1_basic_plates() -> Vec<(FlatPlate, FlatPlateParams, FlatPlateThermal)> {
    let params = FlatPlateParams {
        albedo: 0.0,
        diffuse: 0.5,
    };
    let thermal = FlatPlateThermal {
        emissivity: 1.0,
        heat_capacity_per_area: 600.0,
    };
    vec![
        (
            FlatPlate {
                area: 2.0,
                normal: DVec3::X,
                position: DVec3::new(2.0, 0.0, 0.0),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 2.0,
                normal: -DVec3::Y,
                position: DVec3::new(0.0, -2.0, 0.0),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 2.0,
                normal: -DVec3::X,
                position: DVec3::new(-2.0, 0.0, 0.0),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 2.0,
                normal: DVec3::Y,
                position: DVec3::new(0.0, 2.0, 0.0),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 2.0,
                normal: DVec3::Z,
                position: DVec3::new(0.0, 0.0, 7.5),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 2.0,
                normal: -DVec3::Z,
                position: DVec3::new(0.0, 0.0, -7.5),
            },
            params,
            thermal,
        ),
    ]
}

fn run_srp_basic_test(csv_filename: &str, label: &str) {
    let csv_path = test_data_path(csv_filename);
    assert!(
        csv_path.exists(),
        "SIM_1_BASIC CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_srp_basic_csv(&csv_path);
    assert!(
        !records.is_empty(),
        "{label}: no records found in {csv_filename}"
    );

    // SIM_1_BASIC epoch: 2005-12-31 23:59:50 TAI
    // TAI-UTC = 33s at this date.
    // TAI TJT = (JD_UTC_midnight + tai_seconds/86400) - 2400000.5 - 40000
    let tai_seconds_of_day = 23.0 * 3600.0 + 59.0 * 60.0 + 50.0 + 33.0;
    let tai_tjt = 2_453_736.5 + tai_seconds_of_day / 86400.0 - 2_400_000.5 - 40_000.0;
    let time = SimulationTime::new(tai_tjt, jeod_sim::default_leap_second_table());

    let dt = 1.0; // SIM_1_BASIC logs at 1s intervals
    let mut sim = Simulation::new(time, dt);

    // Sun at origin (SIM_1_BASIC integrates in Sun.inertial frame)
    let sun = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    sim.sun_source = Some(sun);

    // Vehicle at ~1 AU from Sun, zero velocity, 6-plate flat-plate SRP
    let init_temp = 270.0;
    let plates = sim1_basic_plates();
    let num_plates = plates.len();
    sim.add_body(SimBody {
        trans: TranslationalState {
            position: DVec3::new(1.5e11, 0.0, 0.0),
            velocity: DVec3::ZERO,
        },
        mass: Some(jeod_sim::MassProperties::new(1.0)),
        flat_plate_state: Some(FlatPlateState {
            plates,
            temperatures: vec![init_temp; num_plates],
            t_pow4_cached: vec![init_temp.powi(4); num_plates],
        }),
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): SIM_1_BASIC {label}, {} points",
        records.len()
    );

    let mut max_force_err = 0.0_f64;
    let mut max_torque_err = 0.0_f64;

    // Step once to initialize SRP computation before comparing
    sim.step();

    for (i, rec) in records.iter().enumerate() {
        if i > 0 {
            sim.step();
        }

        let body = sim.body(0);
        let rad = body
            .radiation_force
            .as_ref()
            .expect("SRP should produce radiation_force");

        let force_err = (rad.force - rec.force).length();
        let torque_err = (rad.torque - rec.torque).length();
        max_force_err = max_force_err.max(force_err);
        max_torque_err = max_torque_err.max(torque_err);

        if i == 0 || i == records.len() - 1 {
            println!(
                "  t={:.0}s: force=[{:.6e}, {:.6e}, {:.6e}] N",
                rec.time, rad.force.x, rad.force.y, rad.force.z
            );
        }
    }

    println!(
        "  Max force error: {:.6e} N  Max torque error: {:.6e} N*m",
        max_force_err, max_torque_err
    );

    // Flux at 1 AU: ~1361 W/m². Force direction: away from Sun (+X).
    let body = sim.body(0);
    let rad = body.radiation_force.as_ref().unwrap();
    assert!(
        rad.force.x > 0.0,
        "{label}: expected force in +X direction (away from Sun), got force.x={:.6e}",
        rad.force.x
    );

    // Tolerance: 5% above observed max error (~2.4e-8 N for default surface).
    assert!(
        max_force_err < 2.6e-8,
        "{label}: max force error {max_force_err:.3e} N exceeds 2.6e-8 N"
    );
}

#[test]
fn tier3_simulation_srp_basic_default() {
    run_srp_basic_test("srp_basic_srp_basic.csv", "RUN_basic (default surface)");
}

#[test]
fn tier3_simulation_srp_basic_varied_cr() {
    run_srp_basic_test(
        "srp_basic_cr_srp_basic.csv",
        "RUN_basic_cr (varied reflection)",
    );
}
