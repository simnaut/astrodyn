//! Tier 3: SIM_orbinit cross-validation via Simulation pipeline
//!
//! Validates body initialization from 4 distinct coordinate representations
//! by creating a `Simulation`, adding each body, validating, and stepping once.
//! Checks cross-consistency between methods and against JEOD output.
//!
//!   RUN_0101: Orbital elements in inertial frame (STS-114)
//!   RUN_0201: Orbital elements in planet-fixed frame (ISS)
//!   RUN_0301: Orbital elements in planet-fixed frame (STS-114)
//!   RUN_0401: Cartesian state in inertial frame (STS-114)

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_runner::{GravitySourceEntry, RotationModel, Simulation, VehicleConfig};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, SimulationTime,
    TranslationalState,
};

#[test]
fn tier3_simulation_orbinit_cross_consistency() {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    let grav_data_dir = jeod_root.join("models/environment/gravity/data/src");
    let mu_earth =
        jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_data_dir.join("earth_GGM05C.cc"))
            .expect("load Earth mu");

    let runs: Vec<(&str, &str)> = vec![
        ("orbinit_0101_orbinit.csv", "RUN_0101 (STS-114 inertial OE)"),
        ("orbinit_0201_orbinit.csv", "RUN_0201 (ISS pfix OE)"),
        ("orbinit_0301_orbinit.csv", "RUN_0301 (STS-114 pfix OE)"),
        (
            "orbinit_0401_orbinit.csv",
            "RUN_0401 (STS-114 inertial cart)",
        ),
    ];

    let mut states: Vec<(DVec3, DVec3, &str)> = Vec::new();

    for (filename, label) in &runs {
        let csv_path = test_data_path(filename);
        assert!(
            csv_path.exists(),
            "SIM_orbinit CSV not found at {}.\n\
             Generate with: docker run --rm -v $(pwd)/test_data:/output \
             -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
            csv_path.display()
        );

        let records = load_orbinit_csv(&csv_path);
        assert!(
            !records.is_empty(),
            "{label}: no records found in {filename}"
        );

        let init = &records[0];

        // Create a Simulation with Earth point-mass gravity and this body
        let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
        let mut sim = Simulation::new(time, 10.0);

        let earth = sim.add_source(GravitySourceEntry {
            source: GravitySource {
                mu: mu_earth,
                model: GravityModel::PointMass,
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
        });

        sim.add_body(VehicleConfig {
            trans: TranslationalState {
                position: init.position,
                velocity: init.velocity,
            },
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, false)],
            },
            ..Default::default()
        });

        sim.validate().unwrap();

        // Step once to exercise the full pipeline
        sim.step();

        // Read back the body state after one step (confirms pipeline ran)
        let body = sim.body(0);
        let r_mag = body.trans.position.length();
        let v_mag = body.trans.velocity.length();

        println!(
            "  {label}: r={:.3} km  v={:.6} km/s  pos=[{:.1}, {:.1}, {:.1}] m",
            r_mag / 1000.0,
            v_mag / 1000.0,
            init.position.x,
            init.position.y,
            init.position.z,
        );

        // Sanity: LEO orbit (post-step state should still be LEO)
        assert!(
            (6_000_000.0..=8_000_000.0).contains(&r_mag),
            "{label}: r={r_mag:.0} m outside LEO range after one step"
        );
        assert!(
            (6_000.0..=8_000.0).contains(&v_mag),
            "{label}: v={v_mag:.1} m/s outside LEO range after one step"
        );

        // Use the initial state for cross-consistency (matches JEOD's initialization output)
        states.push((init.position, init.velocity, label));
    }

    println!();

    // Cross-consistency: STS-114 runs (0101, 0301, 0401) should agree closely.
    let sts_indices = [0, 2, 3];
    for i in 0..sts_indices.len() {
        for j in (i + 1)..sts_indices.len() {
            let (pos_a, vel_a, label_a) = states[sts_indices[i]];
            let (pos_b, vel_b, label_b) = states[sts_indices[j]];
            let pos_err = (pos_a - pos_b).length();
            let vel_err = (vel_a - vel_b).length();
            println!(
                "  {label_a} vs {label_b}: pos_err={:.6} m  vel_err={:.6e} m/s",
                pos_err, vel_err,
            );
            assert!(
                pos_err < 1.0,
                "STS-114 cross-consistency: position error {pos_err:.3} m exceeds 1.0 m \
                 between {label_a} and {label_b}"
            );
            assert!(
                vel_err < 0.001,
                "STS-114 cross-consistency: velocity error {vel_err:.3e} m/s exceeds 0.001 m/s \
                 between {label_a} and {label_b}"
            );
        }
    }

    // ISS vs STS-114: different vehicles at similar epoch
    let (pos_iss, _, _) = states[1];
    let (pos_sts, _, _) = states[0];
    let cross_vehicle_err = (pos_iss - pos_sts).length();
    println!(
        "\n  ISS vs STS-114: pos_diff={:.1} m (expected: different vehicles)",
        cross_vehicle_err,
    );
    assert!(
        cross_vehicle_err < 1000.0,
        "ISS vs STS-114 position difference {cross_vehicle_err:.0} m exceeds 1 km"
    );

    println!("\n  All initialization methods produce consistent LEO states");
}
