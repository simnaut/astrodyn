//! Tier 3: SIM_dyncomp RUN_3A/3B — Spherical harmonics gravity (4x4 / 8x8 + RNP)

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, GravitySourceEntry, SimBody,
    Simulation, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::crossval_report;

/// JEOD SIM_dyncomp epoch constants (from the original SH test).
const SH_TAI_UTC_S: f64 = 32.0;
const SH_TAI_TO_UT1_S: f64 = -32.469;
const SH_EPOCH_UTC_TJT: f64 = 14424.0;

fn run_sh_simulation_test(
    csv_name: &str,
    degree: usize,
    order: usize,
    label: &str,
    test_name: &str,
) {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    let csv_path = test_data_path(csv_name);
    assert!(
        csv_path.exists(),
        "JEOD trajectory not found at {}",
        csv_path.display()
    );

    // Load GGM02C spherical harmonics coefficients
    let ggm02c_path = jeod_root.join("models/environment/gravity/data/src/earth_GGM02C.cc");
    let sh_data = jeod_sim::coefficients::load_from_jeod_cc(&ggm02c_path).expect("load GGM02C");

    let trajectory = load_trans_trajectory(&csv_path);
    assert!(trajectory.len() > 100);
    let init = &trajectory[0];

    // Build a spherical harmonics gravity source.
    // The Simulation will update planet-fixed rotation each step via RNP.
    let sh_source = GravitySource {
        mu: sh_data.mu,
        model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
    };

    // Initialize Simulation at the SIM_dyncomp epoch.
    // TAI TJT = UTC TJT + TAI-UTC/86400
    let epoch_tai_tjt = SH_EPOCH_UTC_TJT + SH_TAI_UTC_S / 86400.0;
    let mut time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    // Set UT1-TAI offset so GMST computation matches JEOD's time.py configuration
    time.set_ut1_tai_offset(SH_TAI_TO_UT1_S);

    let mut sim = Simulation::new(time, DT);

    // Earth source with planet-fixed rotation (Simulation updates it each step).
    // Initialize with identity — first step() will compute the real rotation.
    let earth = sim.add_source(GravitySourceEntry {
        source: sh_source,
        position: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY), // presence triggers ephemeris update
    });

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_nonspherical(
                earth, degree, order, false,
            )],
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    println!("Tier 3 (Simulation): {label}, {} points", trajectory.len());

    let mut max_pos_error = 0.0_f64;
    let mut max_vel_error = 0.0_f64;

    for record in &trajectory[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);
        let pos_error = (body.trans.position - record.position).length();
        let vel_error = (body.trans.velocity - record.velocity).length();
        max_pos_error = max_pos_error.max(pos_error);
        max_vel_error = max_vel_error.max(vel_error);

        if (record.time % 3600.0).abs() < 30.1 {
            println!(
                "  t={:6.0}s: pos_err={:10.4} m  vel_err={:.6} m/s",
                record.time, pos_error, vel_error
            );
        }
    }

    println!("  Max position error: {:.6e} m", max_pos_error);
    println!("  Max velocity error: {:.6e} m/s", max_vel_error);

    crossval_report(
        test_name,
        &[
            ("position", max_pos_error, 0.5, "m"),
            ("velocity", max_vel_error, 0.001, "m/s"),
        ],
    );

    // Tolerances match existing tier3_spherical_harmonics test
    assert!(
        max_pos_error < 0.5,
        "{label}: position error {max_pos_error:.2} m exceeds 0.5 m"
    );
    assert!(
        max_vel_error < 0.001,
        "{label}: velocity error {max_vel_error:.6} m/s exceeds 0.001 m/s"
    );
}

#[test]
fn tier3_simulation_run3a_sh4x4() {
    run_sh_simulation_test(
        "dyncomp_run3a_state.csv",
        4,
        4,
        "RUN_3A (4x4 SH + RNP)",
        "tier3_simulation_run3a_sh4x4",
    );
}

#[test]
fn tier3_simulation_run3b_sh8x8() {
    run_sh_simulation_test(
        "dyncomp_run3b_state.csv",
        8,
        8,
        "RUN_3B (8x8 SH + RNP)",
        "tier3_simulation_run3b_sh8x8",
    );
}
