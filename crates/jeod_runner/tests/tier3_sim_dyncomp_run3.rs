//! Tier 3: SIM_dyncomp RUN_3A/3B — Spherical harmonics gravity (4x4 / 8x8 + RNP)
//!
//! All simulation parameters (epoch, step size, gravity degree/order) are loaded
//! from the JEOD source files rather than hardcoded, per issue #44.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_runner::{GravitySourceEntry, RotationModel, SimBody, Simulation};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, SimulationTime,
    TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};

/// SIM_dyncomp root directory (relative to JEOD_HOME).
const SIM_DYNCOMP: &str = "verif/SIM_dyncomp";

fn run_sh_simulation_test(
    csv_name: &str,
    run_dir: &str,
    label: &str,
    test_name: &str,
    pos_tol: [f64; 3],
    vel_tol: [f64; 3],
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

    let sim_dir = jeod_root.join(SIM_DYNCOMP);

    // Load epoch and time offsets from JEOD time config
    let time_cfg =
        jeod_test_data::time_config::load_time_config(&sim_dir.join("Modified_data/time.py"));
    let epoch_tai_tjt = time_cfg.tai_tjt();
    let ut1_tai_offset = time_cfg
        .ut1_tai_offset()
        .expect("SIM_dyncomp time.py must specify tai_to_ut1_override_val");

    // Load integration step size from S_define
    let dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));

    // Load gravity control (degree/order) from grav_controls.py + RUN input chain.
    // RUN_3B exec's RUN_3A, so we include both when needed. Files are processed in
    // order; later assignments win.
    let mut grav_files: Vec<std::path::PathBuf> =
        vec![sim_dir.join("Modified_data/grav_controls.py")];
    // RUN_3A is always in the chain (RUN_3B exec's it)
    grav_files.push(sim_dir.join("SET_test/RUN_3A/input.py"));
    if run_dir != "RUN_3A" {
        grav_files.push(sim_dir.join(format!("SET_test/{run_dir}/input.py")));
    }
    let grav_file_refs: Vec<&std::path::Path> = grav_files.iter().map(|p| p.as_path()).collect();
    let grav_cfg = jeod_test_data::gravity_control::load_gravity_control(&grav_file_refs);

    // Load GGM02C spherical harmonics coefficients
    let ggm02c_path = jeod_root.join("models/environment/gravity/data/src/earth_GGM02C.cc");
    let sh_data = jeod_sim::coefficients::load_from_jeod_cc(&ggm02c_path).expect("load GGM02C");

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(trajectory.len() > 100);
    let init = &trajectory[0];

    // Build a spherical harmonics gravity source.
    // The Simulation will update planet-fixed rotation each step via RNP.
    let sh_source = GravitySource {
        mu: sh_data.mu,
        model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
    };

    // Initialize Simulation at the SIM_dyncomp epoch (parsed from time.py).
    let mut time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    // Set UT1-TAI offset so GMST computation matches JEOD's time.py configuration
    time.set_ut1_tai_offset(ut1_tai_offset);

    let mut sim = Simulation::new(time, dt);

    // Earth source with planet-fixed rotation (Simulation updates it each step).
    // Initialize with identity — first step() will compute the real rotation.
    let earth = sim.add_source(GravitySourceEntry {
        source: sh_source,
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY), // presence triggers ephemeris update
        delta_c20: 0.0,
        rotation_model: RotationModel::EarthRNP,
        tidal_config: None,
    });

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_nonspherical(
                earth,
                grav_cfg.degree,
                grav_cfg.order,
                grav_cfg.gradient,
            )],
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    println!("Tier 3 (Simulation): {label}, {} points", trajectory.len());

    // Log our propagated states
    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    for record in &trajectory[1..] {
        sim.step_until(record.time);
        let body = sim.body(0);

        let pos_error = (body.trans.position - record.composite_body.position).length();
        let vel_error = (body.trans.velocity - record.composite_body.velocity).length();
        if (record.time % 3600.0).abs() < 30.1 {
            println!(
                "  t={:6.0}s: pos_err={:10.4} m  vel_err={:.6} m/s",
                record.time, pos_error, vel_error
            );
        }

        our_states.push(StateLog {
            time: record.time,
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            acceleration: Some(body.frame_derivs.trans_accel),
            ang_accel: Some(body.frame_derivs.rot_accel),
            ..Default::default()
        });
    }

    // Reference states from JEOD CSV
    let ref_states: Vec<StateLog> = trajectory[1..]
        .iter()
        .map(|r| StateLog {
            time: r.time,
            position: Some(r.composite_body.position),
            velocity: Some(r.composite_body.velocity),
            acceleration: r.derivs.as_ref().map(|d| d.trans_accel),
            ang_accel: r.derivs.as_ref().map(|d| d.rot_accel),
            ..Default::default()
        })
        .collect();

    // Post-process: compute errors
    let report = CrossvalReport::compute(test_name, &our_states, &ref_states);
    report.write();

    println!(
        "  Max position error: {:.6e} m",
        report.max_position_component()
    );
    println!(
        "  Max velocity error: {:.6e} m/s",
        report.max_velocity_component()
    );

    report.assert_position(pos_tol);
    report.assert_velocity(vel_tol);
}

#[test]
fn tier3_simulation_run3a_sh4x4() {
    run_sh_simulation_test(
        "dyncomp_run3a_state.csv",
        "RUN_3A",
        "RUN_3A (4x4 SH + RNP)",
        "tier3_simulation_run3a_sh4x4",
        [5.3e-2, 1.344e-1, 1.026e-1],
        [6.151e-5, 1.246e-4, 1.24e-4],
    );
}

#[test]
fn tier3_simulation_run3b_sh8x8() {
    run_sh_simulation_test(
        "dyncomp_run3b_state.csv",
        "RUN_3B",
        "RUN_3B (8x8 SH + RNP)",
        "tier3_simulation_run3b_sh8x8",
        [1.325e-1, 2.3e-1, 1.646e-1],
        [1.478e-4, 2.329e-4, 1.892e-4],
    );
}
