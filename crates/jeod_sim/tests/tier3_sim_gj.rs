//! Tier 3: SIM_GJ_test — Gauss-Jackson cross-validation against JEOD.
//!
//! Cross-validates our Gauss-Jackson (Störmer-Cowell) integrator against
//! JEOD's implementation on a circular orbit.
//!
//! Scenario: r₀=[9e6, 0, 0]m, v₀=[0, 8000, 0]m/s, μ=5.76e14, spherical
//! gravity, 300,000s (~83h), logged every 300s (1000 points).
//! Translational dynamics only.
//!
//! Tests vary the GJ order and timestep:
//! - order 8, dt=1s (baseline)
//! - order 4, dt=1s
//! - order 12, dt=1s
//! - order 8, dt=10s

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_sim::{
    GaussJacksonConfig, GravityControl, GravityControls, GravityModel, GravitySource,
    GravitySourceEntry, IntegratorType, SimBody, Simulation, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};

/// Non-standard μ used by SIM_GJ_test (set in input_common.py).
const MU_GJ_TEST: f64 = 5.76e14;

/// Run a GJ cross-validation test with the given config and reference CSV.
///
/// `time_scale` converts CSV sim-time to dynamic time via
/// `dyn_time = sim_time * time_scale`. JEOD's SIM_GJ_test uses
/// `dyn_time.scale_factor` to vary the effective dt: a scale of 10
/// means each CSV sim-second corresponds to 10 dynamic seconds.
/// For dt=1 runs, `time_scale = 1.0`. For dt=10 runs, `time_scale = 10.0`.
fn run_gj_test(
    test_name: &str,
    csv_label: &str,
    config: GaussJacksonConfig,
    dt: f64,
    time_scale: f64,
    pos_tol: [f64; 3],
    vel_tol: [f64; 3],
) {
    let csv_path = test_data_path(&format!("{csv_label}_gj.csv"));
    assert!(
        csv_path.exists(),
        "JEOD GJ reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let trajectory = load_gj_csv(&csv_path);
    assert!(
        trajectory.len() > 100,
        "Expected >100 records, got {}",
        trajectory.len()
    );

    let init = &trajectory[0];

    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_GJ_TEST,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        tidal_config: None,
    });

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        integrator: IntegratorType::GaussJackson(config),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    let final_dyn_time = trajectory.last().unwrap().time * time_scale;
    println!(
        "Tier 3 (Simulation): {test_name}, {} points over {:.0}s, dt={dt}s",
        trajectory.len(),
        final_dyn_time
    );

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    for record in &trajectory[1..] {
        let dyn_time = record.time * time_scale;
        sim.step_until(dyn_time);
        let body = sim.body(0);
        our_states.push(StateLog {
            time: dyn_time,
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            ..Default::default()
        });
    }

    let ref_states: Vec<StateLog> = trajectory[1..]
        .iter()
        .map(|r| StateLog {
            time: r.time * time_scale,
            position: Some(r.position),
            velocity: Some(r.velocity),
            ..Default::default()
        })
        .collect();

    let report = CrossvalReport::compute(test_name, &our_states, &ref_states);
    report.write();

    let max_pos = report.max_position_component();
    let max_vel = report.max_velocity_component();
    println!("  Max position error: {max_pos:.6e} m");
    println!("  Max velocity error: {max_vel:.6e} m/s");

    report.assert_position(pos_tol);
    report.assert_velocity(vel_tol);
}

#[test]
fn tier3_simulation_gj_order8() {
    // Baseline: GJ order 8, dt=1s.
    // Observed: pos [1.258e-4, 1.246e-4, 0] m, vel [1.106e-7, 1.112e-7, 0] m/s.
    run_gj_test(
        "tier3_simulation_gj_order8",
        "integ_gj",
        GaussJacksonConfig::with_order(8),
        1.0,
        1.0,
        [1.321e-4, 1.309e-4, 1e-10],
        [1.161e-7, 1.168e-7, 1e-13],
    );
}

#[test]
fn tier3_simulation_gj_order4() {
    // GJ order 4, dt=1s.
    // Observed: pos [3.676e-5, 3.714e-5, 0] m, vel [3.283e-8, 3.270e-8, 0] m/s.
    run_gj_test(
        "tier3_simulation_gj_order4",
        "integ_gj_order4",
        GaussJacksonConfig::with_order(4),
        1.0,
        1.0,
        [3.860e-5, 3.900e-5, 1e-10],
        [3.447e-8, 3.434e-8, 1e-13],
    );
}

#[test]
fn tier3_simulation_gj_order12() {
    // GJ order 12, dt=1s.
    // Observed: pos [1.851e-4, 1.847e-4, 0] m, vel [1.643e-7, 1.645e-7, 0] m/s.
    run_gj_test(
        "tier3_simulation_gj_order12",
        "integ_gj_order12",
        GaussJacksonConfig::with_order(12),
        1.0,
        1.0,
        [1.943e-4, 1.939e-4, 1e-10],
        [1.725e-7, 1.728e-7, 1e-13],
    );
}

#[test]
fn tier3_simulation_gj_dt10() {
    // GJ order 8, dt=10s. Coarser timestep → larger truncation error.
    // JEOD SIM_GJ_test uses scale_factor=10 → CSV times are sim-seconds.
    // Observed: pos [9.862e-1, 9.846e-1, 0] m, vel [8.751e-4, 8.755e-4, 0] m/s.
    run_gj_test(
        "tier3_simulation_gj_dt10",
        "integ_gj_dt10",
        GaussJacksonConfig::with_order(8),
        10.0,
        10.0,
        [1.036e0, 1.034e0, 1e-10],
        [9.189e-4, 9.193e-4, 1e-13],
    );
}
