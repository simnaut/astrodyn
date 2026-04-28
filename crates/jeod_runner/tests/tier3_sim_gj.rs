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

use jeod_test_data::tier3_csv::{load_gj_csv, test_data_path};

use glam::DVec3;
use jeod_runner::{GravitySourceEntry, RotationModel, Simulation, VehicleConfig};
use jeod_sim::{
    GaussJacksonConfig, GravityControl, GravityControls, GravityModel, GravitySource,
    IntegratorType, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};

/// Non-standard μ used by SIM_GJ_test (set in input_common.py).
const MU_GJ_TEST: f64 = 5.76e14;

/// Run a GJ cross-validation test with the given config and reference CSV.
///
/// JEOD's SIM_GJ_test uses `dyn_time.scale_factor` to vary the effective dt.
/// We match this by setting `SimulationTime::time_scale_factor` and stepping
/// the simulation at `sim_dt` (the sim clock rate), while the integrator
/// internally computes `cycle_dyndt = sim_dt * cycle_scale * time_scale_factor`.
///
/// CSV times are in sim-seconds. For dt=1 runs, `time_scale_factor=1.0` and
/// `sim_dt=1.0`. For dt=10 runs, `time_scale_factor=10.0` and `sim_dt=1.0`
/// (each sim step advances 10s of dynamic time).
fn run_gj_test(
    test_name: &str,
    csv_label: &str,
    config: GaussJacksonConfig,
    sim_dt: f64,
    time_scale_factor: f64,
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

    let mut time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    time.time_scale_factor = time_scale_factor;
    let mut sim = Simulation::new(time, sim_dt);

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: MU_GJ_TEST,
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

    sim.add_body(VehicleConfig {
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

    // CSV times are in sim-seconds; step_until uses simtime.
    let final_sim_time = trajectory.last().unwrap().time;
    let effective_dt = sim_dt * time_scale_factor;
    println!(
        "Tier 3 (Simulation): {test_name}, {} points over {:.0}s sim ({:.0}s dyn), \
         sim_dt={sim_dt}s, tsf={time_scale_factor}, effective_dt={effective_dt}s",
        trajectory.len(),
        final_sim_time,
        final_sim_time * time_scale_factor,
    );

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    for record in &trajectory[1..] {
        sim.step_until(record.time).expect("step_until failed");
        let body = sim.body(0);
        our_states.push(StateLog {
            time: record.time,
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            ..Default::default()
        });
    }

    let ref_states: Vec<StateLog> = trajectory[1..]
        .iter()
        .map(|r| StateLog {
            time: r.time,
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

// non-recipe: SIM_GJ_test uses an artificial μ (5.76e14) and CSV t=0 initial
// conditions; no `recipes::*` building block matches. Helper math for
// integrator agreement / state error is owned by `CrossvalReport`.
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

// non-recipe: same artificial μ as `tier3_simulation_gj_order8`.
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

// non-recipe: same artificial μ as `tier3_simulation_gj_order8`.
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

// non-recipe: same artificial μ as `tier3_simulation_gj_order8`; effective
// dt scaled via `time_scale_factor`.
#[test]
fn tier3_simulation_gj_dt10() {
    // GJ order 8, effective dt=10s. Coarser timestep → larger truncation error.
    // Matches JEOD SIM_GJ_test: sim_dt=1.0, dyn_time.scale_factor=10.
    // The integrator sees cycle_dyndt = 1.0 * cycle_scale * 10.0 = 10.0 * cycle_scale,
    // producing the same physics as dt=10 with tsf=1.0.
    // CSV times are in sim-seconds (log_cycle=30s, terminate=30000s).
    // Observed: pos [9.862e-1, 9.846e-1, 0] m, vel [8.751e-4, 8.755e-4, 0] m/s.
    run_gj_test(
        "tier3_simulation_gj_dt10",
        "integ_gj_dt10",
        GaussJacksonConfig::with_order(8),
        1.0,
        10.0,
        [1.036e0, 1.034e0, 1e-10],
        [9.189e-4, 9.193e-4, 1e-13],
    );
}
