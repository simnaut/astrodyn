//! Tier 3: SIM_GJ_test — Gauss-Jackson (ABM8) vs JEOD's GJ integrator.
//!
//! Cross-validates our Adams-Bashforth-Moulton order 8 integrator against
//! JEOD's Gauss-Jackson (Störmer-Cowell summed-form) on a circular orbit.
//!
//! Scenario: r₀=[9e6, 0, 0]m, v₀=[0, 8000, 0]m/s, μ=5.76e14, spherical
//! gravity, dt=1.0s, 300,000s (~83h), logged every 300s (1000 points).
//! Translational dynamics only.
//!
//! Note: ABM and true GJ are mathematically different 8th-order methods with
//! different error constants. Tolerances are looser than RK4-vs-RK4 tests.

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
/// Dynamics interval matching SIM_GJ_test S_define (#define DYNAMICS 1.00).
const DT_GJ: f64 = 1.0;

#[test]
fn tier3_simulation_gj_order8() {
    let csv_path = test_data_path("integ_gj_gj.csv");
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
    let mut sim = Simulation::new(time, DT_GJ);

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
        integrator: IntegratorType::GaussJackson(GaussJacksonConfig::with_order(8)),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): SIM_GJ_test GJ order 8, {} points over {:.0}s",
        trajectory.len(),
        trajectory.last().unwrap().time
    );

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    for record in &trajectory[1..] {
        sim.step_until(record.time);
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

    let report = CrossvalReport::compute("tier3_simulation_gj_order8", &our_states, &ref_states);
    report.write();

    let max_pos = report.max_position_component();
    let max_vel = report.max_velocity_component();
    println!("  Max position error: {max_pos:.6e} m");
    println!("  Max velocity error: {max_vel:.6e} m/s");

    // ABM8 vs JEOD GJ8: different formulations, same order.
    // Observed: pos [2.227e-4, 2.215e-4, 0] m, vel [1.967e-7, 1.979e-7, 0] m/s.
    // Issue #33 exit criterion: < 1 m over 24h — achieved by 4 orders of magnitude.
    report.assert_position([2.338e-4, 2.326e-4, 1e-10]);
    report.assert_velocity([2.066e-7, 2.078e-7, 1e-13]);
}
