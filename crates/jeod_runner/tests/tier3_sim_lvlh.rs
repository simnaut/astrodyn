//! Tier 3: SIM_LVLH cross-validation (derived_state/verif/SIM_LVLH)
//!
//! Point-mass gravity, 400 km circular LEO (i=45 deg), 24h.
//! The Simulation integrates and computes LVLH frame each step.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_runner::{GravitySourceEntry, RotationModel, SimBody, Simulation};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, SimulationTime,
    TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};

fn load_mu_earth() -> f64 {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );
    jeod_sim::coefficients::load_mu_from_jeod_cc(
        &jeod_root.join("models/environment/gravity/data/src/earth_GGM05C.cc"),
    )
    .expect("load Earth mu from GGM05C")
}

#[test]
fn tier3_simulation_lvlh() {
    let mu_earth = load_mu_earth();
    let csv_path = test_data_path("lvlh_inc_lvlh.csv");
    assert!(
        csv_path.exists(),
        "SIM_LVLH RUN_inc CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_lvlh_csv(&csv_path);
    assert!(records.len() > 100);
    let init = &records[0];

    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );
    let dt = jeod_test_data::s_define::load_dynamics_dt(
        &jeod_root.join("models/dynamics/derived_state/verif/SIM_LVLH/S_define"),
    );

    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

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

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        compute_lvlh: true,
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): SIM_LVLH derived state, {} points",
        records.len()
    );

    let mut our_states = Vec::with_capacity(records.len() - 1);
    let mut ref_states = Vec::with_capacity(records.len() - 1);
    let mut max_mat_err = 0.0_f64;
    let mut max_angvel_err = 0.0_f64;

    for record in &records[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);

        let lvlh = body.lvlh_frame.as_ref().unwrap_or_else(|| {
            panic!("Simulation did not compute LVLH frame at t={}", record.time)
        });

        let mat_err = max_mat_diff(&lvlh.t_parent_this, &record.t_parent_this);
        let angvel_err = (lvlh.ang_vel_this.length() - record.ang_vel_mag).abs();

        max_mat_err = max_mat_err.max(mat_err);
        max_angvel_err = max_angvel_err.max(angvel_err);

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

        if (record.time % 3600.0).abs() < 6.1 {
            let pos_err = (body.trans.position - record.position).length();
            println!(
                "  t={:6.0}s: pos_err={:.4} m  mat_err={:.3e}  angvel_err={:.3e}",
                record.time, pos_err, mat_err, angvel_err
            );
        }
    }

    let max_pos_err = our_states
        .iter()
        .zip(ref_states.iter())
        .map(|(a, b)| (a.position.unwrap() - b.position.unwrap()).length())
        .fold(0.0_f64, f64::max);

    println!("  Max position error:  {:.6e} m", max_pos_err);
    println!("  Max T_parent_this:   {:.6e}", max_mat_err);
    println!("  Max ang_vel error:   {:.6e} rad/s", max_angvel_err);

    let mut report = CrossvalReport::compute("tier3_simulation_lvlh", &our_states, &ref_states);
    report.add_extra("t_parent_this", max_mat_err, "");
    assert!(max_mat_err < 1.42e-11, "t_parent_this");
    report.add_extra("ang_vel", max_angvel_err, "rad/s");
    assert!(max_angvel_err < 3.68e-16, "ang_vel");
    report.write();

    report.assert_position([6.96e-5, 9.448e-5, 6.874e-5]);
}
