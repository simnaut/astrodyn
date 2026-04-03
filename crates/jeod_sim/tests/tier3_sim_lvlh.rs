//! Tier 3: SIM_LVLH cross-validation (derived_state/verif/SIM_LVLH)
//!
//! Point-mass gravity, 400 km circular LEO (i=45 deg), 24h.
//! The Simulation integrates and computes LVLH frame each step.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, GravitySourceEntry, SimBody,
    Simulation, SimulationTime, TranslationalState,
};

#[test]
fn tier3_simulation_lvlh() {
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

    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);

    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        t_inertial_pfix: None,
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

    let mut max_mat_err = 0.0_f64;
    let mut max_angvel_err = 0.0_f64;
    let mut max_pos_err = 0.0_f64;

    for record in &records[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);
        let pos_err = (body.trans.position - record.position).length();
        max_pos_err = max_pos_err.max(pos_err);

        let lvlh = body.lvlh_frame.as_ref().unwrap_or_else(|| {
            panic!("Simulation did not compute LVLH frame at t={}", record.time)
        });

        let mat_err = max_mat_diff(&lvlh.t_parent_this, &record.t_parent_this);
        let angvel_err = (lvlh.ang_vel_this.length() - record.ang_vel_mag).abs();

        max_mat_err = max_mat_err.max(mat_err);
        max_angvel_err = max_angvel_err.max(angvel_err);

        if (record.time % 3600.0).abs() < 6.1 {
            println!(
                "  t={:6.0}s: pos_err={:.4} m  mat_err={:.3e}  angvel_err={:.3e}",
                record.time, pos_err, mat_err, angvel_err
            );
        }
    }

    println!("  Max position error:  {:.4} m", max_pos_err);
    println!("  Max T_parent_this:   {:.6e}", max_mat_err);
    println!("  Max ang_vel error:   {:.6e} rad/s", max_angvel_err);

    assert!(
        max_pos_err < 0.5,
        "Position error {max_pos_err:.2} m exceeds 0.5 m"
    );
    // LVLH frame direction error from ~0.5 m position drift at ~6800 km radius
    // -> angular error ~ 0.5/6.8e6 ~ 7e-8 rad -> matrix element error ~ 7e-8
    assert!(
        max_mat_err < 1e-6,
        "LVLH matrix error {max_mat_err:.3e} exceeds 1e-6"
    );
    assert!(
        max_angvel_err < 1e-10,
        "LVLH ang_vel error {max_angvel_err:.3e} rad/s exceeds 1e-10"
    );
}
