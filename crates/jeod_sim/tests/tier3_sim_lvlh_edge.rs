//! Tier 3: SIM_LVLH edge-case cross-validation
//!
//! RUN_ecc: Eccentric orbit (400 km x 8000 km) — varying orbital rate
//!          exercises LVLH frame computation at different velocities.
//! RUN_equ: Equatorial orbit (i=0) — near-singular LVLH at zero inclination.
//!
//! Point-mass Earth gravity, RK4 at DT=0.03125s, 24h.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, GravitySourceEntry, SimBody,
    Simulation, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::crossval_report;

fn run_lvlh_test(csv_filename: &str, label: &str, test_name: &str) {
    let csv_path = test_data_path(csv_filename);
    assert!(
        csv_path.exists(),
        "SIM_LVLH CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
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
        "Tier 3 (Simulation): SIM_LVLH {label}, {} points",
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

        if (record.time % 7200.0).abs() < 6.1 {
            println!(
                "  t={:6.0}s: pos_err={:.4} m  mat_err={:.3e}  angvel_err={:.3e}",
                record.time, pos_err, mat_err, angvel_err
            );
        }
    }

    println!("  Max position error:  {:.6e} m", max_pos_err);
    println!("  Max T_parent_this:   {:.6e}", max_mat_err);
    println!("  Max ang_vel error:   {:.6e} rad/s", max_angvel_err);

    crossval_report(
        test_name,
        &[
            ("position", max_pos_err, "m"),
            ("T_parent_this", max_mat_err, ""),
            ("ang_vel", max_angvel_err, "rad/s"),
        ],
    );

    assert!(
        max_pos_err < 0.5,
        "{label}: position error {max_pos_err:.2} m exceeds 0.5 m"
    );
    assert!(
        max_mat_err < 1e-6,
        "{label}: LVLH matrix error {max_mat_err:.3e} exceeds 1e-6"
    );
    assert!(
        max_angvel_err < 1e-10,
        "{label}: LVLH ang_vel error {max_angvel_err:.3e} rad/s exceeds 1e-10"
    );
}

#[test]
fn tier3_simulation_lvlh_ecc() {
    run_lvlh_test(
        "lvlh_ecc_lvlh.csv",
        "RUN_ecc (eccentric)",
        "tier3_simulation_lvlh_ecc",
    );
}

#[test]
fn tier3_simulation_lvlh_equ() {
    run_lvlh_test(
        "lvlh_equ_lvlh.csv",
        "RUN_equ (equatorial)",
        "tier3_simulation_lvlh_equ",
    );
}
