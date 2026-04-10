//! Tier 3: SIM_Relative — relative state between two vehicles.
//!
//! Validates `compute_relative_state()` against JEOD SIM_Relative reference.
//! The sim is purely kinematic (no gravity) — vehicles translate and rotate
//! freely, and relative state is computed from their individual states.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_math::JeodQuat;
use jeod_sim::{compute_relative_state, RotationalState, TranslationalState};

#[allow(dead_code)]
struct RelativeRecord {
    time: f64,
    veh_a_pos: DVec3,
    veh_a_vel: DVec3,
    veh_a_quat: JeodQuat,
    veh_a_ang_vel: DVec3,
    veh_b_pos: DVec3,
    veh_b_vel: DVec3,
    veh_b_quat: JeodQuat,
    veh_b_ang_vel: DVec3,
    jeod_rel_pos: DVec3,
    jeod_rel_vel: DVec3,
}

fn load_relative_csv(path: &std::path::Path) -> Vec<RelativeRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_Relative CSV from {}: {e}\n\
             Generate with Docker (see CLAUDE.md).",
            path.display()
        )
    });
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 57 {
            continue;
        }
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        // Columns (57 total, 0-indexed):
        // 0: time
        // 1-6: vehA pos/vel interleaved (pos[0],vel[0],pos[1],vel[1],pos[2],vel[2])
        // 7-22: vehA quaternion (scalar, vector[0-2]) × 4 duplicates
        // 23-25: vehA ang_vel[0-2]
        // 26-31: vehB pos/vel interleaved
        // 32-47: vehB quaternion × 4 duplicates
        // 48-50: vehB ang_vel[0-2]
        // 51-53: rel position[0-2] (grouped)
        // 54-56: rel velocity[0-2] (grouped)
        records.push(RelativeRecord {
            time: p(0),
            veh_a_pos: DVec3::new(p(1), p(3), p(5)),
            veh_a_vel: DVec3::new(p(2), p(4), p(6)),
            veh_a_quat: JeodQuat::new(p(7), p(8), p(9), p(10)),
            veh_a_ang_vel: DVec3::new(p(23), p(24), p(25)),
            veh_b_pos: DVec3::new(p(26), p(28), p(30)),
            veh_b_vel: DVec3::new(p(27), p(29), p(31)),
            veh_b_quat: JeodQuat::new(p(32), p(33), p(34), p(35)),
            veh_b_ang_vel: DVec3::new(p(48), p(49), p(50)),
            jeod_rel_pos: DVec3::new(p(51), p(52), p(53)),
            jeod_rel_vel: DVec3::new(p(54), p(55), p(56)),
        });
    }
    records
}

fn run_relative_scenario(label: &str, csv_name: &str) {
    let csv_path = test_data_path(csv_name);
    let records = load_relative_csv(&csv_path);
    assert!(!records.is_empty(), "{label}: no reference data");

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;

    for rec in &records {
        let ref_trans = TranslationalState {
            position: rec.veh_b_pos,
            velocity: rec.veh_b_vel,
        };
        let ref_rot = RotationalState {
            quaternion: rec.veh_b_quat,
            ang_vel_body: rec.veh_b_ang_vel,
        };
        let subj_trans = TranslationalState {
            position: rec.veh_a_pos,
            velocity: rec.veh_a_vel,
        };
        let subj_rot = RotationalState {
            quaternion: rec.veh_a_quat,
            ang_vel_body: rec.veh_a_ang_vel,
        };

        let rel = compute_relative_state(&ref_trans, Some(&ref_rot), &subj_trans, Some(&subj_rot));

        // JEOD's relative state is "A w.r.t. B in B" — position of A relative to B
        let pos_err = (rel.position - rec.jeod_rel_pos).length();
        let vel_err = (rel.velocity - rec.jeod_rel_vel).length();

        max_pos_err = max_pos_err.max(pos_err);
        max_vel_err = max_vel_err.max(vel_err);
    }

    println!(
        "  {label}: {n} points, max pos err = {max_pos_err:.4e} m, max vel err = {max_vel_err:.4e} m/s",
        n = records.len()
    );

    // Position/velocity expressed in B's body frame, matching JEOD convention.
    // Observed max: 8.04e-14 m position, 7.14e-15 m/s velocity.
    assert!(
        max_pos_err < 8.5e-14,
        "{label}: max position error {max_pos_err:.4e} m exceeds 8.5e-14 m"
    );
    assert!(
        max_vel_err < 7.5e-15,
        "{label}: max velocity error {max_vel_err:.4e} m/s exceeds 7.5e-15 m/s"
    );
}

#[test]
fn tier3_simulation_relative_ab_rot_ab_trans() {
    run_relative_scenario(
        "relative_ab_rot_ab_trans",
        "relative_ab_rot_ab_trans_relative.csv",
    );
}

#[test]
fn tier3_simulation_relative_no_rot_ab_trans() {
    run_relative_scenario(
        "relative_no_rot_ab_trans",
        "relative_no_rot_ab_trans_relative.csv",
    );
}

#[test]
fn tier3_simulation_relative_a_rot_no_trans() {
    run_relative_scenario(
        "relative_a_rot_no_trans",
        "relative_a_rot_no_trans_relative.csv",
    );
}
