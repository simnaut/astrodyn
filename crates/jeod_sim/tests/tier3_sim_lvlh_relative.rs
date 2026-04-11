//! Tier 3: SIM_LvlhRelative — LVLH-relative state cross-validation.
//!
//! Validates `compute_lvlh_relative_state()` against JEOD SIM_LvlhRelative
//! reference data. Compares rectilinear LVLH-relative position and velocity.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_sim::compute_lvlh_relative_state;

struct LvlhRelRecord {
    time: f64,
    ref_pos: DVec3,
    ref_vel: DVec3,
    subj_pos: DVec3,
    subj_vel: DVec3,
    jeod_rel_pos: DVec3,
    jeod_rel_vel: DVec3,
}

fn load_lvlhrel_csv(path: &std::path::Path) -> Vec<LvlhRelRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_LvlhRelative CSV from {}: {e}\n\
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
        assert!(
            f.len() >= 19,
            "line {}: expected >=19 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        // Columns (0-indexed): time, vehA pos/vel (interleaved), vehB pos/vel (interleaved),
        // rel pos[0-2] (grouped), rel vel[0-2] (grouped)
        records.push(LvlhRelRecord {
            time: p(0),
            ref_pos: DVec3::new(p(1), p(3), p(5)),
            ref_vel: DVec3::new(p(2), p(4), p(6)),
            subj_pos: DVec3::new(p(7), p(9), p(11)),
            subj_vel: DVec3::new(p(8), p(10), p(12)),
            jeod_rel_pos: DVec3::new(p(13), p(14), p(15)),
            jeod_rel_vel: DVec3::new(p(16), p(17), p(18)),
        });
    }
    records
}

fn run_lvlhrel_scenario(label: &str, csv_name: &str) {
    let csv_path = test_data_path(csv_name);
    let records = load_lvlhrel_csv(&csv_path);
    assert!(!records.is_empty(), "{label}: no reference data");

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;

    for (i, rec) in records.iter().enumerate() {
        let lvlh_rel =
            compute_lvlh_relative_state(rec.ref_pos, rec.ref_vel, rec.subj_pos, rec.subj_vel);

        let pos_err = (lvlh_rel.position - rec.jeod_rel_pos).length();
        let vel_err = (lvlh_rel.velocity - rec.jeod_rel_vel).length();

        max_pos_err = max_pos_err.max(pos_err);
        max_vel_err = max_vel_err.max(vel_err);

        // LVLH-relative with Coriolis correction matches JEOD convention.
        // Observed max: 2.13e-14 m position, 4.97e-16 m/s velocity.
        assert!(
            pos_err < 2.3e-14,
            "{label} point {i} (t={:.1}): LVLH position error {pos_err:.4e} m",
            rec.time
        );
        assert!(
            vel_err < 5.3e-16,
            "{label} point {i} (t={:.1}): LVLH velocity error {vel_err:.4e} m/s",
            rec.time
        );
    }

    println!(
        "  {label}: {n} points, max pos err = {max_pos_err:.4e} m, max vel err = {max_vel_err:.4e} m/s",
        n = records.len()
    );
}

#[test]
fn tier3_simulation_lvlhrel_test0() {
    run_lvlhrel_scenario("lvlhrel_test0", "lvlhrel_test0_lvlhrel.csv");
}

#[test]
fn tier3_simulation_lvlhrel_test1() {
    run_lvlhrel_scenario("lvlhrel_test1", "lvlhrel_test1_lvlhrel.csv");
}
