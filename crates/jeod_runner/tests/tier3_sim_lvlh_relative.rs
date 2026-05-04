//! Tier 3: SIM_LvlhRelative — LVLH-relative state via Simulation pipeline.
//!
//! Validates `compute_lvlh_relative_state()` against JEOD SIM_LvlhRelative.
//! Two bodies are propagated force-free through `Simulation::step()`, and
//! LVLH-relative state is computed at each checkpoint.

use jeod_test_data::tier3_csv::test_data_path;

use glam::DVec3;
use jeod_runner::Simulation;
use jeod_sim::VehicleConfig;
use jeod_sim::{compute_lvlh_relative_state, SimulationTime, TranslationalState};

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

    let init = &records[0];

    // Create Simulation with 2 bodies, no gravity (kinematic/force-free 3-DOF)
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let dt = if records.len() > 1 {
        records[1].time - records[0].time
    } else {
        1.0
    };
    let mut sim = Simulation::new(time, dt);

    // Body 0: reference vehicle
    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.ref_pos,
            velocity: init.ref_vel,
        },
        ..Default::default()
    });

    // Body 1: subject vehicle
    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.subj_pos,
            velocity: init.subj_vel,
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;

    // Check t=0
    {
        let ref_body = sim.body(0);
        let subj_body = sim.body(1);
        let lvlh_rel = compute_lvlh_relative_state(
            ref_body.trans.position,
            ref_body.trans.velocity,
            subj_body.trans.position,
            subj_body.trans.velocity,
        );
        let pos_err = (lvlh_rel.position.raw_si() - init.jeod_rel_pos).length();
        let vel_err = (lvlh_rel.velocity.raw_si() - init.jeod_rel_vel).length();
        max_pos_err = max_pos_err.max(pos_err);
        max_vel_err = max_vel_err.max(vel_err);
    }

    // Step through remaining records
    for rec in &records[1..] {
        sim.step_until(rec.time).expect("step_until failed");

        let ref_body = sim.body(0);
        let subj_body = sim.body(1);
        let lvlh_rel = compute_lvlh_relative_state(
            ref_body.trans.position,
            ref_body.trans.velocity,
            subj_body.trans.position,
            subj_body.trans.velocity,
        );

        let pos_err = (lvlh_rel.position.raw_si() - rec.jeod_rel_pos).length();
        let vel_err = (lvlh_rel.velocity.raw_si() - rec.jeod_rel_vel).length();
        max_pos_err = max_pos_err.max(pos_err);
        max_vel_err = max_vel_err.max(vel_err);
    }

    println!(
        "  {label}: {n} points, max pos err = {max_pos_err:.4e} m, max vel err = {max_vel_err:.4e} m/s",
        n = records.len()
    );

    // Constant-velocity propagation is exact with RK4 (linear dynamics).
    // LVLH frame computation involves cross products and normalization, so
    // floating-point drift grows with propagation time.
    assert!(
        max_pos_err < 1e-9,
        "{label}: max LVLH position error {max_pos_err:.4e} m exceeds 1e-9 m"
    );
    assert!(
        max_vel_err < 1e-9,
        "{label}: max LVLH velocity error {max_vel_err:.4e} m/s exceeds 1e-9 m/s"
    );
}

// non-recipe: SIM_LvlhRelative seeds two bodies from JEOD CSV t=0 records;
// no `recipes::*` preset matches the bespoke 19-column schema. LVLH error
// metric is `(a - b).length()`, too small to abstract.
#[test]
fn tier3_simulation_lvlhrel_test0() {
    run_lvlhrel_scenario("lvlhrel_test0", "lvlhrel_test0_lvlhrel.csv");
}

// non-recipe: same shape as `tier3_simulation_lvlhrel_test0`.
#[test]
fn tier3_simulation_lvlhrel_test1() {
    run_lvlhrel_scenario("lvlhrel_test1", "lvlhrel_test1_lvlhrel.csv");
}
