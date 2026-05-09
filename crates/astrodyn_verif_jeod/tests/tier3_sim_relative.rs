// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Tier 3: SIM_Relative — relative state between two vehicles via Simulation pipeline.
//!
//! Validates `compute_relative_state()` against JEOD SIM_Relative reference.
//! The sim is purely kinematic (no gravity) — two bodies are propagated force-free
//! through `Simulation::step()`, and relative state is computed at each checkpoint.

use astrodyn_verif_jeod::tier3_csv::test_data_path;

use astrodyn::JeodQuat;
use astrodyn::VehicleConfig;
use astrodyn::{
    compute_relative_state, MassProperties, RotationalState, SelfRef, SimulationTime,
    TranslationalState,
};
use astrodyn_runner::Simulation;
use glam::DVec3;

/// SIM_Relative CSV record. Mirrors the full column layout; not every
/// field is consumed by every assertion.
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
        assert!(
            f.len() >= 57,
            "line {}: expected >=57 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
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

    let init = &records[0];

    // Create Simulation with 2 bodies, no gravity sources (kinematic/force-free)
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let dt = if records.len() > 1 {
        records[1].time - records[0].time
    } else {
        1.0
    };
    let mut sim = Simulation::new(time, dt);

    // Dummy mass — required by validation for rotational dynamics
    let dummy_mass = MassProperties::new(1.0);

    // Body 0: vehicle A (subject)
    sim.add_body(VehicleConfig {
        trans: astrodyn_verif_jeod::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: init.veh_a_pos,
            velocity: init.veh_a_vel,
        }),
        rot: Some(astrodyn_verif_jeod::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: init.veh_a_quat,
                ang_vel_body: init.veh_a_ang_vel,
            }),
        )),
        mass: Some(astrodyn_verif_jeod::typed_bridge::mass_raw_to_self_ref(
            &(dummy_mass),
        )),
        ..Default::default()
    });

    // Body 1: vehicle B (reference)
    sim.add_body(VehicleConfig {
        trans: astrodyn_verif_jeod::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: init.veh_b_pos,
            velocity: init.veh_b_vel,
        }),
        rot: Some(astrodyn_verif_jeod::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: init.veh_b_quat,
                ang_vel_body: init.veh_b_ang_vel,
            }),
        )),
        mass: Some(astrodyn_verif_jeod::typed_bridge::mass_raw_to_self_ref(
            &(dummy_mass),
        )),
        ..Default::default()
    });

    sim.validate().unwrap();

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;

    // Check t=0 (before any steps)
    {
        let a = sim.body(0);
        let b = sim.body(1);
        let a_trans = astrodyn::TranslationalState {
            position: a.trans.position.raw_si(),
            velocity: a.trans.velocity.raw_si(),
        };
        let b_trans = astrodyn::TranslationalState {
            position: b.trans.position.raw_si(),
            velocity: b.trans.velocity.raw_si(),
        };
        let a_rot = a
            .rot
            .as_ref()
            .map(astrodyn_verif_jeod::typed_bridge::rot_typed_to_raw);
        let b_rot = b
            .rot
            .as_ref()
            .map(astrodyn_verif_jeod::typed_bridge::rot_typed_to_raw);
        let rel = compute_relative_state::<SelfRef, SelfRef>(
            &b_trans,
            b_rot.as_ref(),
            &a_trans,
            a_rot.as_ref(),
        );
        // Scenario fixtures cover both branches of `RelativeTranslation`
        // (with/without reference rotational state), so the metric
        // reads through `position_raw`/`velocity_raw` to stay
        // branch-agnostic — the JEOD reference vector is in whichever
        // frame the producer landed in for that scenario.
        let pos_err = (rel.trans.position_raw() - init.jeod_rel_pos).length();
        let vel_err = (rel.trans.velocity_raw() - init.jeod_rel_vel).length();
        max_pos_err = max_pos_err.max(pos_err);
        max_vel_err = max_vel_err.max(vel_err);
    }

    // Step through remaining records
    for rec in &records[1..] {
        sim.step_until(rec.time).expect("step_until failed");

        let a = sim.body(0);
        let b = sim.body(1);
        let a_trans = astrodyn::TranslationalState {
            position: a.trans.position.raw_si(),
            velocity: a.trans.velocity.raw_si(),
        };
        let b_trans = astrodyn::TranslationalState {
            position: b.trans.position.raw_si(),
            velocity: b.trans.velocity.raw_si(),
        };
        let a_rot = a
            .rot
            .as_ref()
            .map(astrodyn_verif_jeod::typed_bridge::rot_typed_to_raw);
        let b_rot = b
            .rot
            .as_ref()
            .map(astrodyn_verif_jeod::typed_bridge::rot_typed_to_raw);

        let rel = compute_relative_state::<SelfRef, SelfRef>(
            &b_trans,
            b_rot.as_ref(),
            &a_trans,
            a_rot.as_ref(),
        );

        let pos_err = (rel.trans.position_raw() - rec.jeod_rel_pos).length();
        let vel_err = (rel.trans.velocity_raw() - rec.jeod_rel_vel).length();

        max_pos_err = max_pos_err.max(pos_err);
        max_vel_err = max_vel_err.max(vel_err);
    }

    println!(
        "  {label}: {n} points, max pos err = {max_pos_err:.4e} m, max vel err = {max_vel_err:.4e} m/s",
        n = records.len()
    );

    // Kinematic propagation with RK4: constant velocity is exact, but constant
    // angular velocity produces nonlinear quaternion dynamics (qdot = 0.5*q*omega),
    // so RK4 has small truncation error that accumulates. Tolerance reflects
    // the propagation-induced quaternion drift affecting body-frame relative state.
    assert!(
        max_pos_err < 3.8e-5,
        "{label}: max position error {max_pos_err:.4e} m exceeds 3.8e-5 m"
    );
    assert!(
        max_vel_err < 3.0e-6,
        "{label}: max velocity error {max_vel_err:.4e} m/s exceeds 3.0e-6 m/s"
    );
}

// non-recipe: SIM_Relative seeds two bodies (rot+trans) from a 57-column
// JEOD CSV with no equivalent recipe preset. Error metric is `(a-b).length()`,
// too small to abstract.
#[test]
fn tier3_simulation_relative_ab_rot_ab_trans() {
    run_relative_scenario(
        "relative_ab_rot_ab_trans",
        "relative_ab_rot_ab_trans_relative.csv",
    );
}

// non-recipe: same shape as `tier3_simulation_relative_ab_rot_ab_trans`.
#[test]
fn tier3_simulation_relative_no_rot_ab_trans() {
    run_relative_scenario(
        "relative_no_rot_ab_trans",
        "relative_no_rot_ab_trans_relative.csv",
    );
}

// non-recipe: same shape as `tier3_simulation_relative_ab_rot_ab_trans`.
#[test]
fn tier3_simulation_relative_a_rot_no_trans() {
    run_relative_scenario(
        "relative_a_rot_no_trans",
        "relative_a_rot_no_trans_relative.csv",
    );
}
