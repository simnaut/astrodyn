//! Tier 3: SIM_Planetary — derived state trajectory in 5 orbit regimes.
//!
//! Validates Simulation trajectory against JEOD SIM_Planetary reference CSVs
//! across LEO inclined, LEO polar, LEO eccentric, LEO equatorial, and GEO orbits.
//! These exercise coordinate singularities (equatorial RAAN, polar LVLH).
//!
//! Note: The CSV only contains position/velocity (orbital element variables
//! were not registered in the SIM_Planetary S_define). Trajectory validation
//! is the primary goal; orbital element computation is validated separately
//! in tier3_sim_orbelem_comprehensive.rs.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, GravitySourceEntry, SimBody,
    Simulation, SimulationTime, TranslationalState,
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

/// Load a planetary state CSV (7 columns: time, pos[3], vel[3]).
fn load_planetary_csv(path: &std::path::Path) -> Vec<StateLog> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_Planetary CSV from {}: {e}\n\
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
            f.len() >= 7,
            "line {}: expected >=7 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(StateLog {
            time: p(0),
            position: Some(DVec3::new(p(1), p(2), p(3))),
            velocity: Some(DVec3::new(p(4), p(5), p(6))),
            ..Default::default()
        });
    }
    records
}

/// Run a SIM_Planetary scenario: point-mass gravity, compare trajectory.
fn run_planetary_scenario(label: &str, csv_name: &str) {
    let mu_earth = load_mu_earth();
    let csv_path = test_data_path(csv_name);
    let ref_states = load_planetary_csv(&csv_path);
    assert!(!ref_states.is_empty(), "{label}: no reference data");

    // Initialize from first reference state
    let init = &ref_states[0];
    let init_pos = init.position.unwrap();
    let init_vel = init.velocity.unwrap();

    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );
    let dt = jeod_test_data::s_define::load_dynamics_dt(
        &jeod_root.join("models/dynamics/derived_state/verif/SIM_Planetary/S_define"),
    );

    let leap_table = jeod_sim::default_leap_second_table();
    let time = SimulationTime::at_j2000(leap_table);
    let mut sim = Simulation::new(time, dt);

    let earth = sim.add_source(GravitySourceEntry::new(
        GravitySource {
            mu: mu_earth,
            model: GravityModel::PointMass,
        },
        DVec3::ZERO,
        None,
    ));

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init_pos,
            velocity: init_vel,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    // Step and collect our states
    let mut our_states = vec![StateLog {
        time: 0.0,
        position: Some(init_pos),
        velocity: Some(init_vel),
        ..Default::default()
    }];

    for record in &ref_states[1..] {
        sim.step_until(record.time);
        let body = sim.body(0);
        our_states.push(StateLog {
            time: record.time,
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            ..Default::default()
        });
    }

    let report = CrossvalReport::compute(
        &format!("tier3_planetary_{label}"),
        &our_states,
        &ref_states[..our_states.len()],
    );
    report.write();

    // SIM_Planetary uses point-mass gravity at J2000 epoch. Our Simulation
    // should match very closely since both use the same force model.
    report.assert_position([1.0, 1.0, 1.0]); // < 1 m per component
    report.assert_velocity([0.001, 0.001, 0.001]); // < 1 mm/s per component
}

#[test]
fn tier3_simulation_planetary_leo_inc() {
    run_planetary_scenario("leo_inc", "planetary_leo_inc_planetary.csv");
}

#[test]
fn tier3_simulation_planetary_leo_polar() {
    run_planetary_scenario("leo_polar", "planetary_leo_polar_planetary.csv");
}

#[test]
fn tier3_simulation_planetary_leo_ecc() {
    run_planetary_scenario("leo_ecc", "planetary_leo_ecc_planetary.csv");
}

#[test]
fn tier3_simulation_planetary_leo_equ() {
    run_planetary_scenario("leo_equ", "planetary_leo_equ_planetary.csv");
}

#[test]
fn tier3_simulation_planetary_geo() {
    run_planetary_scenario("geo", "planetary_geo_planetary.csv");
}
