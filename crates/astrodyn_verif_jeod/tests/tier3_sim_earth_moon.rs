//! Tier 3: SIM_Earth_Moon — Clementine lunar orbit cross-validation.
//!
//! Validates multi-body gravity (Earth + Moon LP150Q 60×60 spherical harmonics,
//! Sun 3rd-body, DE421 BPC libration, cannonball SRP) against the JEOD
//! reference trajectory. Clementine-like orbit, 7 days (604,800 s).
//!
//! Matches JEOD SIM_Earth_Moon RUN_clem configuration:
//! - Integrator: RK4 at 0.03125 s (32 Hz)
//! - Moon gravity: LP150Q 60×60
//! - Moon rotation: DE421 BPC libration (per-step update)
//! - Earth/Sun: point-mass 3rd-body with per-step DE421 ephemeris (JEOD uses DE405)
//! - SRP: cannonball (cx_area=2.1432 m², albedo=1.0, diffuse=0.27)
//! - No drag, no gravity torque
//!
//! The scenario is constructed via
//! [`astrodyn_verif_jeod::setups::earth_moon_clem::earth_moon_clem`] —
//! the same canonical builder consumed by `examples/earth_moon.rs` and
//! the `tier3_perf_runner` binary (issue #447).

use astrodyn_runner::SimulationBuilderExt;
use astrodyn_verif_jeod::crossval::{CrossvalReport, StateLog};
use astrodyn_verif_jeod::setups::earth_moon_clem::earth_moon_clem;
use astrodyn_verif_jeod::setups::earth_moon_rosetta::earth_moon_rosetta;
use astrodyn_verif_jeod::tier3_csv::test_data_path;
use glam::DVec3;

/// Load a state CSV with interleaved columns: time, pos[0], vel[0], pos[1], vel[1], pos[2], vel[2].
fn load_interleaved_csv(path: &std::path::Path, sim_name: &str) -> Vec<StateLog> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read {sim_name} CSV from {}: {e}\n\
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
            position: Some(DVec3::new(p(1), p(3), p(5))),
            velocity: Some(DVec3::new(p(2), p(4), p(6))),
            ..Default::default()
        });
    }
    records
}

/// Clementine lunar orbit: Moon LP150Q 60×60 + Earth 3rd-body + Sun 3rd-body
/// + cannonball SRP, matching JEOD SIM_Earth_Moon RUN_clem.
#[test]
fn tier3_simulation_earth_moon_clem() {
    let csv_path = test_data_path("earth_moon_clem_earth_moon.csv");
    let ref_states = load_interleaved_csv(&csv_path, "SIM_Earth_Moon RUN_clem");
    assert!(
        !ref_states.is_empty(),
        "No reference data for SIM_Earth_Moon RUN_clem"
    );

    // Use JEOD's initial state from the CSV's t=0 row so any future JEOD
    // regen stays the single source of truth.
    let init = &ref_states[0];
    let init_pos = init.position.unwrap();
    let init_vel = init.velocity.unwrap();

    // Build the canonical Earth–Moon Clementine scenario: 32 Hz RK4,
    // Moon LP150Q 60×60 + DE421 BPC libration, Earth/Sun third-body with
    // per-step DE421 ephemeris updates, cannonball SRP. See
    // `astrodyn_verif_jeod::setups::earth_moon_clem` for the full wiring.
    let mut sim = earth_moon_clem(0.03125, Some((init_pos, init_vel)))
        .build()
        .expect("earth_moon_clem scenario must validate");

    let mut our_states = vec![StateLog {
        time: 0.0,
        position: Some(init_pos),
        velocity: Some(init_vel),
        ..Default::default()
    }];

    for (i, record) in ref_states[1..].iter().enumerate() {
        sim.step_until(record.time).expect("step_until failed");
        let body = sim.body(0);
        if i == 0 {
            let jeod_pos = record.position.unwrap();
            let our_pos = body.trans.position.raw_si();
            println!(
                "  t={}: ours=[{:.1}, {:.1}, {:.1}]",
                record.time, our_pos.x, our_pos.y, our_pos.z
            );
            println!(
                "  t={}: JEOD=[{:.1}, {:.1}, {:.1}]",
                record.time, jeod_pos.x, jeod_pos.y, jeod_pos.z
            );
            let err = (body.trans.position.raw_si() - jeod_pos).length();
            println!("  t={}: error={:.1} m", record.time, err);
        }
        our_states.push(StateLog {
            time: record.time,
            position: Some(body.trans.position.raw_si()),
            velocity: Some(body.trans.velocity.raw_si()),
            acceleration: Some(body.trans_accel.raw_si()),
            ang_accel: body.rot_accel.map(|a| a.raw_si()),
            ..Default::default()
        });
    }

    let report = CrossvalReport::compute(
        "tier3_earth_moon_clem",
        &our_states,
        &ref_states[..our_states.len()],
    );
    report.write();

    let max_pos = report.max_position_component();
    println!(
        "  Earth-Moon Clem: max position error = {:.2} m \
         (LP150Q 60x60 + DE421 BPC + cannonball SRP, dt=0.03125s, 7 days)",
        max_pos
    );
    // Residual from DE405/DE421 difference (JEOD uses DE405, we use DE421).
    // Tolerance: observed max × 1.05.
    report.assert_position([0.832, 0.331, 0.972]);
}

/// Rosetta Earth swing-by: Earth point-mass + J2 (degree 2/order 0) +
/// Moon & Sun point-mass third bodies + cannonball SRP, matching JEOD
/// SIM_Earth_Moon RUN_rosetta. Hyperbolic Earth flyby, 15000 s (4.17 hr),
/// Earth-centered inertial. `BCH.02` mission benchmark.
#[test]
fn tier3_simulation_earth_moon_rosetta() {
    let csv_path = test_data_path("earth_moon_rosetta_earth_moon.csv");
    let ref_states = load_interleaved_csv(&csv_path, "SIM_Earth_Moon RUN_rosetta");
    assert!(
        !ref_states.is_empty(),
        "No reference data for SIM_Earth_Moon RUN_rosetta"
    );

    // JEOD t=0 state from the CSV's first row (single source of truth).
    let init = &ref_states[0];
    let init_pos = init.position.unwrap();
    let init_vel = init.velocity.unwrap();

    // 32 Hz RK4, matching the SIM_Earth_Moon S_define dynamics rate.
    let mut sim = earth_moon_rosetta(0.03125, Some((init_pos, init_vel)))
        .build()
        .expect("earth_moon_rosetta scenario must validate");

    let mut our_states = vec![StateLog {
        time: 0.0,
        position: Some(init_pos),
        velocity: Some(init_vel),
        ..Default::default()
    }];

    for record in &ref_states[1..] {
        sim.step_until(record.time).expect("step_until failed");
        let body = sim.body(0);
        our_states.push(StateLog {
            time: record.time,
            position: Some(body.trans.position.raw_si()),
            velocity: Some(body.trans.velocity.raw_si()),
            acceleration: Some(body.trans_accel.raw_si()),
            ang_accel: body.rot_accel.map(|a| a.raw_si()),
            ..Default::default()
        });
    }

    let report = CrossvalReport::compute(
        "tier3_earth_moon_rosetta",
        &our_states,
        &ref_states[..our_states.len()],
    );
    report.write();

    let max_pos = report.max_position_component();
    println!(
        "  Earth-Moon Rosetta: max position error = {max_pos:.2} m \
         (Earth pt-mass+J2 + Moon/Sun 3rd-body + cannonball SRP, dt=0.03125s, 15000s)"
    );
    // Tolerance: observed max × 1.05 (set after first run). Residual is
    // dominated by the DE405 (JEOD) vs DE421 (ours) ephemeris difference
    // over the hyperbolic flyby, same as the Clementine case.
    report.assert_position([ROSETTA_TOL_X, ROSETTA_TOL_Y, ROSETTA_TOL_Z]);
}

// 1.05× observed max per component (CLAUDE.md policy). Observed
// [1.166, 3.577, 3.029] m over the 15000 s flyby — dominated by the
// DE405 (JEOD) vs DE421 (ours) ephemeris difference, as for Clementine.
const ROSETTA_TOL_X: f64 = 1.225;
const ROSETTA_TOL_Y: f64 = 3.756;
const ROSETTA_TOL_Z: f64 = 3.181;
