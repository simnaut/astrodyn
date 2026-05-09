//! Tier 3: SIM_7_time_reversal — time-reversed propagation cross-validation.
//!
//! JEOD propagates forward 60,000 s then sets `scale_factor = -1.0` for another
//! 60,000 sim-seconds. Validates TAI time and trajectory position/velocity
//! during both forward and reverse phases.
//!
//! RUN_1: spherical Earth gravity, RK4 at 0.03125 s.

use astrodyn_verif_jeod::tier3_csv::test_data_path;

use astrodyn::{
    GravityControl, GravityControls, GravityModel, GravitySource, MassProperties, RotationalState,
    SimulationTime, TranslationalState,
};
use astrodyn::{GravitySourceEntry, VehicleConfig};
use astrodyn_runner::Simulation;
use glam::DVec3;

fn load_mu_earth_gemt1() -> f64 {
    astrodyn::gravity_fixtures::load_gemt1().mu
}

#[allow(dead_code)] // position/velocity used by run1, not by time-only run3a/run8b
struct ReversalRecord {
    time: f64,
    position: DVec3,
    velocity: DVec3,
    tai_seconds: f64,
    tai_tjt: f64,
}

fn load_reversal_csv(path: &std::path::Path) -> Vec<ReversalRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_7_time_reversal CSV from {}: {e}\n\
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
            f.len() >= 9,
            "line {}: expected >=9 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(ReversalRecord {
            time: p(0),
            position: DVec3::new(p(1), p(3), p(5)),
            velocity: DVec3::new(p(2), p(4), p(6)),
            tai_seconds: p(7),
            tai_tjt: p(8),
        });
    }
    records
}

/// RUN_1: spherical Earth gravity, forward 60,000 s + reverse 60,000 s.
/// Validates both time and trajectory against JEOD reference.
// non-recipe: SIM_7_time_reversal seeds from a JEOD CSV t=0 record (TAI TJT,
// position, velocity, attitude derived from LVLH+pitch). The bespoke piece
// is the negative-`time_scale_factor` flip at the reversal index — verified
// here as part of the simulation pipeline.
#[test]
fn tier3_sim_time_reversal_run1() {
    let mu_earth_gemt1 = load_mu_earth_gemt1();
    let csv_path = test_data_path("reversal_run1_reversal.csv");
    let records = load_reversal_csv(&csv_path);
    assert!(records.len() > 1, "no reference data");

    let init = &records[0];

    // Epoch: 2007-11-20 00:00:00 UTC. TAI TJT from CSV.
    // Dynamics timestep: 0.03125 s (32 Hz) per
    // models/environment/time/verif/SIM_7_time_reversal/S_define `#define DYNAMICS`.
    let dt = 0.03125_f64;

    let leap_table = astrodyn::default_leap_second_table();
    let time = SimulationTime::new(init.tai_tjt, leap_table);
    let mut sim = Simulation::new(time, dt);

    let earth = sim.add_source("Earth", {
        let mut e = GravitySourceEntry::new(
            GravitySource {
                mu: mu_earth_gemt1,
                model: GravityModel::PointMass,
            },
            astrodyn::Position::<astrodyn::RootInertial>::zero(),
            None,
        );
        e.central = true;
        e
    });

    // JEOD initializes attitude in LVLH: Yaw=0, Pitch=-11.6°, Roll=0, omega=0.
    // Compute the LVLH frame, then apply the Euler rotation.
    let lvlh = astrodyn::compute_body_lvlh_frame(init.position, init.velocity);
    // T_inertial_lvlh = transpose of T_parent_this (LVLH -> inertial)
    let t_inertial_lvlh = lvlh.t_parent_this.transpose();
    // Euler YPR = [0, -11.6°, 0] → rotation about LVLH Y-axis by -11.6°
    let pitch = -11.6_f64.to_radians();
    let t_lvlh_body = glam::DMat3::from_rotation_y(pitch);
    let t_inertial_body = t_inertial_lvlh * t_lvlh_body;
    // Convert rotation matrix to JeodQuat (left-transformation convention)
    let glam_quat = glam::DQuat::from_mat3(&t_inertial_body);
    let init_quat = astrodyn::JeodQuat::new(glam_quat.w, glam_quat.x, glam_quat.y, glam_quat.z);

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        }
        .into(),
        rot: Some(
            RotationalState {
                quaternion: init_quat,
                ang_vel_body: DVec3::ZERO,
            }
            .into(),
        ),
        mass: Some(MassProperties::new(1.0).into()), // mass doesn't affect spherical gravity
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });
    sim.validate().unwrap();

    // Detect reversal point
    let reversal_idx = records
        .windows(2)
        .position(|w| w[1].tai_seconds < w[0].tai_seconds)
        .unwrap_or_else(|| panic!("no reversal point found in CSV"));

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    let mut max_tai_s_err = 0.0_f64;

    for (i, rec) in records.iter().enumerate() {
        if i == 0 {
            continue;
        }

        // Switch to reverse at the reversal point
        if i == reversal_idx + 1 && sim.time.time_scale_factor > 0.0 {
            sim.time.time_scale_factor = -1.0;
        }

        sim.step_until(rec.time).expect("step_until failed");

        let body = sim.body(0);
        let pos_err = (body.trans.position - rec.position).length();
        let vel_err = (body.trans.velocity - rec.velocity).length();
        max_pos_err = max_pos_err.max(pos_err);
        max_vel_err = max_vel_err.max(vel_err);

        let elapsed_jeod = rec.tai_seconds - init.tai_seconds;
        let tai_s_err = (sim.time.tai_seconds - elapsed_jeod).abs();
        max_tai_s_err = max_tai_s_err.max(tai_s_err);
    }

    // Round-trip position: final state should match initial
    let final_body = sim.body(0);
    let roundtrip_pos = (final_body.trans.position - init.position).length();
    let roundtrip_vel = (final_body.trans.velocity - init.velocity).length();

    println!(
        "  reversal_run1: {} points, pos={max_pos_err:.2e}m, vel={max_vel_err:.2e}m/s, \
         TAI={max_tai_s_err:.2e}s, roundtrip_pos={roundtrip_pos:.2e}m, roundtrip_vel={roundtrip_vel:.2e}m/s",
        records.len()
    );

    // Spherical gravity at 0.03125 s, matching JEOD's GEM-T1 mu.
    // Tolerance: observed max × 1.05.
    assert!(
        max_pos_err < 1.46e-5,
        "position error {max_pos_err:.4e} m exceeds 1.46e-5 m"
    );
    assert!(
        max_vel_err < 1.72e-8,
        "velocity error {max_vel_err:.4e} m/s exceeds 1.72e-8 m/s"
    );
    assert!(
        max_tai_s_err < 1e-6,
        "TAI seconds error {max_tai_s_err:.4e} s"
    );
    assert!(
        roundtrip_pos < 1e-3,
        "round-trip position {roundtrip_pos:.4e} m (should return to initial)"
    );
}

/// RUN_3A/RUN_8B: time-only validation (these runs use non-spherical gravity
/// or rotational dynamics that aren't worth duplicating for the time reversal test).
fn run_reversal_time_only(label: &str, csv_name: &str) {
    let csv_path = test_data_path(csv_name);
    let records = load_reversal_csv(&csv_path);
    assert!(records.len() > 1, "{label}: no reference data");

    let init = &records[0];
    let leap_table = astrodyn::default_leap_second_table();
    let mut sim_time = SimulationTime::new(init.tai_tjt, leap_table);

    let reversal_idx = records
        .windows(2)
        .position(|w| w[1].tai_seconds < w[0].tai_seconds)
        .unwrap_or_else(|| panic!("{label}: no reversal point found in CSV"));

    let mut max_tai_s_err = 0.0_f64;

    for (i, rec) in records.iter().enumerate() {
        if i > 0 {
            let sim_dt = rec.time - records[i - 1].time;
            if i == reversal_idx + 1 && sim_time.time_scale_factor > 0.0 {
                sim_time.time_scale_factor = -1.0;
            }
            sim_time.advance(sim_dt);
        }
        let elapsed_jeod = rec.tai_seconds - init.tai_seconds;
        let tai_s_err = (sim_time.tai_seconds - elapsed_jeod).abs();
        max_tai_s_err = max_tai_s_err.max(tai_s_err);
    }

    let final_tai_err = sim_time.tai_seconds.abs();
    println!(
        "  {label}: {} points, TAI_s={max_tai_s_err:.2e}s, round_trip={final_tai_err:.2e}s",
        records.len()
    );

    assert!(
        max_tai_s_err < 1e-6,
        "{label}: TAI error {max_tai_s_err:.4e} s"
    );
    assert!(
        final_tai_err < 1e-9,
        "{label}: round-trip {final_tai_err:.4e} s"
    );
}

// non-recipe: time-only round trip; TAI seconds are read from CSV.
#[test]
fn tier3_sim_time_reversal_run3a() {
    run_reversal_time_only("reversal_run3a", "reversal_run3a_reversal.csv");
}

// non-recipe: time-only round trip; TAI seconds are read from CSV.
#[test]
fn tier3_sim_time_reversal_run8b() {
    run_reversal_time_only("reversal_run8b", "reversal_run8b_reversal.csv");
}
