//! Tier 3: Apollo 8 frame switching cross-validation.
//!
//! Two sub-tests validating integration frame switching:
//!
//! 1. `tier3_apollo8_eci_integ` — Baseline: 100s in Earth-centered inertial frame
//! 2. `tier3_apollo8_frame_switch` — Same IC but switching to Moon-centered inertial
//!    when approaching within 66.1 million meters
//!
//! Reference: `models/dynamics/body_action/verif/SIM_verif_frame_switch/` in JEOD v5.4.
//!
//! Initial conditions: Apollo 8 trans-lunar coast, Dec 23, 1968, 19:38:00 UTC.

#![allow(clippy::excessive_precision)]

use glam::{DMat3, DQuat, DVec3};
use jeod_math::JeodQuat;
use jeod_runner::{FrameSwitchConfig, GravitySourceEntry, Simulation, SwitchSense, VehicleConfig};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, MassProperties, RotationalState,
    SimulationTime,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};

// ── Constants from Modified_data/vehicle.py ──

/// Position in Earth.inertial (m).
const POS_ECI: DVec3 = DVec3::new(
    302_274_887.753_810_17,
    -119_023_818.108_825_01,
    -56_915_743.953_866_437,
);
/// Velocity in Earth.inertial (m/s).
const VEL_ECI: DVec3 = DVec3::new(
    942.182_494_673_019_85,
    -189.920_638_006_114_07,
    -292.959_665_506_469_89,
);
/// Vehicle mass (kg).
const MASS: f64 = 91_589.71;

/// Integration timestep (s). RK4 at 0.5s.
const DT: f64 = 0.5;
/// Total simulation time (s).
const TOTAL_TIME: f64 = 100.0;

/// Frame switch distance threshold (m) — switch to Moon.inertial on approach.
const SWITCH_DISTANCE: f64 = 66.1e6;

// Gravitational parameters — match JEOD's spherical gravity data exactly:
// earth_spherical.cc, moon_spherical.cc, sun_spherical.cc
const MU_EARTH: f64 = 3.986_004_415e14;
const MU_MOON: f64 = 4.902_801_076e12;
const MU_SUN: f64 = 1.327_124_40e20;

fn test_data_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data")
}

/// Build a Simulation configured for the Apollo 8 scenario.
///
/// Returns (sim, body_index).
fn build_apollo8_sim(frame_switches: Vec<FrameSwitchConfig>) -> (Simulation, usize) {
    let data_dir = test_data_dir();
    let bsp_path = data_dir.join("de405.bsp");
    assert!(
        bsp_path.exists(),
        "DE405 ephemeris not found at {}",
        bsp_path.display()
    );

    // Dec 23, 1968, 19:38:00 UTC.
    // JD = 2440213.5 (Dec 23 0h UT) + 19h38m/24h = 2440214.31806
    // TJT = JD - 2440000.5 (JEOD truncated Julian convention).
    let utc_tjt = 2_440_214.318_055_555_5 - 2_440_000.5; // 213.818...
    let leap_table = jeod_sim::default_leap_second_table();
    let tai_tjt = leap_table.utc_to_tai_tjt(utc_tjt);
    let time = SimulationTime::new(tai_tjt, leap_table);

    let ephemeris =
        jeod_sim::Ephemeris::from_bsp(&bsp_path).expect("Failed to load DE405 ephemeris");

    let mut sim = Simulation::new(time, DT);
    sim.ephemeris = Some(ephemeris);

    // Gravity sources: Sun, Earth, Moon (all spherical, matching JEOD config)
    let sun = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: MU_SUN,
                model: GravityModel::PointMass,
            },
            DVec3::ZERO,
            None,
        ),
    );
    sim.set_source_ephemeris(
        sun,
        jeod_sim::EphemerisBody::Sun,
        jeod_sim::EphemerisBody::Earth,
    );

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry::new(
            GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            },
            DVec3::ZERO,
            None,
        ),
    );

    let moon = sim.add_source(
        "Moon",
        GravitySourceEntry::new(
            GravitySource {
                mu: MU_MOON,
                model: GravityModel::PointMass,
            },
            DVec3::ZERO,
            None,
        ),
    );
    sim.set_source_ephemeris(
        moon,
        jeod_sim::EphemerisBody::Moon,
        jeod_sim::EphemerisBody::Earth,
    );

    let body = sim.add_body(VehicleConfig {
        trans: jeod_sim::TranslationalState {
            position: POS_ECI,
            velocity: VEL_ECI,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        }),
        mass: Some({
            // Inertia: diag(100000, 200000, 400000) slug*ft^2; 1 slug*ft^2 = 1.355817948 kg*m^2
            const SLUG_FT2_TO_KG_M2: f64 = 1.355_817_948;
            let inertia = DMat3::from_diagonal(DVec3::new(
                100_000.0 * SLUG_FT2_TO_KG_M2,
                200_000.0 * SLUG_FT2_TO_KG_M2,
                400_000.0 * SLUG_FT2_TO_KG_M2,
            ));
            // CoM offset: [1098, 0, 372] inches; 1 inch = 0.0254 m
            const INCH_TO_M: f64 = 0.0254;
            let com_offset = DVec3::new(1098.0 * INCH_TO_M, 0.0, 372.0 * INCH_TO_M);
            MassProperties::with_inertia(MASS, inertia, com_offset)
        }),
        gravity_controls: GravityControls {
            controls: vec![
                // Earth is the central body for Earth-centered integration.
                GravityControl::new_spherical(earth, false),
                // Sun and Moon are third-body (differential acceleration).
                GravityControl::new_third_body(sun),
                GravityControl::new_third_body(moon),
            ],
        },
        integ_source: None,
        frame_switches,
        ..Default::default()
    });

    sim.validate().expect("validation failed");

    (sim, body)
}

/// Load reference CSV and return position vectors at each timestep.
fn load_reference_positions(filename: &str) -> Vec<DVec3> {
    let path = test_data_dir().join(filename);
    assert!(
        path.exists(),
        "Apollo 8 reference data not found at {}. \
         Generate it with: docker run ... trick/generate_references.sh",
        path.display()
    );

    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));

    let mut positions = Vec::new();
    for line in content.lines().skip(1) {
        // Skip header line
        let vals: Vec<f64> = line
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if vals.len() >= 4 {
            // Columns: time, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], ...
            positions.push(DVec3::new(vals[1], vals[2], vals[3]));
        }
    }
    positions
}

/// A single reference state from the 6-DOF CSV.
struct RefState {
    time: f64,
    position: DVec3,
    velocity: DVec3,
    quaternion: DQuat,
    ang_vel_body: DVec3,
}

/// Load reference CSV in 6-DOF interleaved format.
///
/// Columns: time, pos[0], vel[0], pos[1], vel[1], pos[2], vel[2],
///          q_scalar, q_vec[0], q_vec[1], q_vec[2],
///          ang_vel[0], ang_vel[1], ang_vel[2]
fn load_reference_sixdof(filename: &str) -> Vec<RefState> {
    let path = test_data_dir().join(filename);
    assert!(
        path.exists(),
        "Apollo 8 6-DOF reference data not found at {}. \
         Generate it with: docker run ... trick/generate_references.sh",
        path.display()
    );

    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));

    let mut states = Vec::new();
    for line in content.lines().skip(1) {
        let vals: Vec<f64> = line
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if vals.len() >= 14 {
            // Interleaved: time, pos[0], vel[0], pos[1], vel[1], pos[2], vel[2],
            //              q_scalar, q_vec[0..2], ang_vel[0..2]
            states.push(RefState {
                time: vals[0],
                position: DVec3::new(vals[1], vals[3], vals[5]),
                velocity: DVec3::new(vals[2], vals[4], vals[6]),
                // JEOD scalar-first [q0,q1,q2,q3] -> glam DQuat(x=q1, y=q2, z=q3, w=q0)
                quaternion: DQuat::from_xyzw(vals[8], vals[9], vals[10], vals[7]),
                ang_vel_body: DVec3::new(vals[11], vals[12], vals[13]),
            });
        }
    }
    states
}

#[test]
fn tier3_apollo8_eci_integ() {
    let (mut sim, body_idx) = build_apollo8_sim(vec![]);

    let ref_states = load_reference_sixdof("apollo8_eci_sixdof_state.csv");

    let steps = (TOTAL_TIME / DT).round() as usize;
    let mut our_log = Vec::with_capacity(steps);
    let mut ref_log = Vec::with_capacity(steps);

    for step in 0..steps {
        sim.step();

        let ref_idx = step + 1;
        if ref_idx < ref_states.len() {
            let body = sim.body(body_idx);
            let r = &ref_states[ref_idx];

            our_log.push(StateLog {
                time: r.time,
                position: Some(body.trans.position),
                velocity: Some(body.trans.velocity),
                quaternion: body.rot.as_ref().map(|rot| rot.quaternion.to_glam()),
                ang_vel: body.rot.as_ref().map(|rot| rot.ang_vel_body),
                ..Default::default()
            });
            ref_log.push(StateLog {
                time: r.time,
                position: Some(r.position),
                velocity: Some(r.velocity),
                quaternion: Some(r.quaternion),
                ang_vel: Some(r.ang_vel_body),
                ..Default::default()
            });
        }
    }

    let report = CrossvalReport::compute("tier3_apollo8_eci_integ", &our_log, &ref_log);
    report.write();

    report.assert_position([4.8e-5, 3.9e-5, 3.6e-5]);
    report.assert_velocity([9.6e-7, 7.7e-7, 7.2e-7]);
    report.assert_quat_angle(1e-10);
    report.assert_ang_vel([1e-15, 1e-15, 1e-15]);
}

/// Cross-validate the frame-switch simulation against JEOD's reference data.
/// JEOD logs position in the current integration frame: ECI before the switch,
/// Moon-centered after.
#[test]
fn tier3_apollo8_frame_switch() {
    let (mut sim, body_idx) = build_apollo8_sim(vec![FrameSwitchConfig {
        target_source: 2, // moon source index
        switch_sense: SwitchSense::OnApproach,
        switch_distance: SWITCH_DISTANCE,
        active: true,
    }]);

    let ref_positions = load_reference_positions("apollo8_frame_switch_V_1_State.csv");

    let steps = (TOTAL_TIME / DT).round() as usize;
    let mut max_err_eci = 0.0_f64;
    let mut max_err_moon = 0.0_f64;

    for step in 0..steps {
        sim.step();
        let ref_idx = step + 1;
        if ref_idx >= ref_positions.len() {
            break;
        }
        let our_pos = sim.body(body_idx).trans.position;
        let err = (our_pos - ref_positions[ref_idx]).length();
        if sim.body(body_idx).integ_frame_id == sim.root_frame_id {
            max_err_eci = max_err_eci.max(err);
        } else {
            max_err_moon = max_err_moon.max(err);
        }
    }

    // ECI phase (before switch): ANISE vs JEOD compiled DE405 reader.
    let tol_eci = 1.1e-5; // m (1.0e-5 * 1.05)
    assert!(
        max_err_eci < tol_eci,
        "Frame switch ECI phase: {max_err_eci:.6} m exceeds {tol_eci:.1e} m"
    );

    // Moon-centered phase: 0.31 m constant offset from ANISE vs JEOD's compiled
    // DE405 Chebyshev evaluator producing slightly different Moon positions.
    // The offset is introduced at the frame switch transformation and does not grow.
    let tol_moon = 0.33; // m (0.31 * 1.05)
    assert!(
        max_err_moon < tol_moon,
        "Frame switch Moon phase: {max_err_moon:.6} m exceeds {tol_moon} m"
    );
}
