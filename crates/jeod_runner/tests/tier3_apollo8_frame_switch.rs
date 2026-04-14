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

use glam::DVec3;
use jeod_runner::{
    FrameSwitchConfig, GravitySourceEntry, IntegrationFrame, Simulation, SwitchSense, VehicleConfig,
};
use jeod_sim::{GravityControl, GravityControls, GravityModel, GravitySource, SimulationTime};

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
        "DE421 not found at {}",
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
        jeod_sim::Ephemeris::from_bsp(&bsp_path).expect("Failed to load DE421 ephemeris");

    let mut sim = Simulation::new(time, DT);
    sim.ephemeris = Some(ephemeris);

    // Gravity sources: Sun, Earth, Moon (all spherical, matching JEOD config)
    let sun = sim.add_source(GravitySourceEntry::new(
        GravitySource {
            mu: MU_SUN,
            model: GravityModel::PointMass,
        },
        DVec3::ZERO,
        None,
    ));
    sim.set_source_ephemeris(
        sun,
        jeod_sim::EphemerisBody::Sun,
        jeod_sim::EphemerisBody::Earth,
    );

    let earth = sim.add_source(GravitySourceEntry::new(
        GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        DVec3::ZERO,
        None,
    ));

    let moon = sim.add_source(GravitySourceEntry::new(
        GravitySource {
            mu: MU_MOON,
            model: GravityModel::PointMass,
        },
        DVec3::ZERO,
        None,
    ));
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
        mass: Some(jeod_sim::MassProperties::new(MASS)),
        gravity_controls: GravityControls {
            controls: vec![
                // Earth is the central body for Earth-centered integration.
                GravityControl::new_spherical(earth, false),
                // Sun and Moon are third-body (differential acceleration).
                GravityControl::new_third_body(sun),
                GravityControl::new_third_body(moon),
            ],
        },
        integ_frame: IntegrationFrame::EarthInertial,
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

#[test]
fn tier3_apollo8_eci_integ() {
    let (mut sim, body_idx) = build_apollo8_sim(vec![]);

    let ref_positions = load_reference_positions("apollo8_eci_V_1_State.csv");

    let steps = (TOTAL_TIME / DT).round() as usize;
    let mut max_pos_err = 0.0_f64;

    for step in 0..steps {
        sim.step();

        // ref_positions[0] is at t=0 (initial conditions); after step N we're at t=(N+1)*dt.
        let ref_idx = step + 1;
        if ref_idx < ref_positions.len() {
            let our_pos = sim.body(body_idx).trans.position;
            let err = (our_pos - ref_positions[ref_idx]).length();
            max_pos_err = max_pos_err.max(err);
        }
    }

    // 67 µm over 100s. Residual from ANISE vs JEOD's compiled DE405 reader.
    let tol = 7.1e-5; // m (6.7e-5 * 1.05)
    assert!(
        max_pos_err < tol,
        "Apollo 8 ECI: max position error {max_pos_err:.6} m exceeds tolerance {tol} m"
    );
}

/// Cross-validate the frame-switch simulation against JEOD's reference data.
/// JEOD logs position in the current integration frame: ECI before the switch,
/// Moon-centered after.
#[test]
fn tier3_apollo8_frame_switch() {
    let (mut sim, body_idx) = build_apollo8_sim(vec![FrameSwitchConfig {
        target_frame: IntegrationFrame::MoonInertial,
        switch_sense: SwitchSense::OnApproach,
        switch_distance: SWITCH_DISTANCE,
        active: true,
        central_source: Some(2),
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
        if sim.body(body_idx).integ_frame == IntegrationFrame::EarthInertial {
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

    // Moon-centered phase (60s): 0.31 m from frozen source positions during
    // RK4 sub-stages. JEOD updates its reference frame tree at each derivative
    // evaluation; we freeze source positions per step.
    let tol_moon = 0.33; // m (0.31 * 1.05)
    assert!(
        max_err_moon < tol_moon,
        "Frame switch Moon phase: {max_err_moon:.6} m exceeds {tol_moon} m"
    );
}
