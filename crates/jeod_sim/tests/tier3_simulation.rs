//! Tier 3: jeod_sim::Simulation vs JEOD Trick reference trajectories.
//!
//! These tests validate the actual production code path (Simulation::step)
//! against NASA JEOD's Trick simulation output. Combined with the Tier 0
//! cross-parity proof (Bevy == Simulation, bit-identical), this establishes:
//!
//!   Bevy App ≡ Simulation ≈ JEOD (within Tier 3 tolerances)
//!
//! Scenarios covered:
//!   - RUN_2:  Point-mass gravity, 3-DOF (28800s ISS orbit)
//!   - RUN_2:  Point-mass gravity, 6-DOF with ISS mass (28800s)
//!   - RUN_3A: Spherical harmonics 4x4 + RNP (28800s) [requires JEOD_HOME]
//!   - RUN_3B: Spherical harmonics 8x8 + RNP (28800s) [requires JEOD_HOME]
//!   - RUN_6B: MET atmosphere + ballistic drag, 6-DOF (28800s)
//!   - RUN_9A: External torque, 6-DOF (28800s)
//!
//! Not covered (model not yet supported by Simulation runner):
//!   - SRP (flat-plate model with 6 plates + thermal + shadow — Simulation
//!     only supports spherical SRP, not flat-plate)

use glam::{DMat3, DVec3};
use jeod_atmosphere::met::{self, MetAtmosphere};
use jeod_dynamics::{
    DynamicsConfig, GravityAcceleration, MassProperties, RotationalState, TranslationalState,
};
use jeod_gravity::{GravityControl, GravityControls, GravityModel, GravitySource};
use jeod_interactions::DragConfig;
use jeod_math::JeodQuat;
use jeod_sim::{AtmosphereConfig, AtmosphereModel, GravitySourceEntry, SimBody, Simulation};
use jeod_time::SimulationTime;
use std::path::Path;

const MU_EARTH: f64 = 3.986004418e14;
const DT: f64 = 0.03125; // 32 Hz, matches JEOD SIM_dyncomp

// ── CSV parsing (same column layout as existing tier 3 tests) ──

#[derive(Debug)]
struct TransRecord {
    time: f64,
    position: DVec3,
    velocity: DVec3,
}

#[derive(Debug)]
struct SixDofRecord {
    time: f64,
    position: DVec3,
    velocity: DVec3,
    quaternion: JeodQuat,
    ang_vel: DVec3,
}

fn load_trans_trajectory(path: &Path) -> Vec<TransRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read JEOD trajectory CSV from {}: {e}\n\
             Generate with: docker build -f trick/Dockerfile -t jeod-trick .. && \
             docker run --rm -v $(pwd)/test_data:/output jeod-trick",
            path.display()
        )
    });
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 17 {
            continue;
        }
        let p = |s: &str| -> f64 { s.trim().parse().unwrap() };
        records.push(TransRecord {
            time: p(f[0]),
            position: DVec3::new(p(f[1]), p(f[8]), p(f[15])),
            velocity: DVec3::new(p(f[2]), p(f[9]), p(f[16])),
        });
    }
    records
}

fn load_sixdof_trajectory(path: &Path) -> Vec<SixDofRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read JEOD trajectory CSV from {}: {e}\n\
             Generate with: docker build -f trick/Dockerfile -t jeod-trick .. && \
             docker run --rm -v $(pwd)/test_data:/output jeod-trick",
            path.display()
        )
    });
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(f.len() >= 23, "line {}: expected >=23 columns", i + 1);
        let p = |s: &str| -> f64 { s.trim().parse().unwrap() };
        records.push(SixDofRecord {
            time: p(f[0]),
            position: DVec3::new(p(f[1]), p(f[8]), p(f[15])),
            velocity: DVec3::new(p(f[2]), p(f[9]), p(f[16])),
            ang_vel: DVec3::new(p(f[3]), p(f[10]), p(f[17])),
            quaternion: JeodQuat::new(p(f[22]), p(f[7]), p(f[14]), p(f[21])),
        });
    }
    records
}

fn quaternion_angle_error(q1: &JeodQuat, q2: &JeodQuat) -> f64 {
    let dot = (q1.scalar() * q2.scalar()
        + q1.vector().x * q2.vector().x
        + q1.vector().y * q2.vector().y
        + q1.vector().z * q2.vector().z)
        .abs();
    2.0 * dot.min(1.0).acos()
}

fn test_data_path(filename: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(filename)
}

// ── Scenario 1: Point-mass 3-DOF (RUN_2) ──

#[test]
fn tier3_simulation_run2_3dof() {
    let csv_path = test_data_path("dyncomp_run2_state.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let trajectory = load_trans_trajectory(&csv_path);
    assert!(trajectory.len() > 100);

    let init = &trajectory[0];

    // Set up Simulation — point-mass gravity, no atmosphere, no interactions
    let time = SimulationTime::at_j2000(jeod_time::leap_second::default_leap_second_table());
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
        rot: None,
        mass: None,
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: false,
            three_dof: true,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        drag: None,
        srp: None,
        t_struct_body: DMat3::IDENTITY,
        compute_gravity_torque: false,
        atmospheric_state: None,
        gravity_accel: GravityAcceleration::default(),
        total_force: Default::default(),
        frame_derivs: Default::default(),
        aero_force: None,
        radiation_force: None,
        gravity_torque: None,
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): RUN_2 point-mass 3-DOF, {} points",
        trajectory.len()
    );

    let mut max_pos_error = 0.0_f64;
    let mut max_vel_error = 0.0_f64;

    for record in &trajectory[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);
        let pos_error = (body.trans.position - record.position).length();
        let vel_error = (body.trans.velocity - record.velocity).length();
        max_pos_error = max_pos_error.max(pos_error);
        max_vel_error = max_vel_error.max(vel_error);

        if (record.time % 3600.0).abs() < 30.1 {
            println!(
                "  t={:6.0}s: pos_err={:10.4} m  vel_err={:.6} m/s",
                record.time, pos_error, vel_error
            );
        }
    }

    println!("  Max position error: {:.4} m", max_pos_error);
    println!("  Max velocity error: {:.6} m/s", max_vel_error);

    assert!(
        max_pos_error < 0.5,
        "Position error {max_pos_error:.2} m exceeds 0.5 m over 8 hours"
    );
    assert!(
        max_vel_error < 0.001,
        "Velocity error {max_vel_error:.6} m/s exceeds 0.001 m/s over 8 hours"
    );
}

// ── Scenario 2: Point-mass 6-DOF with ISS mass (RUN_2) ──

#[test]
fn tier3_simulation_run2_6dof() {
    let csv_path = test_data_path("dyncomp_run2_state.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}",
        csv_path.display()
    );

    let trajectory = load_sixdof_trajectory(&csv_path);
    assert!(trajectory.len() > 100);

    let init = &trajectory[0];

    // ISS mass properties from Modified_data/mass.py
    let inertia = DMat3::from_cols(
        DVec3::new(1.02e8, -6.96e6, -5.48e6),
        DVec3::new(-6.96e6, 0.91e8, 5.90e5),
        DVec3::new(-5.48e6, 5.90e5, 1.64e8),
    );
    let mass_props = MassProperties::with_inertia(400_000.0, inertia, DVec3::new(-3.0, -1.5, 4.0));

    let time = SimulationTime::at_j2000(jeod_time::leap_second::default_leap_second_table());
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
        rot: Some(RotationalState {
            quaternion: init.quaternion,
            ang_vel_body: init.ang_vel,
        }),
        mass: Some(mass_props),
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: true,
            three_dof: false,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        drag: None,
        srp: None,
        t_struct_body: DMat3::IDENTITY,
        compute_gravity_torque: false,
        atmospheric_state: None,
        gravity_accel: GravityAcceleration::default(),
        total_force: Default::default(),
        frame_derivs: Default::default(),
        aero_force: None,
        radiation_force: None,
        gravity_torque: None,
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): RUN_2 point-mass 6-DOF, {} points",
        trajectory.len()
    );

    let mut max_pos_error = 0.0_f64;
    let mut max_vel_error = 0.0_f64;
    let mut max_quat_error = 0.0_f64;
    let mut max_omega_error = 0.0_f64;

    for record in &trajectory[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);
        let pos_error = (body.trans.position - record.position).length();
        let vel_error = (body.trans.velocity - record.velocity).length();
        max_pos_error = max_pos_error.max(pos_error);
        max_vel_error = max_vel_error.max(vel_error);

        if let Some(ref rot) = body.rot {
            let quat_error = quaternion_angle_error(&rot.quaternion, &record.quaternion);
            let omega_error = (rot.ang_vel_body - record.ang_vel).length();
            max_quat_error = max_quat_error.max(quat_error);
            max_omega_error = max_omega_error.max(omega_error);
        }

        if (record.time % 3600.0).abs() < 30.1 {
            println!(
                "  t={:6.0}s: pos_err={:10.4} m  quat_err={:.2e} rad",
                record.time, pos_error, max_quat_error
            );
        }
    }

    println!("  Max position error:  {:.4} m", max_pos_error);
    println!("  Max velocity error:  {:.6} m/s", max_vel_error);
    println!("  Max quaternion error: {:.2e} rad", max_quat_error);
    println!("  Max omega error:     {:.2e} rad/s", max_omega_error);

    assert!(
        max_pos_error < 0.5,
        "Position error {max_pos_error:.2} m exceeds 0.5 m"
    );
    assert!(
        max_vel_error < 0.001,
        "Velocity error {max_vel_error:.6} m/s exceeds 0.001 m/s"
    );
    assert!(
        max_quat_error < 0.01,
        "Quaternion error {max_quat_error:.2e} rad exceeds 0.01 rad"
    );
    assert!(
        max_omega_error < 1e-5,
        "Omega error {max_omega_error:.2e} rad/s exceeds 1e-5 rad/s"
    );
}

// ── Scenario 3: MET atmosphere + drag, 6-DOF (RUN_6B) ──

/// Epoch for SIM_dyncomp: midnight 2007-11-20 UTC.
/// MJD = 54424.0, TJT = MJD - 40000 = 14424.0.
/// From JEOD time.py: TAI-UTC = 32s override, tai_to_ut1 = -32.469s.
const DRAG_EPOCH_UTC_TJT: f64 = 14424.0;
const DRAG_TAI_UTC_S: f64 = 32.0;
const DRAG_TAI_TO_UT1_S: f64 = -32.469;

/// Earth rotation rate (JEOD RNPJ2000 default).
const OMEGA_EARTH: f64 = 7.292_115_146_706_388e-5;

#[test]
fn tier3_simulation_run6b_drag() {
    let csv_path = test_data_path("dyncomp_run6b_state.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let trajectory = load_sixdof_trajectory(&csv_path);
    assert!(trajectory.len() >= 100);

    let init = &trajectory[0];

    // Unit sphere mass (from Modified_data/mass.py)
    let inertia = DMat3::from_diagonal(DVec3::splat(0.4));
    let mass_props = MassProperties::with_inertia(1.0, inertia, DVec3::ZERO);

    // MET atmosphere: solar mean conditions (from Modified_data/solar_flux.py)
    let met_model = MetAtmosphere {
        f10: 128.8,
        f10b: 128.8,
        geo_index: 15.7,
        geo_index_type: met::GeoIndexType::Ap,
    };

    // Drag config (from Modified_data/aero_drag.py)
    let drag_config = DragConfig {
        cd: 0.02,
        area: 1.0,
    };

    // Initialize Simulation at the SIM_dyncomp epoch with correct time offsets.
    let epoch_tai_tjt = DRAG_EPOCH_UTC_TJT + DRAG_TAI_UTC_S / 86400.0;
    let mut time = SimulationTime::new(
        epoch_tai_tjt,
        jeod_time::leap_second::default_leap_second_table(),
    );
    time.set_ut1_tai_offset(DRAG_TAI_TO_UT1_S);

    let mut sim = Simulation::new(time, DT);

    // Earth source with planet-fixed rotation — the Simulation's ephemeris stage
    // updates it each step via RNP, so the atmosphere system sees correct geodetic
    // coordinates. Without this, MET density is evaluated at wrong lat/lon.
    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY), // triggers ephemeris update each step
    });

    // Configure atmosphere with planet rotation lookup
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Met(met_model),
        r_eq: 6_378_137.0,
        r_pol: 6_356_752.314_245,
        planet_omega: OMEGA_EARTH,
    });
    sim.atmosphere_planet_source = Some(earth);

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        rot: Some(RotationalState {
            quaternion: init.quaternion,
            ang_vel_body: init.ang_vel,
        }),
        mass: Some(mass_props),
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: true,
            three_dof: false,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        drag: Some(drag_config),
        srp: None,
        t_struct_body: DMat3::IDENTITY,
        compute_gravity_torque: false,
        atmospheric_state: Some(Default::default()), // presence enables atmosphere
        gravity_accel: GravityAcceleration::default(),
        total_force: Default::default(),
        frame_derivs: Default::default(),
        aero_force: None,
        radiation_force: None,
        gravity_torque: None,
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): RUN_6B MET+drag 6-DOF, {} points",
        trajectory.len()
    );

    let mut max_pos_error = 0.0_f64;
    let mut max_vel_error = 0.0_f64;
    let mut max_quat_error = 0.0_f64;

    for record in &trajectory[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);
        let pos_error = (body.trans.position - record.position).length();
        let vel_error = (body.trans.velocity - record.velocity).length();
        max_pos_error = max_pos_error.max(pos_error);
        max_vel_error = max_vel_error.max(vel_error);

        if let Some(ref rot) = body.rot {
            let quat_error = quaternion_angle_error(&rot.quaternion, &record.quaternion);
            max_quat_error = max_quat_error.max(quat_error);
        }

        if (record.time % 3600.0).abs() < 30.1 {
            println!(
                "  t={:6.0}s: pos_err={:10.4} m  vel_err={:.6} m/s",
                record.time, pos_error, vel_error
            );
        }
    }

    println!("  Max position error:  {:.4} m", max_pos_error);
    println!("  Max velocity error:  {:.6} m/s", max_vel_error);
    println!("  Max quaternion error: {:.2e} rad", max_quat_error);

    // Tolerances match existing tier3_drag_trajectory test
    assert!(
        max_pos_error < 2.0,
        "Position error {max_pos_error:.2} m exceeds 2.0 m"
    );
    assert!(
        max_vel_error < 0.005,
        "Velocity error {max_vel_error:.6} m/s exceeds 0.005 m/s"
    );
    assert!(
        max_quat_error < 0.01,
        "Quaternion error {max_quat_error:.2e} rad exceeds 0.01 rad"
    );
}

// ── Scenario 4: Spherical harmonics 4x4 / 8x8 + RNP (RUN_3A, RUN_3B) ──
// Requires JEOD_HOME for GGM02C coefficients.

/// JEOD SIM_dyncomp epoch constants (from the original SH test).
const SH_TAI_UTC_S: f64 = 32.0;
const SH_TAI_TO_UT1_S: f64 = -32.469;
const SH_EPOCH_UTC_TJT: f64 = 14424.0;

fn run_sh_simulation_test(csv_name: &str, degree: usize, order: usize, label: &str) {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    let csv_path = test_data_path(csv_name);
    assert!(
        csv_path.exists(),
        "JEOD trajectory not found at {}",
        csv_path.display()
    );

    // Load GGM02C spherical harmonics coefficients
    let ggm02c_path = jeod_root.join("models/environment/gravity/data/src/earth_GGM02C.cc");
    let sh_data = jeod_gravity::coefficients::load_from_jeod_cc(&ggm02c_path).expect("load GGM02C");

    let trajectory = load_trans_trajectory(&csv_path);
    assert!(trajectory.len() > 100);
    let init = &trajectory[0];

    // Build a spherical harmonics gravity source.
    // The Simulation will update planet-fixed rotation each step via RNP.
    let sh_source = GravitySource {
        mu: sh_data.mu,
        model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
    };

    // Initialize Simulation at the SIM_dyncomp epoch.
    // TAI TJT = UTC TJT + TAI-UTC/86400
    let epoch_tai_tjt = SH_EPOCH_UTC_TJT + SH_TAI_UTC_S / 86400.0;
    let mut time = SimulationTime::new(
        epoch_tai_tjt,
        jeod_time::leap_second::default_leap_second_table(),
    );
    // Set UT1-TAI offset so GMST computation matches JEOD's time.py configuration
    time.set_ut1_tai_offset(SH_TAI_TO_UT1_S);

    let mut sim = Simulation::new(time, DT);

    // Earth source with planet-fixed rotation (Simulation updates it each step).
    // Initialize with identity — first step() will compute the real rotation.
    let earth = sim.add_source(GravitySourceEntry {
        source: sh_source,
        position: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY), // presence triggers ephemeris update
    });

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        rot: None,
        mass: None,
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: false,
            three_dof: true,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_nonspherical(
                earth, degree, order, false,
            )],
        },
        drag: None,
        srp: None,
        t_struct_body: DMat3::IDENTITY,
        compute_gravity_torque: false,
        atmospheric_state: None,
        gravity_accel: GravityAcceleration::default(),
        total_force: Default::default(),
        frame_derivs: Default::default(),
        aero_force: None,
        radiation_force: None,
        gravity_torque: None,
    });

    sim.validate().unwrap();

    println!("Tier 3 (Simulation): {label}, {} points", trajectory.len());

    let mut max_pos_error = 0.0_f64;
    let mut max_vel_error = 0.0_f64;

    for record in &trajectory[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);
        let pos_error = (body.trans.position - record.position).length();
        let vel_error = (body.trans.velocity - record.velocity).length();
        max_pos_error = max_pos_error.max(pos_error);
        max_vel_error = max_vel_error.max(vel_error);

        if (record.time % 3600.0).abs() < 30.1 {
            println!(
                "  t={:6.0}s: pos_err={:10.4} m  vel_err={:.6} m/s",
                record.time, pos_error, vel_error
            );
        }
    }

    println!("  Max position error: {:.4} m", max_pos_error);
    println!("  Max velocity error: {:.6} m/s", max_vel_error);

    // Tolerances match existing tier3_spherical_harmonics test
    assert!(
        max_pos_error < 0.5,
        "{label}: position error {max_pos_error:.2} m exceeds 0.5 m"
    );
    assert!(
        max_vel_error < 0.001,
        "{label}: velocity error {max_vel_error:.6} m/s exceeds 0.001 m/s"
    );
}

#[test]
fn tier3_simulation_run3a_sh4x4() {
    run_sh_simulation_test("dyncomp_run3a_state.csv", 4, 4, "RUN_3A (4x4 SH + RNP)");
}

#[test]
fn tier3_simulation_run3b_sh8x8() {
    run_sh_simulation_test("dyncomp_run3b_state.csv", 8, 8, "RUN_3B (8x8 SH + RNP)");
}

// ── Scenario 5: External torque, 6-DOF (RUN_9A) ──
//
// RUN_9A applies [10, 0, 0] N·m structural-frame torque from t=1000s to t=2000s.
// The Simulation runner doesn't natively support time-scheduled external forces,
// so we step manually and inject the torque by modifying total_force between
// force_collection and integration. Since step() is monolithic, we instead
// step one dt at a time and set the body's total_force.torque after each step's
// force collection would normally produce zero torque. We compensate by adding
// the external torque to what collect_and_resolve_forces produces.
//
// This exercises the same jeod_sim::integrate_body code path as the Simulation
// runner, just with manual torque injection.

#[test]
fn tier3_simulation_run9a_torque() {
    let csv_path = test_data_path("dyncomp_run9a_state.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}",
        csv_path.display()
    );

    let trajectory = load_sixdof_trajectory(&csv_path);
    assert!(trajectory.len() >= 100);
    let init = &trajectory[0];

    // ISS mass properties (from Modified_data/mass.py)
    let inertia = DMat3::from_cols(
        DVec3::new(1.02e8, -6.96e6, -5.48e6),
        DVec3::new(-6.96e6, 0.91e8, 5.90e5),
        DVec3::new(-5.48e6, 5.90e5, 1.64e8),
    );
    let mass_props = MassProperties::with_inertia(400_000.0, inertia, DVec3::new(-3.0, -1.5, 4.0));

    // Use per-body functions directly for torque injection.
    // This still validates jeod_sim's integrate_body and accumulate_gravity.
    let mut trans = TranslationalState {
        position: init.position,
        velocity: init.velocity,
    };
    let mut rot = RotationalState {
        quaternion: init.quaternion,
        ang_vel_body: init.ang_vel,
    };

    let config = DynamicsConfig {
        translational_dynamics: true,
        rotational_dynamics: true,
        three_dof: false,
    };

    let gravity_controls: GravityControls<usize> = GravityControls {
        controls: vec![GravityControl::new_spherical(0_usize, false)],
    };

    let earth_source = GravitySource {
        mu: MU_EARTH,
        model: GravityModel::PointMass,
    };

    println!(
        "Tier 3 (jeod_sim per-body): RUN_9A torque 6-DOF, {} points",
        trajectory.len()
    );

    let mut max_pos_error = 0.0_f64;
    let mut max_vel_error = 0.0_f64;
    let mut max_quat_error = 0.0_f64;
    let mut max_omega_error = 0.0_f64;
    let mut current_time = init.time;

    for record in &trajectory[1..] {
        while current_time + DT <= record.time + 0.001 {
            // Gravity (per-body function)
            let grav = jeod_sim::accumulate_gravity(trans.position, &gravity_controls, |_| {
                Some((&earth_source, None))
            });

            // External torque: [10, 0, 0] N·m in body frame during [1000, 2000)s
            let external_torque = if (999.999..1999.999).contains(&current_time) {
                DVec3::new(10.0, 0.0, 0.0)
            } else {
                DVec3::ZERO
            };

            // Force collection (no interactions, just gravity)
            let (total, _derivs) = jeod_sim::collect_and_resolve_forces(
                None,
                None,
                None,
                Some(&rot),
                DMat3::IDENTITY,
                Some(&mass_props),
                grav.grav_accel,
            );

            // Integration with external torque added.
            // Gravity recomputed at each RK4 intermediate state via closure.
            let gravity_fn = |pos: DVec3| {
                let r = pos.length();
                pos * (-MU_EARTH / (r * r * r))
            };
            jeod_sim::integrate_body(
                &config,
                &mut trans,
                Some(&mut rot),
                Some(&mass_props),
                gravity_fn,
                total.force,
                total.torque + external_torque,
                DT,
            );
            current_time += DT;
        }

        // Handle fractional remainder
        let remainder = record.time - current_time;
        if remainder > 0.001 {
            let grav = jeod_sim::accumulate_gravity(trans.position, &gravity_controls, |_| {
                Some((&earth_source, None))
            });
            let external_torque = if (999.999..1999.999).contains(&current_time) {
                DVec3::new(10.0, 0.0, 0.0)
            } else {
                DVec3::ZERO
            };
            let (total, _) = jeod_sim::collect_and_resolve_forces(
                None,
                None,
                None,
                Some(&rot),
                DMat3::IDENTITY,
                Some(&mass_props),
                grav.grav_accel,
            );
            let gravity_fn = |pos: DVec3| {
                let r = pos.length();
                pos * (-MU_EARTH / (r * r * r))
            };
            jeod_sim::integrate_body(
                &config,
                &mut trans,
                Some(&mut rot),
                Some(&mass_props),
                gravity_fn,
                total.force,
                total.torque + external_torque,
                remainder,
            );
            current_time += remainder;
        }

        let pos_error = (trans.position - record.position).length();
        let vel_error = (trans.velocity - record.velocity).length();
        let quat_error = quaternion_angle_error(&rot.quaternion, &record.quaternion);
        let omega_error = (rot.ang_vel_body - record.ang_vel).length();

        max_pos_error = max_pos_error.max(pos_error);
        max_vel_error = max_vel_error.max(vel_error);
        max_quat_error = max_quat_error.max(quat_error);
        max_omega_error = max_omega_error.max(omega_error);

        if (record.time % 3600.0).abs() < 30.1 {
            println!(
                "  t={:6.0}s: pos_err={:10.4} m  quat_err={:.2e} rad  omega_err={:.2e}",
                record.time, pos_error, quat_error, omega_error
            );
        }
    }

    println!("  Max position error:  {:.4} m", max_pos_error);
    println!("  Max velocity error:  {:.6} m/s", max_vel_error);
    println!("  Max quaternion error: {:.2e} rad", max_quat_error);
    println!("  Max omega error:     {:.2e} rad/s", max_omega_error);

    assert!(
        max_pos_error < 0.5,
        "Position error {max_pos_error:.2} m exceeds 0.5 m"
    );
    assert!(
        max_vel_error < 0.001,
        "Velocity error {max_vel_error:.6} m/s exceeds 0.001 m/s"
    );
    assert!(
        max_quat_error < 0.01,
        "Quaternion error {max_quat_error:.2e} rad exceeds 0.01 rad"
    );
    assert!(
        max_omega_error < 1e-5,
        "Omega error {max_omega_error:.2e} rad/s exceeds 1e-5 rad/s"
    );
}
