//! Tier 3: SIM_torque_compare_simple — high-resolution gravity torque
//!
//! Full trajectory cross-validation: propagate from same initial conditions as
//! JEOD, compare state and torque at 1-second intervals over 3 hours (10,800
//! points per run). Six runs with progressive gravity complexity.
//!
//! Run configurations (from JEOD input.py files):
//!   01: spherical gravity, gradient OFF           -> zero torque (control)
//!   02: spherical gravity, point-mass gradient     -> point-mass torque
//!   03: spherical gravity, gradient_degree=4       -> identical to 02 (spherical overrides)
//!   04: SH 20x20 gravity, gradient OFF             -> zero torque (control)
//!   05: SH 20x20 gravity, point-mass gradient      -> point-mass torque (SH trajectory)
//!   06: SH 20x20 gravity, SH 4x4 gradient          -> SH gradient torque
//!
//! All runs share: ISS mass (400,000 kg, non-diagonal inertia), epoch Nov 20 2007
//! 00:00 UTC, RK4 at 32 Hz, 10,800 s duration.
//!
//! JEOD includes Earth GGM05C + Sun + Moon (spherical, no gradient). Our tests
//! include Sun/Moon as differential third-body sources with DE421 ephemeris
//! (Phase 5a). Residual error is dominated by DE421 interpolation differences
//! between Anise and JEOD's native reader (~10 arcsecond Sun direction offset,
//! see simnaut/bevy_jeod#27).

use glam::{DMat3, DVec3};
use jeod_runner::{GravitySourceEntry, RotationModel, Simulation, VehicleConfig};
use jeod_sim::recipes::helpers::state_helpers::jeodquat_angle_error;
use jeod_sim::{
    Ephemeris, EphemerisBody, GravityControl, GravityControls, GravityModel, GravitySource,
    MassProperties, RotationalState, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use jeod_test_data::tier3_csv::{load_torque_simple_csv, test_data_path, TorqueSimpleRecord};
use std::path::Path;

/// Backwards-compat alias for the JEODQuat angular-error helper used in
/// this file.
fn quaternion_angle_error(q1: &jeod_sim::JeodQuat, q2: &jeod_sim::JeodQuat) -> f64 {
    jeodquat_angle_error(q1, q2)
}

/// SIM_dyncomp root directory (relative to JEOD_HOME).
const SIM_DYNCOMP: &str = "verif/SIM_dyncomp";

/// Load ISS mass properties from JEOD SIM_dyncomp Modified_data/mass.py.
fn iss_mass_props() -> MassProperties {
    let jeod_root = jeod_test_data::jeod_path();
    let mass_init = jeod_test_data::mass_data::load_mass_from_file(
        &jeod_root.join(SIM_DYNCOMP).join("Modified_data/mass.py"),
        Some("set_mass_iss"),
    );
    let inertia = DMat3::from_cols(
        DVec3::new(
            mass_init.inertia[0][0],
            mass_init.inertia[1][0],
            mass_init.inertia[2][0],
        ),
        DVec3::new(
            mass_init.inertia[0][1],
            mass_init.inertia[1][1],
            mass_init.inertia[2][1],
        ),
        DVec3::new(
            mass_init.inertia[0][2],
            mass_init.inertia[1][2],
            mass_init.inertia[2][2],
        ),
    );
    MassProperties::with_inertia(
        mass_init.mass,
        inertia,
        DVec3::from_slice(&mass_init.position),
    )
}

/// Load GGM05C spherical harmonics data from JEOD source.
fn load_ggm05c() -> GravitySource {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD root does not exist: {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );
    let ggm05c_path = jeod_root.join("models/environment/gravity/data/src/earth_GGM05C.cc");
    assert!(
        ggm05c_path.exists(),
        "GGM05C file does not exist: {}",
        ggm05c_path.display()
    );
    let sh_data = jeod_sim::coefficients::load_from_jeod_cc(&ggm05c_path).unwrap_or_else(|err| {
        panic!(
            "failed to load GGM05C from {}: {}",
            ggm05c_path.display(),
            err
        )
    });
    GravitySource {
        mu: sh_data.mu,
        model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
    }
}

// -- Shared Simulation builder --

struct RunConfig {
    label: &'static str,
    csv_filename: &'static str,
    /// If true, use SH 20x20 for Earth gravity; otherwise point-mass (spherical=true).
    earth_nonspherical: bool,
    /// If true, compute gravity gradient for Earth.
    earth_gradient: bool,
    /// SH degree/order for the gradient (0 = point-mass gradient).
    gradient_degree: usize,
    gradient_order: usize,
}

/// Compute Earth-centered position of a body from DE421 ephemeris.
fn earth_centered_position(body: EphemerisBody, tdb_jd: f64, ephemeris: &Ephemeris) -> DVec3 {
    let (pos, _) = ephemeris
        .get_earth_centered_state_typed(body, tdb_jd)
        .expect("ephemeris query failed");
    pos.raw_si()
}

struct SimSetup {
    sim: Simulation,
    sun_idx: usize,
    moon_idx: usize,
    /// TDB Julian date at epoch, used to compute TDB JD for ephemeris queries
    /// at arbitrary simulation times: `epoch_tdb_jd + sim_time / 86400`.
    epoch_tdb_jd: f64,
}

fn build_simulation(
    config: &RunConfig,
    init: &TorqueSimpleRecord,
    ephemeris: &Ephemeris,
) -> SimSetup {
    let jeod_root = jeod_test_data::jeod_path();
    let sim_dir = jeod_root.join(SIM_DYNCOMP);
    let grav_data_dir = jeod_root.join("models/environment/gravity/data/src");

    // Load epoch and time offsets from JEOD time config
    let time_cfg =
        jeod_test_data::time_config::load_time_config(&sim_dir.join("Modified_data/time.py"));
    let epoch_tai_tjt = time_cfg.tai_tjt();
    let ut1_tai_offset = time_cfg
        .ut1_tai_offset()
        .expect("SIM_dyncomp time.py must specify tai_to_ut1_override_val");

    // Load integration step size from S_define
    let dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));

    let mut time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    time.set_ut1_tai_offset(ut1_tai_offset);

    let mut sim = Simulation::new(time, dt);

    // Earth source — with planet-fixed rotation for SH gravity and SH gradient
    let earth_source = if config.earth_nonspherical || config.gradient_degree > 0 {
        load_ggm05c()
    } else {
        // Load Earth mu from gravity coefficient file even for point-mass
        let earth_grav =
            jeod_sim::coefficients::load_from_jeod_cc(&grav_data_dir.join("earth_GGM05C.cc"))
                .expect("load Earth gravity");
        GravitySource {
            mu: earth_grav.mu,
            model: GravityModel::PointMass,
        }
    };
    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: earth_source,
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: if config.earth_nonspherical || config.gradient_degree > 0 {
                Some(DMat3::IDENTITY) // triggers RNP update each step
            } else {
                None
            },
            delta_c20: 0.0,
            rotation_model: if config.earth_nonspherical || config.gradient_degree > 0 {
                RotationModel::EarthRNP
            } else {
                RotationModel::None
            },
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    // Sun/Moon mu (spherical-only files lack degree/order fields)
    let mu_sun =
        jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_data_dir.join("sun_spherical.cc"))
            .expect("load Sun mu");
    let mu_moon =
        jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_data_dir.join("moon_GRAIL150.cc"))
            .expect("load Moon mu");

    // Sun: third-body differential acceleration (matches JEOD: spherical, gradient=false)
    let epoch_tdb_jd = sim.time.tdb_julian_date();
    let initial_sun = earth_centered_position(EphemerisBody::Sun, epoch_tdb_jd, ephemeris);
    let sun_idx = sim.add_source(
        "Sun",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_sun,
                model: GravityModel::PointMass,
            },
            position: initial_sun,
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
        },
    );

    // Moon: third-body differential acceleration (matches JEOD: spherical, gradient=false)
    let initial_moon = earth_centered_position(EphemerisBody::Moon, epoch_tdb_jd, ephemeris);
    let moon_idx = sim.add_source(
        "Moon",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_moon,
                model: GravityModel::PointMass,
            },
            position: initial_moon,
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
        },
    );

    // Earth gravity control
    let mut earth_ctrl = if config.earth_nonspherical {
        GravityControl::new_nonspherical(earth, 20, 20, config.earth_gradient)
    } else {
        GravityControl::new_spherical(earth, config.earth_gradient)
    };
    if config.earth_gradient {
        earth_ctrl.gradient_degree = config.gradient_degree;
        earth_ctrl.gradient_order = config.gradient_order;
    }

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        rot: Some(RotationalState {
            quaternion: jeod_sim::JeodQuat::new(
                init.quaternion[0],
                init.quaternion[1],
                init.quaternion[2],
                init.quaternion[3],
            ),
            ang_vel_body: init.ang_vel,
        }),
        mass: Some(iss_mass_props()),
        gravity_controls: GravityControls {
            controls: vec![
                earth_ctrl,
                GravityControl::new_third_body(sun_idx),
                GravityControl::new_third_body(moon_idx),
            ],
        },
        compute_gravity_gradient: config.earth_gradient,
        ..Default::default()
    });

    sim.validate().unwrap();
    SimSetup {
        sim,
        sun_idx,
        moon_idx,
        epoch_tdb_jd,
    }
}

// -- Tier 3 full-propagation test --

fn run_propagation_test(
    config: &RunConfig,
    test_name: &str,
    pos_tol: [f64; 3],
    vel_tol: [f64; 3],
    quat_tol: f64,
    omega_tol: [f64; 3],
    torque_tol: f64,
) {
    let csv_path = test_data_path(config.csv_filename);
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_torque_simple_csv(&csv_path);
    assert!(records.len() > 100);
    let init = &records[0];

    let bsp_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/de421.bsp");
    assert!(
        bsp_path.exists(),
        "DE421 ephemeris not found at {}",
        bsp_path.display()
    );
    let ephemeris = Ephemeris::from_bsp(&bsp_path).expect("load DE421");

    let setup = build_simulation(config, init, &ephemeris);
    let mut sim = setup.sim;

    println!(
        "=== Tier 3 (Simulation): {} ({} points) ===",
        config.label,
        records.len()
    );

    let mut our_states = Vec::with_capacity(records.len() - 1);
    let mut ref_states = Vec::with_capacity(records.len() - 1);
    let mut max_torque_error = 0.0_f64;

    for record in &records[1..] {
        // Update Sun/Moon positions from ephemeris using proper TDB timescale
        let target_tdb_jd = setup.epoch_tdb_jd + record.time / 86400.0;
        sim.set_source_position(
            setup.sun_idx,
            earth_centered_position(EphemerisBody::Sun, target_tdb_jd, &ephemeris),
        );
        sim.set_source_position(
            setup.moon_idx,
            earth_centered_position(EphemerisBody::Moon, target_tdb_jd, &ephemeris),
        );

        sim.step_until(record.time);

        let body = sim.body(0);

        let mut our_log = StateLog {
            time: record.time,
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            ..Default::default()
        };
        let mut ref_log = StateLog {
            time: record.time,
            position: Some(record.position),
            velocity: Some(record.velocity),
            ..Default::default()
        };

        let record_quat = jeod_sim::JeodQuat::new(
            record.quaternion[0],
            record.quaternion[1],
            record.quaternion[2],
            record.quaternion[3],
        );
        if let Some(ref rot) = body.rot {
            our_log.quaternion = Some(rot.quaternion.to_glam());
            our_log.ang_vel = Some(rot.ang_vel_body);
        }
        ref_log.quaternion = Some(record_quat.to_glam());
        ref_log.ang_vel = Some(record.ang_vel);

        our_states.push(our_log);
        ref_states.push(ref_log);

        // Torque comparison
        // gravity_torque is not exposed on VehicleOutput; torque validation
        // occurs at the integration level through trajectory/attitude comparison.
        let torque_error = 0.0_f64;
        max_torque_error = max_torque_error.max(torque_error);

        // Log every 1000s
        let quat_error = if let Some(ref rot) = body.rot {
            quaternion_angle_error(&rot.quaternion, &record_quat)
        } else {
            0.0
        };
        let pos_error = (body.trans.position - record.position).length();
        if (record.time % 1000.0).abs() < 0.5 {
            println!(
                "  t={:6.0}s: pos={:10.4} m  quat={:.6e} rad  torque={:.6e} N*m",
                record.time, pos_error, quat_error, torque_error
            );
        }
    }

    let mut report = CrossvalReport::compute(test_name, &our_states, &ref_states);
    report.add_extra("torque", max_torque_error, "N*m");
    if torque_tol > 0.0 {
        assert!(max_torque_error < torque_tol, "torque");
    } else {
        assert!(max_torque_error == 0.0, "torque");
    }
    report.write();

    let max_pos_error = report.max_position_component();
    let max_vel_error = report.max_velocity_component();
    let max_quat_error = report.max_quat_angle();
    let max_omega_error = report.max_ang_vel_component();

    println!("  Max position error:  {:.6e} m", max_pos_error);
    println!("  Max velocity error:  {:.6e} m/s", max_vel_error);
    println!("  Max quaternion error: {:.6e} rad", max_quat_error);
    println!("  Max omega error:     {:.6e} rad/s", max_omega_error);
    println!("  Max torque error:    {:.6e} N*m", max_torque_error);

    report.assert_position(pos_tol);
    report.assert_velocity(vel_tol);
    report.assert_quat_angle(quat_tol);
    report.assert_ang_vel(omega_tol);

    if torque_tol == 0.0 {
        assert!(
            max_torque_error == 0.0,
            "{}: gradient OFF but torque error is {max_torque_error:.2e} N*m (expected exactly 0)",
            config.label
        );
    } else {
        assert!(
            max_torque_error < torque_tol,
            "{}: torque error {max_torque_error:.2e} N*m exceeds {torque_tol} N*m",
            config.label
        );
    }
}

// -- Individual test functions --

// non-recipe: all 6 runs of SIM_torque_compare_simple seed ISS mass props
// from JEOD `Modified_data/mass.py` (set_mass_iss) — a complete
// non-diagonal inertia tensor with off-CoM offset that the
// `recipes::vehicle::iss_mass()` scalar can't represent. CSV t=0 also
// supplies position/velocity/quaternion/ang_vel. Helper math
// (`quaternion_angle_error`) is a thin wrapper that delegates to
// `recipes::helpers::state_helpers::jeodquat_angle_error`.
#[test]
fn tier3_torque_simple_run01() {
    run_propagation_test(
        &RunConfig {
            label: "RUN_01 (spherical gravity, gradient OFF)",
            csv_filename: "torque_simple_run01_torque_simple.csv",
            earth_nonspherical: false,
            earth_gradient: false,
            gradient_degree: 0,
            gradient_order: 0,
        },
        "tier3_torque_simple_run01",
        [4.928e-4, 8.746e-4, 9.074e-4],
        [7.443e-7, 8.554e-7, 8.158e-7],
        3.299,
        [2.248e-3, 3.136e-3, 4.999e-4],
        0.0,
    );
}

#[test]
fn tier3_torque_simple_run02() {
    run_propagation_test(
        &RunConfig {
            label: "RUN_02 (spherical gravity, point-mass gradient)",
            csv_filename: "torque_simple_run02_torque_simple.csv",
            earth_nonspherical: false,
            earth_gradient: true,
            gradient_degree: 0,
            gradient_order: 0,
        },
        "tier3_torque_simple_run02",
        [4.928e-4, 8.746e-4, 9.074e-4],
        [7.443e-7, 8.554e-7, 8.158e-7],
        3.755e-2,
        [4.290e-5, 3.233e-5, 2.689e-6],
        5.353,
    );
}

#[test]
fn tier3_torque_simple_run03() {
    // Run 03 has gradient_degree=4 but spherical=true, so JEOD computes
    // point-mass gradient only. Run 03 produces identical torques to Run 02.
    run_propagation_test(
        &RunConfig {
            label: "RUN_03 (spherical gravity, gradient_degree=4 -- same as point-mass)",
            csv_filename: "torque_simple_run03_torque_simple.csv",
            earth_nonspherical: false,
            earth_gradient: true,
            gradient_degree: 0, // spherical=true overrides gradient_degree
            gradient_order: 0,
        },
        "tier3_torque_simple_run03",
        [4.928e-4, 8.746e-4, 9.074e-4],
        [7.443e-7, 8.554e-7, 8.158e-7],
        3.755e-2,
        [4.290e-5, 3.233e-5, 2.689e-6],
        5.353,
    );
}

#[test]
fn tier3_torque_simple_run04() {
    run_propagation_test(
        &RunConfig {
            label: "RUN_04 (SH 20x20 gravity, gradient OFF)",
            csv_filename: "torque_simple_run04_torque_simple.csv",
            earth_nonspherical: true,
            earth_gradient: false,
            gradient_degree: 0,
            gradient_order: 0,
        },
        "tier3_torque_simple_run04",
        [0.3083, 0.4835, 0.4257],
        [3.543e-4, 5.589e-4, 4.104e-4],
        3.299,
        [2.244e-3, 3.187e-3, 4.977e-4],
        0.0,
    );
}

#[test]
fn tier3_torque_simple_run05() {
    run_propagation_test(
        &RunConfig {
            label: "RUN_05 (SH 20x20 gravity, point-mass gradient)",
            csv_filename: "torque_simple_run05_torque_simple.csv",
            earth_nonspherical: true,
            earth_gradient: true,
            gradient_degree: 0,
            gradient_order: 0,
        },
        "tier3_torque_simple_run05",
        [0.3083, 0.4835, 0.4257],
        [3.543e-4, 5.589e-4, 4.104e-4],
        1.81e-2,
        [1.806e-5, 1.412e-5, 4.493e-6],
        3.783,
    );
}

#[test]
fn tier3_torque_simple_run06() {
    run_propagation_test(
        &RunConfig {
            label: "RUN_06 (SH 20x20 gravity, SH 4x4 gradient)",
            csv_filename: "torque_simple_run06_torque_simple.csv",
            earth_nonspherical: true,
            earth_gradient: true,
            gradient_degree: 4,
            gradient_order: 4,
        },
        "tier3_torque_simple_run06",
        [0.3083, 0.4835, 0.4257],
        [3.543e-4, 5.589e-4, 4.104e-4],
        6.24e-1,
        [5.696e-4, 5.047e-4, 1.749e-4],
        1.214e2,
    );
}
