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
//! JEOD includes Earth GGM05C + Sun + Moon (spherical, no gradient); our tests
//! use Earth only because differential 3rd-body acceleration is not yet ported
//! (Phase 5 task 5.40). This causes ~10 m position drift over 3h from the missing
//! ~1e-6 m/s2 Sun/Moon perturbation, which cascades through gravity gradient
//! torque feedback into attitude divergence.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_sim::{
    DynamicsConfig, GravityControl, GravityControls, GravityModel, GravitySource,
    GravitySourceEntry, MassProperties, RotationalState, SimBody, Simulation, SimulationTime,
    TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};

// -- ISS mass properties from JEOD Modified_data/mass/iss.py --

fn iss_mass_props() -> MassProperties {
    let inertia = DMat3::from_cols(
        DVec3::new(1.02e8, -6.96e6, -5.48e6),
        DVec3::new(-6.96e6, 0.91e8, 5.90e5),
        DVec3::new(-5.48e6, 5.90e5, 1.64e8),
    );
    MassProperties::with_inertia(400_000.0, inertia, DVec3::new(-3.0, -1.5, 4.0))
}

// -- Epoch constants --
// Nov 20, 2007 00:00:00 UTC
// JEOD overrides: leap_sec_override_val = 32, tai_to_ut1_override_val = -32.469

const EPOCH_UTC_TJT: f64 = 14424.0;
const TAI_UTC_S: f64 = 32.0;
const TAI_TO_UT1_S: f64 = -32.469;

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

fn build_simulation(config: &RunConfig, init: &TorqueSimpleRecord) -> Simulation {
    let epoch_tai_tjt = EPOCH_UTC_TJT + TAI_UTC_S / 86400.0;
    let mut time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    time.set_ut1_tai_offset(TAI_TO_UT1_S);

    let mut sim = Simulation::new(time, DT);

    // Earth source — with planet-fixed rotation for SH gravity and SH gradient
    let earth_source = if config.earth_nonspherical || config.gradient_degree > 0 {
        load_ggm05c()
    } else {
        GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        }
    };
    let earth = sim.add_source(GravitySourceEntry {
        source: earth_source,
        position: DVec3::ZERO,
        t_inertial_pfix: if config.earth_nonspherical || config.gradient_degree > 0 {
            Some(DMat3::IDENTITY) // triggers RNP update each step
        } else {
            None
        },
    });

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

    // Sun/Moon omitted: 3rd-body differential acceleration is Phase 5 scope.
    // JEOD includes them as spherical point-mass sources (gradient=false).
    // When Phase 5 adds differential acceleration, add Sun/Moon sources here
    // with real mu values and ephemeris-driven positions.

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        rot: Some(RotationalState {
            quaternion: init.quaternion,
            ang_vel_body: init.ang_vel,
        }),
        mass: Some(iss_mass_props()),
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: true,
            three_dof: false,
        },
        gravity_controls: GravityControls {
            controls: vec![earth_ctrl],
        },
        compute_gravity_torque: config.earth_gradient,
        ..Default::default()
    });

    sim.validate().unwrap();
    sim
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

    let mut sim = build_simulation(config, init);

    println!(
        "=== Tier 3 (Simulation): {} ({} points) ===",
        config.label,
        records.len()
    );

    let mut our_states = Vec::with_capacity(records.len() - 1);
    let mut ref_states = Vec::with_capacity(records.len() - 1);
    let mut max_torque_error = 0.0_f64;

    for record in &records[1..] {
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

        if let Some(ref rot) = body.rot {
            our_log.quaternion = Some(rot.quaternion.to_glam());
            our_log.ang_vel = Some(rot.ang_vel_body);
        }
        ref_log.quaternion = Some(record.quaternion.to_glam());
        ref_log.ang_vel = Some(record.ang_vel);

        our_states.push(our_log);
        ref_states.push(ref_log);

        // Torque comparison
        let our_torque = body.gravity_torque.unwrap_or(DVec3::ZERO);
        let torque_error = (our_torque - record.gravity_torque).length();
        max_torque_error = max_torque_error.max(torque_error);

        // Log every 1000s
        let quat_error = if let Some(ref rot) = body.rot {
            quaternion_angle_error(&rot.quaternion, &record.quaternion)
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
        [3.02, 8.51, 1.046e1],
        [3.292e-3, 1.053e-2, 1.026e-2],
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
        [3.02, 8.51, 1.046e1],
        [3.292e-3, 1.053e-2, 1.026e-2],
        3.827e-2,
        [4.372e-5, 3.294e-5, 2.742e-6],
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
        [3.02, 8.51, 1.046e1],
        [3.292e-3, 1.053e-2, 1.026e-2],
        3.827e-2,
        [4.372e-5, 3.294e-5, 2.742e-6],
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
        [2.697, 8.024, 1.008e1],
        [2.918e-3, 1.003e-2, 9.838e-3],
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
        [2.697, 8.024, 1.008e1],
        [2.918e-3, 1.003e-2, 9.838e-3],
        1.845e-2,
        [1.841e-5, 1.439e-5, 4.579e-6],
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
        [2.697, 8.024, 1.008e1],
        [2.918e-3, 1.003e-2, 9.838e-3],
        6.242e-1,
        [5.698e-4, 5.049e-4, 1.75e-4],
        1.214e2,
    );
}
