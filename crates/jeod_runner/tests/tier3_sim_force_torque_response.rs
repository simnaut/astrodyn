//! Tier 3: Analytical verification of SIM_force_torque physics.
//!
//! JEOD's `dyn_body/verif/SIM_force_torque/` is an empty-space test rig that
//! verifies force and torque accumulation, non-transmitted forces, and the
//! equations of motion by scheduling time-stamped force/torque changes and
//! checking that the body traces out a predictable trajectory.
//!
//! Our rig has no "empty space" ephemeris mode — instead we build a central
//! gravity source that the body does not reference (empty `GravityControls`).
//! With no gravity control applied, the body responds only to external
//! force/torque, giving the same behavior as JEOD's EmptySpace mode for the
//! purposes of these analytical checks.
//!
//! JEOD scenario mapping (SIM_force_torque/SET_test/RUN_test/input.py):
//! - The scheduled `trick.add_read` blocks successively apply a z-axis
//!   force that yields 1 m/s^2, 2 m/s^2, 3 m/s^2 on the composite body,
//!   then play with non-transmitted forces, yaw/roll maneuvers, etc.
//! - Our tests below pick out the three primary physics verifications in
//!   that sim: uniform F/m translation, tau/I rotation, and independence
//!   of the two when the force is applied at the center of mass.

mod sim_test_helpers;

use glam::{DMat3, DVec3};
use jeod_runner::{GravitySourceEntry, RotationModel, Simulation, VehicleConfig};
use jeod_sim::{
    GravityControls, GravityModel, GravitySource, JeodQuat, MassProperties, RotationalState,
    SimulationTime, TranslationalState,
};

/// A negligible gravity source kept in the frame tree but not referenced by
/// any body.  Simulation requires a root frame; using `central: true` with
/// `mu=0` gives us the frame without any gravitational effect.
fn add_dummy_central_source(sim: &mut Simulation) {
    sim.add_source(
        "central_point",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );
}

/// Build an empty-space 3-DOF simulation with a single mass-only body.
fn make_free_body_3dof(mass: f64, dt: f64) -> Simulation {
    let mut sim = Simulation::new(
        SimulationTime::at_j2000(jeod_sim::default_leap_second_table()),
        dt,
    );
    add_dummy_central_source(&mut sim);

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        },
        rot: None,
        mass: Some(MassProperties::new(mass)),
        gravity_controls: GravityControls { controls: vec![] },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim
}

/// Build an empty-space 6-DOF simulation with a single rigid body.
fn make_free_body_6dof(mass: f64, inertia: DMat3, dt: f64) -> Simulation {
    let mut sim = Simulation::new(
        SimulationTime::at_j2000(jeod_sim::default_leap_second_table()),
        dt,
    );
    add_dummy_central_source(&mut sim);

    let mass_props = MassProperties::with_inertia(mass, inertia, DVec3::ZERO);
    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        }),
        mass: Some(mass_props),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        ..Default::default()
    });

    sim.validate().unwrap();
    sim
}

// ─── Test 1: F = m*a on a free body ───

/// Constant force on a free body produces uniform acceleration a = F/m.
/// After applying F for T seconds starting from rest:
///   v(T) = F*T/m
///   x(T) = 0.5 * F*T^2/m
// non-recipe: SIM_force_torque uses 100 kg test masses and diag(10,20,20)
// inertia for closed-form verification of F=m·a and τ=I·α. Analytical
// constants don't match any recipe vehicle preset, and the constant-input
// pattern is shorter than `force_torque_profiles::step_input` because
// the on/off times here are step-count expressions, not test parameters.
#[test]
fn tier3_force_constant_acceleration() {
    let mass = 100.0; // kg
    let dt = 0.1; // s
    let t_total = 10.0; // s
    let force = DVec3::new(3.0, -2.0, 1.0); // N (arbitrary direction)

    let mut sim = make_free_body_3dof(mass, dt);
    sim.set_body_external_force(0, force);
    sim.step_n((t_total / dt) as usize);

    let body = sim.body(0);

    let expected_v = force * t_total / mass;
    let expected_x = 0.5 * force * t_total * t_total / mass;

    let err_v = (body.trans.velocity - expected_v).length();
    let err_x = (body.trans.position - expected_x).length();

    let rel_v = err_v / expected_v.length();
    let rel_x = err_x / expected_x.length();

    println!(
        "  F=m*a: v={:?}, expected={:?}, rel_err={rel_v:.3e}",
        body.trans.velocity, expected_v
    );
    println!(
        "         x={:?}, expected={:?}, rel_err={rel_x:.3e}",
        body.trans.position, expected_x
    );

    // RK4 is exact for linear ODEs up to roundoff; tolerances are tight.
    assert!(rel_v < 1.0e-12, "velocity rel err {rel_v:.3e}");
    assert!(rel_x < 1.0e-12, "position rel err {rel_x:.3e}");
}

// ─── Test 2: tau = I*alpha on a symmetric body (constant torque) ───

/// Constant torque about a principal axis of a body with diagonal inertia:
///   omega(T) = tau*T/I (on that axis; zero on others)
///   theta(T) = 0.5 * tau*T^2/I
/// For an initially aligned body, after 10 s with tau_x=1 N·m and I_x=10:
///   omega_x = 1 rad/s, theta_x = 5 rad.
// non-recipe: same analytical-mass setup as `tier3_force_constant_acceleration`.
#[test]
fn tier3_torque_constant_angular_acceleration() {
    let mass = 100.0;
    let i_x = 10.0;
    let i_y = 20.0;
    let i_z = 20.0;
    let inertia = DMat3::from_cols(
        DVec3::new(i_x, 0.0, 0.0),
        DVec3::new(0.0, i_y, 0.0),
        DVec3::new(0.0, 0.0, i_z),
    );
    let dt = 0.01;
    let t_total = 10.0;
    let tau = DVec3::new(1.0, 0.0, 0.0); // N·m about body x-axis

    let mut sim = make_free_body_6dof(mass, inertia, dt);
    sim.set_body_external_torque(0, tau);
    sim.step_n((t_total / dt) as usize);

    let body = sim.body(0);
    let omega = body.rot.as_ref().unwrap().ang_vel_body;

    let expected_omega_x = tau.x * t_total / i_x;

    let rel_err = (omega.x - expected_omega_x).abs() / expected_omega_x.abs();
    println!(
        "  tau=I*alpha: omega={:?}, expected_x={expected_omega_x}, rel_err={rel_err:.3e}",
        omega
    );

    assert!(rel_err < 1.0e-10, "omega_x rel err {rel_err:.3e}");

    // Perpendicular components stay zero (diagonal inertia, torque on principal
    // axis, no initial rate).
    assert!(
        omega.y.abs() < 1.0e-12,
        "omega_y should be zero, got {}",
        omega.y
    );
    assert!(
        omega.z.abs() < 1.0e-12,
        "omega_z should be zero, got {}",
        omega.z
    );
}

// ─── Test 3: force at CoM decouples translation and rotation ───

/// When a force is applied through the center of mass and an independent
/// torque is applied on the body, translation and rotation are decoupled:
///   linear motion follows v = F*t/m regardless of torque
///   angular motion follows omega = tau*t/I regardless of force
// non-recipe: same analytical-mass setup as `tier3_force_constant_acceleration`.
#[test]
fn tier3_force_and_torque_decoupled() {
    let mass = 100.0;
    let i_x = 10.0;
    let i_y = 10.0;
    let i_z = 10.0;
    let inertia = DMat3::from_cols(
        DVec3::new(i_x, 0.0, 0.0),
        DVec3::new(0.0, i_y, 0.0),
        DVec3::new(0.0, 0.0, i_z),
    );
    let dt = 0.01;
    let t_total = 5.0;
    let force = DVec3::new(2.0, 0.0, 0.0); // N along inertial x
    let tau = DVec3::new(0.0, 0.0, 1.0); // N·m about body z

    // Case A: only force.
    let mut sim_a = make_free_body_6dof(mass, inertia, dt);
    sim_a.set_body_external_force(0, force);
    sim_a.step_n((t_total / dt) as usize);
    let v_a = sim_a.body(0).trans.velocity;
    let omega_a = sim_a.body(0).rot.as_ref().unwrap().ang_vel_body;

    // Case B: only torque.
    let mut sim_b = make_free_body_6dof(mass, inertia, dt);
    sim_b.set_body_external_torque(0, tau);
    sim_b.step_n((t_total / dt) as usize);
    let v_b = sim_b.body(0).trans.velocity;
    let omega_b = sim_b.body(0).rot.as_ref().unwrap().ang_vel_body;

    // Case C: both.
    let mut sim_c = make_free_body_6dof(mass, inertia, dt);
    sim_c.set_body_external_force(0, force);
    sim_c.set_body_external_torque(0, tau);
    sim_c.step_n((t_total / dt) as usize);
    let v_c = sim_c.body(0).trans.velocity;
    let omega_c = sim_c.body(0).rot.as_ref().unwrap().ang_vel_body;

    println!("  A (force only): v={:?}, omega={:?}", v_a, omega_a);
    println!("  B (torque only): v={:?}, omega={:?}", v_b, omega_b);
    println!("  C (both): v={:?}, omega={:?}", v_c, omega_c);

    // Decoupling: case C's translation matches A, rotation matches B.
    let dv = (v_c - v_a).length();
    let domega = (omega_c - omega_b).length();

    assert!(
        dv < 1.0e-12,
        "Case C translation should match Case A: |dv|={dv:.3e}"
    );
    assert!(
        domega < 1.0e-12,
        "Case C rotation should match Case B: |domega|={domega:.3e}"
    );

    // Case B should have zero velocity (torque alone doesn't translate).
    assert!(
        v_b.length() < 1.0e-12,
        "Torque-only should produce no translation: |v|={}",
        v_b.length()
    );
    // Case A should have zero angular velocity (force at CoM doesn't rotate).
    assert!(
        omega_a.length() < 1.0e-12,
        "Force at CoM should produce no rotation: |omega|={}",
        omega_a.length()
    );

    // Absolute magnitudes match F=ma and tau=I*alpha analytical predictions.
    let expected_v_x = force.x * t_total / mass;
    let expected_omega_z = tau.z * t_total / i_z;
    assert!((v_a.x - expected_v_x).abs() < 1.0e-12);
    assert!((omega_b.z - expected_omega_z).abs() < 1.0e-12);
}

// ─── Test 4: symmetric +F/-F forcing returns body to rest with residual displacement ───

/// Mirrors JEOD's SIM_force_torque scheduling:
///   From t=7 to t=9 apply force that reverses from +F to -F so that the
///   net velocity change is zero after the symmetric impulse pair.
///
/// Here: 5 seconds at +F, then 5 seconds at -F. Velocity returns to zero,
/// but position remains positive because the body continues to drift while
/// decelerating; analytically, the final displacement is twice the position
/// reached at `t = half_duration`. A symmetric triangle of acceleration is
/// enough to assert that the body behaves linearly.
// non-recipe: symmetric ±F impulse sequence verifies linearity; the
// schedule is two `set_body_external_force` calls bracketing a
// `step_n` count, simpler than instantiating a profile struct.
#[test]
fn tier3_force_symmetric_impulse_returns_to_rest() {
    let mass = 100.0;
    let dt = 0.1;
    let force = DVec3::new(5.0, 0.0, 0.0);
    let half_duration = 5.0;

    let mut sim = make_free_body_3dof(mass, dt);
    sim.set_body_external_force(0, force);
    sim.step_n((half_duration / dt) as usize);
    let v_mid = sim.body(0).trans.velocity;

    sim.set_body_external_force(0, -force);
    sim.step_n((half_duration / dt) as usize);
    let v_end = sim.body(0).trans.velocity;
    let x_end = sim.body(0).trans.position;

    // Velocity should be back to zero within roundoff.
    println!("  v_mid = {v_mid:?}");
    println!("  v_end = {v_end:?}");
    println!("  x_end = {x_end:?}");
    assert!(
        v_end.length() < 1.0e-10,
        "Velocity should return to rest: |v|={}",
        v_end.length()
    );

    // Position must be nonzero (body has drifted).  Analytical:
    //   v during [0,5]:  (F/m)*t peaks at v_mid = 0.25 m/s
    //   x at t=5:        0.5 * a * 5^2 = 0.625 m
    //   v during [5,10]: v_mid - (F/m)*(t-5), zero at t=10
    //   x at t=10:       x(5) + v_mid*5 - 0.5*a*5^2 = 1.25 m
    let expected_x = 2.0 * 0.5 * force.x / mass * half_duration * half_duration;
    let rel = (x_end.x - expected_x).abs() / expected_x;
    println!(
        "  x_end.x = {}, expected = {}, rel_err = {rel:.3e}",
        x_end.x, expected_x
    );
    assert!(
        rel < 1.0e-12,
        "symmetric impulse position rel err {rel:.3e}"
    );
}
