//! Tier 3: Integrator comparison tests.
//!
//! Compares RK4, RKF45, and Gauss-Jackson integrators against each other
//! and verifies energy conservation on the same circular LEO scenario.
//!
//! Scenario: ISS-like circular orbit (a = 6778 km), point-mass Earth gravity,
//! dt = 10s, propagate for 1 orbit (~5400s). 3-DOF (translational only).

mod sim_test_helpers;

use glam::DVec3;
use jeod_runner::{GravitySourceEntry, RotationModel, Simulation, VehicleConfig};
use jeod_sim::{
    GaussJacksonConfig, GravityControl, GravityControls, GravityModel, GravitySource,
    IntegratorType, SimulationTime, TranslationalState,
};

/// Earth gravitational parameter (m^3/s^2) from JEOD earth_GGM05C.
const MU_EARTH: f64 = 398_600.441_50e9;

/// Semi-major axis for ISS-like circular orbit (m).
const SMA: f64 = 6_778_000.0;

/// Circular orbital velocity (m/s): v = sqrt(mu/a).
fn circular_velocity() -> f64 {
    (MU_EARTH / SMA).sqrt()
}

/// Orbital period (s): T = 2*pi*sqrt(a^3/mu).
fn orbital_period() -> f64 {
    2.0 * std::f64::consts::PI * (SMA.powi(3) / MU_EARTH).sqrt()
}

/// Initial position: [a, 0, 0] m.
fn init_position() -> DVec3 {
    DVec3::new(SMA, 0.0, 0.0)
}

/// Initial velocity: [0, v_circ, 0] m/s.
fn init_velocity() -> DVec3 {
    DVec3::new(0.0, circular_velocity(), 0.0)
}

/// Compute specific orbital energy: E = v^2/2 - mu/r.
fn specific_energy(pos: DVec3, vel: DVec3) -> f64 {
    0.5 * vel.length_squared() - MU_EARTH / pos.length()
}

/// Create a simulation with a single body using the given integrator type.
fn make_sim(integrator: IntegratorType, dt: f64) -> Simulation {
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init_position(),
            velocity: init_velocity(),
        },
        integrator,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim
}

/// Propagate for one orbit and return (final_position, final_velocity, max_relative_energy_error).
///
/// Steps by integer multiples of dt only (GJ requires constant step size).
/// The propagation covers `n_steps` full steps where `n_steps = floor(period/dt)`.
fn propagate_one_orbit(integrator: IntegratorType, dt: f64) -> (DVec3, DVec3, f64) {
    let mut sim = make_sim(integrator, dt);
    let period = orbital_period();
    let e0 = specific_energy(init_position(), init_velocity());
    let n_steps = (period / dt).floor() as usize;

    let mut max_rel_energy_err = 0.0_f64;

    for i in 1..=n_steps {
        let t = (i as f64) * dt;
        sim.step_until(t);
        let body = sim.body(0);
        let e = specific_energy(body.trans.position, body.trans.velocity);
        let rel_err = ((e - e0) / e0).abs();
        max_rel_energy_err = max_rel_energy_err.max(rel_err);
    }

    let body = sim.body(0);
    (body.trans.position, body.trans.velocity, max_rel_energy_err)
}

// ══════════════════════════════════════════════════════════════════════════════
// Energy conservation tests
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn tier3_integ_rk4_energy_conservation() {
    let dt = 10.0;
    let (_, _, max_rel_err) = propagate_one_orbit(IntegratorType::Rk4, dt);
    println!("RK4 max relative energy error: {max_rel_err:.6e}");
    // RK4 4th order with dt=10s on a ~5400s orbit: expect < 1e-10 relative.
    assert!(
        max_rel_err < 1e-10,
        "RK4 energy conservation: {max_rel_err:.6e} >= 1e-10"
    );
}

#[test]
fn tier3_integ_rkf45_energy_conservation() {
    let dt = 10.0;
    let (_, _, max_rel_err) = propagate_one_orbit(IntegratorType::Rkf45, dt);
    println!("RKF45 max relative energy error: {max_rel_err:.6e}");
    // RKF45 5th order: should be at least as good as RK4.
    assert!(
        max_rel_err < 1e-10,
        "RKF45 energy conservation: {max_rel_err:.6e} >= 1e-10"
    );
}

#[test]
fn tier3_integ_gj_energy_conservation() {
    let dt = 10.0;
    let (_, _, max_rel_err) = propagate_one_orbit(
        IntegratorType::GaussJackson(GaussJacksonConfig::with_order(8)),
        dt,
    );
    println!("GJ-8 max relative energy error: {max_rel_err:.6e}");
    // GJ-8 with dt=10s over 1 orbit (~540 steps): the bootstrap/priming phase
    // (order+1 RK4 steps) is a significant fraction of the total run, so energy
    // conservation is worse than on long runs where operational mode dominates.
    assert!(
        max_rel_err < 2e-4,
        "GJ-8 energy conservation: {max_rel_err:.6e} >= 2e-4"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Cross-integrator agreement tests
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn tier3_integ_rk4_vs_rkf45_agreement() {
    let dt = 10.0;
    let (pos_rk4, vel_rk4, _) = propagate_one_orbit(IntegratorType::Rk4, dt);
    let (pos_rkf, vel_rkf, _) = propagate_one_orbit(IntegratorType::Rkf45, dt);

    let pos_diff = (pos_rk4 - pos_rkf).length();
    let vel_diff = (vel_rk4 - vel_rkf).length();
    println!("RK4 vs RKF45 after 1 orbit: pos_diff={pos_diff:.6e} m, vel_diff={vel_diff:.6e} m/s");

    // Both are single-step Runge-Kutta methods with dt=10s on a smooth orbit.
    // They should agree well but not exactly (4th vs 5th order).
    assert!(
        pos_diff < 1.0,
        "RK4 vs RKF45 position: {pos_diff:.6e} m >= 1.0 m"
    );
    assert!(
        vel_diff < 1e-3,
        "RK4 vs RKF45 velocity: {vel_diff:.6e} m/s >= 1e-3 m/s"
    );
}

#[test]
fn tier3_integ_rk4_vs_gj_agreement() {
    let dt = 10.0;
    let (pos_rk4, vel_rk4, _) = propagate_one_orbit(IntegratorType::Rk4, dt);
    let (pos_gj, vel_gj, _) = propagate_one_orbit(
        IntegratorType::GaussJackson(GaussJacksonConfig::with_order(8)),
        dt,
    );

    let pos_diff = (pos_rk4 - pos_gj).length();
    let vel_diff = (vel_rk4 - vel_gj).length();
    println!("RK4 vs GJ-8 after 1 orbit: pos_diff={pos_diff:.6e} m, vel_diff={vel_diff:.6e} m/s");

    // RK4 and GJ-8 should agree reasonably well for smooth circular orbit.
    assert!(
        pos_diff < 1.0,
        "RK4 vs GJ-8 position: {pos_diff:.6e} m >= 1.0 m"
    );
    assert!(
        vel_diff < 1e-3,
        "RK4 vs GJ-8 velocity: {vel_diff:.6e} m/s >= 1e-3 m/s"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// Convergence order tests
// ══════════════════════════════════════════════════════════════════════════════

/// Propagate for a fixed duration and return the final position.
fn propagate_fixed_time(integrator: IntegratorType, dt: f64, duration: f64) -> DVec3 {
    let mut sim = make_sim(integrator, dt);
    sim.step_until(duration);
    sim.body(0).trans.position
}

#[test]
fn tier3_integ_rk4_convergence_order() {
    // Run with dt and dt/2. For a p-th order method, the error ratio
    // at halved step size is ~2^p. For RK4 (p=4), expect ratio ~16.
    //
    // We use the analytical solution for circular orbit as reference:
    // after time t, the satellite returns to (a, 0, 0) at t = T.
    // We propagate for half an orbit (T/2) and compare to analytical position.
    let period = orbital_period();
    let half_period = period / 2.0;

    // Analytical position at T/2: (-a, 0, 0) for circular orbit starting at (a,0,0).
    let analytical_pos = DVec3::new(-SMA, 0.0, 0.0);

    let dt1 = 20.0;
    let dt2 = 10.0;

    let pos1 = propagate_fixed_time(IntegratorType::Rk4, dt1, half_period);
    let pos2 = propagate_fixed_time(IntegratorType::Rk4, dt2, half_period);

    let err1 = (pos1 - analytical_pos).length();
    let err2 = (pos2 - analytical_pos).length();

    println!("RK4 convergence: dt={dt1} err={err1:.6e}, dt={dt2} err={err2:.6e}");

    // Avoid division by zero if err2 is extremely small.
    assert!(
        err2 > 1e-15,
        "RK4 dt/2 error too small to measure convergence: {err2:.6e}"
    );

    let ratio = err1 / err2;
    println!("RK4 convergence ratio: {ratio:.2} (expected ~16 for 4th order)");

    // The ratio should be approximately 2^4 = 16 for a 4th-order method.
    // Allow a generous range because the orbit is not exactly polynomial.
    assert!(
        ratio > 8.0,
        "RK4 convergence ratio {ratio:.2} < 8 (expected ~16)"
    );
    assert!(
        ratio < 32.0,
        "RK4 convergence ratio {ratio:.2} > 32 (expected ~16)"
    );
}

#[test]
fn tier3_integ_rkf45_convergence_order() {
    // RKF45 is 5th order. Error ratio at halved step size should be ~2^5 = 32.
    let period = orbital_period();
    let half_period = period / 2.0;
    let analytical_pos = DVec3::new(-SMA, 0.0, 0.0);

    let dt1 = 20.0;
    let dt2 = 10.0;

    let pos1 = propagate_fixed_time(IntegratorType::Rkf45, dt1, half_period);
    let pos2 = propagate_fixed_time(IntegratorType::Rkf45, dt2, half_period);

    let err1 = (pos1 - analytical_pos).length();
    let err2 = (pos2 - analytical_pos).length();

    println!("RKF45 convergence: dt={dt1} err={err1:.6e}, dt={dt2} err={err2:.6e}");

    assert!(
        err2 > 1e-15,
        "RKF45 dt/2 error too small to measure convergence: {err2:.6e}"
    );

    let ratio = err1 / err2;
    println!("RKF45 convergence ratio: {ratio:.2} (expected ~32 for 5th order)");

    // Allow generous range for non-polynomial dynamics.
    assert!(
        ratio > 16.0,
        "RKF45 convergence ratio {ratio:.2} < 16 (expected ~32)"
    );
    assert!(
        ratio < 64.0,
        "RKF45 convergence ratio {ratio:.2} > 64 (expected ~32)"
    );
}
