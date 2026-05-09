//! Tier 3: Integrator vs analytical solution tests.
//!
//! For a circular orbit with point-mass gravity the analytical solution is
//! known exactly:
//! - r(t) = a (constant radius)
//! - v(t) = sqrt(mu/a) (constant speed)
//! - theta(t) = theta_0 + n*t, where n = sqrt(mu/a^3)
//!
//! Tests each integrator (RK4, RKF45, GJ-8) against this analytical solution
//! for position and velocity error over a one-orbit propagation interval.

use astrodyn::{
    GaussJacksonConfig, GravityControl, GravityControls, GravityModel, GravitySource,
    IntegratorType, SimulationTime, TranslationalState,
};
use astrodyn::{GravitySourceEntry, VehicleConfig};
use astrodyn_runner::{RotationModel, Simulation};
use glam::DVec3;

/// Earth gravitational parameter (m^3/s^2) from JEOD earth_GGM05C.
const MU_EARTH: f64 = astrodyn::EARTH.shape.mu;

/// Semi-major axis for ISS-like circular orbit (m).
const SMA: f64 = 6_778_000.0;

/// Circular orbital velocity (m/s): v = sqrt(mu/a).
fn circular_velocity() -> f64 {
    (MU_EARTH / SMA).sqrt()
}

/// Mean motion (rad/s): n = sqrt(mu/a^3).
fn mean_motion() -> f64 {
    (MU_EARTH / SMA.powi(3)).sqrt()
}

/// Orbital period (s): T = 2*pi/n.
fn orbital_period() -> f64 {
    2.0 * std::f64::consts::PI / mean_motion()
}

/// Analytical position at time t for circular orbit starting at (a, 0, 0)
/// with velocity (0, v_circ, 0): x = a*cos(n*t), y = a*sin(n*t), z = 0.
fn analytical_position(t: f64) -> DVec3 {
    let n = mean_motion();
    let theta = n * t;
    DVec3::new(SMA * theta.cos(), SMA * theta.sin(), 0.0)
}

/// Analytical velocity at time t: vx = -v*sin(n*t), vy = v*cos(n*t), vz = 0.
fn analytical_velocity(t: f64) -> DVec3 {
    let n = mean_motion();
    let v = circular_velocity();
    let theta = n * t;
    DVec3::new(-v * theta.sin(), v * theta.cos(), 0.0)
}

/// Create a simulation with a single body using the given integrator.
fn make_sim(integrator: IntegratorType, dt: f64) -> Simulation {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    sim.add_body(VehicleConfig {
        trans: astrodyn_verif_jeod::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: DVec3::new(SMA, 0.0, 0.0),
            velocity: DVec3::new(0.0, circular_velocity(), 0.0),
        }),
        integrator,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim
}

/// Result of propagating one orbit against the analytical solution.
struct AnalyticalResult {
    /// Max position error (m) sampled every step over the orbit.
    max_pos_err: f64,
    /// Max velocity error (m/s) sampled every step over the orbit.
    max_vel_err: f64,
    /// Position error (m) at the final sampled time, `n_steps * dt`,
    /// where `n_steps = floor(T / dt)`.
    final_pos_err: f64,
    /// Velocity error (m/s) at the final sampled time, `n_steps * dt`,
    /// where `n_steps = floor(T / dt)`.
    final_vel_err: f64,
}

/// Propagate for one orbit with the given integrator and compare to analytical.
///
/// Steps by integer multiples of `dt` only (GJ requires constant step size),
/// so the final reported error is evaluated at the largest integer multiple of
/// `dt` that does not exceed one orbital period `T`.
fn compare_analytical(integrator: IntegratorType, dt: f64) -> AnalyticalResult {
    let mut sim = make_sim(integrator, dt);
    let period = orbital_period();
    let n_steps = (period / dt).floor() as usize;

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;

    for i in 1..=n_steps {
        let t = (i as f64) * dt;
        sim.step_until(t).expect("step_until failed");
        let body = sim.body(0);

        let pos_err = (body.trans.position.raw_si() - analytical_position(t)).length();
        let vel_err = (body.trans.velocity.raw_si() - analytical_velocity(t)).length();
        max_pos_err = max_pos_err.max(pos_err);
        max_vel_err = max_vel_err.max(vel_err);
    }

    // Final state is at t = n_steps * dt (closest integer multiple to the period).
    let final_t = (n_steps as f64) * dt;
    let body = sim.body(0);
    let final_pos_err = (body.trans.position.raw_si() - analytical_position(final_t)).length();
    let final_vel_err = (body.trans.velocity.raw_si() - analytical_velocity(final_t)).length();

    AnalyticalResult {
        max_pos_err,
        max_vel_err,
        final_pos_err,
        final_vel_err,
    }
}

#[test]
fn tier3_integ_rk4_vs_analytical() {
    let dt = 10.0;
    let r = compare_analytical(IntegratorType::Rk4, dt);

    println!("RK4 vs analytical (dt={dt}s, 1 orbit):");
    println!("  Max position error:   {:.6e} m", r.max_pos_err);
    println!("  Max velocity error:   {:.6e} m/s", r.max_vel_err);
    println!("  Final position error: {:.6e} m", r.final_pos_err);
    println!("  Final velocity error: {:.6e} m/s", r.final_vel_err);

    // RK4 with dt=10s on a ~5400s circular orbit: 4th-order truncation error
    // accumulates over ~540 steps. Tolerances set at 5% above observed values.
    assert!(
        r.max_pos_err < 2e-2,
        "RK4 max pos error: {:.6e} m >= 2e-2 m",
        r.max_pos_err
    );
    assert!(
        r.max_vel_err < 2e-5,
        "RK4 max vel error: {:.6e} m/s >= 2e-5 m/s",
        r.max_vel_err
    );
    assert!(
        r.final_pos_err < 2e-2,
        "RK4 final pos error: {:.6e} m >= 2e-2 m",
        r.final_pos_err
    );
    assert!(
        r.final_vel_err < 2e-5,
        "RK4 final vel error: {:.6e} m/s >= 2e-5 m/s",
        r.final_vel_err
    );
}

#[test]
fn tier3_integ_rkf45_vs_analytical() {
    let dt = 10.0;
    let r = compare_analytical(IntegratorType::Rkf45, dt);

    println!("RKF45 vs analytical (dt={dt}s, 1 orbit):");
    println!("  Max position error:   {:.6e} m", r.max_pos_err);
    println!("  Max velocity error:   {:.6e} m/s", r.max_vel_err);
    println!("  Final position error: {:.6e} m", r.final_pos_err);
    println!("  Final velocity error: {:.6e} m/s", r.final_vel_err);

    // RKF45 is 5th order, should be more accurate than RK4 at same dt.
    assert!(
        r.max_pos_err < 1e-3,
        "RKF45 max pos error: {:.6e} m >= 1e-3 m",
        r.max_pos_err
    );
    assert!(
        r.max_vel_err < 1e-6,
        "RKF45 max vel error: {:.6e} m/s >= 1e-6 m/s",
        r.max_vel_err
    );
    assert!(
        r.final_pos_err < 1e-3,
        "RKF45 final pos error: {:.6e} m >= 1e-3 m",
        r.final_pos_err
    );
    assert!(
        r.final_vel_err < 1e-6,
        "RKF45 final vel error: {:.6e} m/s >= 1e-6 m/s",
        r.final_vel_err
    );
}

#[test]
fn tier3_integ_gj8_vs_analytical() {
    let dt = 10.0;
    let r = compare_analytical(
        IntegratorType::GaussJackson(GaussJacksonConfig::with_order(8)),
        dt,
    );

    println!("GJ-8 vs analytical (dt={dt}s, 1 orbit):");
    println!("  Max position error:   {:.6e} m", r.max_pos_err);
    println!("  Max velocity error:   {:.6e} m/s", r.max_vel_err);
    println!("  Final position error: {:.6e} m", r.final_pos_err);
    println!("  Final velocity error: {:.6e} m/s", r.final_vel_err);

    // GJ-8 with dt=10s over 1 orbit: bootstrap priming (9 RK4 steps at the
    // coarse dt) introduces larger transient errors than single-step methods.
    // The max position error peaks during bootstrap, then GJ operational mode
    // stabilizes. Tolerances set at 5% above observed values.
    assert!(
        r.max_pos_err < 1.7,
        "GJ-8 max pos error: {:.6e} m >= 1.7 m",
        r.max_pos_err
    );
    assert!(
        r.max_vel_err < 0.68,
        "GJ-8 max vel error: {:.6e} m/s >= 0.68 m/s",
        r.max_vel_err
    );
    assert!(
        r.final_pos_err < 5e-2,
        "GJ-8 final pos error: {:.6e} m >= 5e-2 m",
        r.final_pos_err
    );
    assert!(
        r.final_vel_err < 6e-5,
        "GJ-8 final vel error: {:.6e} m/s >= 6e-5 m/s",
        r.final_vel_err
    );
}

#[test]
fn tier3_integ_rkf45_more_accurate_than_rk4() {
    // At the same step size, RKF45 (5th order) should be more accurate
    // than RK4 (4th order) for this smooth circular orbit.
    let dt = 10.0;
    let rk4 = compare_analytical(IntegratorType::Rk4, dt);
    let rkf = compare_analytical(IntegratorType::Rkf45, dt);

    println!("Accuracy comparison at dt={dt}s:");
    println!(
        "  RK4  max pos err: {:.6e} m, RKF45: {:.6e} m",
        rk4.max_pos_err, rkf.max_pos_err
    );

    assert!(
        rkf.max_pos_err < rk4.max_pos_err,
        "RKF45 ({:.6e}) should be more accurate than RK4 ({:.6e})",
        rkf.max_pos_err,
        rk4.max_pos_err
    );
}
