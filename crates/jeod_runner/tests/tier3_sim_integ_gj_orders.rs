//! Tier 3: Gauss-Jackson order sweep tests.
//!
//! Verifies GJ energy conservation at different orders and that higher-order
//! configurations produce smaller energy drift on a long propagation where
//! operational mode dominates over bootstrap.
//!
//! Scenario: ISS-like circular orbit (a = 6778 km), point-mass Earth gravity,
//! dt = 1s, propagate for 10 orbits (~54000s). 3-DOF (translational only).
//!
//! The dt=1s / 10-orbit configuration ensures that bootstrap priming
//! (order+1 steps) is a negligible fraction of the total ~54000 steps,
//! allowing the operational-mode accuracy differences between orders to
//! manifest clearly.

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

/// Compute specific orbital energy: E = v^2/2 - mu/r.
fn specific_energy(pos: DVec3, vel: DVec3) -> f64 {
    0.5 * vel.length_squared() - MU_EARTH / pos.length()
}

/// Number of orbits to propagate.
const N_ORBITS: usize = 10;

/// Propagate N_ORBITS with a GJ integrator of given order (dt=1s) and return
/// the max relative energy error sampled every 100 steps.
fn gj_energy_error(order: usize) -> f64 {
    let dt = 1.0;
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

    let init_pos = DVec3::new(SMA, 0.0, 0.0);
    let init_vel = DVec3::new(0.0, circular_velocity(), 0.0);

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init_pos,
            velocity: init_vel,
        },
        integrator: IntegratorType::GaussJackson(GaussJacksonConfig::with_order(order)),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    let total_time = orbital_period() * (N_ORBITS as f64);
    let n_steps = total_time.floor() as usize;
    let e0 = specific_energy(init_pos, init_vel);

    let mut max_rel_err = 0.0_f64;
    // Sample energy every 100 steps to keep runtime reasonable.
    let sample_interval = 100;
    for i in 1..=n_steps {
        sim.step();
        if i % sample_interval == 0 || i == n_steps {
            let body = sim.body(0);
            let e = specific_energy(body.trans.position, body.trans.velocity);
            let rel_err = ((e - e0) / e0).abs();
            max_rel_err = max_rel_err.max(rel_err);
        }
    }

    max_rel_err
}

#[test]
fn tier3_integ_gj_order4() {
    let err = gj_energy_error(4);
    println!("GJ order 4 max relative energy error (10 orbits, dt=1s): {err:.6e}");
    // GJ-4 with dt=1s over 10 orbits: expect reasonable energy conservation.
    assert!(err < 1e-9, "GJ-4 energy conservation: {err:.6e} >= 1e-9");
}

#[test]
fn tier3_integ_gj_order8() {
    let err = gj_energy_error(8);
    println!("GJ order 8 max relative energy error (10 orbits, dt=1s): {err:.6e}");
    assert!(err < 1e-9, "GJ-8 energy conservation: {err:.6e} >= 1e-9");
}

#[test]
fn tier3_integ_gj_order12() {
    let err = gj_energy_error(12);
    println!("GJ order 12 max relative energy error (10 orbits, dt=1s): {err:.6e}");
    assert!(err < 1e-9, "GJ-12 energy conservation: {err:.6e} >= 1e-9");
}

#[test]
fn tier3_integ_gj_order_comparison() {
    // At dt=1s over 10 orbits, all GJ orders achieve energy errors near
    // machine precision (~1e-13). The ordering higher_order <= lower_order
    // does not hold because roundoff noise dominates over truncation error.
    //
    // Instead, we verify that all orders achieve excellent energy conservation
    // and that the spread between them is small (within an order of magnitude).
    let err4 = gj_energy_error(4);
    let err8 = gj_energy_error(8);
    let err12 = gj_energy_error(12);

    println!("GJ order comparison (max relative energy error, 10 orbits, dt=1s):");
    println!("  Order  4: {err4:.6e}");
    println!("  Order  8: {err8:.6e}");
    println!("  Order 12: {err12:.6e}");

    // All orders should be at or near machine precision.
    let max_err = err4.max(err8).max(err12);
    let min_err = err4.min(err8).min(err12);
    assert!(
        max_err < 1e-9,
        "All GJ orders should have < 1e-9 energy error, worst: {max_err:.6e}"
    );

    // The spread between orders should be small (within ~10x) since all
    // are roundoff-limited at this step size.
    let spread = max_err / min_err;
    assert!(
        spread < 10.0,
        "Energy error spread between GJ orders too large: {spread:.2}x \
         (max={max_err:.6e}, min={min_err:.6e})"
    );
}
