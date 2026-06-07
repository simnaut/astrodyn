//! Tier 3: Gauss-Jackson order sweep tests.

#![allow(
    clippy::float_cmp,
    reason = "Tier 3 tests assert bit-exact recovery of literal-built / analytic state values"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Tier 3 step counts and indices fit exactly in f64 mantissa and usize"
)]
//!
//! Verifies GJ energy conservation at different orders by comparing the
//! maximum relative energy error over the full propagation, including the
//! bootstrap priming phase.
//!
//! Scenario: ISS-like circular orbit (a = 6778 km), point-mass Earth gravity,
//! dt = 1s, propagate for 10 orbits (~54000s). 3-DOF (translational only).
//!
//! The dt=1s / 10-orbit configuration yields a long operational-mode
//! propagation (~54000 steps total), but the peak energy error is produced
//! during the bootstrap priming phase (order+1 RK4 steps). Higher-order GJ
//! needs more priming steps, so the asserted max error can be larger at
//! higher order due to larger bootstrap transients.

use astrodyn::{
    GaussJacksonConfig, GravityControl, GravityControls, GravityGradient, GravityModel,
    GravitySource, IntegratorType, SimulationTime, TranslationalState,
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
/// the max relative energy error computed every step.
fn gj_energy_error(order: usize) -> f64 {
    let dt = 1.0;
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

    let _earth = sim.add_source(
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
            marker_only: false,
        },
    );

    let init_pos = DVec3::new(SMA, 0.0, 0.0);
    let init_vel = DVec3::new(0.0, circular_velocity(), 0.0);

    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: init_pos,
            velocity: init_vel,
        }),
        integrator: IntegratorType::GaussJackson(
            // JEOD-faithful warn-and-continue (#485 C1): synthetic GJ-order
            // sweep occasionally trips convergence on shorter orders; the
            // test asserts integrator order-dependent error, not numerical
            // panic-by-default behavior.
            GaussJacksonConfig::with_order(order).with_allow_non_convergence(true),
        ),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("tier3-sim-integ-gj-orders-0")
    });

    sim.validate().unwrap();

    let total_time = orbital_period() * (N_ORBITS as f64);
    let n_steps = total_time.floor() as usize;
    let e0 = specific_energy(init_pos, init_vel);

    let mut max_rel_err = 0.0_f64;
    for _ in 1..=n_steps {
        sim.step().expect("step failed");
        let body = sim.body(0);
        let e = specific_energy(body.trans.position.raw_si(), body.trans.velocity.raw_si());
        let rel_err = ((e - e0) / e0).abs();
        max_rel_err = max_rel_err.max(rel_err);
    }

    max_rel_err
}

/// Compute energy errors for GJ orders 4, 8, and 12 in a single pass.
/// Each order runs the full 10-orbit propagation exactly once.
fn gj_order_errors() -> (f64, f64, f64) {
    static CACHE: std::sync::OnceLock<(f64, f64, f64)> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        let err4 = gj_energy_error(4);
        let err8 = gj_energy_error(8);
        let err12 = gj_energy_error(12);
        (err4, err8, err12)
    })
}

#[test]
fn tier3_integ_gj_order4() {
    let (err, _, _) = gj_order_errors();
    println!("GJ order 4 max relative energy error (10 orbits, dt=1s): {err:.6e}");
    // GJ-4 with dt=1s over 10 orbits: bootstrap priming (5 RK4 steps) produces
    // a transient energy spike; operational mode is much better. Tolerance
    // 5% above observed max.
    assert!(
        err < 4.1e-7,
        "GJ-4 energy conservation: {err:.6e} >= 4.1e-7"
    );
}

#[test]
fn tier3_integ_gj_order8() {
    let (_, err, _) = gj_order_errors();
    println!("GJ order 8 max relative energy error (10 orbits, dt=1s): {err:.6e}");
    // GJ-8: 9 RK4 priming steps -> larger bootstrap spike. Tolerance
    // 5% above observed max.
    assert!(
        err < 1.76e-6,
        "GJ-8 energy conservation: {err:.6e} >= 1.76e-6"
    );
}

#[test]
fn tier3_integ_gj_order12() {
    let (_, _, err) = gj_order_errors();
    println!("GJ order 12 max relative energy error (10 orbits, dt=1s): {err:.6e}");
    // GJ-12: 13 RK4 priming steps -> largest bootstrap spike. Tolerance
    // 5% above observed max.
    assert!(
        err < 1.31e-5,
        "GJ-12 energy conservation: {err:.6e} >= 1.31e-5"
    );
}

#[test]
fn tier3_integ_gj_order_comparison() {
    // At dt=1s over 10 orbits, the peak energy error for each order is
    // dominated by the bootstrap priming phase (order+1 RK4 steps at the
    // coarse dt). Higher orders require more priming steps and thus exhibit
    // *larger* peak energy errors — the opposite of what one might expect
    // from truncation error alone.
    //
    // We verify that all orders conserve energy within acceptable bounds
    // and that the spread between them is bounded.
    let (err4, err8, err12) = gj_order_errors();

    println!("GJ order comparison (max relative energy error, 10 orbits, dt=1s):");
    println!("  Order  4: {err4:.6e}");
    println!("  Order  8: {err8:.6e}");
    println!("  Order 12: {err12:.6e}");

    // All orders should conserve energy well despite bootstrap transients.
    // Higher orders have more priming steps and thus larger bootstrap spikes.
    let max_err = err4.max(err8).max(err12);
    let min_err = err4.min(err8).min(err12);
    assert!(
        max_err < 1.31e-5,
        "All GJ orders should have < 1.31e-5 energy error, worst: {max_err:.6e}"
    );

    // The spread between orders reflects different bootstrap spike magnitudes.
    // Higher-order GJ needs more priming steps, producing larger transient
    // energy errors. Observed spread ~32x; tolerance at 5% above.
    if min_err > 0.0 {
        let spread = max_err / min_err;
        assert!(
            spread < 34.0,
            "Energy error spread between GJ orders too large: {spread:.2}x \
             (max={max_err:.6e}, min={min_err:.6e})"
        );
    } else {
        assert!(
            max_err <= f64::EPSILON,
            "Energy errors include an exact zero while worst error is not \
             effectively zero (max={max_err:.6e}, min={min_err:.6e})"
        );
    }
}
