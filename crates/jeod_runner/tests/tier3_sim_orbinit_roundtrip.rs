//! Tier 3: Orbit initialization round-trip tests
//!
//! For each orbit family, initializes from orbital elements, propagates through
//! `Simulation::step()`, then recovers orbital elements from the propagated
//! state and verifies they match the originals.
//!
//! This exercises the full pipeline end-to-end: element-to-Cartesian
//! initialization, RK4 integration with point-mass gravity, and
//! Cartesian-to-element recovery -- ensuring no information is lost
//! or corrupted through the pipeline.

mod sim_test_helpers;

use glam::DVec3;
use jeod_math::OrbitalElements;
use jeod_runner::{GravitySourceEntry, RotationModel, Simulation, VehicleConfig};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, SimulationTime,
    TranslationalState,
};

/// Earth gravitational parameter (m^3/s^2) -- GGM05C value.
const MU_EARTH: f64 = 3.986_004_415e14;

/// Earth equatorial radius (m).
const R_EARTH: f64 = 6_378_137.0;

/// Initialize from classical elements using OrbitalElements directly.
fn state_from_elements(
    a: f64,
    e: f64,
    i: f64,
    raan: f64,
    argp: f64,
    nu: f64,
    mu: f64,
) -> TranslationalState {
    let mut oe = OrbitalElements::default();
    oe.semi_major_axis = a;
    oe.e_mag = e;
    oe.inclination = i;
    oe.long_asc_node = raan;
    oe.arg_periapsis = argp;
    oe.true_anom = nu;

    if e < 1.0 {
        oe.semiparam = a * (1.0 - e * e);
    } else {
        oe.semiparam = a.abs() * (e * e - 1.0);
    }
    oe.nu_to_anomalies();

    let (position, velocity) = oe.to_cartesian(mu).expect("to_cartesian failed");
    TranslationalState { position, velocity }
}

/// Build a Simulation, propagate exactly one full orbit (for bound orbits),
/// recover elements, and verify they match the originals.
///
/// For point-mass gravity, after exactly one orbit all elements should be
/// preserved (the orbit is a fixed Keplerian ellipse).
#[allow(clippy::too_many_arguments)]
fn roundtrip_via_simulation(
    a: f64,
    e: f64,
    i: f64,
    raan: f64,
    argp: f64,
    nu: f64,
    dt: f64,
    n_steps: usize,
    label: &str,
    a_tol: f64,
    e_tol: f64,
    angle_tol: f64,
) {
    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);

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
        trans,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim.step_n(n_steps);

    let body = sim.body(0);
    let oe_recovered =
        OrbitalElements::from_cartesian(MU_EARTH, body.trans.position, body.trans.velocity)
            .expect("from_cartesian failed after propagation");

    println!("  {label}: recovered elements after {n_steps} steps");
    println!(
        "    a: {:.6e} (expected {:.6e}, err {:.3e})",
        oe_recovered.semi_major_axis,
        a,
        (oe_recovered.semi_major_axis - a).abs()
    );
    println!(
        "    e: {:.10} (expected {:.10}, err {:.3e})",
        oe_recovered.e_mag,
        e,
        (oe_recovered.e_mag - e).abs()
    );
    println!(
        "    i: {:.8} rad (expected {:.8}, err {:.3e})",
        oe_recovered.inclination,
        i,
        (oe_recovered.inclination - i).abs()
    );

    // Semi-major axis (relative error)
    let a_err = (oe_recovered.semi_major_axis - a).abs() / a.abs();
    assert!(
        a_err < a_tol,
        "{label}: semi_major_axis relative error {a_err:.6e} exceeds tolerance {a_tol:.1e}"
    );

    // Eccentricity (absolute error -- e can be small or zero)
    let e_err = (oe_recovered.e_mag - e).abs();
    assert!(
        e_err < e_tol,
        "{label}: eccentricity error {e_err:.6e} exceeds tolerance {e_tol:.1e}"
    );

    // Inclination (absolute error in radians)
    let i_err = (oe_recovered.inclination - i).abs();
    assert!(
        i_err < angle_tol,
        "{label}: inclination error {i_err:.6e} rad exceeds tolerance {angle_tol:.1e}"
    );

    // RAAN -- only check for non-equatorial orbits (singular when i~0)
    if i > 1e-6 && (std::f64::consts::PI - i) > 1e-6 {
        let raan_err = angle_diff(oe_recovered.long_asc_node, raan);
        assert!(
            raan_err < angle_tol,
            "{label}: RAAN error {raan_err:.6e} rad exceeds tolerance {angle_tol:.1e}"
        );
    }

    // Argument of periapsis -- only check for non-circular orbits
    if e > 1e-6 {
        let argp_err = angle_diff(oe_recovered.arg_periapsis, argp);
        assert!(
            argp_err < angle_tol,
            "{label}: arg_periapsis error {argp_err:.6e} rad exceeds tolerance {angle_tol:.1e}"
        );
    }
}

use sim_test_helpers::angle_diff;

// ======================================================================
// Circular LEO round-trip
// ======================================================================

#[test]
fn tier3_orbinit_roundtrip_circular() {
    let a = R_EARTH + 400_000.0;
    let e = 0.0;
    let i = 51.6_f64.to_radians();
    let raan = 30.0_f64.to_radians();
    let argp = 0.0;
    let nu = 0.0;

    let period = 2.0 * std::f64::consts::PI * (a * a * a / MU_EARTH).sqrt();
    let dt = 10.0;
    let n_steps = (period / dt).round() as usize;

    println!("Tier 3 round-trip: Circular LEO (period={period:.1} s, {n_steps} steps)");

    // For circular orbit, a and e are special: JEOD uses r_mag for a, e~0
    // After one orbit, the state returns close to the start, so recovered
    // elements should match. Use relaxed e_tol since e~0 is degenerate.
    roundtrip_via_simulation(
        a,
        e,
        i,
        raan,
        argp,
        nu,
        dt,
        n_steps,
        "circular_leo",
        1e-10, // a relative
        1e-8,  // e absolute (degenerate for circular)
        1e-8,  // angle
    );
}

// ======================================================================
// Eccentric orbit round-trip
// ======================================================================

#[test]
fn tier3_orbinit_roundtrip_eccentric() {
    let a = R_EARTH + 2_000_000.0;
    let e = 0.3;
    let i = 28.5_f64.to_radians();
    let raan = 45.0_f64.to_radians();
    let argp = 90.0_f64.to_radians();
    let nu = 0.0; // start at periapsis for clean round-trip

    let period = 2.0 * std::f64::consts::PI * (a * a * a / MU_EARTH).sqrt();
    let dt = 10.0;
    let n_steps = (period / dt).round() as usize;

    println!("Tier 3 round-trip: Eccentric e=0.3 (period={period:.1} s, {n_steps} steps)");

    roundtrip_via_simulation(
        a,
        e,
        i,
        raan,
        argp,
        nu,
        dt,
        n_steps,
        "eccentric_e03",
        1e-10,
        1e-10,
        1e-8,
    );
}

// ======================================================================
// Retrograde orbit round-trip
// ======================================================================

#[test]
fn tier3_orbinit_roundtrip_retrograde() {
    let a = R_EARTH + 800_000.0;
    let e = 0.05;
    let i = 150.0_f64.to_radians();
    let raan = 200.0_f64.to_radians();
    let argp = 30.0_f64.to_radians();
    let nu = 0.0;

    let period = 2.0 * std::f64::consts::PI * (a * a * a / MU_EARTH).sqrt();
    let dt = 10.0;
    let n_steps = (period / dt).round() as usize;

    println!("Tier 3 round-trip: Retrograde (period={period:.1} s, {n_steps} steps)");

    roundtrip_via_simulation(
        a,
        e,
        i,
        raan,
        argp,
        nu,
        dt,
        n_steps,
        "retrograde",
        1e-10,
        1e-10,
        1e-8,
    );
}

// ======================================================================
// Equatorial orbit round-trip
// ======================================================================

#[test]
fn tier3_orbinit_roundtrip_equatorial() {
    let a = R_EARTH + 600_000.0;
    let e = 0.1;
    let i = 0.0;
    let raan = 0.0;
    let argp = 45.0_f64.to_radians();
    let nu = 0.0;

    let period = 2.0 * std::f64::consts::PI * (a * a * a / MU_EARTH).sqrt();
    let dt = 10.0;
    let n_steps = (period / dt).round() as usize;

    println!("Tier 3 round-trip: Equatorial (period={period:.1} s, {n_steps} steps)");

    roundtrip_via_simulation(
        a,
        e,
        i,
        raan,
        argp,
        nu,
        dt,
        n_steps,
        "equatorial",
        1e-10,
        1e-10,
        1e-8,
    );
}

// ======================================================================
// Polar orbit round-trip
// ======================================================================

#[test]
fn tier3_orbinit_roundtrip_polar() {
    let a = R_EARTH + 500_000.0;
    let e = 0.02;
    let i = 90.0_f64.to_radians();
    let raan = 60.0_f64.to_radians();
    let argp = 0.0;
    let nu = 0.0;

    let period = 2.0 * std::f64::consts::PI * (a * a * a / MU_EARTH).sqrt();
    let dt = 10.0;
    let n_steps = (period / dt).round() as usize;

    println!("Tier 3 round-trip: Polar (period={period:.1} s, {n_steps} steps)");

    roundtrip_via_simulation(
        a, e, i, raan, argp, nu, dt, n_steps, "polar", 1e-10, 1e-10, 1e-8,
    );
}

// ======================================================================
// Highly eccentric (Molniya-like) round-trip
// ======================================================================

#[test]
fn tier3_orbinit_roundtrip_molniya() {
    let a = R_EARTH + 10_000_000.0;
    let e = 0.7;
    let i = 63.4_f64.to_radians();
    let raan = 120.0_f64.to_radians();
    let argp = 270.0_f64.to_radians();
    let nu = 0.0;

    let period = 2.0 * std::f64::consts::PI * (a * a * a / MU_EARTH).sqrt();
    let dt = 10.0;
    let n_steps = (period / dt).round() as usize;

    println!("Tier 3 round-trip: Molniya (period={period:.1} s, {n_steps} steps)");

    roundtrip_via_simulation(
        a, e, i, raan, argp, nu, dt, n_steps, "molniya",
        1e-9, // relaxed for high eccentricity
        1e-9, 1e-7,
    );
}

// ======================================================================
// Hyperbolic orbit round-trip (short propagation, recover elements)
// ======================================================================

#[test]
fn tier3_orbinit_roundtrip_hyperbolic() {
    let e = 1.5;
    let r_peri = R_EARTH + 300_000.0;
    let a = -(r_peri / (e - 1.0));
    let i = 30.0_f64.to_radians();
    let raan = 0.0;
    let argp = 0.0;
    let nu = 0.1;

    // Short propagation: 100 steps of 1 second
    let dt = 1.0;
    let n_steps = 100;

    println!("Tier 3 round-trip: Hyperbolic (a={a:.0} m, e={e})");

    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);

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
        trans,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim.step_n(n_steps);

    let body = sim.body(0);
    let oe = OrbitalElements::from_cartesian(MU_EARTH, body.trans.position, body.trans.velocity)
        .expect("from_cartesian failed");

    // For hyperbolic orbits, a is negative
    let a_err = (oe.semi_major_axis - a).abs() / a.abs();
    let e_err = (oe.e_mag - e).abs();
    let i_err = (oe.inclination - i).abs();

    println!("  hyperbolic: a_err={a_err:.3e}, e_err={e_err:.3e}, i_err={i_err:.3e}");

    assert!(
        a_err < 1e-10,
        "Hyperbolic a relative error {a_err:.6e} exceeds tolerance"
    );
    assert!(
        e_err < 1e-10,
        "Hyperbolic e error {e_err:.6e} exceeds tolerance"
    );
    assert!(
        i_err < 1e-10,
        "Hyperbolic i error {i_err:.6e} exceeds tolerance"
    );
}
