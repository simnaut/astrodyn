//! Tier 3: Orbit initialization families -- conservation verification
//!
//! Exercises `Simulation::step()` end-to-end for diverse orbit families:
//! circular, eccentric, retrograde, equatorial, polar, hyperbolic, and
//! near-parabolic. Since no Docker reference data exists for these cases,
//! verification uses analytical invariants:
//!
//! - Specific orbital energy conservation: E = v^2/2 - mu/r
//! - Specific angular momentum conservation: |h| = |r x v|
//! - Radius constancy for circular orbits
//! - Periapsis/apoapsis radius bounds for elliptic orbits

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

/// Build a Simulation with point-mass Earth gravity and a single body
/// at the given translational state. Returns the simulation ready to step.
fn build_sim(trans: TranslationalState, dt: f64) -> Simulation {
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
    sim
}

/// Compute specific orbital energy: E = v^2/2 - mu/r.
fn specific_energy(pos: DVec3, vel: DVec3, mu: f64) -> f64 {
    vel.length_squared() / 2.0 - mu / pos.length()
}

/// Compute specific angular momentum magnitude: |h| = |r x v|.
fn specific_ang_momentum(pos: DVec3, vel: DVec3) -> f64 {
    pos.cross(vel).length()
}

/// Initialize from classical elements using OrbitalElements directly.
/// Works for all eccentricities including e >= 1.
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
        // Hyperbolic: a is negative, p = a(1 - e^2) = |a|(e^2 - 1)
        oe.semiparam = a.abs() * (e * e - 1.0);
    }
    oe.nu_to_anomalies();

    let (position, velocity) = oe.to_cartesian(mu).expect("to_cartesian failed");
    TranslationalState { position, velocity }
}

/// Propagate and verify energy and angular momentum conservation.
///
/// Returns (max_energy_error, max_h_rel_error) for additional assertions.
/// Energy error is relative when |E₀| is large, but switches to absolute
/// error normalized by mu/r₀ when |E₀| is small (near-parabolic orbits
/// where E₀ ≈ 0 makes relative error ill-conditioned).
fn verify_conservation(
    sim: &mut Simulation,
    n_steps: usize,
    label: &str,
    energy_tol: f64,
    h_tol: f64,
) -> (f64, f64) {
    let body0 = sim.body(0);
    let e0 = specific_energy(body0.trans.position, body0.trans.velocity, MU_EARTH);
    let h0 = specific_ang_momentum(body0.trans.position, body0.trans.velocity);

    // For near-parabolic orbits, |E₀| can be near zero, making relative
    // energy error ill-conditioned (inf/NaN). Use mu/r₀ as a stable scale.
    let r0 = body0.trans.position.length();
    let energy_scale = if e0.abs() > MU_EARTH / r0 * 1e-6 {
        e0.abs() // standard relative error
    } else {
        MU_EARTH / r0 // stable scale for near-parabolic
    };

    let mut max_e_err = 0.0_f64;
    let mut max_h_err = 0.0_f64;

    for step in 1..=n_steps {
        sim.step();
        let body = sim.body(0);
        let e_now = specific_energy(body.trans.position, body.trans.velocity, MU_EARTH);
        let h_now = specific_ang_momentum(body.trans.position, body.trans.velocity);

        let e_err = ((e_now - e0) / energy_scale).abs();
        let h_rel = ((h_now - h0) / h0).abs();

        max_e_err = max_e_err.max(e_err);
        max_h_err = max_h_err.max(h_rel);

        if step % 100 == 0 || step == n_steps {
            println!(
                "  {label} step {step}/{n_steps}: E_err={e_err:.3e}, h_rel={h_rel:.3e}, \
                 r={:.1} km, v={:.3} km/s",
                body.trans.position.length() / 1000.0,
                body.trans.velocity.length() / 1000.0,
            );
        }
    }

    println!(
        "  {label}: max E_err={max_e_err:.3e} (tol {energy_tol:.1e}), \
         max h_rel={max_h_err:.3e} (tol {h_tol:.1e})"
    );

    assert!(
        max_e_err < energy_tol,
        "{label}: energy conservation failed: max error {max_e_err:.6e} \
         exceeds tolerance {energy_tol:.1e}"
    );
    assert!(
        max_h_err < h_tol,
        "{label}: angular momentum conservation failed: max relative error {max_h_err:.6e} \
         exceeds tolerance {h_tol:.1e}"
    );

    (max_e_err, max_h_err)
}

// ======================================================================
// Circular LEO
// ======================================================================

#[test]
fn tier3_orbinit_circular_leo() {
    let alt = 400_000.0; // 400 km
    let r = R_EARTH + alt;
    let a = r;
    let e = 0.0;
    let i = 51.6_f64.to_radians(); // ISS-like inclination
    let raan = 30.0_f64.to_radians();
    let argp = 0.0;
    let nu = 0.0;

    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);

    // Orbital period ~ 2*pi*sqrt(a^3/mu) ~ 5554 s. Propagate 2 orbits.
    let dt = 10.0;
    let period = 2.0 * std::f64::consts::PI * (a * a * a / MU_EARTH).sqrt();
    let n_steps = (2.0 * period / dt).ceil() as usize;

    let mut sim = build_sim(trans, dt);

    println!(
        "Tier 3: Circular LEO (a={a:.0} m, e={e}, i={:.1} deg)",
        i.to_degrees()
    );
    println!("  Period={period:.1} s, dt={dt} s, n_steps={n_steps}");

    verify_conservation(&mut sim, n_steps, "circular_leo", 1e-10, 1e-10);

    // Additional check: radius should stay nearly constant for circular orbit
    // over the full propagation window, not just at the final sample.
    let mut radius_sim = build_sim(trans, dt);
    let mut min_r = r;
    let mut max_r = r;

    for _ in 0..n_steps {
        radius_sim.step();
        let body = radius_sim.body(0);
        let r_now = body.trans.position.length();
        min_r = min_r.min(r_now);
        max_r = max_r.max(r_now);
    }

    let max_rel_err = ((max_r - r).abs().max((r - min_r).abs())) / r;
    println!(
        "  Radius: initial={r:.1} m, min={min_r:.1} m, max={max_r:.1} m, max_rel_err={max_rel_err:.3e}"
    );
    assert!(
        max_rel_err < 1e-8,
        "Circular orbit radius varied during propagation: min={min_r:.6e}, max={max_r:.6e}, max_rel_err={max_rel_err:.6e}"
    );
}

// ======================================================================
// Eccentric orbit (e=0.3)
// ======================================================================

#[test]
fn tier3_orbinit_eccentric() {
    let a = R_EARTH + 2_000_000.0; // ~8378 km semi-major axis
    let e = 0.3;
    let i = 28.5_f64.to_radians(); // Cape Canaveral latitude
    let raan = 45.0_f64.to_radians();
    let argp = 90.0_f64.to_radians();
    let nu = 60.0_f64.to_radians();

    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);

    let dt = 10.0;
    let period = 2.0 * std::f64::consts::PI * (a * a * a / MU_EARTH).sqrt();
    let n_steps = (2.0 * period / dt).ceil() as usize;

    let mut sim = build_sim(trans, dt);

    println!(
        "Tier 3: Eccentric orbit (a={:.0} m, e={e}, i={:.1} deg)",
        a,
        i.to_degrees()
    );

    verify_conservation(&mut sim, n_steps, "eccentric_e03", 2.2e-10, 1e-10);

    // Verify periapsis/apoapsis bounds over the full propagation window.
    let r_peri = a * (1.0 - e);
    let r_apo = a * (1.0 + e);
    let mut bounds_sim = build_sim(trans, dt);
    let mut min_r = f64::MAX;
    let mut max_r = 0.0_f64;

    for _ in 0..n_steps {
        bounds_sim.step();
        let body = bounds_sim.body(0);
        let r_now = body.trans.position.length();
        min_r = min_r.min(r_now);
        max_r = max_r.max(r_now);
    }

    println!(
        "  Radius bounds: min={min_r:.1} m (peri={r_peri:.1}), max={max_r:.1} m (apo={r_apo:.1})"
    );
    assert!(
        min_r >= r_peri * 0.999 && max_r <= r_apo * 1.001,
        "Radius outside [{r_peri:.0}, {r_apo:.0}] m bounds: min={min_r:.0}, max={max_r:.0}"
    );
}

// ======================================================================
// Highly eccentric orbit (e=0.7)
// ======================================================================

#[test]
fn tier3_orbinit_highly_eccentric() {
    let a = R_EARTH + 10_000_000.0; // ~16378 km
    let e = 0.7;
    let i = 63.4_f64.to_radians(); // Molniya inclination
    let raan = 120.0_f64.to_radians();
    let argp = 270.0_f64.to_radians();
    let nu = 0.0; // at periapsis

    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);

    let dt = 10.0;
    let period = 2.0 * std::f64::consts::PI * (a * a * a / MU_EARTH).sqrt();
    let n_steps = (1.0 * period / dt).ceil() as usize; // 1 orbit

    let mut sim = build_sim(trans, dt);

    println!(
        "Tier 3: Highly eccentric (a={:.0} m, e={e}, i={:.1} deg)",
        a,
        i.to_degrees()
    );

    verify_conservation(&mut sim, n_steps, "eccentric_e07", 5.2e-9, 1e-10);
}

// ======================================================================
// Retrograde orbit (i > 90 deg)
// ======================================================================

#[test]
fn tier3_orbinit_retrograde() {
    let a = R_EARTH + 800_000.0; // ~7178 km
    let e = 0.05;
    let i = 150.0_f64.to_radians(); // retrograde
    let raan = 200.0_f64.to_radians();
    let argp = 30.0_f64.to_radians();
    let nu = 180.0_f64.to_radians(); // at apoapsis

    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);

    let dt = 10.0;
    let period = 2.0 * std::f64::consts::PI * (a * a * a / MU_EARTH).sqrt();
    let n_steps = (2.0 * period / dt).ceil() as usize;

    let mut sim = build_sim(trans, dt);

    println!(
        "Tier 3: Retrograde orbit (a={:.0} m, e={e}, i={:.1} deg)",
        a,
        i.to_degrees()
    );

    verify_conservation(&mut sim, n_steps, "retrograde", 1e-10, 1e-10);

    // Verify orbit is retrograde: angular momentum Z component should be negative
    let body = sim.body(0);
    let h = body.trans.position.cross(body.trans.velocity);
    assert!(
        h.z < 0.0,
        "Retrograde orbit should have negative h_z, got {:.3e}",
        h.z
    );
}

// ======================================================================
// Equatorial orbit (i ~ 0, RAAN is singular)
// ======================================================================

#[test]
fn tier3_orbinit_equatorial() {
    let a = R_EARTH + 600_000.0;
    let e = 0.1;
    let i = 0.0; // equatorial
    let raan = 0.0; // undefined for equatorial, set to 0
    let argp = 45.0_f64.to_radians();
    let nu = 90.0_f64.to_radians();

    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);

    let dt = 10.0;
    let period = 2.0 * std::f64::consts::PI * (a * a * a / MU_EARTH).sqrt();
    let n_steps = (2.0 * period / dt).ceil() as usize;

    let mut sim = build_sim(trans, dt);

    println!("Tier 3: Equatorial orbit (a={:.0} m, e={e}, i=0)", a);

    verify_conservation(&mut sim, n_steps, "equatorial", 1e-10, 1e-10);

    // Verify orbit stays in equatorial plane over the full propagation window.
    let mut eq_sim = build_sim(trans, dt);
    let mut max_z_frac = 0.0_f64;
    for _ in 0..n_steps {
        eq_sim.step();
        let body = eq_sim.body(0);
        let z_frac = body.trans.position.z.abs() / body.trans.position.length();
        max_z_frac = max_z_frac.max(z_frac);
    }
    println!("  Equatorial: max |z|/r = {max_z_frac:.3e}");
    assert!(
        max_z_frac < 1e-12,
        "Equatorial orbit left the equatorial plane: max |z|/r={max_z_frac:.3e}"
    );
}

// ======================================================================
// Polar orbit (i = 90 deg)
// ======================================================================

#[test]
fn tier3_orbinit_polar() {
    let a = R_EARTH + 500_000.0;
    let e = 0.02;
    let i = 90.0_f64.to_radians(); // polar
    let raan = 60.0_f64.to_radians();
    let argp = 0.0;
    let nu = 45.0_f64.to_radians();

    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);

    let dt = 10.0;
    let period = 2.0 * std::f64::consts::PI * (a * a * a / MU_EARTH).sqrt();
    let n_steps = (2.0 * period / dt).ceil() as usize;

    let mut sim = build_sim(trans, dt);

    println!("Tier 3: Polar orbit (a={:.0} m, e={e}, i=90 deg)", a);

    verify_conservation(&mut sim, n_steps, "polar", 1e-10, 1e-10);

    // Propagate another 2 orbits to verify the orbit passes over the poles:
    // track the maximum |z|/r ratio across steps.
    let mut max_z_frac = 0.0_f64;
    for _ in 0..n_steps {
        sim.step();
        let body = sim.body(0);
        let z_frac = body.trans.position.z.abs() / body.trans.position.length();
        max_z_frac = max_z_frac.max(z_frac);
    }
    println!("  Polar: max |z|/r = {max_z_frac:.4}");
    assert!(
        max_z_frac > 0.5,
        "Polar orbit should pass over the poles (|z| > r/2): max |z|/r = {max_z_frac:.4}"
    );

    // For polar orbit, angular momentum should be perpendicular to Z:
    // h_z = x*vy - y*vx should be ~0 for i=90.
    let body = sim.body(0);
    let r_mag = body.trans.position.length();
    let h_z = body.trans.position.x * body.trans.velocity.y
        - body.trans.position.y * body.trans.velocity.x;
    let h_mag = specific_ang_momentum(body.trans.position, body.trans.velocity);
    let h_z_frac = h_z.abs() / h_mag;
    println!(
        "  Polar: h_z/|h| = {h_z_frac:.3e}, r={:.1} km",
        r_mag / 1000.0
    );
    assert!(
        h_z_frac < 1e-10,
        "Polar orbit h_z should be ~0: h_z/|h|={h_z_frac:.3e}"
    );
}

// ======================================================================
// Hyperbolic orbit (e > 1)
// ======================================================================

#[test]
fn tier3_orbinit_hyperbolic() {
    // Hyperbolic: a < 0, e > 1
    let e = 1.5;
    let r_peri = R_EARTH + 300_000.0; // periapsis at 300 km altitude
    let a = -(r_peri / (e - 1.0)); // a < 0
    let i = 30.0_f64.to_radians();
    let raan = 0.0;
    let argp = 0.0;
    let nu = 0.1; // just past periapsis

    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);

    // Hyperbolic: propagate for a short time (~10 minutes)
    let dt = 1.0;
    let n_steps = 600;

    let mut sim = build_sim(trans, dt);

    println!(
        "Tier 3: Hyperbolic orbit (a={:.0} m, e={e}, r_peri={:.0} km)",
        a,
        r_peri / 1000.0
    );

    // Energy should be positive for hyperbolic
    let body0 = sim.body(0);
    let e0 = specific_energy(body0.trans.position, body0.trans.velocity, MU_EARTH);
    assert!(
        e0 > 0.0,
        "Hyperbolic orbit should have positive energy: E={e0:.6e}"
    );

    verify_conservation(&mut sim, n_steps, "hyperbolic", 1e-10, 1e-10);

    // Verify the body is moving away (radius increasing after periapsis)
    let body = sim.body(0);
    let r_final = body.trans.position.length();
    assert!(
        r_final > r_peri * 1.1,
        "Hyperbolic orbit should be escaping: r_final={:.0} m, r_peri={:.0} m",
        r_final,
        r_peri
    );
}

// ======================================================================
// Near-parabolic orbit (e ~ 1)
// ======================================================================

#[test]
fn tier3_orbinit_near_parabolic() {
    // Near-parabolic: e is slightly above 1.0 but still within
    // ORBIT_SWITCH_TOL (1e-2), so this remains in JEOD's near-parabolic branch.
    let e = 1.005;
    let r_peri = R_EARTH + 500_000.0;
    let a = -(r_peri / (e - 1.0)); // very large |a|
    let i = 10.0_f64.to_radians();
    let raan = 0.0;
    let argp = 0.0;
    let nu = 0.05; // near periapsis

    let trans = state_from_elements(a, e, i, raan, argp, nu, MU_EARTH);

    // Short propagation -- near-parabolic orbit moves slowly near periapsis
    let dt = 1.0;
    let n_steps = 300;

    let mut sim = build_sim(trans, dt);

    println!(
        "Tier 3: Near-parabolic orbit (a={:.0} m, e={e}, r_peri={:.0} km)",
        a,
        r_peri / 1000.0
    );

    // Near-parabolic orbits have near-zero energy
    let body0 = sim.body(0);
    let e0 = specific_energy(body0.trans.position, body0.trans.velocity, MU_EARTH);
    println!("  Initial energy: {e0:.6e} J/kg (should be near zero)");

    // Relaxed tolerance for near-parabolic: numerical sensitivity is higher
    verify_conservation(&mut sim, n_steps, "near_parabolic", 1e-9, 1e-10);
}
