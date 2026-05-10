//! Tier 3: Extended solar-beta tests (analytical).
//!
//! These tests build controlled orbit/Sun geometries and propagate through
//! `Simulation::step()` with a fake Sun source at a chosen position. The
//! reported solar-beta angle is checked against the closed-form value
//! β = asin(ĥ · ŝ).
//!
//! * `tier3_solar_beta_equatorial_at_equinox` — equatorial orbit with Sun in
//!   the orbital (equatorial) plane → β ≈ 0.
//! * `tier3_solar_beta_polar_orbit` — polar orbit. Over one year, β can swing
//!   across the full [-90°, +90°] range. We verify three snapshot geometries.
//! * `tier3_solar_beta_iss_orbit` — ISS inclination (51.6°); for Sun on the
//!   +Z axis β = π/2 − inclination, for Sun in the orbital plane β = 0.
//! * `tier3_solar_beta_sun_in_orbital_plane` — Sun direction in the orbital
//!   plane regardless of orbit inclination → β = 0.
//! * `tier3_solar_beta_sun_perpendicular_to_plane` — Sun along the orbit
//!   normal → |β| = 90°.
//! * `tier3_solar_beta_bounded` — for every propagated checkpoint of a mid-
//!   inclination orbit with a fixed Sun position, |β| ≤ 90°.
//!
//! No Docker reference data required.

use astrodyn::Vec3Ext;
use astrodyn::{DerivedStateConfig, GravitySourceEntry, VehicleConfig};
use astrodyn::{
    GravityControl, GravityControls, GravityModel, GravityRole, GravitySource, SimulationTime,
    TranslationalState,
};
use astrodyn_runner::{RotationModel, Simulation};
use glam::DVec3;

fn load_mu_earth() -> f64 {
    astrodyn::gravity_fixtures::load_ggm05c().mu
}

/// Sun at a cartoon distance in the +X direction produces `sun_direction = +X`
/// for any body near the origin; at this scale the relative-Sun vector from
/// any LEO body is effectively parallel to the Sun position vector.
const SUN_DISTANCE_M: f64 = 1.495_978_707e11; // 1 AU

/// Build a Simulation with:
///   * central-body Earth at origin (point-mass gravity),
///   * a "Sun" source at `sun_position` with mu = 0 (kinematic only),
///   * a single vehicle configured with `solar_beta: true`.
///
/// Returns the simulation ready to be `validate`d and stepped.
fn build_solar_beta_sim(
    mu_earth: f64,
    dt: f64,
    sun_position: DVec3,
    body_state: TranslationalState,
) -> Simulation {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_earth,
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

    let sun = sim.add_source(
        "Sun",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: sun_position.m_at::<astrodyn::RootInertial>(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
            marker_only: false,
        },
    );
    sim.sun_source = Some(sun);

    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&body_state),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityRole::Central)],
        },
        derived: DerivedStateConfig {
            solar_beta: true,
            ..Default::default()
        },
        ..Default::default()
    });

    sim
}

#[test]
fn tier3_solar_beta_equatorial_at_equinox() {
    // Equatorial circular orbit → orbit normal = +Z.
    // Sun in the equatorial (x–y) plane → ŝ ⊥ ĥ → β = 0.
    let mu_earth = load_mu_earth();
    let r = 6_778_137.0;
    let v = (mu_earth / r).sqrt();

    let mut sim = build_solar_beta_sim(
        mu_earth,
        10.0,
        DVec3::new(SUN_DISTANCE_M, 0.0, 0.0),
        TranslationalState {
            position: DVec3::new(r, 0.0, 0.0),
            velocity: DVec3::new(0.0, v, 0.0),
        },
    );
    sim.validate().unwrap();

    let period = 2.0 * std::f64::consts::PI * (r * r * r / mu_earth).sqrt();
    let n_steps = (period / 10.0) as usize;

    let mut max_beta = 0.0_f64;
    for step in 1..=n_steps {
        sim.step_until(step as f64 * 10.0)
            .expect("step_until failed");
        let beta = sim.body(0).solar_beta.expect("solar beta not computed");
        max_beta = max_beta.max(beta.abs());
    }

    // The Sun-direction deviates slightly from +X as the body orbits (since
    // sun_direction = sun_position - body_position). For r/SUN_DISTANCE ~ 4.5e-5
    // the worst-case projection onto ĥ = +Z is zero (coplanar), so β is
    // limited by floating-point noise and the small offset: well under 1e-4 rad.
    assert!(
        max_beta < 1e-4,
        "equatorial+in-plane sun: max |beta| = {max_beta:.3e} rad exceeds 1e-4"
    );
}

#[test]
fn tier3_solar_beta_polar_orbit() {
    // Polar orbit about Earth — orbit plane contains +Z, so orbit normal is
    // perpendicular to +Z. Over a full year the Sun sweeps the ecliptic; we
    // verify three snapshot geometries by instantiating distinct simulations.
    //
    //   (1) Sun along +X (equinox-like):   ĥ·ŝ = 0 → β = 0
    //   (2) Sun along +Y:                  β = ±90° if orbit normal = ±Y
    //   (3) Sun along +Z:                  Sun is in the orbital plane → β = 0
    let mu_earth = load_mu_earth();
    let r = 6_778_137.0;
    let v = (mu_earth / r).sqrt();

    // Polar orbit: position +X, velocity +Z → h = r × v = r v (+Y)
    let body = TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        velocity: DVec3::new(0.0, 0.0, v),
    };

    let cases = [
        (DVec3::new(SUN_DISTANCE_M, 0.0, 0.0), 0.0_f64), // Sun along +X
        (DVec3::new(0.0, SUN_DISTANCE_M, 0.0), 90.0_f64), // Sun along +Y
        (DVec3::new(0.0, 0.0, SUN_DISTANCE_M), 0.0_f64), // Sun along +Z
    ];

    for (sun_pos, expected_deg) in cases {
        let mut sim = build_solar_beta_sim(mu_earth, 10.0, sun_pos, body);
        sim.validate().unwrap();
        sim.step_until(10.0).expect("step_until failed"); // derived state is populated after first step
        let beta = sim.body(0).solar_beta.expect("solar beta not computed");
        let expected = expected_deg.to_radians();
        assert!(
            (beta.abs() - expected).abs() < 1e-4,
            "polar orbit, Sun at {sun_pos:?}: |beta| = {} rad, expected {}",
            beta.abs(),
            expected
        );
    }
}

#[test]
fn tier3_solar_beta_iss_orbit() {
    // ISS-like circular orbit at 51.6° inclination. With
    //   r = (r, 0, 0), v = (0, v cos i, v sin i)
    //   h = r × v = (0, -r v sin i, r v cos i)
    //   ĥ = (0, -sin i, cos i)
    //
    // For Sun along +Z:    ĥ·ẑ = cos i  →  β = asin(cos i) = π/2 − i
    // For Sun along +X:    ĥ·x̂ = 0      →  β = 0
    // For Sun along −Y:    ĥ·(−ŷ) = sin i →  β = asin(sin i) = i
    //
    // We verify all three cases.
    let mu_earth = load_mu_earth();
    let r = 6_778_137.0;
    let v = (mu_earth / r).sqrt();
    let inc = 51.6_f64.to_radians();

    let body = TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        velocity: DVec3::new(0.0, v * inc.cos(), v * inc.sin()),
    };

    // Case 1: Sun in the equatorial plane (+X). β ≈ 0.
    {
        let mut sim =
            build_solar_beta_sim(mu_earth, 10.0, DVec3::new(SUN_DISTANCE_M, 0.0, 0.0), body);
        sim.validate().unwrap();
        sim.step_until(10.0).expect("step_until failed");
        let beta = sim.body(0).solar_beta.expect("solar beta not computed");
        assert!(
            beta.abs() < 1e-4,
            "ISS orbit + Sun in equatorial plane: |beta| = {beta} rad (expected ≈0)"
        );
    }

    // Case 2: Sun along +Z → β = π/2 − i.
    {
        let mut sim =
            build_solar_beta_sim(mu_earth, 10.0, DVec3::new(0.0, 0.0, SUN_DISTANCE_M), body);
        sim.validate().unwrap();
        sim.step_until(10.0).expect("step_until failed");
        let beta = sim.body(0).solar_beta.expect("solar beta not computed");
        let expected = std::f64::consts::FRAC_PI_2 - inc;
        assert!(
            (beta - expected).abs() < 1e-4,
            "ISS orbit + Sun along +Z: beta = {beta} rad (expected {expected} = π/2 − i)"
        );
    }

    // Case 3: Sun along −Y → β = i. This exercises the bound that β can
    // reach the full inclination angle at a favorable Sun direction.
    {
        let mut sim =
            build_solar_beta_sim(mu_earth, 10.0, DVec3::new(0.0, -SUN_DISTANCE_M, 0.0), body);
        sim.validate().unwrap();
        sim.step_until(10.0).expect("step_until failed");
        let beta = sim.body(0).solar_beta.expect("solar beta not computed");
        assert!(
            (beta - inc).abs() < 1e-4,
            "ISS orbit + Sun along −Y: beta = {beta} rad (expected {inc} = inclination)"
        );
    }
}

#[test]
fn tier3_solar_beta_sun_in_orbital_plane() {
    // Sun direction exactly in the orbital plane → ĥ · ŝ = 0 → β = 0.
    // Check this for a 30° inclination orbit where the Sun sits exactly at
    // the ascending node direction (in the equatorial plane the node is +X).
    let mu_earth = load_mu_earth();
    let r = 7_000_000.0;
    let v = (mu_earth / r).sqrt();
    let inc = 30.0_f64.to_radians();

    // Position along +X (ascending node), velocity tipped into the orbit plane.
    let body = TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        velocity: DVec3::new(0.0, v * inc.cos(), v * inc.sin()),
    };
    // Orbit normal: h = r × v = (r, 0, 0) × (0, v cos i, v sin i)
    //              = (0·v sin i − 0·v cos i, 0·0 − r·v sin i, r·v cos i − 0·0)
    //              = (0, −r v sin i, r v cos i). Direction = (0, −sin i, cos i).
    // Sun along +X lies in the plane since +X · (orbit normal) = 0.
    let mut sim = build_solar_beta_sim(mu_earth, 10.0, DVec3::new(SUN_DISTANCE_M, 0.0, 0.0), body);
    sim.validate().unwrap();
    sim.step_until(10.0).expect("step_until failed");
    let beta = sim.body(0).solar_beta.expect("solar beta not computed");
    assert!(
        beta.abs() < 1e-4,
        "Sun in orbital plane: |beta| = {beta} rad exceeds 1e-4"
    );
}

#[test]
fn tier3_solar_beta_sun_perpendicular_to_plane() {
    // Sun along the orbit normal → ĥ · ŝ = ±1 → |β| = π/2.
    // Construct an equatorial orbit (normal = +Z) and place Sun along +Z.
    let mu_earth = load_mu_earth();
    let r = 7_000_000.0;
    let v = (mu_earth / r).sqrt();

    let body = TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        velocity: DVec3::new(0.0, v, 0.0),
    };

    let mut sim = build_solar_beta_sim(mu_earth, 10.0, DVec3::new(0.0, 0.0, SUN_DISTANCE_M), body);
    sim.validate().unwrap();
    sim.step_until(10.0).expect("step_until failed");
    let beta = sim.body(0).solar_beta.expect("solar beta not computed");

    let pi_2 = std::f64::consts::FRAC_PI_2;
    // Tolerance accounts for the small LEO offset from the origin vs the Sun
    // at 1 AU: the body-Sun direction is slightly off +Z by r/AU ≈ 4.7e-5 rad.
    assert!(
        (beta.abs() - pi_2).abs() < 1e-4,
        "Sun perpendicular to orbit plane: |beta| = {} rad (expected π/2 = {})",
        beta.abs(),
        pi_2
    );
}

#[test]
fn tier3_solar_beta_bounded() {
    // For any orbit and any Sun position, |β| ≤ π/2. We propagate an
    // inclined orbit for several periods and verify this bound at every
    // integration step. (The tighter bound mentioned in the spec —
    // π/2 + inclination — does not apply to β itself, which is always a
    // signed angle in [-π/2, +π/2] by construction; our asserted bound is
    // the mathematical limit |β| ≤ π/2.)
    let mu_earth = load_mu_earth();
    let r = 7_000_000.0;
    let v = (mu_earth / r).sqrt();
    let inc = 45.0_f64.to_radians();

    let body = TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        velocity: DVec3::new(0.0, v * inc.cos(), v * inc.sin()),
    };

    let mut sim = build_solar_beta_sim(
        mu_earth,
        10.0,
        // Arbitrary Sun direction with components in all three axes.
        DVec3::new(
            0.7 * SUN_DISTANCE_M,
            0.5 * SUN_DISTANCE_M,
            0.2 * SUN_DISTANCE_M,
        ),
        body,
    );
    sim.validate().unwrap();

    let period = 2.0 * std::f64::consts::PI * (r * r * r / mu_earth).sqrt();
    let n_steps = (3.0 * period / 10.0) as usize;

    let pi_2 = std::f64::consts::FRAC_PI_2;
    let mut max_abs_beta = 0.0_f64;
    for step in 1..=n_steps {
        sim.step_until(step as f64 * 10.0)
            .expect("step_until failed");
        let beta = sim.body(0).solar_beta.expect("solar beta not computed");
        // Absolute bound from the asin definition.
        assert!(
            beta.abs() <= pi_2 + 1e-12,
            "step {step}: |beta| = {} exceeds π/2",
            beta.abs()
        );
        max_abs_beta = max_abs_beta.max(beta.abs());
    }

    // Confirm the β swing is nontrivial for this geometry so the test
    // actually exercises the bound rather than asserting near zero.
    assert!(
        max_abs_beta > 0.05,
        "max |beta| = {max_abs_beta} rad — geometry too degenerate to test bound"
    );
}
