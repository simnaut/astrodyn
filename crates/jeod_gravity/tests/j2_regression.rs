//! Test J2 nodal regression rate against the analytical formula.
//!
//! Propagates an ISS-like orbit with J2-only gravity and measures the
//! RAAN change over time. Compares to the analytical formula:
//!   dΩ/dt = -3nJ₂R²cos(i) / (2p²)
//! where n=mean motion, J₂=1.0826e-3, R=equatorial radius, p=semi-latus rectum.
//!
//! Requires JEOD_HOME for GGM02C coefficients.

use glam::DVec3;
use jeod_dynamics::{rk4_translational_step, TranslationalState};
use jeod_gravity::coefficients;
use jeod_math::OrbitalElements;
use jeod_test_data::jeod_path;
use std::f64::consts::PI;

#[test]
fn j2_nodal_regression_rate() {
    let root = jeod_path();
    assert!(root.exists(), "JEOD source not found");

    let ggm02c_path = root.join("models/environment/gravity/data/src/earth_GGM02C.cc");
    let sh_data = coefficients::load_from_jeod_cc(&ggm02c_path).expect("load GGM02C coefficients");
    let mu = sh_data.mu;
    let r_eq = sh_data.radius;

    // ISS-like orbit: 400 km altitude, 51.6° inclination
    let altitude = 400_000.0;
    let r0 = r_eq + altitude;
    let inclination = 51.6 * PI / 180.0;

    // Initial state: circular orbit in the inclined plane
    let v_circ = (mu / r0).sqrt();
    let state_initial = TranslationalState {
        position: DVec3::new(r0, 0.0, 0.0),
        velocity: DVec3::new(0.0, v_circ * inclination.cos(), v_circ * inclination.sin()),
    };

    // Truncate to J2-only (degree=2, order=0) for clean comparison
    let degree = 2;
    let order = 0;

    // Gravity acceleration function: J2 + point-mass
    let accel_fn = |s: &TranslationalState| -> DVec3 {
        let pm = jeod_gravity::calc_spherical(mu, s.position);
        let sh = jeod_gravity::calc_nonspherical(
            &sh_data, s.position, degree, order, false, 0, 0,
        );
        pm.grav_accel + sh.grav_accel
    };

    // Propagate for 1 day
    let dt = 10.0; // seconds
    let total_time = 86400.0; // 1 day
    let steps = (total_time / dt) as usize;

    let mut state = state_initial;
    for _ in 0..steps {
        state = rk4_translational_step(&state, &accel_fn, dt);
    }

    // Compute RAAN at start and end
    let elems_start =
        OrbitalElements::from_cartesian(mu, state_initial.position, state_initial.velocity)
            .unwrap();
    let elems_end = OrbitalElements::from_cartesian(mu, state.position, state.velocity).unwrap();

    let raan_start = elems_start.long_asc_node;
    let raan_end = elems_end.long_asc_node;
    let mut d_raan = raan_end - raan_start;
    // Handle wrapping
    if d_raan > PI {
        d_raan -= 2.0 * PI;
    }
    if d_raan < -PI {
        d_raan += 2.0 * PI;
    }

    let d_raan_per_day = d_raan; // already 1 day propagation

    // Analytical J2 regression rate:
    //   dΩ/dt = -3nJ₂R²cos(i) / (2p²)
    // where J₂ is unnormalized = -sqrt(5) * C20
    let c20 = sh_data.cnm[2][0];
    let j2 = -(5.0_f64).sqrt() * c20;
    let a = elems_start.semi_major_axis;
    let e = elems_start.e_mag;
    let p = a * (1.0 - e * e);
    let n = (mu / (a * a * a)).sqrt(); // mean motion, rad/s
    let d_raan_analytical = -1.5 * n * j2 * r_eq * r_eq * inclination.cos() / (p * p);
    let d_raan_analytical_per_day = d_raan_analytical * 86400.0;

    let relative_error = ((d_raan_per_day - d_raan_analytical_per_day) / d_raan_analytical_per_day).abs();

    eprintln!("  RAAN change (numerical):   {:.6} deg/day", d_raan_per_day * 180.0 / PI);
    eprintln!("  RAAN change (analytical):  {:.6} deg/day", d_raan_analytical_per_day * 180.0 / PI);
    eprintln!("  Relative error: {:.4e}", relative_error);

    assert!(
        relative_error < 0.01,
        "J2 regression rate error {:.4e} exceeds 1%\n  numerical: {:.6} deg/day\n  analytical: {:.6} deg/day",
        relative_error,
        d_raan_per_day * 180.0 / PI,
        d_raan_analytical_per_day * 180.0 / PI,
    );
}
