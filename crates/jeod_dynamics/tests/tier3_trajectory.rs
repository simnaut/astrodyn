//! Tier 3 trajectory cross-validation tests.
//!
//! These tests propagate orbits over extended periods and validate the results
//! against known analytical properties and JEOD reference data where available.
//! They exercise the full integration pipeline (RK4 + point-mass gravity).

use glam::DVec3;
use jeod_dynamics::{rk4_translational_step, TranslationalState};
use std::f64::consts::PI;

const MU_EARTH: f64 = 3.986004418e14; // m^3/s^2
const R_EARTH: f64 = 6_378_137.0; // m

fn point_mass_accel(mu: f64, pos: DVec3) -> DVec3 {
    let r_sq = pos.length_squared();
    let r_mag = r_sq.sqrt();
    pos * (-mu / (r_sq * r_mag))
}

fn specific_energy(pos: DVec3, vel: DVec3, mu: f64) -> f64 {
    0.5 * vel.length_squared() - mu / pos.length()
}

fn angular_momentum(pos: DVec3, vel: DVec3) -> DVec3 {
    pos.cross(vel)
}

/// Propagate a state for the given duration at the given timestep.
fn propagate(state: &TranslationalState, mu: f64, dt: f64, duration: f64) -> Vec<(f64, TranslationalState)> {
    let steps = (duration / dt).ceil() as usize;
    let mut trajectory = Vec::with_capacity(steps + 1);
    let mut current = *state;
    trajectory.push((0.0, current));

    for i in 1..=steps {
        current = rk4_translational_step(&current, |s| point_mass_accel(mu, s.position), dt);
        trajectory.push((i as f64 * dt, current));
    }
    trajectory
}

// ========================================================================
// Test 1: Energy and angular momentum conservation over 10 orbits
// ========================================================================
#[test]
fn tier3_energy_conservation_10_orbits() {
    let r0 = R_EARTH + 400_000.0; // 400 km altitude
    let v0 = (MU_EARTH / r0).sqrt(); // circular velocity

    let state = TranslationalState {
        position: DVec3::new(r0, 0.0, 0.0),
        velocity: DVec3::new(0.0, v0, 0.0),
    };

    let period = 2.0 * PI * (r0.powi(3) / MU_EARTH).sqrt();
    let dt = 10.0; // seconds
    let duration = 10.0 * period;

    let trajectory = propagate(&state, MU_EARTH, dt, duration);

    let e0 = specific_energy(state.position, state.velocity, MU_EARTH);
    let h0 = angular_momentum(state.position, state.velocity);

    let mut max_energy_drift = 0.0_f64;
    let mut max_h_drift = 0.0_f64;

    for (t, s) in &trajectory {
        let e = specific_energy(s.position, s.velocity, MU_EARTH);
        let h = angular_momentum(s.position, s.velocity);

        let energy_drift = ((e - e0) / e0.abs()).abs();
        let h_drift = ((h - h0).length() / h0.length()).abs();

        max_energy_drift = max_energy_drift.max(energy_drift);
        max_h_drift = max_h_drift.max(h_drift);

        // Per-step assertions with generous bounds
        assert!(
            energy_drift < 1e-7,
            "Energy drift {:.2e} at t={:.0}s exceeds 1e-7",
            energy_drift, t
        );
        assert!(
            h_drift < 1e-7,
            "Angular momentum drift {:.2e} at t={:.0}s exceeds 1e-7",
            h_drift, t
        );
    }

    // Tighter final bounds
    assert!(
        max_energy_drift < 1e-8,
        "Max relative energy drift {:.2e} exceeds 1e-8",
        max_energy_drift
    );
    assert!(
        max_h_drift < 1e-8,
        "Max relative angular momentum drift {:.2e} exceeds 1e-8",
        max_h_drift
    );

    println!("10-orbit conservation (dt={}s):", dt);
    println!("  Max relative energy drift:  {:.2e}", max_energy_drift);
    println!("  Max relative h drift:       {:.2e}", max_h_drift);
    println!("  Total steps:                {}", trajectory.len());
}

// ========================================================================
// Test 2: Orbital period accuracy
// ========================================================================
#[test]
fn tier3_orbital_period_accuracy() {
    let r0 = R_EARTH + 400_000.0;
    let v0 = (MU_EARTH / r0).sqrt();

    let state = TranslationalState {
        position: DVec3::new(r0, 0.0, 0.0),
        velocity: DVec3::new(0.0, v0, 0.0),
    };

    let analytical_period = 2.0 * PI * (r0.powi(3) / MU_EARTH).sqrt();
    let dt = 1.0; // 1-second timestep for precision

    // Propagate for slightly more than one period
    let trajectory = propagate(&state, MU_EARTH, dt, analytical_period * 1.1);

    // Find the time when y crosses zero going positive (completes one orbit)
    // Initial state: x=r0, y=0, vy>0, so y starts positive.
    // After one orbit, y crosses zero from negative to positive.
    let mut crossing_time = None;
    for window in trajectory.windows(2) {
        let (t0, s0) = &window[0];
        let (t1, s1) = &window[1];

        // Look for y going from negative to non-negative, with x > 0
        if s0.position.y < 0.0 && s1.position.y >= 0.0 && s1.position.x > 0.0 {
            // Linear interpolation for more precise crossing
            let frac = (-s0.position.y) / (s1.position.y - s0.position.y);
            crossing_time = Some(t0 + frac * (t1 - t0));
            break;
        }
    }

    let measured_period = crossing_time.expect("Did not find zero-crossing for period measurement");
    let period_error = ((measured_period - analytical_period) / analytical_period).abs();

    println!("Orbital period test (dt={}s):", dt);
    println!("  Analytical period: {:.6} s", analytical_period);
    println!("  Measured period:   {:.6} s", measured_period);
    println!("  Relative error:    {:.2e}", period_error);

    assert!(
        period_error < 1e-4,
        "Period error {:.2e} exceeds 1e-4 (0.01%)",
        period_error
    );
}

// ========================================================================
// Test 3: Position return accuracy after one orbit
// ========================================================================
#[test]
fn tier3_position_return_after_one_orbit() {
    let r0 = R_EARTH + 400_000.0;
    let v0 = (MU_EARTH / r0).sqrt();

    let initial = TranslationalState {
        position: DVec3::new(r0, 0.0, 0.0),
        velocity: DVec3::new(0.0, v0, 0.0),
    };

    let period = 2.0 * PI * (r0.powi(3) / MU_EARTH).sqrt();
    let dt = 10.0;
    let full_steps = (period / dt).floor() as usize;
    let remainder = period - (full_steps as f64 * dt);

    // Propagate full steps + a fractional final step to land exactly at one period
    let mut state = initial;
    for _ in 0..full_steps {
        state = rk4_translational_step(&state, |s| point_mass_accel(MU_EARTH, s.position), dt);
    }
    if remainder > 1e-12 {
        state = rk4_translational_step(&state, |s| point_mass_accel(MU_EARTH, s.position), remainder);
    }

    let pos_error = (state.position - initial.position).length();
    let vel_error = (state.velocity - initial.velocity).length();

    println!("One-orbit return test (dt={}s, {} full steps + {:.3}s remainder):", dt, full_steps, remainder);
    println!("  Position error: {:.3} m", pos_error);
    println!("  Velocity error: {:.6} m/s", vel_error);

    assert!(
        pos_error < 100.0,
        "Position return error {:.1} m exceeds 100 m",
        pos_error
    );
    assert!(
        vel_error < 0.1,
        "Velocity return error {:.4} m/s exceeds 0.1 m/s",
        vel_error
    );
}

// ========================================================================
// Test 4: Eccentric orbit (e=0.3) maintains correct apoapsis/periapsis
// ========================================================================
#[test]
fn tier3_eccentric_orbit_apse_distances() {
    let a = R_EARTH + 1_000_000.0; // semi-major axis
    let e = 0.3;
    let r_periapsis = a * (1.0 - e);
    let r_apoapsis = a * (1.0 + e);

    // Start at periapsis: r = a(1-e), v = sqrt(mu * (2/r - 1/a))
    let r0 = r_periapsis;
    let v0 = (MU_EARTH * (2.0 / r0 - 1.0 / a)).sqrt();

    let state = TranslationalState {
        position: DVec3::new(r0, 0.0, 0.0),
        velocity: DVec3::new(0.0, v0, 0.0),
    };

    let period = 2.0 * PI * (a.powi(3) / MU_EARTH).sqrt();
    let dt = 10.0;

    let trajectory = propagate(&state, MU_EARTH, dt, period);

    let mut min_r = f64::MAX;
    let mut max_r = 0.0_f64;

    for (_, s) in &trajectory {
        let r = s.position.length();
        min_r = min_r.min(r);
        max_r = max_r.max(r);
    }

    let periapsis_error = ((min_r - r_periapsis) / r_periapsis).abs();
    let apoapsis_error = ((max_r - r_apoapsis) / r_apoapsis).abs();

    println!("Eccentric orbit apse test (e={}, dt={}s):", e, dt);
    println!("  Analytical periapsis: {:.1} m, measured: {:.1} m, error: {:.2e}", r_periapsis, min_r, periapsis_error);
    println!("  Analytical apoapsis:  {:.1} m, measured: {:.1} m, error: {:.2e}", r_apoapsis, max_r, apoapsis_error);

    assert!(
        periapsis_error < 1e-6,
        "Periapsis error {:.2e} exceeds 1e-6",
        periapsis_error
    );
    assert!(
        apoapsis_error < 1e-6,
        "Apoapsis error {:.2e} exceeds 1e-6",
        apoapsis_error
    );
}

// ========================================================================
// Test 5: ISS orbit from JEOD orbital elements — 24-hour propagation
// ========================================================================
#[test]
fn tier3_iss_24h_propagation() {
    // ISS orbital elements from JEOD: trans_Orbit_inertial_body_set01.py
    // a = 6732.90120152 km, e = 0.00129073350, i = 51.670450765 deg
    let a = 6_732_901.20152; // m
    let e = 0.00129073350;
    let r_peri = a * (1.0 - e);

    // Start at periapsis for simplicity
    let v_peri = (MU_EARTH * (2.0 / r_peri - 1.0 / a)).sqrt();

    let state = TranslationalState {
        position: DVec3::new(r_peri, 0.0, 0.0),
        velocity: DVec3::new(0.0, v_peri, 0.0),
    };

    let dt = 10.0;
    let duration = 86400.0; // 24 hours
    let period = 2.0 * PI * (a.powi(3) / MU_EARTH).sqrt();
    let n_orbits = duration / period;

    let trajectory = propagate(&state, MU_EARTH, dt, duration);

    let e0 = specific_energy(state.position, state.velocity, MU_EARTH);

    // Check energy conservation over 24 hours
    let final_state = &trajectory.last().unwrap().1;
    let ef = specific_energy(final_state.position, final_state.velocity, MU_EARTH);
    let relative_energy_drift = ((ef - e0) / e0.abs()).abs();

    // Check altitude stays within expected bounds
    let expected_min_alt = a * (1.0 - e) - R_EARTH;
    let expected_max_alt = a * (1.0 + e) - R_EARTH;

    let mut min_alt = f64::MAX;
    let mut max_alt = f64::MIN;
    for (_, s) in &trajectory {
        let alt = s.position.length() - R_EARTH;
        min_alt = min_alt.min(alt);
        max_alt = max_alt.max(alt);
    }

    println!("ISS 24-hour propagation (dt={}s, {:.1} orbits):", dt, n_orbits);
    println!("  Relative energy drift:     {:.2e}", relative_energy_drift);
    println!("  Expected altitude range:   {:.1} - {:.1} km",
        expected_min_alt / 1000.0, expected_max_alt / 1000.0);
    println!("  Measured altitude range:   {:.1} - {:.1} km",
        min_alt / 1000.0, max_alt / 1000.0);
    println!("  Total steps:               {}", trajectory.len());

    assert!(
        relative_energy_drift < 1e-7,
        "24h energy drift {:.2e} exceeds 1e-7",
        relative_energy_drift
    );

    // Altitude should stay within expected bounds (with small numerical tolerance)
    let alt_tolerance = 1000.0; // 1 km tolerance for discretization
    assert!(
        min_alt > expected_min_alt - alt_tolerance,
        "Min altitude {:.1} km below expected {:.1} km",
        min_alt / 1000.0, (expected_min_alt - alt_tolerance) / 1000.0
    );
    assert!(
        max_alt < expected_max_alt + alt_tolerance,
        "Max altitude {:.1} km above expected {:.1} km",
        max_alt / 1000.0, (expected_max_alt + alt_tolerance) / 1000.0
    );
}

// ========================================================================
// Test 6: Cross-validate Rust vs C point-mass at JEOD test positions
// ========================================================================
#[test]
fn tier3_cross_validate_gravity_at_jeod_positions() {
    let jeod_root = jeod_test_data::jeod_path();
    if !jeod_root.exists() {
        eprintln!("JEOD not found, skipping cross-validation");
        return;
    }

    let cases = jeod_test_data::gravity_verif::load_gravity_test_cases(&jeod_root);

    let mu_earth = 3.986004418e14;

    let mut full_count = 0;
    let mut perturb_count = 0;

    for case in &cases {
        let our_result = jeod_gravity::compute_point_mass_gravity(mu_earth, case.position);
        let point_mass_mag = our_result.accel.length();

        if case.perturb_only {
            // perturbOnly=1: JEOD acceleration is harmonics perturbation only
            // (total minus point-mass). Perturbation should be small relative
            // to point-mass (J2 ≈ 0.1% for LEO).
            let perturbation_mag = case.acceleration.length();
            let ratio = perturbation_mag / point_mass_mag;
            assert!(
                ratio < 0.01,
                "Case {}: perturbation {:.6e} is > 1% of point-mass {:.6e}",
                case.case_num, perturbation_mag, point_mass_mag
            );
            perturb_count += 1;
        } else {
            // perturbOnly=0: JEOD acceleration is TOTAL gravity (point-mass +
            // harmonics). Our point-mass should be close — within ~1% for LEO,
            // larger for high-altitude cases where harmonics contribute more
            // relative error.
            let jeod_mag = case.acceleration.length();
            let relative_diff = ((point_mass_mag - jeod_mag) / jeod_mag).abs();
            assert!(
                relative_diff < 0.01,
                "Case {}: point-mass {:.6e} vs JEOD total {:.6e}, diff {:.2e}",
                case.case_num, point_mass_mag, jeod_mag, relative_diff
            );

            // Direction should agree (both point roughly toward center).
            let our_dir = our_result.accel.normalize();
            let jeod_dir = case.acceleration.normalize();
            let cos_angle = our_dir.dot(jeod_dir);
            assert!(
                cos_angle > 0.999,
                "Case {}: direction mismatch, cos(angle) = {:.6}",
                case.case_num, cos_angle
            );
            full_count += 1;
        }

        // In all cases, point-mass should be anti-radial.
        let cos_radial = case.position.normalize().dot(our_result.accel.normalize());
        assert!(
            cos_radial < -0.999,
            "Case {}: point-mass not anti-radial, cos = {:.6}",
            case.case_num, cos_radial
        );
    }

    println!(
        "Cross-validated point-mass against {} JEOD positions ({} full, {} perturbation-only)",
        cases.len(), full_count, perturb_count
    );
}
