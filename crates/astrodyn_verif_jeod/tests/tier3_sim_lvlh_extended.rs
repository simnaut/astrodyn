//! Tier 3: Extended LVLH-frame tests (analytical).
//!
//! These tests propagate through `Simulation::step()` and verify analytical
//! properties of the LVLH frame at each checkpoint.
//!
//! * `tier3_lvlh_retrograde_orbit` — retrograde circular orbit: the orbit
//!   normal flips sign, so the LVLH Y-axis (row 1 of T_parent_this) flips
//!   sign relative to a prograde orbit with the same radius.
//! * `tier3_lvlh_eccentric_orbit` — elliptical orbit: LVLH angular velocity
//!   |ω| = |h| / r² varies with r; verify the observed magnitudes at perigee
//!   and apogee match the closed-form values.
//! * `tier3_lvlh_periodicity` — after one orbital period the LVLH frame
//!   returns to its initial orientation to high precision.
//!
//! No Docker reference data required.

use astrodyn::recipes::helpers::state_helpers::max_mat_diff;
use astrodyn::{DerivedStateConfig, GravitySourceEntry, VehicleConfig};
use astrodyn::{
    GravityControl, GravityControls, GravityModel, GravitySource, SimulationTime,
    TranslationalState,
};
use astrodyn_runner::{RotationModel, Simulation};
use glam::DVec3;

fn load_mu_earth() -> f64 {
    astrodyn::gravity_fixtures::load_ggm05c().mu
}

fn make_earth_lvlh_sim(dt: f64, mu_earth: f64, body: TranslationalState) -> Simulation {
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

    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&body),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        derived: DerivedStateConfig {
            lvlh: true,
            ..Default::default()
        },
        ..Default::default()
    });

    sim
}

/// Extract the LVLH Y-hat axis (row 1 of T_parent_this) expressed in the
/// parent (inertial) frame.
fn lvlh_y_hat(t_parent_this: &glam::DMat3) -> DVec3 {
    // Row 1 is the Y-axis expressed in parent: columns of DMat3 are
    // column-major, so row 1 is (col(0).y, col(1).y, col(2).y).
    DVec3::new(
        t_parent_this.col(0).y,
        t_parent_this.col(1).y,
        t_parent_this.col(2).y,
    )
}

#[test]
fn tier3_lvlh_retrograde_orbit() {
    // Prograde orbit: position +X, velocity +Y → h = +Z → Y-hat = -Z.
    // Retrograde orbit: position +X, velocity -Y → h = -Z → Y-hat = +Z.
    //
    // We propagate both for one step, read the Y-hat from the LVLH frame,
    // and verify the sign flip.
    let mu_earth = load_mu_earth();
    let r = 6_778_137.0;
    let v = (mu_earth / r).sqrt();
    let dt = 10.0;

    // Prograde (equatorial)
    let mut sim_prograde = make_earth_lvlh_sim(
        dt,
        mu_earth,
        TranslationalState {
            position: DVec3::new(r, 0.0, 0.0),
            velocity: DVec3::new(0.0, v, 0.0),
        },
    );
    sim_prograde.validate().unwrap();
    sim_prograde.step_until(dt).expect("step_until failed");
    let lvlh_p = sim_prograde
        .body(0)
        .lvlh_frame
        .expect("prograde: LVLH not computed");
    let y_p = lvlh_y_hat(&lvlh_p.t_parent_this);

    // Retrograde (equatorial, velocity reversed)
    let mut sim_retro = make_earth_lvlh_sim(
        dt,
        mu_earth,
        TranslationalState {
            position: DVec3::new(r, 0.0, 0.0),
            velocity: DVec3::new(0.0, -v, 0.0),
        },
    );
    sim_retro.validate().unwrap();
    sim_retro.step_until(dt).expect("step_until failed");
    let lvlh_r = sim_retro
        .body(0)
        .lvlh_frame
        .expect("retrograde: LVLH not computed");
    let y_r = lvlh_y_hat(&lvlh_r.t_parent_this);

    // Y-hat = -h_hat, so prograde Y ≈ -Ẑ and retrograde Y ≈ +Ẑ.
    assert!(
        (y_p.z + 1.0).abs() < 1e-8,
        "prograde Y-hat Z-component: {:.6e}, expected -1",
        y_p.z
    );
    assert!(
        (y_r.z - 1.0).abs() < 1e-8,
        "retrograde Y-hat Z-component: {:.6e}, expected +1",
        y_r.z
    );
    // Sum should be ≈ 0 (one flip of sign).
    let sum = y_p + y_r;
    assert!(
        sum.length() < 1e-8,
        "Y-hat did not flip sign: y_p + y_r = {sum:?}"
    );

    // Angular velocities have equal magnitude but opposite sign on the LVLH
    // Y-axis: ω_LVLH = [0, -wmag, 0]. For retrograde, wmag is the same
    // (|h|/r²), and the LVLH frame Y-hat points the other way, so the angular
    // velocity vector (expressed in LVLH coords) remains [0, -wmag, 0] in both
    // cases, but the inertial orbit winds the other way. Sanity-check the
    // magnitudes are equal.
    let wmag_p = lvlh_p.ang_vel_this.length();
    let wmag_r = lvlh_r.ang_vel_this.length();
    assert!(
        (wmag_p - wmag_r).abs() / wmag_p < 1e-10,
        "LVLH ang vel magnitudes differ: prograde {wmag_p}, retrograde {wmag_r}"
    );
}

#[test]
fn tier3_lvlh_eccentric_orbit() {
    // Elliptical orbit with perigee at r_p = 6778 km, apogee at r_a = 20000 km.
    // |ω_LVLH| = |h| / r². At perigee r = r_p, velocity is perpendicular to
    // position, so h = r_p * v_p. Same at apogee. We start at perigee.
    let mu_earth = load_mu_earth();
    let r_p = 6_778_137.0;
    let r_a = 20_000_000.0;
    let a = 0.5 * (r_p + r_a);
    let e = (r_a - r_p) / (r_a + r_p);
    let v_p = (mu_earth * (1.0 + e) / (a * (1.0 - e))).sqrt();
    let v_a = (mu_earth * (1.0 - e) / (a * (1.0 + e))).sqrt();
    let dt = 10.0;

    let body = TranslationalState {
        position: DVec3::new(r_p, 0.0, 0.0),
        velocity: DVec3::new(0.0, v_p, 0.0),
    };
    let mut sim = make_earth_lvlh_sim(dt, mu_earth, body);
    sim.validate().unwrap();

    let period = 2.0 * std::f64::consts::PI * (a * a * a / mu_earth).sqrt();
    // Angular momentum magnitude (constant along orbit).
    let h_mag = r_p * v_p;
    let expected_omega_peri = h_mag / (r_p * r_p);
    let expected_omega_apo = h_mag / (r_a * r_a);

    // Step through first full period, tracking observed omega near perigee
    // (small r) and apogee (large r) by finding the radius extrema.
    let n_steps = (period / dt) as usize;
    let mut min_r = f64::INFINITY;
    let mut max_r = 0.0_f64;
    let mut omega_at_min_r = 0.0_f64;
    let mut omega_at_max_r = 0.0_f64;
    for step in 1..=n_steps {
        sim.step_until(step as f64 * dt).expect("step_until failed");
        let body = sim.body(0);
        let r = body.trans.position.raw_si().length();
        let lvlh = body.lvlh_frame.expect("LVLH not computed");
        let omega = lvlh.ang_vel_this.length();
        if r < min_r {
            min_r = r;
            omega_at_min_r = omega;
        }
        if r > max_r {
            max_r = r;
            omega_at_max_r = omega;
        }
    }

    // Perigee check. Perigee is close to but might not exactly hit r_p on
    // our sampled grid; allow a small relative tolerance.
    assert!(
        ((min_r - r_p).abs() / r_p) < 5e-4,
        "perigee radius drift: min_r = {min_r}, expected {r_p}"
    );
    assert!(
        ((max_r - r_a).abs() / r_a) < 5e-4,
        "apogee radius drift: max_r = {max_r}, expected {r_a}"
    );
    // Omega ratio (perigee/apogee) should match (r_a/r_p)².
    let expected_ratio = (r_a / r_p).powi(2);
    let observed_ratio = omega_at_min_r / omega_at_max_r;
    assert!(
        ((observed_ratio - expected_ratio).abs() / expected_ratio) < 5e-3,
        "omega ratio: observed {observed_ratio}, expected {expected_ratio}"
    );
    // Absolute omega values within a few parts per thousand (sampling error).
    assert!(
        ((omega_at_min_r - expected_omega_peri).abs() / expected_omega_peri) < 5e-3,
        "omega at perigee: {omega_at_min_r}, expected {expected_omega_peri}"
    );
    assert!(
        ((omega_at_max_r - expected_omega_apo).abs() / expected_omega_apo) < 5e-3,
        "omega at apogee: {omega_at_max_r}, expected {expected_omega_apo}"
    );
    // Also sanity-check v_a against v_p via conservation of angular momentum:
    // h = r_p v_p = r_a v_a. (Use it to prevent dead_code warnings on v_a.)
    let predicted_v_a = h_mag / r_a;
    assert!(
        (predicted_v_a - v_a).abs() / v_a < 1e-12,
        "v_a consistency: predicted {predicted_v_a}, closed-form {v_a}"
    );
}

#[test]
fn tier3_lvlh_periodicity() {
    // After one orbital period, a circular Keplerian orbit returns to its
    // initial state, so the LVLH frame (which is a function of r,v) must
    // return to its initial orientation. Because the simulation pipeline
    // stepper advances the inertial state first and then computes derived
    // state, we perform a "cold" LVLH computation from the initial conditions
    // and compare it to the pipeline-reported LVLH after exactly one period.
    //
    // We also tune the step size so that the configured period is an exact
    // integer multiple of dt — otherwise any LVLH mismatch is dominated by
    // rounding the stop time to a grid point rather than by physics error.
    let mu_earth = load_mu_earth();
    let r = 6_778_137.0;
    let v = (mu_earth / r).sqrt();
    let period = 2.0 * std::f64::consts::PI * (r * r * r / mu_earth).sqrt();
    // Integer number of steps; choose n_steps ≈ 560 then set dt = period/n_steps.
    let n_steps: usize = 560;
    let dt = period / n_steps as f64;

    let initial_state = TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        velocity: DVec3::new(0.0, v, 0.0),
    };
    let mut sim = make_earth_lvlh_sim(dt, mu_earth, initial_state);
    sim.validate().unwrap();

    // Reference LVLH from the initial conditions (independent of the sim).
    let lvlh_reference =
        astrodyn::compute_body_lvlh_frame(initial_state.position, initial_state.velocity)
            .t_parent_this;

    // Propagate exactly one orbital period.
    sim.step_until(n_steps as f64 * dt)
        .expect("step_until failed");
    let lvlh_final = sim
        .body(0)
        .lvlh_frame
        .expect("LVLH not computed")
        .t_parent_this;

    // For an exact-integer-step period, the only residual is RK4 truncation
    // over one orbit on a 10-s scale step, which is well under 1e-5 for a
    // circular LEO.
    let max_diff = max_mat_diff(&lvlh_reference, &lvlh_final);
    assert!(
        max_diff < 1e-5,
        "LVLH frame failed to return to initial orientation: max diff = {max_diff:.3e}"
    );

    // Additionally verify the orbit position at one period is close to the
    // initial position (confirms we did indeed complete ~1 full revolution).
    let pos_return_err = (sim.body(0).trans.position.raw_si() - initial_state.position).length();
    // RK4 truncation over one orbit on a dt ≈ period/560 is tens of cm.
    assert!(
        pos_return_err < 1.0,
        "position not returned to initial within 1 m after 1 period: \
         err = {pos_return_err:.3e} m"
    );
}
