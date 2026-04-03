//! Integration tests for interaction physics.
//!
//! These tests propagate orbits using the pure `jeod_*` crates (no Bevy)
//! and verify that interaction forces produce physically correct behavior.

use glam::{DMat3, DVec3};
use jeod_atmosphere::met;
use jeod_dynamics::{
    rk4_sixdof_step, rk4_translational_step, MassProperties, RotationalState, SixDofState,
    TranslationalState,
};
use jeod_interactions::{
    compute_ballistic_drag, compute_flat_plate_srp_thermal, compute_gravity_torque,
    compute_shadow_fraction, solar_flux_at_distance, DragConfig, FlatPlate, FlatPlateParams,
    FlatPlateThermal,
};
use jeod_math::geodetic::cartesian_to_geodetic;
use jeod_math::JeodQuat;

const MU_EARTH: f64 = 3.986_004_415e14; // m^3/s^2
const R_EARTH: f64 = 6_378_137.0; // m
const R_EARTH_POL: f64 = R_EARTH * (1.0 - 1.0 / 298.257_223_563); // JEOD: r_eq * (1 - flat_coeff)
const R_SUN: f64 = 6.98e8; // m

/// Compute GMST in radians (same formula as MET model / JEOD atmos_MET_TME.cc).
fn compute_gmst(tjt: f64) -> f64 {
    let tjt_prev_midnight = tjt.floor();
    let fraction_of_day = tjt - tjt_prev_midnight;
    let century_days = tjt_prev_midnight + 24980.0;
    let century_frac = (century_days + 0.5) / 36525.0;
    let minutes_of_day = fraction_of_day * 1440.0;
    let greenwich_mean_position = (99.6909833
        + 36000.76892 * century_frac
        + 0.00038708 * century_frac * century_frac
        + 0.250684477 * minutes_of_day)
        .rem_euclid(360.0);
    greenwich_mean_position * 0.017453293
}

/// Compute point-mass gravity acceleration.
fn gravity_accel(pos: DVec3) -> DVec3 {
    let r = pos.length();
    -MU_EARTH / (r * r * r) * pos
}

/// Compute point-mass gravity gradient tensor at a position.
fn gravity_gradient(pos: DVec3) -> DMat3 {
    let r = pos.length();
    let r2 = r * r;
    let r5 = r2 * r2 * r;
    let mu_r5 = MU_EARTH / r5;

    // G_ij = mu * (3*r_i*r_j / r^5 - delta_ij / r^3)
    let mu_r3 = MU_EARTH / (r2 * r);
    DMat3::from_cols(
        DVec3::new(
            mu_r5 * 3.0 * pos.x * pos.x - mu_r3,
            mu_r5 * 3.0 * pos.y * pos.x,
            mu_r5 * 3.0 * pos.z * pos.x,
        ),
        DVec3::new(
            mu_r5 * 3.0 * pos.x * pos.y,
            mu_r5 * 3.0 * pos.y * pos.y - mu_r3,
            mu_r5 * 3.0 * pos.z * pos.y,
        ),
        DVec3::new(
            mu_r5 * 3.0 * pos.x * pos.z,
            mu_r5 * 3.0 * pos.y * pos.z,
            mu_r5 * 3.0 * pos.z * pos.z - mu_r3,
        ),
    )
}

/// LEO orbit with drag: altitude should decrease over time.
///
/// Propagate an ISS-like orbit (400 km, circular) with MET atmosphere
/// (solar mean) and ballistic drag for 24 hours. Verify that the
/// semi-major axis decreases by a physically realistic amount.
#[test]
fn drag_causes_altitude_decay() {
    let altitude = 400_000.0; // m
    let r0 = R_EARTH + altitude;
    let v0 = (MU_EARTH / r0).sqrt();

    let mut state = TranslationalState {
        position: DVec3::new(r0, 0.0, 0.0),
        velocity: DVec3::new(0.0, v0, 0.0),
    };

    let drag_config = DragConfig {
        cd: 2.2,
        area: 1900.0, // ISS-like
        constant_density: None,
    };

    let atmos = met::SOLAR_MEAN;
    let mass = 420_000.0; // kg (ISS mass)
    let tjt_start = jeod_time::epoch::J2000_TAI_TJT;

    let dt = 60.0; // 1-minute steps
    let total_time = 86400.0; // 24 hours
    let steps = (total_time / dt) as usize;

    let initial_energy = 0.5 * v0 * v0 - MU_EARTH / r0;

    for step in 0..steps {
        let tjt = tjt_start + (step as f64 * dt) / 86400.0;
        let new_state = rk4_translational_step(
            &state,
            |s| {
                let grav = gravity_accel(s.position);

                // Rotate inertial → planet-fixed via GMST, then geodetic coords.
                // Matches JEOD's PlanetFixedPosition → MET pipeline.
                let gmst = compute_gmst(tjt);
                let (cos_g, sin_g) = (gmst.cos(), gmst.sin());
                let pfix = DVec3::new(
                    cos_g * s.position.x + sin_g * s.position.y,
                    -sin_g * s.position.x + cos_g * s.position.y,
                    s.position.z,
                );
                let geo = cartesian_to_geodetic(pfix, R_EARTH, R_EARTH_POL);
                let atmos_state =
                    atmos.density(geo.altitude / 1000.0, geo.latitude, geo.longitude, tjt);

                let drag = compute_ballistic_drag(
                    &drag_config,
                    &atmos_state,
                    s.velocity,
                    &DMat3::IDENTITY,
                );

                grav + drag.force / mass
            },
            dt,
        );
        state = new_state;
    }

    let final_r = state.position.length();
    let final_v = state.velocity.length();
    let final_energy = 0.5 * final_v * final_v - MU_EARTH / final_r;

    // Drag should reduce energy (make it more negative)
    assert!(
        final_energy < initial_energy,
        "Drag should reduce orbital energy: initial={initial_energy}, final={final_energy}"
    );

    // Semi-major axis should decrease
    let initial_sma = -MU_EARTH / (2.0 * initial_energy);
    let final_sma = -MU_EARTH / (2.0 * final_energy);
    let sma_decay = initial_sma - final_sma;

    assert!(sma_decay > 0.0, "Semi-major axis should decrease with drag");

    // ISS loses ~50-300 m/day depending on solar activity.
    // MET solar mean at 400 km should produce realistic decay.
    assert!(
        sma_decay > 50.0 && sma_decay < 1000.0,
        "SMA should decay ~100-300 m over 24h with MET solar mean, got {} m",
        sma_decay
    );
}

/// SRP force causes secular eccentricity change on a Sun-pointing orbit.
///
/// For a circular orbit in the ecliptic plane with the Sun along +X,
/// SRP creates a constant anti-Sun force. This should cause the orbit
/// to develop eccentricity over time.
#[test]
fn srp_changes_eccentricity() {
    let altitude = 36_000_000.0; // GEO-like (high orbit, SRP is more significant)
    let r0 = R_EARTH + altitude;
    let v0 = (MU_EARTH / r0).sqrt();

    let mut state = TranslationalState {
        position: DVec3::new(r0, 0.0, 0.0),
        velocity: DVec3::new(0.0, v0, 0.0),
    };

    // Single flat plate facing +X (toward Sun) as a simple absorber
    let plates = vec![(
        FlatPlate {
            area: 100.0,
            normal: DVec3::X,
            position: DVec3::ZERO,
        },
        FlatPlateParams {
            albedo: 0.0,
            diffuse: 0.0,
        },
        FlatPlateThermal {
            emissivity: 0.0,
            heat_capacity_per_area: 50.0,
        },
    )];
    let t_pow4_cached = vec![270.0_f64.powi(4)];

    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);
    let mass = 1000.0; // kg

    let dt = 60.0;
    let total_time = 86400.0 * 7.0; // 7 days
    let steps = (total_time / dt) as usize;

    for _ in 0..steps {
        let new_state = rk4_translational_step(
            &state,
            |s| {
                let grav = gravity_accel(s.position);
                let sun_to_vehicle = s.position - sun_pos;
                let dist = sun_to_vehicle.length();
                if dist < 1.0 {
                    return grav;
                }
                let flux_hat = sun_to_vehicle / dist;
                let flux_mag = solar_flux_at_distance(dist);
                let srp = compute_flat_plate_srp_thermal(
                    &plates,
                    &t_pow4_cached,
                    flux_hat,
                    flux_mag,
                    DVec3::ZERO,
                    1.0,
                );
                grav + srp.force / mass
            },
            dt,
        );
        state = new_state;
    }

    // Compute eccentricity from state vectors
    let r_vec = state.position;
    let v_vec = state.velocity;
    let r = r_vec.length();
    let h = r_vec.cross(v_vec);
    let e_vec = v_vec.cross(h) / MU_EARTH - r_vec / r;
    let eccentricity = e_vec.length();

    // SRP should have introduced non-zero eccentricity
    assert!(
        eccentricity > 1e-8,
        "SRP should introduce eccentricity, got {eccentricity}"
    );

    // But it should still be small (SRP is a weak force)
    assert!(
        eccentricity < 0.01,
        "Eccentricity should be small, got {eccentricity}"
    );
}

/// Gravity gradient torque causes attitude oscillation for a nadir-pointing body.
///
/// A body with asymmetric inertia at zero angular velocity should experience
/// gravity gradient torque that creates libration (pendulum-like attitude motion).
#[test]
fn gravity_torque_causes_libration() {
    let altitude = 400_000.0;
    let r0 = R_EARTH + altitude;
    let v0 = (MU_EARTH / r0).sqrt();

    // Asymmetric inertia (ISS-like: long axis along velocity)
    let inertia = DMat3::from_diagonal(DVec3::new(1e7, 5e7, 5e7));
    let inverse_inertia = DMat3::from_diagonal(DVec3::new(1e-7, 2e-8, 2e-8));
    let mass_props = MassProperties {
        mass: 420_000.0,
        inverse_mass: 1.0 / 420_000.0,
        inertia,
        inverse_inertia,
        position: DVec3::ZERO,
    };

    // Start with body tilted 5° from nadir about Y
    let theta = 5.0_f64.to_radians();
    let q = JeodQuat::new((theta / 2.0).cos(), 0.0, (theta / 2.0).sin(), 0.0);

    let mut state = SixDofState {
        trans: TranslationalState {
            position: DVec3::new(r0, 0.0, 0.0),
            velocity: DVec3::new(0.0, v0, 0.0),
        },
        rot: RotationalState {
            quaternion: q,
            ang_vel_body: DVec3::ZERO,
        },
    };

    let dt = 1.0;
    let steps = 3600; // 1 hour
    let mut max_omega = 0.0_f64;

    for _ in 0..steps {
        let grad = gravity_gradient(state.trans.position);
        let t_parent_this = state.rot.quaternion.left_quat_to_transformation();
        let grav_torque = compute_gravity_torque(&grad, &t_parent_this, &inertia);

        let new_state = rk4_sixdof_step(
            &state,
            |s| gravity_accel(s.trans.position),
            |_s| grav_torque, // constant torque per step
            &mass_props,
            dt,
        );

        max_omega = max_omega.max(new_state.rot.ang_vel_body.length());
        state = new_state;
    }

    // Gravity gradient torque should have induced angular velocity
    assert!(
        max_omega > 1e-6,
        "Gravity torque should cause angular motion, max omega = {max_omega}"
    );

    // But libration should be bounded (pendulum, not runaway)
    assert!(
        max_omega < 0.1,
        "Libration should be bounded, max omega = {max_omega} rad/s"
    );
}

/// Shadow detection: vehicle in LEO experiences eclipse each orbit.
///
/// A vehicle in a 400 km circular orbit should spend ~36% of each orbit
/// in Earth's shadow (for a Sun along +X, equatorial orbit).
#[test]
fn leo_eclipse_fraction() {
    let altitude = 400_000.0;
    let r0 = R_EARTH + altitude;
    let v0 = (MU_EARTH / r0).sqrt();
    let period = 2.0 * std::f64::consts::PI * (r0.powi(3) / MU_EARTH).sqrt();

    let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);
    let body_pos = DVec3::ZERO; // Earth at origin

    let dt = 10.0;
    let steps = (period / dt) as usize;
    let mut shadow_count = 0;

    let mut state = TranslationalState {
        position: DVec3::new(r0, 0.0, 0.0),
        velocity: DVec3::new(0.0, v0, 0.0),
    };

    for _ in 0..steps {
        let frac = compute_shadow_fraction(state.position, sun_pos, body_pos, R_EARTH, R_SUN);
        if frac < 0.5 {
            shadow_count += 1;
        }
        state = rk4_translational_step(&state, |s| gravity_accel(s.position), dt);
    }

    let eclipse_fraction = shadow_count as f64 / steps as f64;

    // For equatorial orbit, ~35-37% in shadow
    assert!(
        eclipse_fraction > 0.25 && eclipse_fraction < 0.45,
        "Eclipse fraction should be ~35%, got {:.1}%",
        eclipse_fraction * 100.0
    );
}
