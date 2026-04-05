//! Tier 3: Cross-validation of ballistic drag + MET atmosphere against
//! JEOD SIM_dyncomp RUN_6B reference data.
//!
//! RUN_6B configuration (from SET_test/RUN_6B/input.py):
//! - Spherical gravity (point-mass Earth, mu=3.986_004_415e14 m^3/s^2)
//! - MET atmosphere: F10.7=128.8, F10B=128.8, Ap=15.7 (solar mean)
//! - Ballistic drag: Cd=0.02, Area=1.0 m^2, DRAG_OPT_CD
//! - Unit sphere mass: mass=1.0 kg, I=diag(0.4, 0.4, 0.4), CoM at origin
//! - No gravity gradient torque, no external torques
//! - Elliptical orbit starting at ~430 km altitude
//! - Structural-to-body transform: identity (eigen_angle = 0.0)
//! - RK4, dt=0.03125s, 28800s (8 hours), 481 data points at 60s
//!
//! This validates:
//! 1. MET atmosphere density computation at varying altitudes/positions
//! 2. Ballistic drag force magnitude and direction
//! 3. Coupled gravity + drag trajectory over 8 hours
//! 4. Correct velocity-to-body-frame transformation for drag

use glam::{DMat3, DVec3};
use jeod_atmosphere::met;
use jeod_atmosphere::AtmosphereState;
use jeod_dynamics::{
    rk4_sixdof_step, MassProperties, RotationalState, SixDofState, TranslationalState,
};
use jeod_interactions::{compute_ballistic_drag, DragConfig};
use jeod_math::geodetic::cartesian_to_geodetic;
use jeod_math::JeodQuat;
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use jeod_test_data::dyncomp_csv::load_dyncomp_csv;
use std::path::Path;

const MU_EARTH: f64 = 3.986_004_415e14;
const R_EARTH_EQ: f64 = 6_378_137.0; // WGS84 equatorial radius (m)
const R_EARTH_POL: f64 = R_EARTH_EQ * (1.0 - 1.0 / 298.257_223_563); // JEOD: r_eq * (1 - flat_coeff)

/// Earth rotation rate from JEOD RNPJ2000 default data (GEM-T1 gravity model).
/// See models/environment/RNP/RNPJ2000/data/src/data_rnp_j2000.cc
const OMEGA_EARTH: f64 = 7.292_115_146_706_388e-5; // rad/s

/// Compute Greenwich Mean Sidereal Time (GMST) in radians using the same formula
/// as the MET atmosphere model (from JEOD atmos_MET_TME.cc).
///
/// This is needed to convert inertial longitude to planet-fixed longitude for
/// the MET model, which expects Earth-fixed coordinates.
fn compute_gmst(tjt: f64) -> f64 {
    let tjt_prev_midnight = tjt.floor();
    let fraction_of_day = tjt - tjt_prev_midnight;

    // Days since 1900-01-01: TJT epoch is 24980 days after 1900-01-01
    let century_days = tjt_prev_midnight + 24980.0;
    let century_frac = (century_days + 0.5) / 36525.0;

    let minutes_of_day = fraction_of_day * 1440.0;

    // Coefficients from MET model (same as Almanac GMST formula)
    let a1: f64 = 99.6909833;
    let a2: f64 = 36000.76892;
    let a3: f64 = 0.00038708;
    let a4: f64 = 0.250684477;

    let greenwich_mean_position =
        (a1 + a2 * century_frac + a3 * century_frac * century_frac + a4 * minutes_of_day)
            .rem_euclid(360.0);

    // Use the same DEG_TO_RAD as MET (Jacchia's truncated constant)
    greenwich_mean_position * 0.017453293
}

/// Epoch TJT for SIM_dyncomp: midnight 2007-11-20 UTC.
///
/// JD = 2454424.5, MJD = 54424.0, TJT = MJD - 40000 = 14424.0.
///
/// From Modified_data/time.py:
///   jeod_time.time_utc.set_date_and_time(2007, 11, 20, 0, 0, 0.0)
const EPOCH_TJT: f64 = 14424.0;

#[test]
fn tier3_drag_trajectory_run6b() {
    let csv_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/dyncomp_run6b_state.csv");

    assert!(
        csv_path.exists(),
        "Tier 3 reference data not found at {}.\n\
         Generate with:\n  \
         docker build -f trick/Dockerfile -t jeod-trick ..\n  \
         docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(
        trajectory.len() >= 100,
        "Expected at least 100 data points, got {}",
        trajectory.len()
    );

    // Unit sphere mass properties from Modified_data/mass.py (set_mass_sphere):
    // mass = 1.0 kg, CoM at origin, I = 0.4 * identity
    let inertia = DMat3::from_cols(
        DVec3::new(0.4, 0.0, 0.0),
        DVec3::new(0.0, 0.4, 0.0),
        DVec3::new(0.0, 0.0, 0.4),
    );
    let mass_props = MassProperties::with_inertia(1.0, inertia, DVec3::ZERO);
    let mass = 1.0_f64;

    // MET atmosphere: solar mean conditions from Modified_data/solar_flux.py
    let met_model = met::MetAtmosphere {
        f10: 128.8,
        f10b: 128.8,
        geo_index: 15.7,
        geo_index_type: met::GeoIndexType::Ap,
    };

    // Ballistic drag config from Modified_data/aero_drag.py (set_aero_drag_ballistic)
    let drag_config = DragConfig {
        cd: 0.02,
        area: 1.0,
        constant_density: None,
    };

    // Initialize from first JEOD record
    let init = &trajectory[0];
    let mut state = SixDofState {
        trans: TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        },
        rot: RotationalState {
            quaternion: JeodQuat::from_glam(init.composite_body.quaternion),
            ang_vel_body: init.composite_body.ang_vel,
        },
    };

    let dt = 0.03125; // match JEOD's SIM_dyncomp integration rate (32 Hz)
    let mut current_time = init.time;

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    let ref_states: Vec<StateLog> = trajectory
        .iter()
        .skip(1)
        .map(|r| StateLog {
            time: r.time,
            position: Some(r.composite_body.position),
            velocity: Some(r.composite_body.velocity),
            acceleration: r.derivs.as_ref().map(|d| d.trans_accel),
            quaternion: Some(r.composite_body.quaternion),
            ang_vel: Some(r.composite_body.ang_vel),
            ang_accel: r.derivs.as_ref().map(|d| d.rot_accel),
        })
        .collect();

    // Gravity + drag acceleration closure
    let gravity_plus_drag_accel = |s: &SixDofState, sim_time: f64| -> DVec3 {
        // Point-mass gravity
        let r_sq = s.trans.position.length_squared();
        let r_mag = r_sq.sqrt();
        let gravity = s.trans.position * (-MU_EARTH / (r_sq * r_mag));

        // MET atmosphere density at current position.
        // JEOD passes planet-fixed (PCPF) geodetic coordinates to the MET model.
        // We rotate the inertial position about Z by -GMST to get planet-fixed
        // Cartesian, then convert to geodetic (ellipsoidal) coordinates. This
        // matches JEOD's PlanetaryDerivedState -> cart_to_ellip() pipeline.
        let tjt = EPOCH_TJT + sim_time / 86400.0;
        let gmst = compute_gmst(tjt);
        let cos_g = gmst.cos();
        let sin_g = gmst.sin();
        let pfix_pos = DVec3::new(
            cos_g * s.trans.position.x + sin_g * s.trans.position.y,
            -sin_g * s.trans.position.x + cos_g * s.trans.position.y,
            s.trans.position.z,
        );
        let geo = cartesian_to_geodetic(pfix_pos, R_EARTH_EQ, R_EARTH_POL);
        let met_state = met_model.density(
            geo.altitude / 1000.0, // convert m to km
            geo.latitude,
            geo.longitude,
            tjt,
        );

        // Atmospheric co-rotation wind: omega_earth x position (in inertial frame).
        // From JEOD wind_velocity.cc: wind[0] = -omega*pos[1], wind[1] = omega*pos[0], wind[2] = 0.
        // Modified_data/uniform_wind.py sets omega = earth.rnp.planet_omega with scale=1.0.
        let wind = DVec3::new(
            -OMEGA_EARTH * s.trans.position.y,
            OMEGA_EARTH * s.trans.position.x,
            0.0,
        );

        let atmos = AtmosphereState {
            density: met_state.density,
            temperature: met_state.temperature,
            pressure: met_state.pressure,
            wind,
        };

        // Compute drag force in the body/structural frame
        // T_inertial_struct: rotation matrix from inertial to body frame
        let t_inertial_struct = s.rot.quaternion.left_quat_to_transformation();
        let aero =
            compute_ballistic_drag(&drag_config, &atmos, s.trans.velocity, &t_inertial_struct);

        // Transform drag force from body frame back to inertial frame
        // T_inertial_struct transforms inertial->body, so transpose goes body->inertial
        let drag_force_inertial = t_inertial_struct.transpose() * aero.force;
        let drag_accel = drag_force_inertial / mass;

        gravity + drag_accel
    };

    for record in trajectory.iter().skip(1) {
        // Integrate forward to this record's time
        while current_time + dt <= record.time + 0.001 {
            let t = current_time;
            state = rk4_sixdof_step(
                &state,
                |s| gravity_plus_drag_accel(s, t),
                |_s| DVec3::ZERO, // no torques (ballistic drag, no gravity gradient)
                &mass_props,
                dt,
            );
            current_time += dt;
        }
        let remainder = record.time - current_time;
        if remainder > 0.001 {
            let t = current_time;
            state = rk4_sixdof_step(
                &state,
                |s| gravity_plus_drag_accel(s, t),
                |_s| DVec3::ZERO,
                &mass_props,
                remainder,
            );
            current_time += remainder;
        }

        our_states.push(StateLog {
            time: record.time,
            position: Some(state.trans.position),
            velocity: Some(state.trans.velocity),
            quaternion: Some(state.rot.quaternion.to_glam()),
            ang_vel: Some(state.rot.ang_vel_body),
            ..Default::default()
        });

        // Compare for logging
        let pos_error = (state.trans.position - record.composite_body.position).length();
        let vel_error = (state.trans.velocity - record.composite_body.velocity).length();

        // Log progress hourly with position error, density, drag force
        let log_hourly = (record.time % 3600.0).abs() < 30.1;
        if log_hourly {
            let tjt = EPOCH_TJT + record.time / 86400.0;
            let gmst = compute_gmst(tjt);
            let cos_g = gmst.cos();
            let sin_g = gmst.sin();
            let pfix_pos = DVec3::new(
                cos_g * state.trans.position.x + sin_g * state.trans.position.y,
                -sin_g * state.trans.position.x + cos_g * state.trans.position.y,
                state.trans.position.z,
            );
            let geo = cartesian_to_geodetic(pfix_pos, R_EARTH_EQ, R_EARTH_POL);
            let met_state =
                met_model.density(geo.altitude / 1000.0, geo.latitude, geo.longitude, tjt);
            let wind = DVec3::new(
                -OMEGA_EARTH * state.trans.position.y,
                OMEGA_EARTH * state.trans.position.x,
                0.0,
            );
            let atmos = AtmosphereState {
                density: met_state.density,
                temperature: met_state.temperature,
                pressure: met_state.pressure,
                wind,
            };

            let t_inertial_struct = state.rot.quaternion.left_quat_to_transformation();
            let aero = compute_ballistic_drag(
                &drag_config,
                &atmos,
                state.trans.velocity,
                &t_inertial_struct,
            );

            println!(
                "  t={:6.0}s ({:.1}h): pos_err={:10.4}m  vel_err={:.6}m/s  \
                 alt={:.1}km  density={:.3e}kg/m3  drag={:.3e}N",
                record.time,
                record.time / 3600.0,
                pos_error,
                vel_error,
                geo.altitude / 1000.0,
                met_state.density,
                aero.force.length(),
            );
        }
    }

    let mut report =
        CrossvalReport::compute("tier3_drag_trajectory_run6b", &our_states, &ref_states);
    report.position_tol = Some([2.0; 3]);
    report.velocity_tol = Some([0.005; 3]);
    report.quat_angle_tol = Some(0.01);
    report.write();

    let max_pos_error = report.max_position_error();
    let max_vel_error = report.max_velocity_error();
    let max_quat_error = report.max_quat_angle_error();

    println!();
    println!("=== Tier 3 Drag Trajectory Cross-Validation (RUN_6B) ===");
    println!(
        "Duration: {} s ({} data points)",
        trajectory.last().unwrap().time,
        trajectory.len()
    );
    println!("Atmosphere: MET solar mean (F10.7=128.8, Ap=15.7)");
    println!("Drag: Cd=0.02, Area=1.0 m^2, mass=1.0 kg");
    println!("Max position error:   {:.6e} m", max_pos_error);
    println!("Max velocity error:   {:.6e} m/s", max_vel_error);
    println!(
        "Max quaternion error: {:.6e} rad ({:.4} deg)",
        max_quat_error,
        max_quat_error.to_degrees()
    );

    // Position threshold: tightened to 2x actual (~0.8 m) to catch regression.
    // PLAN.md exit criterion is 100 m, but actual performance is sub-meter
    // thanks to matching JEOD's geodetic pipeline, co-rotation wind, and GMST.
    assert!(
        max_pos_error < 2.0,
        "Position error {:.4} m exceeds 2.0 m threshold (regression?)",
        max_pos_error
    );
    assert!(
        max_vel_error < 0.005,
        "Velocity error {:.6} m/s exceeds 0.005 m/s threshold (regression?)",
        max_vel_error
    );

    // Quaternion threshold: the sphere has no torques, so attitude should
    // remain nearly constant (only numerical drift). Use generous threshold.
    assert!(
        max_quat_error < 0.01,
        "Quaternion angular error {:.6e} rad exceeds 0.01 rad threshold",
        max_quat_error
    );
}
