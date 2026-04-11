//! Tier 3: SIM_dyncomp RUN_4 — Spherical gravity + Sun/Moon third-body
//!
//! This test validates differential (third-body) gravity acceleration.
//! Earth uses standard point-mass gravity; Sun and Moon use differential
//! acceleration (vehicle toward Sun/Moon minus Earth toward Sun/Moon).
//!
//! Scenario from JEOD SIM_dyncomp RUN_4:
//! - Earth: spherical gravity (central body)
//! - Sun/Moon: spherical gravity (third-body, differential)
//! - No drag, no gravity gradient torque
//! - ISS mass/orbit, 28800s (8h), 60s logging
//!
//! Sun and Moon positions are queried from the DE421 ephemeris at each
//! logged 60s sample.
//!
//! All simulation parameters (epoch, step size, mu values, mass) are loaded
//! from the JEOD source files rather than hardcoded, per issue #44.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_sim::{
    DynamicsConfig, Ephemeris, EphemerisBody, GravityControl, GravityControls, GravityModel,
    GravitySource, GravitySourceEntry, JeodQuat, MassProperties, RotationModel, RotationalState,
    SimBody, Simulation, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use std::path::Path;

/// SIM_dyncomp root directory (relative to JEOD_HOME).
const SIM_DYNCOMP: &str = "verif/SIM_dyncomp";

/// Compute Earth-centered position of a body from DE421 ephemeris.
fn earth_centered_position(body: EphemerisBody, tdb_jd: f64, ephemeris: &Ephemeris) -> DVec3 {
    let (pos, _) = ephemeris
        .get_earth_centered_state(body, tdb_jd)
        .expect("ephemeris query failed");
    pos
}

#[test]
fn tier3_simulation_run4_3rd_body() {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    let csv_path = test_data_path("dyncomp_run4_state.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let bsp_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/de421.bsp");
    assert!(
        bsp_path.exists(),
        "DE421 ephemeris not found at {}",
        bsp_path.display()
    );
    let ephemeris = Ephemeris::from_bsp(&bsp_path).expect("load DE421");

    let sim_dir = jeod_root.join(SIM_DYNCOMP);
    let grav_data_dir = jeod_root.join("models/environment/gravity/data/src");

    // Load epoch and time offsets from JEOD time config
    let time_cfg =
        jeod_test_data::time_config::load_time_config(&sim_dir.join("Modified_data/time.py"));
    let epoch_tai_tjt = time_cfg.tai_tjt();
    let ut1_tai_offset = time_cfg
        .ut1_tai_offset()
        .expect("SIM_dyncomp time.py must specify tai_to_ut1_override_val");

    // Load integration step size from S_define
    let dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));

    // Load mu values from JEOD gravity coefficient files.
    // Sun/Moon use load_mu (spherical-only files lack degree/order fields).
    let earth_grav =
        jeod_sim::coefficients::load_from_jeod_cc(&grav_data_dir.join("earth_GGM05C.cc"))
            .expect("load Earth gravity");
    let mu_sun =
        jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_data_dir.join("sun_spherical.cc"))
            .expect("load Sun mu");
    let mu_moon =
        jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_data_dir.join("moon_GRAIL150.cc"))
            .expect("load Moon mu");

    // Load ISS mass properties from SIM_dyncomp mass.py
    let mass_init = jeod_test_data::mass_data::load_mass_from_file(
        &sim_dir.join("Modified_data/mass.py"),
        Some("set_mass_iss"),
    );

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(trajectory.len() > 100);

    let init = &trajectory[0];

    // Initialize at the SIM_dyncomp epoch (parsed from time.py) so DE421 Sun/Moon
    // queries match the JEOD reference sim's absolute time.
    let mut time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    time.set_ut1_tai_offset(ut1_tai_offset);
    let mut sim = Simulation::new(time, dt);

    // Earth: central body at origin (not differential)
    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: earth_grav.mu,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });

    // Sun: third-body (differential acceleration)
    let tdb_jd = sim.time.tdb_julian_date();
    let initial_sun = earth_centered_position(EphemerisBody::Sun, tdb_jd, &ephemeris);
    let sun = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: mu_sun,
            model: GravityModel::PointMass,
        },
        position: initial_sun,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });

    // Moon: third-body (differential acceleration)
    let initial_moon = earth_centered_position(EphemerisBody::Moon, tdb_jd, &ephemeris);
    let moon = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: mu_moon,
            model: GravityModel::PointMass,
        },
        position: initial_moon,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });

    // ISS mass properties (parsed from Modified_data/mass.py)
    let inertia = glam::DMat3::from_cols(
        DVec3::new(
            mass_init.inertia[0][0],
            mass_init.inertia[1][0],
            mass_init.inertia[2][0],
        ),
        DVec3::new(
            mass_init.inertia[0][1],
            mass_init.inertia[1][1],
            mass_init.inertia[2][1],
        ),
        DVec3::new(
            mass_init.inertia[0][2],
            mass_init.inertia[1][2],
            mass_init.inertia[2][2],
        ),
    );
    let mass_props = MassProperties::with_inertia(
        mass_init.mass,
        inertia,
        DVec3::from_slice(&mass_init.position),
    );

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::from_glam(init.composite_body.quaternion),
            ang_vel_body: init.composite_body.ang_vel,
        }),
        mass: Some(mass_props),
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: true,
            three_dof: false,
        },
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_spherical(earth, false),
                GravityControl::new_third_body(sun),
                GravityControl::new_third_body(moon),
            ],
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): RUN_4 spherical + Sun/Moon 3rd-body, {} points",
        trajectory.len()
    );

    // Propagate, updating Sun/Moon positions from ephemeris each logging interval.
    // Per-step updates were tested but give identical error (~37 m) — the residual
    // is from DE421 interpolation differences between Anise and JEOD's native
    // reader (~10 arcsecond Sun direction offset, see simnaut/bevy_jeod#27).
    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    for record in &trajectory[1..] {
        // Update ephemeris-driven source positions before stepping.
        // Compute TDB JD for the target time using the epoch's TDB JD plus
        // elapsed simulation days. This uses the proper TDB timescale.
        let target_tdb_jd = tdb_jd + record.time / 86400.0;
        sim.sources[sun].position =
            earth_centered_position(EphemerisBody::Sun, target_tdb_jd, &ephemeris);
        sim.sources[moon].position =
            earth_centered_position(EphemerisBody::Moon, target_tdb_jd, &ephemeris);

        sim.step_until(record.time);

        let body = sim.body(0);
        let rot = body.rot.as_ref().unwrap();
        our_states.push(StateLog {
            time: record.time,
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            acceleration: Some(body.frame_derivs.trans_accel),
            quaternion: Some(rot.quaternion.to_glam()),
            ang_vel: Some(rot.ang_vel_body),
            ang_accel: Some(body.frame_derivs.rot_accel),
        });
    }

    // Reference states from JEOD CSV
    let ref_states: Vec<StateLog> = trajectory[1..]
        .iter()
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

    // Post-process: compute errors
    let report =
        CrossvalReport::compute("tier3_simulation_run4_3rd_body", &our_states, &ref_states);
    report.write();

    let max_pos = report.max_position_component();
    let max_vel = report.max_velocity_component();
    let max_quat = report.max_quat_angle();
    let max_omega = report.max_ang_vel_component();

    println!("  Max position error:   {max_pos:.6e} m");
    println!("  Max velocity error:   {max_vel:.6e} m/s");
    println!("  Max quaternion error: {max_quat:.6e} rad");
    println!("  Max omega error:      {max_omega:.6e} rad/s");

    // Tolerances: 5% above observed max error per component.
    // With the correct SIM_dyncomp epoch (2007-11-20 UTC), errors are ~2e-3 m
    // (vs ~37 m with the wrong J2000 epoch). Residual is from DE421 Anise vs
    // JEOD native reader interpolation differences (see simnaut/bevy_jeod#27).
    report.assert_position([1.644e-3, 2.098e-3, 2.025e-3]);
    report.assert_velocity([1.762e-6, 2.082e-6, 2.400e-6]);
    report.assert_quat_angle(4.426e-8);
    report.assert_ang_vel([2.619e-18, 1.367e-18, 7.969e-19]);
}
