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

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_sim::{
    DynamicsConfig, Ephemeris, EphemerisBody, GravityControl, GravityControls, GravityModel,
    GravitySource, GravitySourceEntry, JeodQuat, MassProperties, RotationalState, SimBody,
    Simulation, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use std::path::Path;

const MU_SUN: f64 = 1.327_124_40e20;
const MU_MOON: f64 = 4902.79980693169e9;

/// SIM_dyncomp epoch: 2007-11-20 midnight UTC, same as all other RUN_* tests.
const DYNCOMP_UTC_TJT: f64 = 14424.0;
const DYNCOMP_TAI_UTC_S: f64 = 32.0;
const DYNCOMP_UT1_TAI_S: f64 = -32.469;

/// Compute Earth-centered position of a body from DE421 ephemeris.
fn earth_centered_position(body: EphemerisBody, tdb_jd: f64, ephemeris: &Ephemeris) -> DVec3 {
    let (pos, _) = ephemeris
        .get_earth_centered_state(body, tdb_jd)
        .expect("ephemeris query failed");
    pos
}

#[test]
fn tier3_simulation_run4_3rd_body() {
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

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(trajectory.len() > 100);

    let init = &trajectory[0];

    // Initialize at the SIM_dyncomp epoch (2007-11-20 UTC) so DE421 Sun/Moon
    // queries match the JEOD reference sim's absolute time.
    let epoch_tai_tjt = DYNCOMP_UTC_TJT + DYNCOMP_TAI_UTC_S / 86400.0;
    let mut time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    time.set_ut1_tai_offset(DYNCOMP_UT1_TAI_S);
    let mut sim = Simulation::new(time, DT);

    // Earth: central body at origin (not differential)
    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        tidal_config: None,
    });

    // Sun: third-body (differential acceleration)
    let tdb_jd = sim.time.tdb_julian_date();
    let initial_sun = earth_centered_position(EphemerisBody::Sun, tdb_jd, &ephemeris);
    let sun = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_SUN,
            model: GravityModel::PointMass,
        },
        position: initial_sun,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        tidal_config: None,
    });

    // Moon: third-body (differential acceleration)
    let initial_moon = earth_centered_position(EphemerisBody::Moon, tdb_jd, &ephemeris);
    let moon = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_MOON,
            model: GravityModel::PointMass,
        },
        position: initial_moon,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        tidal_config: None,
    });

    // ISS mass properties (same as RUN_2 6-DOF test)
    let inertia = glam::DMat3::from_cols(
        DVec3::new(1.02e8, -6.96e6, -5.48e6),
        DVec3::new(-6.96e6, 0.91e8, 5.90e5),
        DVec3::new(-5.48e6, 5.90e5, 1.64e8),
    );
    let mass_props = MassProperties::with_inertia(400_000.0, inertia, DVec3::new(-3.0, -1.5, 4.0));

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
