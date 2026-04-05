//! Tier 3: SIM_dyncomp RUN_7A–7D — Spherical harmonics + Sun/Moon third-body (± drag)
//!
//! Validates combined high-fidelity gravity (SH 4x4 or 8x8 + Sun/Moon
//! differential 3rd-body acceleration) with and without aerodynamic drag.
//!
//! Scenarios from JEOD SIM_dyncomp:
//! - RUN_7A: 4x4 SH + Sun/Moon, no drag, elliptical orbit, 8h
//! - RUN_7B: 8x8 SH + Sun/Moon, no drag, elliptical orbit, 8h
//! - RUN_7C: 4x4 SH + Sun/Moon + drag (Cd=0.02, A=1 m²), elliptical orbit, 8h
//! - RUN_7D: 8x8 SH + Sun/Moon + drag, elliptical orbit, 8h
//!
//! All runs use sphere mass (1 kg, I=0.4*Identity), MET atmosphere (solar mean),
//! and GGM02C gravity coefficients with RNP Earth rotation.
//!
//! Sun and Moon positions are queried from DE421 ephemeris at each logged 60s sample.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_sim::{
    met_atmosphere, AtmosphereConfig, AtmosphereModel, DragConfig, DynamicsConfig, Ephemeris,
    EphemerisBody, GravityControl, GravityControls, GravityModel, GravitySource,
    GravitySourceEntry, JeodQuat, MassProperties, MetAtmosphere, RotationalState, SimBody,
    Simulation, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use std::path::Path;

const MU_SUN: f64 = 1.327_124_40e20;
const MU_MOON: f64 = 4902.79980693169e9;

/// Epoch constants (same as other SIM_dyncomp tests: Nov 20 2007 00:00 UTC).
const EPOCH_UTC_TJT: f64 = 14424.0;
const TAI_UTC_S: f64 = 32.0;
const TAI_TO_UT1_S: f64 = -32.469;

fn earth_centered_position(body: EphemerisBody, tdb_jd: f64, ephemeris: &Ephemeris) -> DVec3 {
    let (pos, _) = ephemeris
        .get_earth_centered_state(body, tdb_jd)
        .expect("ephemeris query failed");
    pos
}

#[allow(clippy::too_many_arguments)]
fn run_7_test(
    csv_name: &str,
    degree: usize,
    order: usize,
    with_drag: bool,
    label: &str,
    test_name: &str,
    pos_tol: [f64; 3],
    vel_tol: [f64; 3],
) {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    let csv_path = test_data_path(csv_name);
    assert!(
        csv_path.exists(),
        "JEOD trajectory not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let bsp_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/de421.bsp");
    assert!(
        bsp_path.exists(),
        "DE421 not found at {}",
        bsp_path.display()
    );
    let ephemeris = Ephemeris::from_bsp(&bsp_path).expect("load DE421");

    // Load GGM02C spherical harmonics coefficients (same as RUN_3 tests)
    let ggm02c_path = jeod_root.join("models/environment/gravity/data/src/earth_GGM02C.cc");
    let sh_data = jeod_sim::coefficients::load_from_jeod_cc(&ggm02c_path).expect("load GGM02C");

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(trajectory.len() > 100);
    let init = &trajectory[0];

    // Initialize Simulation at the SIM_dyncomp epoch
    let epoch_tai_tjt = EPOCH_UTC_TJT + TAI_UTC_S / 86400.0;
    let mut time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    time.set_ut1_tai_offset(TAI_TO_UT1_S);

    let mut sim = Simulation::new(time, DT);

    // Earth source with SH gravity and planet-fixed rotation
    let sh_source = GravitySource {
        mu: sh_data.mu,
        model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
    };
    let earth = sim.add_source(GravitySourceEntry {
        source: sh_source,
        position: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY),
    });

    // Sun and Moon: third-body differential acceleration
    let tdb_jd = sim.time.tdb_julian_date();
    let sun = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_SUN,
            model: GravityModel::PointMass,
        },
        position: earth_centered_position(EphemerisBody::Sun, tdb_jd, &ephemeris),
        t_inertial_pfix: None,
    });
    let moon = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_MOON,
            model: GravityModel::PointMass,
        },
        position: earth_centered_position(EphemerisBody::Moon, tdb_jd, &ephemeris),
        t_inertial_pfix: None,
    });

    // Drag configuration (only for RUN_7C/7D)
    let drag_config = if with_drag {
        // MET atmosphere: solar mean (from Modified_data/solar_flux.py)
        let met_model = MetAtmosphere {
            f10: 128.8,
            f10b: 128.8,
            geo_index: 15.7,
            geo_index_type: met_atmosphere::GeoIndexType::Ap,
        };
        sim.atmosphere = Some(AtmosphereConfig {
            model: AtmosphereModel::Met(met_model),
            r_eq: 6_378_137.0,
            r_pol: 6_378_137.0 * (1.0 - 1.0 / 298.257_223_563),
            planet_omega: OMEGA_EARTH,
        });
        sim.atmosphere_planet_source = Some(earth);

        Some(DragConfig {
            cd: 0.02,
            area: 1.0,
            constant_density: None,
        })
    } else {
        None
    };

    // Sphere mass: 1 kg, inertia 0.4*I, CoM at origin (from Modified_data/mass.py)
    let sphere_mass =
        MassProperties::with_inertia(1.0, DMat3::from_diagonal(DVec3::splat(0.4)), DVec3::ZERO);

    // Drag requires RotationalState for inertial-to-body frame transform.
    // Even for a sphere (isotropic drag), the code path needs orientation.
    let (rot, config) = if with_drag {
        (
            Some(RotationalState {
                quaternion: JeodQuat::from_glam(init.composite_body.quaternion),
                ang_vel_body: init.composite_body.ang_vel,
            }),
            DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            },
        )
    } else {
        (None, DynamicsConfig::default())
    };

    let mut body = SimBody {
        trans: TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        },
        rot,
        mass: Some(sphere_mass),
        config,
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_nonspherical(earth, degree, order, false),
                GravityControl::new_third_body(sun),
                GravityControl::new_third_body(moon),
            ],
        },
        ..Default::default()
    };

    if let Some(dc) = drag_config {
        body.drag = Some(dc);
        body.atmospheric_state = Some(Default::default());
    }

    sim.add_body(body);
    sim.validate().unwrap();

    println!("Tier 3 (Simulation): {label}, {} points", trajectory.len());

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    for record in &trajectory[1..] {
        // Update Sun/Moon positions from ephemeris
        let target_tdb_jd = tdb_jd + record.time / 86400.0;
        sim.sources[sun].position =
            earth_centered_position(EphemerisBody::Sun, target_tdb_jd, &ephemeris);
        sim.sources[moon].position =
            earth_centered_position(EphemerisBody::Moon, target_tdb_jd, &ephemeris);

        sim.step_until(record.time);
        let body = sim.body(0);

        if (record.time % 3600.0).abs() < 30.1 {
            let pos_error = (body.trans.position - record.composite_body.position).length();
            println!("  t={:6.0}s: pos_err={:10.4} m", record.time, pos_error);
        }

        our_states.push(StateLog {
            time: record.time,
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            acceleration: Some(body.frame_derivs.trans_accel),
            ang_accel: Some(body.frame_derivs.rot_accel),
            ..Default::default()
        });
    }

    let ref_states: Vec<StateLog> = trajectory[1..]
        .iter()
        .map(|r| StateLog {
            time: r.time,
            position: Some(r.composite_body.position),
            velocity: Some(r.composite_body.velocity),
            acceleration: r.derivs.as_ref().map(|d| d.trans_accel),
            ang_accel: r.derivs.as_ref().map(|d| d.rot_accel),
            ..Default::default()
        })
        .collect();

    let report = CrossvalReport::compute(test_name, &our_states, &ref_states);
    report.write();

    println!(
        "  Max position error: {:.6e} m",
        report.max_position_component()
    );
    println!(
        "  Max velocity error: {:.6e} m/s",
        report.max_velocity_component()
    );

    report.assert_position(pos_tol);
    report.assert_velocity(vel_tol);
}

// Provisional tolerances: set high initially, tighten after first run.

#[test]
fn tier3_simulation_run7a_sh4x4_3rd_body() {
    run_7_test(
        "dyncomp_run7a_state.csv",
        4,
        4,
        false,
        "RUN_7A (4x4 SH + Sun/Moon, no drag)",
        "tier3_simulation_run7a",
        [5.13e-2, 1.316e-1, 9.986e-2],
        [6.041e-5, 1.206e-4, 1.218e-4],
    );
}

#[test]
fn tier3_simulation_run7b_sh8x8_3rd_body() {
    run_7_test(
        "dyncomp_run7b_state.csv",
        8,
        8,
        false,
        "RUN_7B (8x8 SH + Sun/Moon, no drag)",
        "tier3_simulation_run7b",
        [1.28e-1, 2.25e-1, 1.597e-1],
        [1.447e-4, 2.25e-4, 1.856e-4],
    );
}

#[test]
fn tier3_simulation_run7c_sh4x4_3rd_body_drag() {
    run_7_test(
        "dyncomp_run7c_state.csv",
        4,
        4,
        true,
        "RUN_7C (4x4 SH + Sun/Moon + drag)",
        "tier3_simulation_run7c",
        [6.988e-1, 1.038, 8.523e-1],
        [7.06e-4, 1.156e-3, 9.565e-4],
    );
}

#[test]
fn tier3_simulation_run7d_sh8x8_3rd_body_drag() {
    run_7_test(
        "dyncomp_run7d_state.csv",
        8,
        8,
        true,
        "RUN_7D (8x8 SH + Sun/Moon + drag)",
        "tier3_simulation_run7d",
        [7.735e-1, 1.126, 9.118e-1],
        [7.898e-4, 1.259e-3, 1.03e-3],
    );
}
