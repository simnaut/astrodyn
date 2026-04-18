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
//!
//! Simulation parameters (epoch, step size, mu values) are loaded from the JEOD
//! source files rather than hardcoded, per issue #44.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_runner::{GravitySourceEntry, RotationModel, Simulation, VehicleConfig};
use jeod_sim::{
    met_atmosphere, AtmosphereConfig, AtmosphereModel, DragConfig, Ephemeris, EphemerisBody,
    GravityControl, GravityControls, GravityModel, GravitySource, JeodQuat, MassProperties,
    MetAtmosphere, RotationalState, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use std::path::Path;

/// SIM_dyncomp root directory (relative to JEOD_HOME).
const SIM_DYNCOMP: &str = "verif/SIM_dyncomp";

fn earth_centered_position(body: EphemerisBody, tdb_jd: f64, ephemeris: &Ephemeris) -> DVec3 {
    let (pos, _) = ephemeris
        .get_earth_centered_state(body, tdb_jd)
        .expect("ephemeris query failed");
    pos
}

fn run_7_test(
    csv_name: &str,
    run_dir: &str,
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

    // Load gravity control (degree/order) from the exec chain.
    // All RUN_7* start from RUN_7A (which sets 4x4). RUN_7B/7D override to 8x8.
    // RUN_7C exec's RUN_7A, RUN_7D exec's RUN_7B.
    let mut grav_files: Vec<std::path::PathBuf> =
        vec![sim_dir.join("Modified_data/grav_controls.py")];
    // RUN_7A is always in the chain
    grav_files.push(sim_dir.join("SET_test/RUN_7A/input.py"));
    // For 8x8 variants (7B, 7D), RUN_7B adds the override
    if run_dir == "RUN_7B" || run_dir == "RUN_7D" {
        grav_files.push(sim_dir.join("SET_test/RUN_7B/input.py"));
    }
    // 7C and 7D have their own files too (mostly drag, no gravity changes)
    if run_dir == "RUN_7C" || run_dir == "RUN_7D" {
        grav_files.push(sim_dir.join(format!("SET_test/{run_dir}/input.py")));
    }
    let grav_file_refs: Vec<&std::path::Path> = grav_files.iter().map(|p| p.as_path()).collect();
    let grav_cfg = jeod_test_data::gravity_control::load_gravity_control(&grav_file_refs);

    // Load GGM02C spherical harmonics coefficients
    let ggm02c_path = grav_data_dir.join("earth_GGM02C.cc");
    let sh_data = jeod_sim::coefficients::load_from_jeod_cc(&ggm02c_path).expect("load GGM02C");

    // Load Sun/Moon mu (spherical-only files lack degree/order fields)
    let mu_sun =
        jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_data_dir.join("sun_spherical.cc"))
            .expect("load Sun mu");
    let mu_moon =
        jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_data_dir.join("moon_GRAIL150.cc"))
            .expect("load Moon mu");

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(trajectory.len() > 100);
    let init = &trajectory[0];

    // Initialize Simulation at the SIM_dyncomp epoch (parsed from time.py)
    let mut time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    time.set_ut1_tai_offset(ut1_tai_offset);

    let mut sim = Simulation::new(time, dt);

    // Earth source with SH gravity and planet-fixed rotation
    let sh_source = GravitySource {
        mu: sh_data.mu,
        model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
    };
    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: sh_source,
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: Some(DMat3::IDENTITY),
            delta_c20: 0.0,
            rotation_model: RotationModel::EarthRNP,
            tidal_config: None,
            planet_omega: OMEGA_EARTH,
            central: true,
        },
    );

    // Sun and Moon: third-body differential acceleration (mu from JEOD gravity files)
    let tdb_jd = sim.time.tdb_julian_date();
    let sun = sim.add_source(
        "Sun",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_sun,
                model: GravityModel::PointMass,
            },
            position: earth_centered_position(EphemerisBody::Sun, tdb_jd, &ephemeris),
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
        },
    );
    let moon = sim.add_source(
        "Moon",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_moon,
                model: GravityModel::PointMass,
            },
            position: earth_centered_position(EphemerisBody::Moon, tdb_jd, &ephemeris),
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
        },
    );

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
            r_eq: jeod_sim::EARTH.shape.r_eq,
            r_pol: jeod_sim::EARTH.shape.r_pol,
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

    // Sphere mass (loaded from JEOD Modified_data/mass.py, function set_mass_sphere)
    let mass_init = jeod_test_data::mass_data::load_mass_from_file(
        &sim_dir.join("Modified_data/mass.py"),
        Some("set_mass_sphere"),
    );
    let sphere_inertia = DMat3::from_cols(
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
    let sphere_mass = MassProperties::with_inertia(
        mass_init.mass,
        sphere_inertia,
        DVec3::from_slice(&mass_init.position),
    );

    // Drag requires RotationalState for inertial-to-body frame transform.
    // Even for a sphere (isotropic drag), the code path needs orientation.
    let rot = if with_drag {
        Some(RotationalState {
            quaternion: JeodQuat::from_glam(init.composite_body.quaternion),
            ang_vel_body: init.composite_body.ang_vel,
        })
    } else {
        None
    };

    let mut body = VehicleConfig {
        trans: TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        },
        rot,
        mass: Some(sphere_mass),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_nonspherical(
                    earth,
                    grav_cfg.degree,
                    grav_cfg.order,
                    grav_cfg.gradient,
                ),
                GravityControl::new_third_body(sun),
                GravityControl::new_third_body(moon),
            ],
        },
        ..Default::default()
    };

    if let Some(dc) = drag_config {
        body.drag = Some(dc);
    }

    sim.add_body(body);
    sim.validate().unwrap();

    println!("Tier 3 (Simulation): {label}, {} points", trajectory.len());

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    for record in &trajectory[1..] {
        // Update Sun/Moon positions from ephemeris
        let target_tdb_jd = tdb_jd + record.time / 86400.0;
        sim.set_source_position(
            sun,
            earth_centered_position(EphemerisBody::Sun, target_tdb_jd, &ephemeris),
        );
        sim.set_source_position(
            moon,
            earth_centered_position(EphemerisBody::Moon, target_tdb_jd, &ephemeris),
        );

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
            acceleration: None,
            ang_accel: None,
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

#[test]
fn tier3_simulation_run7a_sh4x4_3rd_body() {
    run_7_test(
        "dyncomp_run7a_state.csv",
        "RUN_7A",
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
        "RUN_7B",
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
        "RUN_7C",
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
        "RUN_7D",
        true,
        "RUN_7D (8x8 SH + Sun/Moon + drag)",
        "tier3_simulation_run7d",
        [7.735e-1, 1.126, 9.118e-1],
        [7.898e-4, 1.259e-3, 1.03e-3],
    );
}
