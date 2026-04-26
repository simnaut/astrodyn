//! `VerificationCase` constructors for the SIM_dyncomp Tier 3 family.
//!
//! Each constructor returns a fully-populated
//! [`VerificationCase`](jeod_sim::recipes::verification::VerificationCase)
//! whose scenario closure loads its initial conditions from JEOD source
//! files (Modified_data/*.py, S_define, gravity coefficient files, plus
//! the t=0 row of the matching reference CSV — all "JEOD source data"
//! per CLAUDE.md). The scenario builds a [`SimulationBuilder`] that the
//! `run_and_assert` machinery materializes into a runtime
//! [`Simulation`](crate::Simulation).

use std::path::PathBuf;

use glam::{DMat3, DVec3};
use jeod_sim::met_atmosphere;
use jeod_sim::recipes::verification::{CsvReference, Tolerances, VerificationCase};
use jeod_sim::{
    coefficients::load_from_jeod_cc, default_leap_second_table, AtmosphereConfig, AtmosphereModel,
    DragConfig, GravityControl, GravityControls, GravityModel, GravitySource, GravitySourceEntry,
    JeodQuat, MassProperties, MetAtmosphere, RotationModel, RotationalState, SimulationBuilder,
    SimulationTime, TranslationalState, VehicleConfig, EARTH,
};
use jeod_test_data::tier3_csv::test_data_path;
use uom::si::f64::Time;
use uom::si::time::second;

const SIM_DYNCOMP: &str = "verif/SIM_dyncomp";

/// Earth's sidereal angular velocity, sourced from
/// [`PlanetConfig::omega`](jeod_sim::PlanetConfig).
const OMEGA_EARTH: f64 = jeod_sim::planet_config::EARTH.omega;

fn jeod_root() -> PathBuf {
    let r = jeod_test_data::jeod_path();
    assert!(
        r.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        r.display()
    );
    r
}

fn dyncomp_csv(name: &str) -> Vec<jeod_test_data::dyncomp_csv::DyncompRecord> {
    let path = test_data_path(name);
    assert!(
        path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        path.display()
    );
    let records = jeod_test_data::dyncomp_csv::load_dyncomp_csv(&path);
    assert!(records.len() > 100, "{}: too few records", name);
    records
}

fn point_mass_earth_source(mu: f64) -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
        planet_omega: 0.0,
        central: true,
    }
}

// ── RUN_2: Point-mass 3-DOF / 6-DOF ────────────────────────────────────────

fn build_run2_3dof() -> SimulationBuilder {
    let jeod = jeod_root();
    let sim_dir = jeod.join(SIM_DYNCOMP);
    let dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    let earth_grav =
        load_from_jeod_cc(&jeod.join("models/environment/gravity/data/src/earth_GGM05C.cc"))
            .expect("load Earth gravity");
    let trajectory = dyncomp_csv("dyncomp_run2_state.csv");
    let init = &trajectory[0];

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source("Earth", point_mass_earth_source(earth_grav.mu));
    sb.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });
    sb
}

fn build_run2_6dof() -> SimulationBuilder {
    let jeod = jeod_root();
    let sim_dir = jeod.join(SIM_DYNCOMP);
    let dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    let earth_grav =
        load_from_jeod_cc(&jeod.join("models/environment/gravity/data/src/earth_GGM05C.cc"))
            .expect("load Earth gravity");
    let mass_init = jeod_test_data::mass_data::load_mass_from_file(
        &sim_dir.join("Modified_data/mass.py"),
        Some("set_mass_iss"),
    );
    let trajectory = dyncomp_csv("dyncomp_run2_state.csv");
    let init = &trajectory[0];

    let inertia = DMat3::from_cols(
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

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source("Earth", point_mass_earth_source(earth_grav.mu));
    sb.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::from_glam(init.composite_body.quaternion),
            ang_vel_body: init.composite_body.ang_vel,
        }),
        mass: Some(mass_props),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });
    sb
}

/// SIM_dyncomp RUN_2 — point-mass 3-DOF (translational only).
pub fn run2_3dof() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run2_3dof",
        scenario: build_run2_3dof,
        reference: CsvReference::Dyncomp("dyncomp_run2_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [1.37e-6, 2.154e-6, 1.826e-6],
            velocity_m_s: [1.446e-9, 2.389e-9, 1.814e-9],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
    }
}

/// SIM_dyncomp RUN_2 — point-mass 6-DOF (with ISS mass properties).
pub fn run2_6dof() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run2_6dof",
        scenario: build_run2_6dof,
        reference: CsvReference::Dyncomp("dyncomp_run2_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [1.37e-6, 2.154e-6, 1.826e-6],
            velocity_m_s: [1.446e-9, 2.389e-9, 1.814e-9],
            quat_angle_rad: 4.426e-8,
            ang_vel_rad_s: [2.619e-18, 1.367e-18, 7.969e-19],
            extras: &[],
        },
        extras: None,
    }
}

// ── Shared helpers used by RUN_3+ (SH gravity, RNP rotation, time/UT1) ─────

fn iss_mass_properties() -> MassProperties {
    let jeod = jeod_root();
    let sim_dir = jeod.join(SIM_DYNCOMP);
    let mass_init = jeod_test_data::mass_data::load_mass_from_file(
        &sim_dir.join("Modified_data/mass.py"),
        Some("set_mass_iss"),
    );
    let inertia = DMat3::from_cols(
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
    MassProperties::with_inertia(
        mass_init.mass,
        inertia,
        DVec3::from_slice(&mass_init.position),
    )
}

fn sphere_mass_properties() -> MassProperties {
    let jeod = jeod_root();
    let sim_dir = jeod.join(SIM_DYNCOMP);
    let mass_init = jeod_test_data::mass_data::load_mass_from_file(
        &sim_dir.join("Modified_data/mass.py"),
        Some("set_mass_sphere"),
    );
    let inertia = DMat3::from_cols(
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
    MassProperties::with_inertia(
        mass_init.mass,
        inertia,
        DVec3::from_slice(&mass_init.position),
    )
}

/// Simulation time anchored at the SIM_dyncomp epoch (parsed from
/// `Modified_data/time.py`), with the UT1-TAI offset applied.
fn dyncomp_time() -> SimulationTime {
    let jeod = jeod_root();
    let sim_dir = jeod.join(SIM_DYNCOMP);
    let time_cfg =
        jeod_test_data::time_config::load_time_config(&sim_dir.join("Modified_data/time.py"));
    let mut time = SimulationTime::new(time_cfg.tai_tjt(), default_leap_second_table());
    let ut1_tai_offset = time_cfg
        .ut1_tai_offset()
        .expect("SIM_dyncomp time.py must specify tai_to_ut1_override_val");
    time.set_ut1_tai_offset(ut1_tai_offset);
    time
}

/// Earth gravity source backed by GGM02C spherical-harmonics coefficients,
/// with the EarthRNP rotation model so the planet-fixed frame updates each
/// step. Used by the RUN_3A / RUN_3B / RUN_5* / RUN_6* configurations.
fn earth_sh_with_rnp() -> GravitySourceEntry {
    let jeod = jeod_root();
    let sh_data =
        load_from_jeod_cc(&jeod.join("models/environment/gravity/data/src/earth_GGM02C.cc"))
            .expect("load GGM02C");
    GravitySourceEntry {
        source: GravitySource {
            mu: sh_data.mu,
            model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
        },
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY),
        delta_c20: 0.0,
        rotation_model: RotationModel::EarthRNP,
        tidal_config: None,
        planet_omega: OMEGA_EARTH,
        central: true,
    }
}

/// Earth point-mass with the EarthRNP rotation model (used by drag /
/// atmosphere runs that need geodetic coordinates each step).
fn earth_pm_with_rnp(mu: f64) -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY),
        delta_c20: 0.0,
        rotation_model: RotationModel::EarthRNP,
        tidal_config: None,
        planet_omega: OMEGA_EARTH,
        central: true,
    }
}

// ── RUN_3A / RUN_3B: spherical-harmonics gravity (4x4 / 8x8) + RNP ─────────

fn build_run3_3dof(run_dir: &str, csv_name: &str) -> SimulationBuilder {
    let jeod = jeod_root();
    let sim_dir = jeod.join(SIM_DYNCOMP);
    let dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));

    // Gravity controls degree/order from RUN input chain.
    let mut grav_files: Vec<PathBuf> = vec![sim_dir.join("Modified_data/grav_controls.py")];
    grav_files.push(sim_dir.join("SET_test/RUN_3A/input.py"));
    if run_dir != "RUN_3A" {
        grav_files.push(sim_dir.join(format!("SET_test/{run_dir}/input.py")));
    }
    let grav_file_refs: Vec<&std::path::Path> = grav_files.iter().map(|p| p.as_path()).collect();
    let grav_cfg = jeod_test_data::gravity_control::load_gravity_control(&grav_file_refs);

    let trajectory = dyncomp_csv(csv_name);
    let init = &trajectory[0];

    let mut sb = SimulationBuilder::new(dyncomp_time(), dt);
    let earth = sb.add_source("Earth", earth_sh_with_rnp());
    sb.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_nonspherical(
                earth,
                grav_cfg.degree,
                grav_cfg.order,
                grav_cfg.gradient,
            )],
        },
        ..Default::default()
    });
    sb
}

fn build_run3a() -> SimulationBuilder {
    build_run3_3dof("RUN_3A", "dyncomp_run3a_state.csv")
}

fn build_run3b() -> SimulationBuilder {
    build_run3_3dof("RUN_3B", "dyncomp_run3b_state.csv")
}

/// SIM_dyncomp RUN_3A — 4×4 spherical-harmonics gravity + RNP rotation.
pub fn run3a_sh4x4() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run3a_sh4x4",
        scenario: build_run3a,
        reference: CsvReference::Dyncomp("dyncomp_run3a_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [5.3e-2, 1.344e-1, 1.026e-1],
            velocity_m_s: [6.151e-5, 1.246e-4, 1.24e-4],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
    }
}

/// SIM_dyncomp RUN_3B — 8×8 spherical-harmonics gravity + RNP rotation.
pub fn run3b_sh8x8() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run3b_sh8x8",
        scenario: build_run3b,
        reference: CsvReference::Dyncomp("dyncomp_run3b_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [1.325e-1, 2.3e-1, 1.646e-1],
            velocity_m_s: [1.478e-4, 2.329e-4, 1.892e-4],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
    }
}

// ── RUN_5B / RUN_5C: elliptical, no drag, gradient=true ────────────────────

fn build_run5(csv_name: &str) -> SimulationBuilder {
    let jeod = jeod_root();
    let sim_dir = jeod.join(SIM_DYNCOMP);
    let dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    let mu_earth = jeod_sim::coefficients::load_mu_from_jeod_cc(
        &jeod.join("models/environment/gravity/data/src/earth_GGM05C.cc"),
    )
    .expect("load Earth mu");
    let mass_props = iss_mass_properties();
    let trajectory = dyncomp_csv(csv_name);
    let init = &trajectory[0];

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source("Earth", point_mass_earth_source(mu_earth));
    sb.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::from_glam(init.composite_body.quaternion),
            ang_vel_body: init.composite_body.ang_vel,
        }),
        mass: Some(mass_props),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, true)], // gradient=true
        },
        ..Default::default()
    });
    sb
}

fn build_run5b() -> SimulationBuilder {
    build_run5("dyncomp_run5b_state.csv")
}

fn build_run5c() -> SimulationBuilder {
    build_run5("dyncomp_run5c_state.csv")
}

/// SIM_dyncomp RUN_5B — elliptical orbit, point-mass + gravity gradient,
/// 6-DOF (drag off, MET solar mean configuration without effect).
pub fn run5b_atmosphere_mean() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run5b_atmosphere_mean",
        scenario: build_run5b,
        reference: CsvReference::Dyncomp("dyncomp_run5b_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [5.374e-7, 8.376e-7, 6.318e-7],
            velocity_m_s: [5.179e-10, 9.311e-10, 7.361e-10],
            quat_angle_rad: 4.426e-8,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
    }
}

/// SIM_dyncomp RUN_5C — same physics as RUN_5B with the solar-max MET
/// configuration (no effect on dynamics; drag is off).
pub fn run5c_atmosphere_max() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run5c_atmosphere_max",
        scenario: build_run5c,
        reference: CsvReference::Dyncomp("dyncomp_run5c_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [5.374e-7, 8.376e-7, 6.318e-7],
            velocity_m_s: [5.179e-10, 9.311e-10, 7.361e-10],
            quat_angle_rad: 4.426e-8,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
    }
}

// ── RUN_6A / RUN_6B: drag (constant-density / MET) ─────────────────────────

fn met_solar_mean() -> MetAtmosphere {
    MetAtmosphere {
        f10: 128.8,
        f10b: 128.8,
        geo_index: 15.7,
        geo_index_type: met_atmosphere::GeoIndexType::Ap,
    }
}

fn build_run6_drag(csv_name: &str, drag_config: DragConfig) -> SimulationBuilder {
    let jeod = jeod_root();
    let sim_dir = jeod.join(SIM_DYNCOMP);
    let dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    let earth_grav =
        load_from_jeod_cc(&jeod.join("models/environment/gravity/data/src/earth_GGM05C.cc"))
            .expect("load Earth gravity");
    let mass_props = sphere_mass_properties();
    let trajectory = dyncomp_csv(csv_name);
    let init = &trajectory[0];

    let mut sb = SimulationBuilder::new(dyncomp_time(), dt);
    let earth = sb.add_source("Earth", earth_pm_with_rnp(earth_grav.mu));
    sb = sb.atmosphere(
        AtmosphereConfig {
            model: AtmosphereModel::Met(met_solar_mean()),
            r_eq: EARTH.shape.r_eq,
            r_pol: EARTH.shape.r_pol,
            planet_omega: OMEGA_EARTH,
        },
        earth,
    );
    sb.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::from_glam(init.composite_body.quaternion),
            ang_vel_body: init.composite_body.ang_vel,
        }),
        mass: Some(mass_props),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        drag: Some(drag_config),
        ..Default::default()
    });
    sb
}

fn build_run6a() -> SimulationBuilder {
    build_run6_drag(
        "dyncomp_run6a_state.csv",
        DragConfig {
            cd: 0.02,
            area: 1.0,
            constant_density: Some(1.4e-12),
        },
    )
}

fn build_run6b() -> SimulationBuilder {
    build_run6_drag(
        "dyncomp_run6b_state.csv",
        DragConfig {
            cd: 0.02,
            area: 1.0,
            constant_density: None,
        },
    )
}

/// SIM_dyncomp RUN_6A — sphere mass with constant-density drag (1.4e-12
/// kg/m³). Atmosphere pipeline still runs for wind/co-rotation, but the
/// constant-density override drives the drag force.
pub fn run6a_const_density_drag() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run6a_const_density_drag",
        scenario: build_run6a,
        reference: CsvReference::Dyncomp("dyncomp_run6a_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [4.366e-4, 6.84e-4, 5.325e-4],
            velocity_m_s: [4.942e-7, 7.482e-7, 6.155e-7],
            quat_angle_rad: 4.426e-8,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
    }
}

/// SIM_dyncomp RUN_6B — sphere mass with MET-solar-mean drag.
pub fn run6b_drag() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run6b_drag",
        scenario: build_run6b,
        reference: CsvReference::Dyncomp("dyncomp_run6b_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [7.971e-1, 1.114, 8.945e-1],
            velocity_m_s: [7.861e-4, 1.254e-3, 1.003e-3],
            quat_angle_rad: 4.426e-8,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
    }
}

// ── RUN_10A / RUN_10C / RUN_10D: gravity-gradient torque ───────────────────

fn cylinder_mass_properties() -> MassProperties {
    let jeod = jeod_root();
    let sim_dir = jeod.join(SIM_DYNCOMP);
    let mass_init = jeod_test_data::mass_data::load_mass_from_file(
        &sim_dir.join("Modified_data/mass.py"),
        Some("set_mass_cylinder"),
    );
    let inertia = DMat3::from_cols(
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
    MassProperties::with_inertia(
        mass_init.mass,
        inertia,
        DVec3::from_slice(&mass_init.position),
    )
}

fn build_run10(csv_name: &str) -> SimulationBuilder {
    let jeod = jeod_root();
    let sim_dir = jeod.join(SIM_DYNCOMP);
    let dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    let earth_grav =
        load_from_jeod_cc(&jeod.join("models/environment/gravity/data/src/earth_GGM05C.cc"))
            .expect("load Earth gravity");
    let mass_props = cylinder_mass_properties();
    let trajectory = dyncomp_csv(csv_name);
    let init = &trajectory[0];

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    let earth = sb.add_source("Earth", point_mass_earth_source(earth_grav.mu));
    sb.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::from_glam(init.composite_body.quaternion),
            ang_vel_body: init.composite_body.ang_vel,
        }),
        mass: Some(mass_props),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, true)], // gradient=true
        },
        compute_gravity_gradient: true,
        ..Default::default()
    });
    sb
}

fn build_run10a() -> SimulationBuilder {
    build_run10("dyncomp_run10a_state.csv")
}

fn build_run10c() -> SimulationBuilder {
    build_run10("dyncomp_run10c_state.csv")
}

fn build_run10d() -> SimulationBuilder {
    build_run10("dyncomp_run10d_state.csv")
}

/// SIM_dyncomp RUN_10A — gravity-gradient torque, cylinder mass, 6-DOF.
/// Initial attitude is 85° pitch + 1° yaw from LVLH; tests gravity
/// gradient libration.
pub fn run10a_gravity_torque() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run10a_gravity_torque",
        scenario: build_run10a,
        reference: CsvReference::Dyncomp("dyncomp_run10a_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [1.37e-6, 2.154e-6, 1.826e-6],
            velocity_m_s: [1.446e-9, 2.389e-9, 1.814e-9],
            quat_angle_rad: 7.556e-5,
            ang_vel_rad_s: [1e-15, 1.172e-7, 9.301e-8],
            extras: &[],
        },
        extras: None,
    }
}

/// SIM_dyncomp RUN_10C — gravity-gradient torque, elliptical orbit,
/// zero initial body rate.
pub fn run10c_gravity_torque_elliptical() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run10c_gravity_torque_elliptical",
        scenario: build_run10c,
        reference: CsvReference::Dyncomp("dyncomp_run10c_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [5.374e-7, 8.376e-7, 6.318e-7],
            velocity_m_s: [5.179e-10, 9.311e-10, 7.361e-10],
            quat_angle_rad: 7.978e-5,
            ang_vel_rad_s: [1e-15, 1.243e-7, 9.646e-8],
            extras: &[],
        },
        extras: None,
    }
}

// ── RUN_5A: MET atmosphere validation (drag off, atmosphere live) ─────────

fn build_run5a_met() -> SimulationBuilder {
    let jeod = jeod_root();
    let sim_dir = jeod.join(SIM_DYNCOMP);
    let dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    let mu_earth = jeod_sim::coefficients::load_mu_from_jeod_cc(
        &jeod.join("models/environment/gravity/data/src/earth_GGM05C.cc"),
    )
    .expect("load Earth mu");
    let mass_props = iss_mass_properties();
    let csv_path = test_data_path("dyncomp_run5a_atmos_atmos_traj.csv");
    assert!(
        csv_path.exists(),
        "Atmosphere trajectory CSV not found at {}.\n\
         Generate with Docker (see CLAUDE.md).",
        csv_path.display()
    );
    let records = jeod_test_data::tier3_csv::load_atmos_traj_csv(&csv_path);
    assert!(records.len() > 100, "Expected >100 records");
    let init = &records[0];

    // RUN_5A: minimum solar activity (F10.7 = 70, Ap = 0)
    let met_model = MetAtmosphere {
        f10: 70.0,
        f10b: 70.0,
        geo_index: 0.0,
        geo_index_type: met_atmosphere::GeoIndexType::Ap,
    };

    let mut sb = SimulationBuilder::new(dyncomp_time(), dt);
    let earth = sb.add_source("Earth", earth_pm_with_rnp(mu_earth));
    sb = sb.atmosphere(
        AtmosphereConfig {
            model: AtmosphereModel::Met(met_model),
            r_eq: EARTH.shape.r_eq,
            r_pol: EARTH.shape.r_pol,
            planet_omega: OMEGA_EARTH,
        },
        earth,
    );
    sb.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        }),
        mass: Some(mass_props),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, true)],
        },
        ..Default::default()
    });
    sb
}

/// SIM_dyncomp RUN_5A — MET atmosphere live (minimum solar activity),
/// drag off, ISS mass with identity attitude. Validates atmosphere
/// pipeline matches JEOD reference trajectory (atmosphere has no
/// dynamic effect since drag is off).
pub fn run5a_met() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_met_run5a",
        scenario: build_run5a_met,
        reference: CsvReference::AtmosTraj("dyncomp_run5a_atmos_atmos_traj.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [2.5e-6, 2.5e-6, 2.5e-6],
            velocity_m_s: [0.0; 3], // verbatim from existing test (no velocity assert)
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
    }
}

// ── RUN_6B aero-trajectory variants (sphere + drag, position+velocity only) ─

/// Build a RUN_6B-style sphere/drag scenario reading initial conditions
/// from an aero-trajectory CSV (`dyncomp_run6b_aero_aero_traj.csv` or the
/// rotated-frame variant). Uses `MassProperties::new(1.0)` (1 kg sphere
/// with no explicit inertia) per JEOD's RUN_6B sphere replacement.
fn build_run6b_aero_traj(csv_name: &'static str, t_struct_body: DMat3) -> SimulationBuilder {
    let jeod = jeod_root();
    let sim_dir = jeod.join(SIM_DYNCOMP);
    let dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    let mu_earth = jeod_sim::coefficients::load_mu_from_jeod_cc(
        &jeod.join("models/environment/gravity/data/src/earth_GGM05C.cc"),
    )
    .expect("load Earth mu");

    let csv_path = test_data_path(csv_name);
    assert!(
        csv_path.exists(),
        "Aero-trajectory CSV not found at {}.\n\
         Generate with Docker (see CLAUDE.md).",
        csv_path.display()
    );
    let records = jeod_test_data::tier3_csv::load_aero_traj_csv(&csv_path);
    assert!(records.len() > 100, "{csv_name}: too few records");
    let init = &records[0];

    let mut sb = SimulationBuilder::new(dyncomp_time(), dt);
    let earth = sb.add_source("Earth", earth_pm_with_rnp(mu_earth));
    sb = sb.atmosphere(
        AtmosphereConfig {
            model: AtmosphereModel::Met(met_solar_mean()),
            r_eq: EARTH.shape.r_eq,
            r_pol: EARTH.shape.r_pol,
            planet_omega: OMEGA_EARTH,
        },
        earth,
    );
    sb.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        }),
        mass: Some(MassProperties::new(1.0)),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        drag: Some(DragConfig {
            cd: 0.02,
            area: 1.0,
            constant_density: None,
        }),
        t_struct_body,
        ..Default::default()
    });
    sb
}

fn build_run6b_aero_identity() -> SimulationBuilder {
    build_run6b_aero_traj("dyncomp_run6b_aero_aero_traj.csv", DMat3::IDENTITY)
}

fn build_run6b_aero_rotated() -> SimulationBuilder {
    // 15° rotation about [1,1,1]/√3 — matches DYNCOMP_AERO_ROT_SNIPPET in
    // generate_references.sh. For ballistic drag on a sphere the inertial
    // force is rotation-invariant, so the trajectory must match the
    // identity case; any divergence indicates a frame-transform bug.
    let eigen_angle = 15.0_f64.to_radians();
    let eigen_axis = DVec3::new(1.0, 1.0, 1.0).normalize();
    let q_struct_body = JeodQuat::left_quat_from_eigen_rotation(eigen_angle, eigen_axis);
    build_run6b_aero_traj(
        "dyncomp_run6b_rot_aero_traj.csv",
        q_struct_body.left_quat_to_transformation(),
    )
}

/// SIM_dyncomp RUN_6B — sphere + drag, identity structural frame.
/// Validates the drag pipeline against the aero-trajectory CSV
/// (position+velocity only; aero_force is not exposed on
/// `VehicleOutput`).
pub fn run6b_drag_aero_traj() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_drag_run6b",
        scenario: build_run6b_aero_identity,
        reference: CsvReference::AeroTraj("dyncomp_run6b_aero_aero_traj.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [1.12, 1.12, 1.12],
            velocity_m_s: [1.254e-3, 1.254e-3, 1.254e-3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
    }
}

/// SIM_dyncomp RUN_6B with a 15° rotation about [1,1,1]/√3 applied to
/// the structural-to-body transform. Should match the identity case
/// since ballistic drag on a sphere is rotation-invariant.
pub fn run6b_drag_rotated_struct() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_drag_run6b_rotated",
        scenario: build_run6b_aero_rotated,
        reference: CsvReference::AeroTraj("dyncomp_run6b_rot_aero_traj.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [0.798, 1.114, 0.895],
            velocity_m_s: [7.861e-4, 1.254e-3, 1.003e-3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
    }
}

/// SIM_dyncomp RUN_10D — gravity-gradient torque, elliptical orbit,
/// non-zero initial body rate.
pub fn run10d_gravity_torque_elliptical_rate() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run10d_gravity_torque_elliptical_rate",
        scenario: build_run10d,
        reference: CsvReference::Dyncomp("dyncomp_run10d_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [5.374e-7, 8.376e-7, 6.318e-7],
            velocity_m_s: [5.179e-10, 9.311e-10, 7.361e-10],
            quat_angle_rad: 1.106e-4,
            ang_vel_rad_s: [1e-15, 1.825e-7, 1.196e-7],
            extras: &[],
        },
        extras: None,
    }
}
