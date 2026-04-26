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
use jeod_sim::recipes::verification::{CsvReference, Tolerances, VerificationCase};
use jeod_sim::{
    coefficients::load_from_jeod_cc, default_leap_second_table, GravityControl, GravityControls,
    GravityModel, GravitySource, GravitySourceEntry, JeodQuat, MassProperties, RotationModel,
    RotationalState, SimulationBuilder, SimulationTime, TranslationalState, VehicleConfig,
};
use jeod_test_data::tier3_csv::test_data_path;
use uom::si::f64::Time;
use uom::si::time::second;

const SIM_DYNCOMP: &str = "verif/SIM_dyncomp";

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
    }
}
