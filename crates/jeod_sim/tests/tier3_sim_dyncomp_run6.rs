//! Tier 3: SIM_dyncomp RUN_6A/6B — Drag (constant density and MET atmosphere)

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_sim::{
    met_atmosphere, AtmosphereConfig, AtmosphereModel, DragConfig, DynamicsConfig, GravityControl,
    GravityControls, GravityModel, GravitySource, GravitySourceEntry, MassProperties,
    MetAtmosphere, RotationalState, SimBody, Simulation, SimulationTime, TranslationalState,
};

/// Epoch for SIM_dyncomp: midnight 2007-11-20 UTC.
/// MJD = 54424.0, TJT = MJD - 40000 = 14424.0.
/// From JEOD time.py: TAI-UTC = 32s override, tai_to_ut1 = -32.469s.
const DRAG_EPOCH_UTC_TJT: f64 = 14424.0;
const DRAG_TAI_UTC_S: f64 = 32.0;
const DRAG_TAI_TO_UT1_S: f64 = -32.469;

// ── RUN_6B: MET atmosphere + drag, 6-DOF ──

#[test]
fn tier3_simulation_run6b_drag() {
    let csv_path = test_data_path("dyncomp_run6b_state.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let trajectory = load_sixdof_trajectory(&csv_path);
    assert!(trajectory.len() >= 100);

    let init = &trajectory[0];

    // Unit sphere mass (from Modified_data/mass.py)
    let inertia = DMat3::from_diagonal(DVec3::splat(0.4));
    let mass_props = MassProperties::with_inertia(1.0, inertia, DVec3::ZERO);

    // MET atmosphere: solar mean conditions (from Modified_data/solar_flux.py)
    let met_model = MetAtmosphere {
        f10: 128.8,
        f10b: 128.8,
        geo_index: 15.7,
        geo_index_type: met_atmosphere::GeoIndexType::Ap,
    };

    // Drag config (from Modified_data/aero_drag.py)
    let drag_config = DragConfig {
        cd: 0.02,
        area: 1.0,
        constant_density: None,
    };

    // Initialize Simulation at the SIM_dyncomp epoch with correct time offsets.
    let epoch_tai_tjt = DRAG_EPOCH_UTC_TJT + DRAG_TAI_UTC_S / 86400.0;
    let mut time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    time.set_ut1_tai_offset(DRAG_TAI_TO_UT1_S);

    let mut sim = Simulation::new(time, DT);

    // Earth source with planet-fixed rotation — the Simulation's ephemeris stage
    // updates it each step via RNP, so the atmosphere system sees correct geodetic
    // coordinates. Without this, MET density is evaluated at wrong lat/lon.
    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY), // triggers ephemeris update each step
    });

    // Configure atmosphere with planet rotation lookup
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Met(met_model),
        r_eq: 6_378_137.0,
        r_pol: 6_378_137.0 * (1.0 - 1.0 / 298.257_223_563),
        planet_omega: OMEGA_EARTH,
    });
    sim.atmosphere_planet_source = Some(earth);

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        rot: Some(RotationalState {
            quaternion: init.quaternion,
            ang_vel_body: init.ang_vel,
        }),
        mass: Some(mass_props),
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: true,
            three_dof: false,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        drag: Some(drag_config),
        atmospheric_state: Some(Default::default()), // presence enables atmosphere
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): RUN_6B MET+drag 6-DOF, {} points",
        trajectory.len()
    );

    let mut max_pos_error = 0.0_f64;
    let mut max_vel_error = 0.0_f64;
    let mut max_quat_error = 0.0_f64;

    for record in &trajectory[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);
        let pos_error = (body.trans.position - record.position).length();
        let vel_error = (body.trans.velocity - record.velocity).length();
        max_pos_error = max_pos_error.max(pos_error);
        max_vel_error = max_vel_error.max(vel_error);

        if let Some(ref rot) = body.rot {
            let quat_error = quaternion_angle_error(&rot.quaternion, &record.quaternion);
            max_quat_error = max_quat_error.max(quat_error);
        }

        if (record.time % 3600.0).abs() < 30.1 {
            println!(
                "  t={:6.0}s: pos_err={:10.4} m  vel_err={:.6} m/s",
                record.time, pos_error, vel_error
            );
        }
    }

    println!("  Max position error:  {:.4} m", max_pos_error);
    println!("  Max velocity error:  {:.6} m/s", max_vel_error);
    println!("  Max quaternion error: {:.2e} rad", max_quat_error);

    // Tolerances match existing tier3_drag_trajectory test
    assert!(
        max_pos_error < 2.0,
        "Position error {max_pos_error:.2} m exceeds 2.0 m"
    );
    assert!(
        max_vel_error < 0.005,
        "Velocity error {max_vel_error:.6} m/s exceeds 0.005 m/s"
    );
    assert!(
        max_quat_error < 0.01,
        "Quaternion error {max_quat_error:.2e} rad exceeds 0.01 rad"
    );
}

// ── RUN_6A: Constant-density drag, sphere mass ──
//
// Same as RUN_6B but with constant atmospheric density = 1.4e-12 kg/m³
// (JEOD `AerodynamicDrag::constant_density = True`, `density = 1.4e-12`).
// Isolates drag force computation from the MET atmosphere model.

#[test]
fn tier3_simulation_run6a_const_density_drag() {
    let csv_path = test_data_path("dyncomp_run6a_state.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let trajectory = load_sixdof_trajectory(&csv_path);
    assert!(trajectory.len() >= 100);
    let init = &trajectory[0];

    // Unit sphere mass (from Modified_data/mass.py)
    let inertia = DMat3::from_diagonal(DVec3::splat(0.4));
    let mass_props = MassProperties::with_inertia(1.0, inertia, DVec3::ZERO);

    // MET atmosphere config — the Simulation still runs the atmosphere pipeline
    // for wind (co-rotation), but constant_density overrides the MET density.
    let met_model = MetAtmosphere {
        f10: 128.8,
        f10b: 128.8,
        geo_index: 15.7,
        geo_index_type: met_atmosphere::GeoIndexType::Ap,
    };

    // Drag config with constant density = 1.4e-12 kg/m³
    // (from Modified_data/aero_drag.py: set_aero_const_density_drag)
    let drag_config = DragConfig {
        cd: 0.02,
        area: 1.0,
        constant_density: Some(1.4e-12),
    };

    let epoch_tai_tjt = DRAG_EPOCH_UTC_TJT + DRAG_TAI_UTC_S / 86400.0;
    let mut time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    time.set_ut1_tai_offset(DRAG_TAI_TO_UT1_S);

    let mut sim = Simulation::new(time, DT);

    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        t_inertial_pfix: Some(DMat3::IDENTITY),
    });

    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Met(met_model),
        r_eq: 6_378_137.0,
        r_pol: 6_378_137.0 * (1.0 - 1.0 / 298.257_223_563),
        planet_omega: OMEGA_EARTH,
    });
    sim.atmosphere_planet_source = Some(earth);

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        rot: Some(RotationalState {
            quaternion: init.quaternion,
            ang_vel_body: init.ang_vel,
        }),
        mass: Some(mass_props),
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: true,
            three_dof: false,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        drag: Some(drag_config),
        atmospheric_state: Some(Default::default()),
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): RUN_6A constant-density drag, {} points",
        trajectory.len()
    );

    let mut max_pos_error = 0.0_f64;
    let mut max_vel_error = 0.0_f64;
    let mut max_quat_error = 0.0_f64;

    for record in &trajectory[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);
        let pos_error = (body.trans.position - record.position).length();
        let vel_error = (body.trans.velocity - record.velocity).length();
        max_pos_error = max_pos_error.max(pos_error);
        max_vel_error = max_vel_error.max(vel_error);

        if let Some(ref rot) = body.rot {
            let quat_error = quaternion_angle_error(&rot.quaternion, &record.quaternion);
            max_quat_error = max_quat_error.max(quat_error);
        }

        if (record.time % 7200.0).abs() < 30.1 {
            println!(
                "  t={:6.0}s: pos_err={:.3e} m  vel_err={:.3e} m/s",
                record.time, pos_error, vel_error
            );
        }
    }

    println!("  Max position error:  {:.6e} m", max_pos_error);
    println!("  Max velocity error:  {:.6e} m/s", max_vel_error);
    println!("  Max quaternion error: {:.2e} rad", max_quat_error);

    assert!(
        max_pos_error < 0.5,
        "RUN_6A: position error {max_pos_error:.3e} m exceeds 0.5 m"
    );
    assert!(
        max_vel_error < 0.001,
        "RUN_6A: velocity error {max_vel_error:.3e} m/s exceeds 0.001 m/s"
    );
    assert!(
        max_quat_error < 0.01,
        "RUN_6A: quaternion error {max_quat_error:.2e} rad exceeds 0.01 rad"
    );
}
