//! Tier 3: SIM_dyncomp RUN_6A/6B — Drag (constant density and MET atmosphere)

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_sim::{
    met_atmosphere, AtmosphereConfig, AtmosphereModel, DragConfig, DynamicsConfig, GravityControl,
    GravityControls, GravityModel, GravitySource, GravitySourceEntry, JeodQuat, MassProperties,
    MetAtmosphere, RotationalState, SimBody, Simulation, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};

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

    let trajectory = load_dyncomp_csv(&csv_path);
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

    // Log our propagated states
    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    for record in &trajectory[1..] {
        sim.step_until(record.time);
        let body = sim.body(0);
        let rot = body.rot.as_ref().unwrap();

        let pos_error = (body.trans.position - record.composite_body.position).length();
        let vel_error = (body.trans.velocity - record.composite_body.velocity).length();
        if (record.time % 3600.0).abs() < 30.1 {
            println!(
                "  t={:6.0}s: pos_err={:10.4} m  vel_err={:.6} m/s",
                record.time, pos_error, vel_error
            );
        }

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
    let report = CrossvalReport::compute("tier3_simulation_run6b_drag", &our_states, &ref_states);
    report.write();

    println!(
        "  Max position error:  {:.6e} m",
        report.max_position_component()
    );
    println!(
        "  Max velocity error:  {:.6e} m/s",
        report.max_velocity_component()
    );
    println!(
        "  Max quaternion error: {:.6e} rad",
        report.max_quat_angle()
    );

    report.assert_position([7.971e-1, 1.114, 8.945e-1]);
    report.assert_velocity([7.861e-4, 1.254e-3, 1.003e-3]);
    report.assert_quat_angle(4.426e-8);
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

    let trajectory = load_dyncomp_csv(&csv_path);
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

    // Log our propagated states
    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    for record in &trajectory[1..] {
        sim.step_until(record.time);
        let body = sim.body(0);
        let rot = body.rot.as_ref().unwrap();

        let pos_error = (body.trans.position - record.composite_body.position).length();
        let vel_error = (body.trans.velocity - record.composite_body.velocity).length();
        if (record.time % 7200.0).abs() < 30.1 {
            println!(
                "  t={:6.0}s: pos_err={:.3e} m  vel_err={:.3e} m/s",
                record.time, pos_error, vel_error
            );
        }

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
    let report = CrossvalReport::compute(
        "tier3_simulation_run6a_const_density_drag",
        &our_states,
        &ref_states,
    );
    report.write();

    println!(
        "  Max position error:  {:.6e} m",
        report.max_position_component()
    );
    println!(
        "  Max velocity error:  {:.6e} m/s",
        report.max_velocity_component()
    );
    println!(
        "  Max quaternion error: {:.6e} rad",
        report.max_quat_angle()
    );

    report.assert_position([4.366e-4, 6.84e-4, 5.325e-4]);
    report.assert_velocity([4.942e-7, 7.482e-7, 6.155e-7]);
    report.assert_quat_angle(4.426e-8);
}
