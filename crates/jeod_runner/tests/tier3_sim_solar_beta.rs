//! Tier 3: SIM_SolarBeta cross-validation (derived_state/verif/SIM_SolarBeta)
//!
//! Validates solar beta wiring via the RUN_2 point-mass trajectory (8h,
//! validated to < 0.5 m against JEOD) with DE421 ephemeris for Sun direction.
//! Self-consistency is verified to bit-identical precision.
//!
//! Sun has mu=0 because this test compares against RUN_2 (Earth-only gravity).
//! The Sun source is used solely for solar beta direction, not gravitational
//! perturbation. For 3rd-body gravity validation, see `tier3_sim_dyncomp_run4`.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_runner::{
    DerivedStateConfig, GravitySourceEntry, RotationModel, Simulation, VehicleConfig,
};
use jeod_sim::{
    Ephemeris, EphemerisBody, GravityControl, GravityControls, GravityModel, GravitySource,
    SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use std::path::Path;

fn load_mu_earth() -> f64 {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );
    jeod_sim::coefficients::load_mu_from_jeod_cc(
        &jeod_root.join("models/environment/gravity/data/src/earth_GGM05C.cc"),
    )
    .expect("load Earth mu from GGM05C")
}

#[test]
fn tier3_simulation_solar_beta() {
    let mu_earth = load_mu_earth();
    let csv_path = test_data_path("dyncomp_run2_state.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}",
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

    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );
    let dt =
        jeod_test_data::s_define::load_dynamics_dt(&jeod_root.join("verif/SIM_dyncomp/S_define"));

    // J2000.0 epoch (TJT = 0.0 for J2000)
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: mu_earth,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });

    // Sun source -- position from DE421 at J2000.0.
    // mu=0 matches the JEOD RUN_2 reference (Earth-only gravity). Sun is used
    // solely for solar beta direction. 3rd-body gravity validated separately
    // by tier3_sim_dyncomp_run4 and tier3_sim_torque_simple.
    // J2000.0 = JD 2451545.0
    let j2000_jd = 2_451_545.0;
    let (initial_sun, _) = ephemeris
        .get_earth_centered_state(EphemerisBody::Sun, j2000_jd)
        .expect("Sun position at J2000");
    let sun = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: initial_sun,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    sim.sun_source = Some(sun);

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        derived: DerivedStateConfig {
            solar_beta: true,
            ..Default::default()
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): solar beta via RUN_2 + DE421, {} points",
        trajectory.len()
    );

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    let mut ref_states = Vec::with_capacity(trajectory.len() - 1);

    for record in &trajectory[1..] {
        // Update Sun position from ephemeris
        let tdb_jd = j2000_jd + record.time / 86_400.0;
        let (sun_pos, _) = ephemeris
            .get_earth_centered_state(EphemerisBody::Sun, tdb_jd)
            .expect("Sun position query");
        sim.sources[sun].position = sun_pos;

        sim.step_until(record.time);

        let body = sim.body(0);

        let beta = body.solar_beta.unwrap_or_else(|| {
            panic!("Simulation did not compute solar beta at t={}", record.time)
        });

        // Self-consistency: verify against manual computation from Simulation's
        // own state. Must be bit-identical (same code path).
        let expected =
            jeod_sim::compute_body_solar_beta(body.trans.position, body.trans.velocity, sun_pos);
        assert_eq!(
            beta.to_bits(),
            expected.to_bits(),
            "Solar beta self-consistency failed at t={}: sim={}, manual={}",
            record.time,
            beta,
            expected
        );

        our_states.push(StateLog {
            time: record.time,
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            acceleration: None,
            ang_accel: None,
            ..Default::default()
        });
        ref_states.push(StateLog {
            time: record.time,
            position: Some(record.composite_body.position),
            velocity: Some(record.composite_body.velocity),
            acceleration: record.derivs.as_ref().map(|d| d.trans_accel),
            ang_accel: record.derivs.as_ref().map(|d| d.rot_accel),
            ..Default::default()
        });

        if (record.time % 3600.0).abs() < 30.1 {
            let pos_err = (body.trans.position - record.composite_body.position).length();
            println!(
                "  t={:6.0}s: pos_err={:.4} m  beta={:.4} deg ({:.6} rad)",
                record.time,
                pos_err,
                beta.to_degrees(),
                beta
            );
        }
    }

    let report = CrossvalReport::compute("tier3_simulation_solar_beta", &our_states, &ref_states);
    report.write();

    let max_pos_err = report.max_position_component();
    println!("  Max position error: {:.6e} m", max_pos_err);

    report.assert_position([1.37e-6, 2.154e-6, 1.826e-6]);
}
