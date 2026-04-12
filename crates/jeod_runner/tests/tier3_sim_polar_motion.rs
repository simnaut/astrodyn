//! Tier 3: Polar motion regression check (point-mass gravity).
//!
//! Validates that enabling `Simulation::polar_motion` does not break
//! point-mass propagation. With point-mass gravity (`t_inertial_pfix: None`),
//! the planet-fixed rotation is never used, so polar motion has **zero**
//! trajectory effect — errors should match RUN_2 exactly.
//!
//! A meaningful polar motion validation requires spherical-harmonic gravity
//! where the planet-fixed rotation enters the gravity computation (see
//! RUN_3* and RUN_7* tests).
//!
//! We propagate our Simulation with polar motion enabled using the IERS
//! values from the JEOD input file (xp=0.06806", yp=0.24156" converted
//! to radians) and compare against the RUN_2P reference trajectory.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_runner::{GravitySourceEntry, RotationModel, SimBody, Simulation};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, SimulationTime,
    TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};

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

/// Arcseconds to radians conversion factor (from JEOD polar_motion data).
const ARCSEC_TO_RAD: f64 = 4.848_136_811_095_36e-6;

/// Polar motion values from JEOD SIM_RNP_J2000_prop RUN_J2000_RNP_prop input.py.
/// These are the IERS values for 1999-03-03 (the epoch is different from
/// SIM_dyncomp's J2000.0, but the polar motion table is static — JEOD's
/// SIM_dyncomp uses the same table lookup at whatever epoch).
///
/// For SIM_dyncomp epoch (2007-11-20), the xp/yp values come from the
/// IERS table at MJD ~54424. We use the values that JEOD computes
/// internally via table interpolation. Since we're comparing our
/// polar-motion-ON trajectory against JEOD's polar-motion-ON trajectory,
/// small differences in the exact xp/yp values are absorbed into the
/// tolerance.
///
/// Note: we use constant xp/yp (no time-varying interpolation) which
/// introduces a small error over 8h since polar motion evolves slowly.
/// At ~0.3 arcsec/month, 8h drift is < 0.001 arcsec = negligible.
const XP_ARCSEC: f64 = 0.06806;
const YP_ARCSEC: f64 = 0.24156;

#[test]
fn tier3_simulation_run2p_polar_motion() {
    let csv_path = test_data_path("dyncomp_run2p_state.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro \
         -v /path/to/jeod/verif/SIM_dyncomp/SET_test/RUN_2P:/jeod/verif/SIM_dyncomp/SET_test/RUN_2P \
         jeod-trick",
        csv_path.display()
    );

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(trajectory.len() > 100);
    let init = &trajectory[0];

    let mu_earth = load_mu_earth();
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );
    let dt =
        jeod_test_data::s_define::load_dynamics_dt(&jeod_root.join("verif/SIM_dyncomp/S_define"));
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

    // Enable polar motion
    let xp = XP_ARCSEC * ARCSEC_TO_RAD;
    let yp = YP_ARCSEC * ARCSEC_TO_RAD;
    sim.polar_motion = Some((xp, yp));

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

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): RUN_2P polar motion, {} points",
        trajectory.len()
    );

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    for record in &trajectory[1..] {
        sim.step_until(record.time);
        let body = sim.body(0);
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

    let report = CrossvalReport::compute(
        "tier3_simulation_run2p_polar_motion",
        &our_states,
        &ref_states,
    );
    report.write();

    let max_pos = report.max_position_component();
    let max_vel = report.max_velocity_component();
    println!("  Max position error: {max_pos:.6e} m");
    println!("  Max velocity error: {max_vel:.6e} m/s");

    // RUN_2P uses point-mass gravity where the planet-fixed rotation is not
    // used — polar motion has zero trajectory effect. Errors match RUN_2
    // exactly (~1.3e-6 m from RK4 floating-point accumulation).
    // A meaningful polar motion trajectory test requires SH gravity; this
    // test validates that enabling polar motion doesn't break point-mass.
    report.assert_position([1.37e-6, 2.154e-6, 1.826e-6]);
    report.assert_velocity([1.446e-9, 2.389e-9, 1.814e-9]);
}
