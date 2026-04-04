//! Tier 3: SIM_SolarBeta cross-validation (derived_state/verif/SIM_SolarBeta)
//!
//! Validates solar beta wiring via the RUN_2 point-mass trajectory (8h,
//! validated to < 0.5 m against JEOD) with DE421 ephemeris for Sun direction.
//! Self-consistency is verified to bit-identical precision.
//!
//! Once Phase 5 delivers third-body gravity, this test should be upgraded to
//! run the full SIM_SolarBeta scenario (10 days, 8x8 SH + Sun/Moon).

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_sim::{
    Ephemeris, EphemerisBody, GravityControl, GravityControls, GravityModel, GravitySource,
    GravitySourceEntry, SimBody, Simulation, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::crossval_report;
use std::path::Path;

#[test]
fn tier3_simulation_solar_beta() {
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
    let trajectory = load_trans_trajectory(&csv_path);
    assert!(trajectory.len() > 100);
    let init = &trajectory[0];

    // J2000.0 epoch (TJT = 0.0 for J2000)
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);

    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        t_inertial_pfix: None,
    });

    // Sun source -- position from DE421 at J2000.0
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
        t_inertial_pfix: None,
    });
    sim.sun_source = Some(sun);

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        compute_solar_beta: true,
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): solar beta via RUN_2 + DE421, {} points",
        trajectory.len()
    );

    let mut max_pos_err = 0.0_f64;

    for record in &trajectory[1..] {
        // Update Sun position from ephemeris
        let tdb_jd = j2000_jd + record.time / 86_400.0;
        let (sun_pos, _) = ephemeris
            .get_earth_centered_state(EphemerisBody::Sun, tdb_jd)
            .expect("Sun position query");
        sim.sources[sun].position = sun_pos;

        sim.step_until(record.time);

        let body = sim.body(0);
        let pos_err = (body.trans.position - record.position).length();
        max_pos_err = max_pos_err.max(pos_err);

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

        if (record.time % 3600.0).abs() < 30.1 {
            println!(
                "  t={:6.0}s: pos_err={:.4} m  beta={:.4} deg ({:.6} rad)",
                record.time,
                pos_err,
                beta.to_degrees(),
                beta
            );
        }
    }

    println!("  Max position error: {:.6e} m", max_pos_err);

    crossval_report(
        "tier3_simulation_solar_beta",
        &[("position", max_pos_err, "m")],
    );

    // Position tracks JEOD RUN_2 trajectory
    assert!(
        max_pos_err < 0.5,
        "Position error {max_pos_err:.2} m exceeds 0.5 m"
    );
}
