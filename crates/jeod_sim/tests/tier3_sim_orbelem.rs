//! Tier 3: SIM_OrbElem cross-validation (derived_state/verif/SIM_OrbElem)
//!
//! Point-mass gravity, eccentric orbit (e=0.36), 24h, dt=0.03125s.
//! The Simulation integrates the orbit and computes orbital elements each step.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, GravitySourceEntry,
    RotationModel, SimBody, Simulation, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};

fn load_mu_earth() -> f64 {
    let jeod_root = jeod_test_data::jeod_path();
    jeod_sim::coefficients::load_mu_from_jeod_cc(
        &jeod_root.join("models/environment/gravity/data/src/earth_GGM05C.cc"),
    )
    .expect("load Earth mu from GGM05C")
}

#[test]
fn tier3_simulation_orbelem() {
    let mu_earth = load_mu_earth();
    let csv_path = test_data_path("orbelem_ecc_orbelem.csv");
    assert!(
        csv_path.exists(),
        "SIM_OrbElem RUN_ecc CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_orbelem_csv(&csv_path);
    assert!(records.len() > 100);
    let init = &records[0];

    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );
    let dt = jeod_test_data::s_define::load_dynamics_dt(
        &jeod_root.join("models/dynamics/derived_state/verif/SIM_OrbElem/S_define"),
    );

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

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        orbital_elements_source: Some(earth),
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): SIM_OrbElem derived state, {} points",
        records.len()
    );

    let mut our_states = Vec::with_capacity(records.len() - 1);
    let mut ref_states = Vec::with_capacity(records.len() - 1);
    let mut max_sma_err = 0.0_f64;
    let mut max_ecc_err = 0.0_f64;
    let mut max_inc_err = 0.0_f64;
    let mut max_aop_err = 0.0_f64;
    let mut max_lan_err = 0.0_f64;
    let mut max_ta_err = 0.0_f64;
    let mut max_ma_err = 0.0_f64;

    for record in &records[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);

        let oe = body.orbital_elements.as_ref().unwrap_or_else(|| {
            panic!(
                "Simulation did not compute orbital elements at t={}",
                record.time
            )
        });

        let sma_err = (oe.semi_major_axis - record.semi_major_axis).abs();
        let ecc_err = (oe.e_mag - record.e_mag).abs();
        let inc_err = (oe.inclination - record.inclination).abs();
        let aop_err = angle_diff(oe.arg_periapsis, record.arg_periapsis);
        let lan_err = angle_diff(oe.long_asc_node, record.long_asc_node);
        let ta_err = angle_diff(oe.true_anom, record.true_anom);
        let ma_err = angle_diff(oe.mean_anom, record.mean_anom);

        max_sma_err = max_sma_err.max(sma_err);
        max_ecc_err = max_ecc_err.max(ecc_err);
        max_inc_err = max_inc_err.max(inc_err);
        max_aop_err = max_aop_err.max(aop_err);
        max_lan_err = max_lan_err.max(lan_err);
        max_ta_err = max_ta_err.max(ta_err);
        max_ma_err = max_ma_err.max(ma_err);

        our_states.push(StateLog {
            time: record.time,
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            ..Default::default()
        });
        ref_states.push(StateLog {
            time: record.time,
            position: Some(record.position),
            velocity: Some(record.velocity),
            ..Default::default()
        });

        if (record.time % 3600.0).abs() < 6.1 {
            let pos_err = (body.trans.position - record.position).length();
            println!(
                "  t={:6.0}s: pos_err={:.4} m  sma_err={:.3e} m  ecc_err={:.3e}",
                record.time, pos_err, sma_err, ecc_err
            );
        }
    }

    let max_pos_err = our_states
        .iter()
        .zip(ref_states.iter())
        .map(|(a, b)| (a.position.unwrap() - b.position.unwrap()).length())
        .fold(0.0_f64, f64::max);

    println!("  Max position error:  {:.6e} m", max_pos_err);
    println!("  Max SMA error:       {:.6e} m", max_sma_err);
    println!("  Max eccentricity:    {:.6e}", max_ecc_err);
    println!("  Max inclination:     {:.6e} rad", max_inc_err);
    println!("  Max arg_periapsis:   {:.6e} rad", max_aop_err);
    println!("  Max long_asc_node:   {:.6e} rad", max_lan_err);
    println!("  Max true_anom:       {:.6e} rad", max_ta_err);
    println!("  Max mean_anom:       {:.6e} rad", max_ma_err);

    let mut report = CrossvalReport::compute("tier3_simulation_orbelem", &our_states, &ref_states);
    report.add_extra("sma", max_sma_err, "m");
    assert!(max_sma_err < 2.613e-6, "sma");
    report.add_extra("eccentricity", max_ecc_err, "");
    assert!(max_ecc_err < 1.496e-13, "eccentricity");
    report.add_extra("inclination", max_inc_err, "rad");
    assert!(max_inc_err < 8.436e-17, "inclination");
    report.add_extra("arg_periapsis", max_aop_err, "rad");
    assert!(max_aop_err < 1.78e-12, "arg_periapsis");
    report.add_extra("long_asc_node", max_lan_err, "rad");
    assert!(max_lan_err < 9.513e-14, "long_asc_node");
    report.add_extra("true_anom", max_ta_err, "rad");
    assert!(max_ta_err < 1.136e-11, "true_anom");
    report.add_extra("mean_anom", max_ma_err, "rad");
    assert!(max_ma_err < 5.642e-12, "mean_anom");
    report.write();

    report.assert_position([6.556e-5, 5.15e-5, 5.478e-8]);
}
