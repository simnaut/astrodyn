//! Tier 3: SIM_OrbElem cross-validation (derived_state/verif/SIM_OrbElem)
//!
//! Point-mass gravity, eccentric orbit (e=0.36), 24h, dt=0.03125s.
//! The Simulation integrates the orbit and computes orbital elements each step.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, GravitySourceEntry, SimBody,
    Simulation, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::crossval_report;

#[test]
fn tier3_simulation_orbelem() {
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

    let mut max_sma_err = 0.0_f64;
    let mut max_ecc_err = 0.0_f64;
    let mut max_inc_err = 0.0_f64;
    let mut max_aop_err = 0.0_f64;
    let mut max_lan_err = 0.0_f64;
    let mut max_ta_err = 0.0_f64;
    let mut max_ma_err = 0.0_f64;
    let mut max_pos_err = 0.0_f64;

    for record in &records[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);
        let pos_err = (body.trans.position - record.position).length();
        max_pos_err = max_pos_err.max(pos_err);

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

        if (record.time % 3600.0).abs() < 6.1 {
            println!(
                "  t={:6.0}s: pos_err={:.4} m  sma_err={:.3e} m  ecc_err={:.3e}",
                record.time, pos_err, sma_err, ecc_err
            );
        }
    }

    println!("  Max position error:  {:.6e} m", max_pos_err);
    println!("  Max SMA error:       {:.6e} m", max_sma_err);
    println!("  Max eccentricity:    {:.6e}", max_ecc_err);
    println!("  Max inclination:     {:.6e} rad", max_inc_err);
    println!("  Max arg_periapsis:   {:.6e} rad", max_aop_err);
    println!("  Max long_asc_node:   {:.6e} rad", max_lan_err);
    println!("  Max true_anom:       {:.6e} rad", max_ta_err);
    println!("  Max mean_anom:       {:.6e} rad", max_ma_err);

    crossval_report(
        "tier3_simulation_orbelem",
        &[
            ("position", max_pos_err, 0.5, "m"),
            ("sma", max_sma_err, 1.0, "m"),
            ("eccentricity", max_ecc_err, 1e-10, ""),
            ("inclination", max_inc_err, 1e-10, "rad"),
            ("arg_periapsis", max_aop_err, 1e-8, "rad"),
            ("long_asc_node", max_lan_err, 1e-8, "rad"),
            ("true_anom", max_ta_err, 1e-8, "rad"),
            ("mean_anom", max_ma_err, 1e-8, "rad"),
        ],
    );

    // Position tolerance (same as RUN_2 point-mass test)
    assert!(
        max_pos_err < 0.5,
        "Position error {max_pos_err:.2} m exceeds 0.5 m"
    );
    // Orbital element tolerances account for integration-induced position drift.
    // SMA: ~0.5 m position error -> ~0.1 m SMA error via vis-viva.
    // Angular elements: near machine precision since the math is validated.
    assert!(
        max_sma_err < 1.0,
        "SMA error {max_sma_err:.3e} m exceeds 1.0 m"
    );
    assert!(
        max_ecc_err < 1e-10,
        "Eccentricity error {max_ecc_err:.3e} exceeds 1e-10"
    );
    assert!(
        max_inc_err < 1e-10,
        "Inclination error {max_inc_err:.3e} rad exceeds 1e-10"
    );
    assert!(
        max_aop_err < 1e-8,
        "Arg periapsis error {max_aop_err:.3e} rad exceeds 1e-8"
    );
    assert!(
        max_lan_err < 1e-8,
        "Long asc node error {max_lan_err:.3e} rad exceeds 1e-8"
    );
    assert!(
        max_ta_err < 1e-8,
        "True anomaly error {max_ta_err:.3e} rad exceeds 1e-8"
    );
    assert!(
        max_ma_err < 1e-8,
        "Mean anomaly error {max_ma_err:.3e} rad exceeds 1e-8"
    );
}
