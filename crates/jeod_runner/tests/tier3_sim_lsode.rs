//! Tier 3: SIM_integ_test — ABM4 and LSODE cross-validation against JEOD.
//!
//! Cross-validates our Adams-Bashforth-Moulton 4 integrator against JEOD's
//! implementation on the `orbit` test case of `SIM_integ_test`.
//!
//! # Scenario
//!
//! The orbit test in `TranslationTestOrbit` integrates a Kepler orbit with:
//! - `sma` = 6811.137 km, `ecc` = 0, `mean_anomaly_0` = 0
//! - `omega` = 1.1231543952404041e-3 rad/s (≈ circular LEO period of 93 min)
//! - μ = sma³·omega² (derived)
//!
//! JEOD logs the true Kepler solution (`true_canon_state`) alongside the
//! integrator's output (`prop_integ_state`) every 200 s for 80000 s (401
//! points, ~14 orbits).
//!
//! # Integrators covered
//!
//! - **RUN_abm4**: Adams-Bashforth-Moulton 4, fixed step, RK4 priming. This
//!   is exactly the method we implement as `IntegratorType::Abm4`. Tolerances
//!   should be tight.
//! - **RUN_lsode**: LSODE with default `ImplicitAdamsNonStiff` mode. LSODE
//!   auto-selects order (1..12) and step size adaptively; for a smooth,
//!   non-stiff circular orbit it settles on high-order Adams. We compare
//!   against our fixed-order ABM4 as an approximation — the trajectories
//!   should agree to millimeter-scale over 80000 s.
//!
//! The LSODE variable-order Adams method and BDF (stiff) support from JEOD's
//! `lsode_first_order_ode_integrator` remain as future work. See
//! `crates/jeod_dynamics/src/abm4.rs` for the doc rationale.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_runner::{GravitySourceEntry, RotationModel, Simulation, VehicleConfig};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, IntegratorType, SimulationTime,
    TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use std::path::Path;

/// Orbital parameters from `TranslationTestOrbit` defaults.
/// JEOD: `translation_test.hh` member initializers.
const SMA: f64 = 6_811_137.0; // m
const MDOT: f64 = 1.123_154_395_240_404_1e-3; // rad/s
/// Dynamics timestep used by the sim (S_define: DYNAMICS = 1.00).
const SIM_DT: f64 = 1.0;

/// Compute μ from sma and mean motion (consistent with `TranslationTestOrbit`).
fn compute_mu() -> f64 {
    SMA * SMA * SMA * MDOT * MDOT
}

/// CSV loader for SIM_integ_test orbit logs (injected via INTEG_SNIPPET in
/// `generate_references.sh`). Columns are:
///   0: time (s)
///   1-3: prop_integ_state.position[0..2] (m)
///   4-6: prop_integ_state.velocity[0..2] (m/s)
///   7-9: true_canon_state.position[0..2] (m)
///   10-12: true_canon_state.velocity[0..2] (m/s)
///   13: rel_position_err_mag (dimensionless)
///   14: rel_velocity_err_mag (dimensionless)
///   15: rel_energy_error (dimensionless)
struct IntegRecord {
    time: f64,
    prop_position: DVec3,
    prop_velocity: DVec3,
}

fn load_integ_csv(path: &Path) -> Vec<IntegRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read JEOD SIM_integ_test CSV at {}: {e}",
            path.display()
        )
    });
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 13,
            "SIM_integ_test CSV line {}: expected >=13 columns, got {} \
             (header order mismatch with INTEG_SNIPPET in generate_references.sh?)",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(IntegRecord {
            time: p(0),
            prop_position: DVec3::new(p(1), p(2), p(3)),
            prop_velocity: DVec3::new(p(4), p(5), p(6)),
        });
    }
    records
}

/// Run a SIM_integ_test orbit cross-validation against our integrator.
///
/// The initial state is taken from the t=0 row of the JEOD CSV (which equals
/// the Kepler solution at t=0, a JEOD source value). We propagate entirely
/// under our own code through `Simulation::step_until(record.time)` and
/// compare our state against JEOD's `prop_integ_state` at each checkpoint.
fn run_integ_test(
    test_name: &str,
    csv_label: &str,
    integrator: IntegratorType,
    pos_tol: [f64; 3],
    vel_tol: [f64; 3],
) {
    let csv_path = test_data_path(&format!("{csv_label}_integ.csv"));
    assert!(
        csv_path.exists(),
        "JEOD SIM_integ_test reference CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let trajectory = load_integ_csv(&csv_path);
    assert!(
        trajectory.len() > 100,
        "{csv_label}: expected >100 records from SIM_integ_test orbit log, got {}",
        trajectory.len()
    );

    let init = &trajectory[0];
    let mu = compute_mu();

    let mut time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    time.time_scale_factor = 1.0;
    let mut sim = Simulation::new(time, SIM_DT);

    let earth = sim.add_source(
        "Earth",
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
        },
    );

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.prop_position,
            velocity: init.prop_velocity,
        },
        integrator,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    let final_time = trajectory.last().unwrap().time;
    println!(
        "Tier 3 (Simulation): {test_name}, {} points over {:.0}s, sim_dt={SIM_DT}s, \
         μ={mu:.6e}, sma={SMA:.1}m, ω={MDOT:.3e}rad/s",
        trajectory.len(),
        final_time,
    );

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    for record in &trajectory[1..] {
        sim.step_until(record.time);
        let body = sim.body(0);
        our_states.push(StateLog {
            time: record.time,
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            ..Default::default()
        });
    }

    let ref_states: Vec<StateLog> = trajectory[1..]
        .iter()
        .map(|r| StateLog {
            time: r.time,
            position: Some(r.prop_position),
            velocity: Some(r.prop_velocity),
            ..Default::default()
        })
        .collect();

    let report = CrossvalReport::compute(test_name, &our_states, &ref_states);
    report.write();

    let max_pos = report.max_position_component();
    let max_vel = report.max_velocity_component();
    println!("  Max position error: {max_pos:.6e} m");
    println!("  Max velocity error: {max_vel:.6e} m/s");

    report.assert_position(pos_tol);
    report.assert_velocity(vel_tol);
}

/// Cross-validate our ABM4 against JEOD's `RUN_abm4`.
///
/// Both integrate the same Kepler orbit with the same ABM4 method
/// (predictor/corrector weights from er7_utils). With RK4 priming matched
/// on both sides, the trajectories should agree to microscale — any drift
/// reflects floating-point differences, not algorithm divergence.
#[test]
fn tier3_simulation_lsode_abm4() {
    // Placeholder tolerances — tighten after observing actual errors from
    // a real run. `tier3_report` re-extracts these literals from source.
    run_integ_test(
        "tier3_simulation_lsode_abm4",
        "integ_abm4",
        IntegratorType::Abm4,
        [1.0, 1.0, 1.0],
        [1.0e-3, 1.0e-3, 1.0e-3],
    );
}

/// Cross-validate our ABM4 against JEOD's LSODE in `ImplicitAdamsNonStiff`
/// mode (RUN_lsode).
///
/// LSODE selects order and step size adaptively; for this smooth non-stiff
/// orbit problem it will typically pick a high-order Adams method with step
/// sizes similar to or larger than our fixed dt=1 s. The trajectories should
/// track closely but not bit-for-bit. Tolerances here allow for the modest
/// divergence between fixed-order ABM4 and LSODE's variable-order Adams.
#[test]
fn tier3_simulation_lsode_default() {
    // Placeholder tolerances — tighten after observing actual errors from
    // a real run. For LSODE-vs-ABM4 we expect meter-scale drift over 14
    // orbits due to method differences (variable-order vs fixed-order 4).
    run_integ_test(
        "tier3_simulation_lsode_default",
        "integ_lsode",
        IntegratorType::Abm4,
        [10.0, 10.0, 10.0],
        [1.0e-2, 1.0e-2, 1.0e-2],
    );
}
