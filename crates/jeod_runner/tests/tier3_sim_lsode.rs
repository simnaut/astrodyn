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
//! - Random (but deterministic in regression mode) rotation between
//!   canonical and integration frames — the logged `prop_integ_state` lives
//!   in a frame that is deterministically rotated relative to the Kepler
//!   solution, so the CSV's t=0 values must be used as initial conditions.
//!
//! JEOD logs the true Kepler solution (`true_canon_state`) alongside the
//! integrator's output (`prop_integ_state`) every 200 s for 80000 s (401
//! points, ~14 orbits).
//!
//! # Dynamic-time scaling
//!
//! `SIM_integ_test` uses Trick's `TimeDyn::scale_factor` so that every 1 s
//! of sim time corresponds to `π / (180 · omega)` s of dynamic time — one
//! degree of orbital phase per sim step. `IntegrationTest::initialize`
//! (models/utils/integration/verif/src/integration_test.cc:167) computes
//! `delta_t = omega_dt / omega` and `time_scale = delta_t / sim_dt`, with
//! `omega_dt = 1°` coming from `run_common.py` and `omega = mdot` from
//! `TranslationTestOrbit::pre_initialize`. The CSV's time column is
//! sim-seconds, but the integrator internally advances `dyn_dt = sim_dt *
//! time_scale ≈ 15.54 s` per step. We match this by setting
//! [`SimulationTime::time_scale_factor`] to the same value.
//!
//! # Integrators covered
//!
//! - **RUN_abm4**: Adams-Bashforth-Moulton 4, fixed step, RK4 priming. This
//!   is exactly the method we implement as `IntegratorType::Abm4`. With
//!   matched dyn-dt the trajectories agree to ~0.3 mm over 14 orbits,
//!   dominated by floating-point reduction differences.
//! - **RUN_lsode**: LSODE with default `ImplicitAdamsNonStiff` mode. LSODE
//!   auto-selects order (1..12) and step size adaptively; for a smooth,
//!   non-stiff circular orbit it settles on a very high-order Adams method
//!   with `rel_position_err_mag` ≈ 5e-9. We compare against our fixed-order
//!   ABM4 and tolerate km-scale drift over 14 orbits — this is the expected
//!   order-4 vs order-~12 truncation difference, not an algorithm bug.
//!
//! The LSODE variable-order Adams method and BDF (stiff) support from JEOD's
//! `lsode_first_order_ode_integrator` remain as future work. See
//! `crates/jeod_dynamics/src/abm4.rs` for the doc rationale.

use jeod_test_data::tier3_csv::test_data_path;

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

/// Dynamic-time scale factor used by JEOD's `SIM_integ_test`.
///
/// `IntegrationTest::initialize` (integration_test.cc:167-172) computes
/// `delta_t = omega_dt / omega` and `time_scale = delta_t / sim_dt` when
/// both `omega_dt` (set from `run_common.py` to 1 degree) and `omega`
/// (set by `TranslationTestOrbit::pre_initialize` to `mdot`) are positive.
///
/// This drives Trick's `TimeDyn::scale_factor`: dynamic time advances by
/// `sim_dt * time_scale` per sim step, so each step represents 1° of
/// orbital phase. The CSV's time column remains sim-time, but the orbital
/// dynamics inside each step run at the scaled dt.
fn compute_time_scale() -> f64 {
    // omega_dt = 1° = π/180 rad, omega = MDOT rad/s, sim_dt = 1 s.
    (std::f64::consts::PI / 180.0) / MDOT / SIM_DT
}

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
    let time_scale = compute_time_scale();

    let mut time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    // JEOD's IntegrationTest sets TimeDyn::scale_factor so that one sim-second
    // corresponds to `time_scale` dynamic-seconds. The CSV's time column is
    // sim-time; the integrator internally uses dyn-dt = sim_dt * time_scale.
    time.time_scale_factor = time_scale;
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
/// on both sides, the trajectories agree to sub-millimeter scale — any
/// drift reflects floating-point reduction order differences, not algorithm
/// divergence.
// non-recipe: SIM_integ_test derives μ from sma/mean-motion (computed in
// `compute_mu`), uses CSV t=0 prop_integ_state as IC (deterministically
// rotated), and applies `time_scale_factor`. None of these match a recipe
// preset; cross-validation math is owned by `CrossvalReport`.
#[test]
fn tier3_simulation_lsode_abm4() {
    // Observed max-component errors over 14 orbits (80000 s sim time):
    //   position [3.406e-4, 3.260e-4, 2.152e-4] m
    //   velocity [3.870e-7, 3.553e-7, 2.362e-7] m/s
    // Tolerances set to 5% above observed (CLAUDE.md tolerance policy).
    run_integ_test(
        "tier3_simulation_lsode_abm4",
        "integ_abm4",
        IntegratorType::Abm4,
        [3.576e-4, 3.423e-4, 2.260e-4],
        [4.064e-7, 3.731e-7, 2.480e-7],
    );
}

/// Cross-validate our ABM4 against JEOD's LSODE in `ImplicitAdamsNonStiff`
/// mode (RUN_lsode).
///
/// LSODE selects order and step size adaptively; for this smooth non-stiff
/// orbit problem it settles on a very high-order Adams method (JEOD's log
/// reports `rel_position_err_mag ≈ 5e-9` — ~34 mm over 80000 s). Our
/// fixed-order ABM4 at the same dyn-dt is much less accurate: the drift
/// between the two methods reaches ~9.5 km per component over 14 orbits,
/// which is the expected order-4 vs order-~12 truncation difference. This
/// test documents that drift rather than asserting agreement between
/// dissimilar integrators.
// non-recipe: same derived μ + CSV-rotated IC as `tier3_simulation_lsode_abm4`.
#[test]
fn tier3_simulation_lsode_default() {
    // Observed max-component errors (ABM4 vs LSODE reference, 14 orbits):
    //   position [9.485e3, 9.130e3, 6.028e3] m
    //   velocity [1.082e1, 9.908e0, 6.581e0] m/s
    // Tolerances set to 5% above observed (CLAUDE.md tolerance policy).
    // The drift is dominated by ABM4's fixed order-4 truncation error; a
    // future port of LSODE's variable-order Adams scheme would shrink these
    // to the same floating-point-noise level as `tier3_simulation_lsode_abm4`.
    run_integ_test(
        "tier3_simulation_lsode_default",
        "integ_lsode",
        IntegratorType::Abm4,
        [9.960e3, 9.587e3, 6.330e3],
        [1.137e1, 1.041e1, 6.910e0],
    );
}
