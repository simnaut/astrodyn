//! Tier 3: SIM_orbinit docker cross-validation (t=0 initialization)

#![allow(
    clippy::float_cmp,
    reason = "Tier 3 tests assert bit-exact recovery of literal-built / analytic state values"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Tier 3 step counts and indices fit exactly in f64 mantissa and usize"
)]
//!
//! JEOD's SIM_orbinit is an initialization-only sim
//! (`exec_set_terminate_time(0)`) that writes the post-initialization
//! state of `composite_body` into a CSV at t=0. Each RUN exercises a
//! different orbital-element set or coordinate frame.
//!
//! These tests build each scenario through its `sim_orbinit_docker`
//! recipe, which performs the orbital-element-to-Cartesian conversion
//! (or direct-Cartesian pass-through for RUN_0401) from JEOD source
//! fixtures and feeds the result into a point-mass-Earth `Simulation`.
//! The pre-propagation `body(0)` state is compared against JEOD's
//! logged t=0 row with the same tight tolerances the inline test used
//! (single-digit nanometre position on the inertial RUNs, tens of
//! micrometres on the pfix RUNs where RNP-series drift dominates).
//! The synthetic-cadence checkpoint then propagates one tick so the
//! integrator + frame-propagation stages run end-to-end, exercising
//! the full `Simulation` pipeline.
//!
//! Per CLAUDE.md: initial conditions from JEOD source files are
//! permitted; JEOD output (CSV values) is never fed back into our
//! computation. The Docker reference CSVs are read here only as the
//! comparison target for the recipe's computed initial state.
//!
//! Scenarios:
//!   RUN_0001: ISS orbital elements in inertial frame (set01, time_periapsis)
//!   RUN_0003: ISS orbital elements in inertial frame (set03, slr + true anomaly)
//!   RUN_0103: STS-114 orbital elements in inertial frame (set03, slr + true anomaly)
//!   RUN_0004: ISS orbital elements in inertial frame (set04, altitudes + true anomaly)
//!   RUN_0104: STS-114 orbital elements in inertial frame (set04, altitudes + true anomaly)
//!   RUN_0005: ISS orbital elements in inertial frame (set05, altitudes + time_periapsis)
//!   RUN_0105: STS-114 orbital elements in inertial frame (set05, altitudes + time_periapsis)
//!   RUN_0101: STS-114 orbital elements in inertial frame (set01, time_periapsis)
//!   RUN_0201: ISS orbital elements in planet-fixed (pfix) frame (set01)
//!   RUN_0301: STS-114 orbital elements in planet-fixed (pfix) frame (set01)
//!   RUN_0401: STS-114 direct Cartesian state in inertial frame
//!
//! All scenarios share the same JEOD epoch: 2005-07-28 10:09:59 UT1.
//! The SIM disables polar motion (`earth.rnp.enable_polar = False`).
//! Gravity uses `earth_GGM05C` with `mu = 3.9860044150e14 m^3/s^2`.
//!
//! The `Simulation` construction lives in the `sim_orbinit_docker`
//! recipe module so the parity wrapper (`bevy_parity_orbinit_docker.rs`)
//! can drive the same scenarios through the Bevy adapter for the
//! `runner ↔ bevy` half of the transitivity argument.

use astrodyn_runner::builder::SimulationBuilderExt;
use astrodyn_runner::Simulation;
use astrodyn_verif_jeod::run_verification::sim_orbinit_docker;
use astrodyn_verif_jeod::tier3_csv::{load_orbinit_csv, test_data_path};
use astrodyn_verif_jeod::verification::{CsvReference, InitialConditions, VerificationCase};

/// Build the recipe's `Simulation` exactly the way the parity trait
/// does — call the scenario factory with a default `InitialConditions`
/// (the recipes compute their initial state from committed body-init
/// fixtures and don't read `InitialConditions`), then `.build()` — so
/// the runner-side propagation here and the Bevy-side propagation in
/// `bevy_parity_orbinit_docker.rs` see the same initial state
/// bit-pattern.
fn build_sim(case: &VerificationCase) -> Simulation {
    (case.scenario)(&InitialConditions::default())
        .build()
        .unwrap_or_else(|e| panic!("scenario `{}` build failed: {e:?}", case.name))
}

/// Pull `(dt, num_steps)` off a recipe's
/// [`CsvReference::SyntheticTimes`] reference. Every recipe in
/// `sim_orbinit_docker` uses this variant because the orbinit Docker
/// CSVs are initialization-only (one row at t=0); panicking on any
/// other variant surfaces a future recipe-shape drift here rather
/// than producing a silently-truncated propagation. Returning both
/// halves of the cadence lets callers assert that the `dt` they're
/// stepping at (`sim.dt`) matches the cadence the recipe declared.
fn synthetic_cadence(case: &VerificationCase) -> (f64, usize) {
    match &case.reference {
        CsvReference::SyntheticTimes { dt, num_steps } => (*dt, *num_steps),
        _ => panic!("`{}`: expected SyntheticTimes reference", case.name),
    }
}

/// Build a Simulation from the recipe, compare its pre-propagation
/// initial state against the JEOD-logged t=0 row, then propagate the
/// recipe's synthetic cadence so the integrator + frame-propagation
/// stages run end-to-end. The initial-state comparison is the
/// substantive assertion — the orbital-element-to-Cartesian
/// conversion runs inside the recipe's scenario factory, so this
/// check exercises the same code paths the parity wrapper drives on
/// both runtimes.
fn assert_orbinit_match(
    case: VerificationCase,
    csv_filename: &str,
    label: &str,
    pos_tol: f64,
    vel_tol: f64,
) {
    let csv_path = test_data_path(csv_filename);
    assert!(
        csv_path.exists(),
        "{label}: JEOD reference CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/crates/astrodyn_verif_jeod/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );
    let records = load_orbinit_csv(&csv_path);
    assert!(
        !records.is_empty(),
        "{label}: no records in {}",
        csv_path.display()
    );
    let jeod = &records[0];
    assert_eq!(jeod.time, 0.0, "{label}: expected t=0 row in CSV");

    let mut sim = build_sim(&case);
    let (dt, n_steps) = synthetic_cadence(&case);
    assert_eq!(
        dt, sim.dt,
        "`{}`: recipe SyntheticTimes dt ({dt}) and Simulation dt ({}) drifted apart",
        case.name, sim.dt,
    );

    // Read the pre-propagation initial state — this is what the recipe
    // built from the orbital-element fixture (and optional pfix
    // rotation), and it's what the parity wrapper integrates from.
    let init_pos = sim.body(0).trans.position.raw_si();
    let init_vel = sim.body(0).trans.velocity.raw_si();

    let pos_err = (init_pos - jeod.position).length();
    let vel_err = (init_vel - jeod.velocity).length();

    println!(
        "  {label}: our pos=[{:.3}, {:.3}, {:.3}] m",
        init_pos.x, init_pos.y, init_pos.z
    );
    println!(
        "  {label}: JEOD pos=[{:.3}, {:.3}, {:.3}] m",
        jeod.position.x, jeod.position.y, jeod.position.z
    );
    println!("  {label}: pos_err = {pos_err:.6e} m  vel_err = {vel_err:.6e} m/s");

    assert!(
        pos_err < pos_tol,
        "{label}: position error {pos_err:.6e} m exceeds tolerance {pos_tol:.1e} m"
    );
    assert!(
        vel_err < vel_tol,
        "{label}: velocity error {vel_err:.6e} m/s exceeds tolerance {vel_tol:.1e} m/s"
    );

    // Drive the integrator + frame-propagation stages end-to-end at
    // the recipe's synthetic cadence so the pipeline runs through.
    // `step_n` advances exactly `n_steps` whole steps. (`step_until`
    // has a 1 ms slop and may stop one step short.)
    sim.step_n(n_steps).expect("step_n failed");
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0001: ISS orbital elements in inertial frame
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0001_iss_inertial() {
    // RUN_0001: ISS, SmaEccIncAscnodeArgperTimeperi, reference=Earth.inertial.
    // No frame rotation required — recipe output is already in inertial.
    // Observed: pos=6.25e-9 m, vel=6.19e-12 m/s (5% above → listed).
    assert_orbinit_match(
        sim_orbinit_docker::run_0001(),
        "orbinit_0001_orbinit.csv",
        "RUN_0001 (ISS inertial set01)",
        6.56e-9,
        6.50e-12,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0101: STS-114 orbital elements in inertial frame
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0101_sts_inertial() {
    // Observed: pos=1.04e-9 m, vel=2.27e-13 m/s (5% above → listed).
    assert_orbinit_match(
        sim_orbinit_docker::run_0101(),
        "orbinit_0101_orbinit.csv",
        "RUN_0101 (STS-114 inertial set01)",
        1.10e-9,
        2.39e-13,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0002 / RUN_0102: set02 (mean-anomaly parameterization), inertial frame.
// Exercises `init_from_mean_anomaly` directly (distinct from set01's
// time-periapsis → mean-anomaly derivation). Tolerances are 1.05× observed.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0002_iss_inertial() {
    // Observed: pos=3.26e-9 m, vel=3.40e-12 m/s (5% above → listed).
    assert_orbinit_match(
        sim_orbinit_docker::run_0002(),
        "orbinit_0002_orbinit.csv",
        "RUN_0002 (ISS inertial set02, mean anomaly)",
        3.42e-9,
        3.57e-12,
    );
}

#[test]
fn tier3_orbinit_docker_run0102_sts_inertial() {
    // Observed: pos=1.68e-9 m, vel=2.33e-12 m/s (5% above → listed).
    assert_orbinit_match(
        sim_orbinit_docker::run_0102(),
        "orbinit_0102_orbinit.csv",
        "RUN_0102 (STS-114 inertial set02, mean anomaly)",
        1.76e-9,
        2.45e-12,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0003 / RUN_0103: set03 (semi-latus rectum + true-anomaly), inertial
// frame. Exercises `init_from_semi_latus_rectum_true_anomaly` directly —
// JEOD's `SlrEccIncAscnodeArgperTanom` branch uses the deck's semi-latus
// rectum as `semiparam` verbatim (no sma round-trip). Tolerances 1.05× observed.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0003_iss_inertial() {
    // Observed: pos=5.21e-10 m, vel=2.27e-13 m/s (5% above → listed).
    assert_orbinit_match(
        sim_orbinit_docker::run_0003(),
        "orbinit_0003_orbinit.csv",
        "RUN_0003 (ISS inertial set03, slr + true anomaly)",
        5.47e-10,
        2.39e-13,
    );
}

#[test]
fn tier3_orbinit_docker_run0103_sts_inertial() {
    // Observed: pos=1.40e-9 m, vel=9.37e-13 m/s (5% above → listed).
    assert_orbinit_match(
        sim_orbinit_docker::run_0103(),
        "orbinit_0103_orbinit.csv",
        "RUN_0103 (STS-114 inertial set03, slr + true anomaly)",
        1.47e-9,
        9.84e-13,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0004 / RUN_0104: set04 (apo/peri altitudes + true-anomaly), inertial
// frame. Exercises `init_from_altitudes_true_anomaly`: JEOD's `ShapeAltitudes`
// branch derives sma/ecc from the altitudes (referenced to Earth's equatorial
// radius `r_eq = 6_378_137 m`) before resolving the true anomaly. Tolerances
// are 1.05× observed.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0004_iss_inertial() {
    // Observed: pos=4.66e-10 m, vel=0 m/s (5% above → listed; vel floored at
    // 1e-13 m/s since the exact-zero observed residual leaves no headroom).
    assert_orbinit_match(
        sim_orbinit_docker::run_0004(),
        "orbinit_0004_orbinit.csv",
        "RUN_0004 (ISS inertial set04, altitudes + true anomaly)",
        4.89e-10,
        1.0e-13,
    );
}

#[test]
fn tier3_orbinit_docker_run0104_sts_inertial() {
    // Observed: pos=2.13e-9 m, vel=2.27e-13 m/s (5% above → listed).
    assert_orbinit_match(
        sim_orbinit_docker::run_0104(),
        "orbinit_0104_orbinit.csv",
        "RUN_0104 (STS-114 inertial set04, altitudes + true anomaly)",
        2.24e-9,
        2.39e-13,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0005 / RUN_0105: set05 (apo/peri altitudes + time-periapsis), inertial
// frame. Exercises `init_from_altitudes_time_periapsis`: sma/ecc from the
// altitudes as in set04, then `time_periapsis → mean anomaly` exactly as
// set01's derivation. Tolerances are 1.05× observed.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0005_iss_inertial() {
    // Observed: pos=5.35e-9 m, vel=5.57e-12 m/s (5% above → listed).
    assert_orbinit_match(
        sim_orbinit_docker::run_0005(),
        "orbinit_0005_orbinit.csv",
        "RUN_0005 (ISS inertial set05, altitudes + time periapsis)",
        5.62e-9,
        5.85e-12,
    );
}

#[test]
fn tier3_orbinit_docker_run0105_sts_inertial() {
    // Observed: pos=4.01e-9 m, vel=4.51e-12 m/s (5% above → listed).
    assert_orbinit_match(
        sim_orbinit_docker::run_0105(),
        "orbinit_0105_orbinit.csv",
        "RUN_0105 (STS-114 inertial set05, altitudes + time periapsis)",
        4.21e-9,
        4.73e-12,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0201: ISS orbital elements in planet-fixed frame
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0201_iss_pfix() {
    // RUN_0201: ISS pfix set01. Requires RNP rotation at the SIM epoch
    // (handled inside the recipe). Observed: pos=1.51e-5 m, vel=1.17e-8 m/s
    // (5% above → listed). The residual reflects tiny differences
    // between our RNP series and JEOD's over the ~11 000 km Earth
    // rotation arm from 2005-07-28.
    assert_orbinit_match(
        sim_orbinit_docker::run_0201(),
        "orbinit_0201_orbinit.csv",
        "RUN_0201 (ISS pfix set01)",
        1.59e-5,
        1.23e-8,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0301: STS-114 orbital elements in planet-fixed frame
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0301_sts_pfix() {
    // Observed: pos=1.51e-5 m, vel=1.17e-8 m/s (5% above → listed).
    assert_orbinit_match(
        sim_orbinit_docker::run_0301(),
        "orbinit_0301_orbinit.csv",
        "RUN_0301 (STS-114 pfix set01)",
        1.59e-5,
        1.23e-8,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0401: STS-114 direct Cartesian state in inertial frame
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0401_sts_trans_state() {
    // RUN_0401 uses DynBodyInitTransState (direct Cartesian input in
    // inertial). The JEOD input.py sets position and velocity directly;
    // recipe initialization is a pass-through to the body state. The
    // CSV has only ~10-char precision for RUN_0401 inputs; allow 1 µm
    // / 1 nm/s.
    assert_orbinit_match(
        sim_orbinit_docker::run_0401(),
        "orbinit_0401_orbinit.csv",
        "RUN_0401 (STS-114 inertial cart)",
        1.0e-6,
        1.0e-9,
    );
}
