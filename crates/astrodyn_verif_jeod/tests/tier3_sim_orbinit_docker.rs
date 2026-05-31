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
//!   RUN_0006: ISS orbital elements in inertial frame (set06, arg-latitude + radial-vel)
//!   RUN_0106: STS-114 orbital elements in inertial frame (set06, arg-latitude + radial-vel)
//!   RUN_0010: ISS orbital elements in inertial frame (set10, sma/ecc + true anomaly)
//!   RUN_0110: STS-114 orbital elements in inertial frame (set10, sma/ecc + true anomaly)
//!   RUN_0011: ISS orbital elements in inertial frame (set11, altitudes + true anomaly)
//!   RUN_0111: STS-114 orbital elements in inertial frame (set11, altitudes + true anomaly)
//!   RUN_0101: STS-114 orbital elements in inertial frame (set01, time_periapsis)
//!   RUN_0201: ISS orbital elements in planet-fixed (pfix) frame (set01)
//!   RUN_0301: STS-114 orbital elements in planet-fixed (pfix) frame (set01)
//!   RUN_0202: ISS orbital elements in pfix frame (set02, mean anomaly)
//!   RUN_0302: STS-114 orbital elements in pfix frame (set02, mean anomaly)
//!   RUN_0203: ISS orbital elements in pfix frame (set03, slr + true anomaly)
//!   RUN_0303: STS-114 orbital elements in pfix frame (set03, slr + true anomaly)
//!   RUN_0204: ISS orbital elements in pfix frame (set04, altitudes + true anomaly)
//!   RUN_0304: STS-114 orbital elements in pfix frame (set04, altitudes + true anomaly)
//!   RUN_0205: ISS orbital elements in pfix frame (set05, altitudes + time_periapsis)
//!   RUN_0305: STS-114 orbital elements in pfix frame (set05, altitudes + time_periapsis)
//!   RUN_0206: ISS orbital elements in pfix frame (set06, arg-latitude + radial-vel)
//!   RUN_0306: STS-114 orbital elements in pfix frame (set06, arg-latitude + radial-vel)
//!   RUN_0210: ISS orbital elements in pfix frame (set10, sma/ecc + true anomaly)
//!   RUN_0310: STS-114 orbital elements in pfix frame (set10, sma/ecc + true anomaly)
//!   RUN_0211: ISS orbital elements in pfix frame (set11, altitudes + true anomaly)
//!   RUN_0311: STS-114 orbital elements in pfix frame (set11, altitudes + true anomaly)
//!   RUN_0401: STS-114 direct Cartesian state in inertial frame
//!   RUN_0400: ISS direct Cartesian state in inertial frame
//!   RUN_0410: ISS direct Cartesian state in planet-fixed (pfix) frame
//!   RUN_0411: STS-114 direct Cartesian state in planet-fixed (pfix) frame
//!   RUN_2100: ISS inertial Cartesian + direct inertial attitude/rate init
//!             (`DynBodyInitRotState`) — full-state (pos/vel/quat/rate)
//!   RUN_1230: ISS inertial Cartesian + LVLH-relative attitude/rate init
//!             (`DynBodyInitLvlhRotState`) — full-state (pos/vel/quat/rate)
//!
//! RUN_2100 / RUN_1230 are the first *rotational* RUNs: they attach a
//! 6-DOF body (ISS mass properties) and cross-validate the attitude
//! quaternion and body-frame angular velocity in addition to position
//! and velocity, against a dedicated full-state reference CSV.
//!
//! All scenarios share the same JEOD epoch: 2005-07-28 10:09:59 UT1.
//! The SIM disables polar motion (`earth.rnp.enable_polar = False`).
//! Gravity uses `earth_GGM05C` with `mu = 3.9860044150e14 m^3/s^2`.
//!
//! The `Simulation` construction lives in the `sim_orbinit_docker`
//! recipe module so the parity wrapper (`bevy_parity_orbinit_docker.rs`)
//! can drive the same scenarios through the Bevy adapter for the
//! `runner ↔ bevy` half of the transitivity argument.

use astrodyn::JeodQuat;
use astrodyn_runner::builder::SimulationBuilderExt;
use astrodyn_runner::Simulation;
use astrodyn_verif_jeod::run_verification::sim_orbinit_docker;
use astrodyn_verif_jeod::tier3_csv::{
    load_orbinit_csv, load_orbinit_full_state_csv, test_data_path,
};
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

/// Angle (rad) between two JEOD scalar-first quaternions, accounting
/// for the double-cover sign ambiguity (`q` and `−q` are the same
/// attitude). Mirrors the bespoke helper in
/// `tier3_sim_apollo_trajectory.rs`.
fn quat_angle_between(a: JeodQuat, b: JeodQuat) -> f64 {
    let av = a.vector();
    let bv = b.vector();
    let dot = a.scalar() * b.scalar() + av.x * bv.x + av.y * bv.y + av.z * bv.z;
    2.0 * dot.abs().min(1.0).acos()
}

/// Full-state cross-validation for the rotational-init RUNs: in
/// addition to the position / velocity comparison [`assert_orbinit_match`]
/// performs, this reads the JEOD-logged attitude quaternion and
/// body-frame angular velocity at t=0 and compares them against the
/// recipe's computed rotational state. RUN_2100's attitude is the
/// non-identity Yaw-Pitch-Roll triple `[77.59, -30.60, -46.10]` deg,
/// so the quaternion-angle assertion is a genuine convention check
/// (Euler sequence, deg→rad, scalar-first↔scalar-last), not a trivial
/// identity pass.
#[allow(
    clippy::too_many_arguments,
    reason = "four per-component tolerances (pos/vel/quat-angle/ang-vel) plus case/csv/label mirror the assert_orbinit_match signature; grouping them would obscure the call sites"
)]
fn assert_orbinit_full_state(
    case: VerificationCase,
    csv_filename: &str,
    label: &str,
    pos_tol: f64,
    vel_tol: f64,
    quat_angle_tol: f64,
    ang_vel_tol: f64,
) {
    let csv_path = test_data_path(csv_filename);
    assert!(
        csv_path.exists(),
        "{label}: JEOD reference CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/crates/astrodyn_verif_jeod/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );
    let records = load_orbinit_full_state_csv(&csv_path);
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

    let body = sim.body(0);
    let init_pos = body.trans.position.raw_si();
    let init_vel = body.trans.velocity.raw_si();
    let rot = body
        .rot
        .as_ref()
        .unwrap_or_else(|| panic!("{label}: expected a 6-DOF body with rotational state"));
    let our_quat = rot.q_inertial_body.to_jeod_quat();
    let our_ang_vel = rot.ang_vel_body.raw_si();

    let pos_err = (init_pos - jeod.position).length();
    let vel_err = (init_vel - jeod.velocity).length();
    let jeod_quat = JeodQuat::from_array(jeod.quaternion);
    let quat_angle_err = quat_angle_between(our_quat, jeod_quat);
    let ang_vel_err = (our_ang_vel - jeod.ang_vel_body).length();

    println!(
        "  {label}: pos_err={pos_err:.3e} m  vel_err={vel_err:.3e} m/s  \
         quat_angle_err={quat_angle_err:.3e} rad  ang_vel_err={ang_vel_err:.3e} rad/s"
    );

    assert!(
        pos_err < pos_tol,
        "{label}: position error {pos_err:.6e} m exceeds tolerance {pos_tol:.1e} m"
    );
    assert!(
        vel_err < vel_tol,
        "{label}: velocity error {vel_err:.6e} m/s exceeds tolerance {vel_tol:.1e} m/s"
    );
    assert!(
        quat_angle_err < quat_angle_tol,
        "{label}: attitude angle error {quat_angle_err:.6e} rad exceeds tolerance \
         {quat_angle_tol:.1e} rad"
    );
    assert!(
        ang_vel_err < ang_vel_tol,
        "{label}: angular-velocity error {ang_vel_err:.6e} rad/s exceeds tolerance \
         {ang_vel_tol:.1e} rad/s"
    );

    // Drive the pipeline end-to-end at the synthetic cadence.
    sim.step_n(n_steps).expect("step_n failed");
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0001: ISS orbital elements in inertial frame
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0001_iss_inertial() {
    // RUN_0001: ISS, SmaEccIncAscnodeArgperTimeperi, reference=Earth.inertial.
    // No frame rotation required — recipe output is already in inertial.
    // Tolerances 1.05× observed max (CLAUDE.md).
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
    // Tolerances 1.05× observed max (CLAUDE.md).
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
    // Tolerances 1.05× observed max (CLAUDE.md).
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
    // Tolerances 1.05× observed max (CLAUDE.md).
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
    // Tolerances 1.05× observed max (CLAUDE.md).
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
    // Tolerances 1.05× observed max (CLAUDE.md).
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
    // Tolerances 1.05× observed max; vel floored at 1e-13 m/s since the
    // exact-zero observed residual leaves no headroom.
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
    // Tolerances 1.05× observed max (CLAUDE.md).
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
    // Tolerances 1.05× observed max (CLAUDE.md).
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
    // Tolerances 1.05× observed max (CLAUDE.md).
    assert_orbinit_match(
        sim_orbinit_docker::run_0105(),
        "orbinit_0105_orbinit.csv",
        "RUN_0105 (STS-114 inertial set05, altitudes + time periapsis)",
        4.21e-9,
        4.73e-12,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0006 / RUN_0106: set06 (arg-latitude + radial-vel), inertial frame.
// Exercises `init_from_arg_latitude_radial_vel`: JEOD's
// `SmaIncAscnodeArglatRadRadvel` branch derives (e, ν, ω) from the orbital
// radius / radial-velocity pair via the eccentric-anomaly identities, then
// resolves the sma + true-anomaly shape. Tolerances are 1.05× observed.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0006_iss_inertial() {
    // Tolerances 1.05× observed max (CLAUDE.md).
    assert_orbinit_match(
        sim_orbinit_docker::run_0006(),
        "orbinit_0006_orbinit.csv",
        "RUN_0006 (ISS inertial set06, arg-latitude + radial-vel)",
        4.89e-10,
        2.39e-13,
    );
}

#[test]
fn tier3_orbinit_docker_run0106_sts_inertial() {
    // Tolerances 1.05× observed max; vel floored at 1e-13 m/s since the
    // exact-zero observed residual leaves no headroom.
    assert_orbinit_match(
        sim_orbinit_docker::run_0106(),
        "orbinit_0106_orbinit.csv",
        "RUN_0106 (STS-114 inertial set06, arg-latitude + radial-vel)",
        6.91e-10,
        1.0e-13,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0010 / RUN_0110: set10 (sma/ecc + true-anomaly), inertial frame.
// Exercises `init_from_orbital_elements` directly — JEOD's
// `SmaEccIncAscnodeArgperTanom` branch derives semiparam = a·(1−e²) and
// resolves the true anomaly. Tolerances are 1.05× observed.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0010_iss_inertial() {
    // Tolerances 1.05× observed max (CLAUDE.md).
    assert_orbinit_match(
        sim_orbinit_docker::run_0010(),
        "orbinit_0010_orbinit.csv",
        "RUN_0010 (ISS inertial set10, sma/ecc + true anomaly)",
        6.91e-10,
        4.77e-13,
    );
}

#[test]
fn tier3_orbinit_docker_run0110_sts_inertial() {
    // Tolerances 1.05× observed max; vel floored at 1e-13 m/s since the
    // exact-zero observed residual leaves no headroom.
    assert_orbinit_match(
        sim_orbinit_docker::run_0110(),
        "orbinit_0110_orbinit.csv",
        "RUN_0110 (STS-114 inertial set10, sma/ecc + true anomaly)",
        1.20e-9,
        1.0e-13,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0011 / RUN_0111: set11 (apo/peri altitudes + true-anomaly), inertial
// frame. JEOD's `CaseEleven` is the same option as set04, so this reuses
// `init_from_altitudes_true_anomaly`. Tolerances are 1.05× observed.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0011_iss_inertial() {
    // Tolerances 1.05× observed max; vel floored at 1e-13 m/s. Matches
    // RUN_0004 exactly — set11 (CaseEleven) is the same JEOD option as
    // set04 with the same ISS elements.
    assert_orbinit_match(
        sim_orbinit_docker::run_0011(),
        "orbinit_0011_orbinit.csv",
        "RUN_0011 (ISS inertial set11, altitudes + true anomaly)",
        4.89e-10,
        1.0e-13,
    );
}

#[test]
fn tier3_orbinit_docker_run0111_sts_inertial() {
    // Tolerances 1.05× observed max. Matches RUN_0104 exactly — set11
    // (CaseEleven) is the same JEOD option as set04 with the same
    // STS-114 elements.
    assert_orbinit_match(
        sim_orbinit_docker::run_0111(),
        "orbinit_0111_orbinit.csv",
        "RUN_0111 (STS-114 inertial set11, altitudes + true anomaly)",
        2.24e-9,
        2.39e-13,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0201: ISS orbital elements in planet-fixed frame
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0201_iss_pfix() {
    // RUN_0201: ISS pfix set01. Requires RNP rotation at the SIM epoch
    // (handled inside the recipe). Tolerances 1.05× observed max. The
    // residual reflects tiny differences between our RNP series and
    // JEOD's over the ~11 000 km Earth rotation arm from 2005-07-28.
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
    // Tolerances 1.05× observed max (CLAUDE.md).
    assert_orbinit_match(
        sim_orbinit_docker::run_0301(),
        "orbinit_0301_orbinit.csv",
        "RUN_0301 (STS-114 pfix set01)",
        1.59e-5,
        1.23e-8,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0202 / RUN_0302: set02 (mean-anomaly) in planet-fixed (pfix) frame.
// Same converter as RUN_0002/0102, but the elements are interpreted in
// Earth.pfix and rotated to inertial at the SIM epoch, so the residual
// reflects RNP-series drift over the Earth-rotation arm. Tolerances 1.05× observed.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0202_iss_pfix() {
    // Tolerances 1.05× observed max. The residual reflects RNP-series
    // drift over the Earth-rotation arm.
    assert_orbinit_match(
        sim_orbinit_docker::run_0202(),
        "orbinit_0202_orbinit.csv",
        "RUN_0202 (ISS pfix set02, mean anomaly)",
        1.59e-5,
        1.23e-8,
    );
}

#[test]
fn tier3_orbinit_docker_run0302_sts_pfix() {
    // Tolerances 1.05× observed max (CLAUDE.md).
    assert_orbinit_match(
        sim_orbinit_docker::run_0302(),
        "orbinit_0302_orbinit.csv",
        "RUN_0302 (STS-114 pfix set02, mean anomaly)",
        1.59e-5,
        1.23e-8,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0203 / RUN_0303: set03 (semi-latus rectum + true-anomaly) in pfix frame.
// Tolerances 1.05× observed.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0203_iss_pfix() {
    // Tolerances 1.05× observed max (CLAUDE.md).
    assert_orbinit_match(
        sim_orbinit_docker::run_0203(),
        "orbinit_0203_orbinit.csv",
        "RUN_0203 (ISS pfix set03, slr + true anomaly)",
        1.59e-5,
        1.23e-8,
    );
}

#[test]
fn tier3_orbinit_docker_run0303_sts_pfix() {
    // Tolerances 1.05× observed max (CLAUDE.md).
    assert_orbinit_match(
        sim_orbinit_docker::run_0303(),
        "orbinit_0303_orbinit.csv",
        "RUN_0303 (STS-114 pfix set03, slr + true anomaly)",
        1.59e-5,
        1.23e-8,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0204 / RUN_0304: set04 (apo/peri altitudes + true-anomaly) in pfix frame.
// Tolerances 1.05× observed.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0204_iss_pfix() {
    // Tolerances 1.05× observed max (CLAUDE.md).
    assert_orbinit_match(
        sim_orbinit_docker::run_0204(),
        "orbinit_0204_orbinit.csv",
        "RUN_0204 (ISS pfix set04, altitudes + true anomaly)",
        1.59e-5,
        1.23e-8,
    );
}

#[test]
fn tier3_orbinit_docker_run0304_sts_pfix() {
    // Tolerances 1.05× observed max (CLAUDE.md).
    assert_orbinit_match(
        sim_orbinit_docker::run_0304(),
        "orbinit_0304_orbinit.csv",
        "RUN_0304 (STS-114 pfix set04, altitudes + true anomaly)",
        1.59e-5,
        1.23e-8,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0205 / RUN_0305: set05 (apo/peri altitudes + time-periapsis) in pfix
// frame. Tolerances 1.05× observed.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0205_iss_pfix() {
    // Tolerances 1.05× observed max (CLAUDE.md).
    assert_orbinit_match(
        sim_orbinit_docker::run_0205(),
        "orbinit_0205_orbinit.csv",
        "RUN_0205 (ISS pfix set05, altitudes + time periapsis)",
        1.59e-5,
        1.23e-8,
    );
}

#[test]
fn tier3_orbinit_docker_run0305_sts_pfix() {
    // Tolerances 1.05× observed max (CLAUDE.md).
    assert_orbinit_match(
        sim_orbinit_docker::run_0305(),
        "orbinit_0305_orbinit.csv",
        "RUN_0305 (STS-114 pfix set05, altitudes + time periapsis)",
        1.59e-5,
        1.23e-8,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0206 / RUN_0306: set06 (arg-latitude + radial-vel) in pfix frame.
// Same converter as RUN_0006/0106, but the elements are interpreted in
// Earth.pfix and rotated to inertial at the SIM epoch. Tolerances 1.05× observed.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0206_iss_pfix() {
    // Tolerances 1.05× observed max. The residual reflects RNP-series
    // drift over the Earth-rotation arm.
    assert_orbinit_match(
        sim_orbinit_docker::run_0206(),
        "orbinit_0206_orbinit.csv",
        "RUN_0206 (ISS pfix set06, arg-latitude + radial-vel)",
        1.59e-5,
        1.23e-8,
    );
}

#[test]
fn tier3_orbinit_docker_run0306_sts_pfix() {
    // Tolerances 1.05× observed max. The residual reflects RNP-series
    // drift over the Earth-rotation arm.
    assert_orbinit_match(
        sim_orbinit_docker::run_0306(),
        "orbinit_0306_orbinit.csv",
        "RUN_0306 (STS-114 pfix set06, arg-latitude + radial-vel)",
        1.59e-5,
        1.23e-8,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0210 / RUN_0310: set10 (sma/ecc + true-anomaly) in pfix frame.
// Same converter as RUN_0010/0110, rotated to inertial at the SIM epoch.
// Tolerances 1.05× observed.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0210_iss_pfix() {
    // Tolerances 1.05× observed max. The residual reflects RNP-series
    // drift over the Earth-rotation arm.
    assert_orbinit_match(
        sim_orbinit_docker::run_0210(),
        "orbinit_0210_orbinit.csv",
        "RUN_0210 (ISS pfix set10, sma/ecc + true anomaly)",
        1.59e-5,
        1.23e-8,
    );
}

#[test]
fn tier3_orbinit_docker_run0310_sts_pfix() {
    // Tolerances 1.05× observed max. The residual reflects RNP-series
    // drift over the Earth-rotation arm.
    assert_orbinit_match(
        sim_orbinit_docker::run_0310(),
        "orbinit_0310_orbinit.csv",
        "RUN_0310 (STS-114 pfix set10, sma/ecc + true anomaly)",
        1.59e-5,
        1.23e-8,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0211 / RUN_0311: set11 (altitudes + true-anomaly) in pfix frame. JEOD's
// CaseEleven is the same option as set04, so this reuses
// `init_from_altitudes_true_anomaly`, rotated to inertial at the SIM epoch.
// Tolerances 1.05× observed.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0211_iss_pfix() {
    // Tolerances 1.05× observed max. The residual reflects RNP-series
    // drift over the Earth-rotation arm.
    assert_orbinit_match(
        sim_orbinit_docker::run_0211(),
        "orbinit_0211_orbinit.csv",
        "RUN_0211 (ISS pfix set11, altitudes + true anomaly)",
        1.59e-5,
        1.23e-8,
    );
}

#[test]
fn tier3_orbinit_docker_run0311_sts_pfix() {
    // Tolerances 1.05× observed max. The residual reflects RNP-series
    // drift over the Earth-rotation arm.
    assert_orbinit_match(
        sim_orbinit_docker::run_0311(),
        "orbinit_0311_orbinit.csv",
        "RUN_0311 (STS-114 pfix set11, altitudes + true anomaly)",
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

// ───────────────────────────────────────────────────────────────────────────
// RUN_0400: ISS direct Cartesian state in inertial frame
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0400_iss_trans_state() {
    // RUN_0400 uses DynBodyInitTransState (direct Cartesian input in
    // inertial). Recipe initialization is a pass-through to the body
    // state and the JEOD CSV logs the input to full precision, so the
    // residual is exactly zero; both tolerances are floored at 1e-13.
    assert_orbinit_match(
        sim_orbinit_docker::run_0400(),
        "orbinit_0400_orbinit.csv",
        "RUN_0400 (ISS inertial cart)",
        1.0e-13,
        1.0e-13,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0410 / RUN_0411: direct Cartesian state in planet-fixed (pfix) frame.
// Unlike the orbital-element pfix RUNs, the direct trans-state path composes
// the planet-fixed state into inertial through the full reference-frame
// relation (including the planet-rotation `ω × r` velocity term). The residual
// reflects RNP-series drift over the Earth-rotation arm plus the ~10-char CSV
// input precision. Tolerances 1.05× observed max (CLAUDE.md).
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0410_iss_pfix_trans_state() {
    // Tolerances 1.05× observed max (CLAUDE.md).
    assert_orbinit_match(
        sim_orbinit_docker::run_0410(),
        "orbinit_0410_orbinit.csv",
        "RUN_0410 (ISS pfix cart)",
        1.589e-5,
        1.225e-8,
    );
}

#[test]
fn tier3_orbinit_docker_run0411_sts_pfix_trans_state() {
    // Tolerances 1.05× observed max (CLAUDE.md).
    assert_orbinit_match(
        sim_orbinit_docker::run_0411(),
        "orbinit_0411_orbinit.csv",
        "RUN_0411 (STS-114 pfix cart)",
        1.589e-5,
        1.225e-8,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_2100 / RUN_1230: rotational initialization (first attitude RUNs in
// SIM_orbinit). Full-state cross-validation — position, velocity, attitude
// quaternion angle, and body-frame angular velocity — against the JEOD-logged
// t=0 row. RUN_2100 carries a non-identity Yaw-Pitch-Roll attitude, so the
// quaternion-angle assertion is a genuine Euler-sequence / deg→rad /
// scalar-first↔scalar-last convention check. Per-component tolerances are
// 1.05× the observed max on a clean run (CLAUDE.md); the attitude and rate
// floors are tight (sub-nanoradian / sub-pico-rad-per-second), reflecting
// that the initialization conversion reproduces JEOD's t=0 rotational state
// to floating-point precision against the ~10-character CSV input.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run2100_iss_inertial_att_rate() {
    // RUN_2100: ISS, DynBodyInitRotState, reference=Earth.inertial.
    // Attitude Yaw-Pitch-Roll [77.59, -30.60, -46.10] deg; body-frame
    // inertial rate from the ISS / LVLH rate decks.
    // Tolerances 1.05× observed max (CLAUDE.md).
    assert_orbinit_full_state(
        sim_orbinit_docker::run_2100(),
        "orbinit_2100_orbinit.csv",
        "RUN_2100 (ISS inertial att+rate)",
        1.0e-13,
        1.0e-13,
        1.0e-12,
        5.0e-19,
    );
}

#[test]
fn tier3_orbinit_docker_run1230_iss_lvlh_att_rate() {
    // RUN_1230: ISS, DynBodyInitLvlhRotState, planet=Earth.
    // Body aligned with LVLH (identity LVLH→body), LVLH-relative body
    // rate [0.002, 0.006, -0.003] deg/s.
    // Tolerances 1.05× observed max (CLAUDE.md).
    assert_orbinit_full_state(
        sim_orbinit_docker::run_1230(),
        "orbinit_1230_orbinit.csv",
        "RUN_1230 (ISS LVLH att+rate)",
        1.0e-13,
        1.0e-13,
        1.0e-12,
        5.0e-19,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// Double-vehicle relative-init RUNs (STS-114 chaser relative to ISS target).
// The chaser state is composed with the (default-initialized) ISS inertial
// state through `RefFrameState::incr_left`. Full-state cross-validation:
// position / velocity / attitude-angle / angular-rate against the JEOD-logged
// chaser composite-body t=0 row. Tolerances are 1.05× observed (CLAUDE.md);
// for the translation-only RUNs the chaser carries identity attitude / zero
// rate (deck `rotational_dynamics = False`), so the rotational tolerances are
// floors. RUN_3771's non-identity Pitch_Roll_Yaw [90,0,0] attitude makes the
// quaternion-angle assertion a genuine convention check.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0441_sts_body_relative() {
    // RUN_0441: STS-114 translation in the ISS composite-body frame.
    // Tolerances 1.05× observed; rotational tolerances are floors
    // (chaser attitude/rate are identity/zero in this translation-only RUN).
    assert_orbinit_full_state(
        sim_orbinit_docker::run_0441(),
        "orbinit_0441_orbinit.csv",
        "RUN_0441 (STS body-relative trans)",
        5.47e-10,
        9.55e-13,
        1.0e-12,
        1.0e-18,
    );
}

#[test]
fn tier3_orbinit_docker_run0571_sts_lvlh_relative() {
    // RUN_0571: STS-114 translation in the ISS LVLH frame (ω×r in velocity).
    // Tolerances 1.05× observed; rotational tolerances are floors.
    assert_orbinit_full_state(
        sim_orbinit_docker::run_0571(),
        "orbinit_0571_orbinit.csv",
        "RUN_0571 (STS LVLH-relative trans)",
        5.47e-10,
        1.07e-12,
        1.0e-12,
        1.0e-18,
    );
}

#[test]
fn tier3_orbinit_docker_run0681_sts_ned_relative() {
    // RUN_0681: STS-114 translation in the NED frame relative to ISS
    // (spherical lat/lon, composed pfix→inertial).
    // Tolerances 1.05× observed; rotational tolerances are floors.
    assert_orbinit_full_state(
        sim_orbinit_docker::run_0681(),
        "orbinit_0681_orbinit.csv",
        "RUN_0681 (STS NED-relative trans)",
        2.20e-9,
        3.06e-12,
        1.0e-12,
        1.0e-18,
    );
}

#[test]
fn tier3_orbinit_docker_run3771_sts_lvlh_full_state() {
    // RUN_3771: STS-114 full state in the ISS LVLH frame — Pitch_Roll_Yaw
    // [90,0,0] attitude + LVLH-relative body rate.
    // Tolerances 1.05× observed; quat-angle floored (observed exactly 0 —
    // the non-identity attitude composes bit-exactly), ang-vel 1.05× observed.
    assert_orbinit_full_state(
        sim_orbinit_docker::run_3771(),
        "orbinit_3771_orbinit.csv",
        "RUN_3771 (STS LVLH full state)",
        5.47e-10,
        1.07e-12,
        1.0e-12,
        4.59e-19,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_3822: single-vehicle full-state NED initialization at a geodetic ground
// point (PAD_39A, elliptical/ellipsoid lat/lon). The body is aligned with and
// at rest in the local NED frame, but the NED frame rotates with the Earth — so
// the body's *inertial* attitude is the non-trivial inertial→NED rotation at
// 28.6°N / −80.6°E (a genuine NED-axis / scalar-first↔scalar-last convention
// check, NOT a trivial identity pass) and its *inertial* angular velocity
// recovers ω_earth. Tolerances are 1.05× the observed max (CLAUDE.md).
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run3822_pad39a_ned_full_state() {
    // RUN_3822: PAD_39A, DynBodyInitNedState full state, elliptical/geodetic
    // NED at lat 28.6082° / lon −80.6040° / alt 3 m, body 10 m Down, aligned
    // with NED. The inertial attitude is the non-trivial NED→inertial rotation;
    // the inertial body rate recovers ω_earth.
    // Tolerances 1.05× observed max (CLAUDE.md).
    // The position residual is dominated by the geodetic-ellipsoid inversion
    // (our Borkowski solver vs JEOD `update_from_ellip`). The inertial
    // attitude composes bit-exactly and the inertial body rate recovers
    // ω_earth bit-exactly, so the quat-angle and ang-vel tolerances are
    // floors (the observed residuals leave no headroom).
    assert_orbinit_full_state(
        sim_orbinit_docker::run_3822(),
        "orbinit_3822_orbinit.csv",
        "RUN_3822 (PAD_39A NED full state)",
        1.58e-5,
        8.83e-10,
        1.0e-12,
        1.0e-19,
    );
}
