//! Tier 3: SIM_orbinit docker cross-validation (t=0 initialization)
//!
//! JEOD's SIM_orbinit is an initialization-only sim (`exec_set_terminate_time(0)`)
//! that writes the post-initialization state of `composite_body` into a CSV at t=0.
//! Each RUN exercises a different orbital-element set or coordinate frame.
//!
//! These tests start from JEOD source files (the `Modified_data/*.py` orbital
//! element specifications plus the epoch in `earth.py`) and reproduce JEOD's
//! initialization output through our own implementation of
//! `DynBodyInitOrbit::apply()`. The CSV's t=0 row is compared to our computed
//! inertial position/velocity.
//!
//! Per CLAUDE.md: initial conditions from JEOD source files are permitted;
//! JEOD output (CSV values) is never fed back into our computation.
//!
//! Scenarios:
//!   RUN_0001: ISS orbital elements in inertial frame (set01, time_periapsis)
//!   RUN_0101: STS-114 orbital elements in inertial frame (set01, time_periapsis)
//!   RUN_0201: ISS orbital elements in planet-fixed (pfix) frame (set01)
//!   RUN_0301: STS-114 orbital elements in planet-fixed (pfix) frame (set01)
//!   RUN_0401: STS-114 direct Cartesian state in inertial frame
//!
//! All scenarios share the same JEOD epoch: 2005-07-28 10:09:59 UT1.
//! The SIM disables polar motion (`earth.rnp.enable_polar = False`).
//! Gravity uses `earth_GGM05C` with `mu = 3.9860044150e14 m^3/s^2`.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_dynamics::init_from_mean_anomaly;
use jeod_sim::{
    calendar_to_tjt, compute_t_parent_this_from_tjt, default_leap_second_table, CalendarDate,
    TranslationalState,
};

/// SIM_orbinit epoch: 2005-07-28 10:09:59 UT1 (from `Modified_data/earth.py`).
const ORBINIT_YEAR: i32 = 2005;
const ORBINIT_MONTH: i32 = 7;
const ORBINIT_DAY: i32 = 28;
const ORBINIT_HOUR: i32 = 10;
const ORBINIT_MINUTE: i32 = 9;
const ORBINIT_SECOND: f64 = 59.0;

/// Compute the inertial-to-planet-fixed rotation matrix at the SIM_orbinit epoch.
///
/// SIM_orbinit uses `initializer = "UT1"` with `set_date_and_time(2005,7,28,10,9,59)`
/// and `earth.rnp.enable_polar = False`. Following JEOD `rnp.update_rnp(tt, gmst, ut1)`,
/// the rotation uses precession+nutation (via TT) and GAST (via GMST).
///
/// GMST is derived from UT1 directly. TT = TAI + 32.184 s, and we use the default
/// leap second table to compute TAI-UTC at this epoch (= 32 s for 2005), giving
/// TAI = UT1 + (TAI-UTC) when UT1-UTC ≈ 0.
fn compute_t_inertial_pfix_at_orbinit_epoch() -> DMat3 {
    // UT1 TJT for the calendar date.
    let ut1_cal = CalendarDate::new(
        ORBINIT_YEAR,
        ORBINIT_MONTH,
        ORBINIT_DAY,
        ORBINIT_HOUR,
        ORBINIT_MINUTE,
        ORBINIT_SECOND,
    );
    let ut1_tjt = calendar_to_tjt(&ut1_cal);

    // For SIM_orbinit, UT1-UTC≈0 (no override). So UTC_tjt ≈ UT1_tjt, and
    // TAI_tjt = UTC_tjt + TAI-UTC/86400.
    let leap = default_leap_second_table();
    let tai_utc_s = leap.tai_utc_at_utc_tjt(ut1_tjt);
    let tai_tjt = ut1_tjt + tai_utc_s / 86_400.0;

    // TT = TAI + 32.184 s
    let tt_tjt = tai_tjt + 32.184 / 86_400.0;

    // GMST seconds since J2000 noon UT1, computed from UT1 directly.
    // Matches SimulationTime::recompute_derived(): du = ut1_tjt - 11544.5
    let du = ut1_tjt - 11_544.5;
    let gmst_seconds = jeod_time::time_converter_ut1_gmst::ut1_to_gmst_seconds(du);

    // SIM_orbinit sets enable_polar = False → no polar motion
    compute_t_parent_this_from_tjt(gmst_seconds, tt_tjt)
}

/// Load an orbit initialization record from a JEOD SIM_orbinit CSV and assert
/// the expected initial state matches our reproduction.
///
/// For `time_periapsis`-parameterized orbits, mean anomaly is computed as
/// `M = t_peri * sqrt(mu/a^3)` (JEOD `dyn_body_init_orbit.cc:295`).
///
/// For pfix orbits, the orbital elements are interpreted in the planet-fixed
/// frame; we build the state there and rotate to inertial via `T_pfix_to_inertial`
/// (no ω×r term — JEOD `dyn_body_init_orbit.cc:331-332` rotates position and
/// velocity as pure 3-vectors).
fn assert_orbinit_match(
    jeod_root: &std::path::Path,
    vehicle: &str,
    init_name: &str,
    csv_filename: &str,
    label: &str,
    pos_tol: f64,
    vel_tol: f64,
) {
    let grav_data_dir = jeod_root.join("models/environment/gravity/data/src");
    let mu_earth =
        jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_data_dir.join("earth_GGM05C.cc"))
            .expect("load Earth mu from earth_GGM05C.cc");

    // Load JEOD orbital elements input (from input.py -> Modified_data/*.py).
    let init = jeod_test_data::orbital_init::load_orbital_init(jeod_root, vehicle, init_name);

    // JEOD input.py for set01 uses time_periapsis (M = n * t_peri).
    let t_peri = init
        .time_periapsis
        .unwrap_or_else(|| panic!("{label}: set01 expected time_periapsis in {init_name}.py"));
    let a = init.semi_major_axis;
    let n = (mu_earth / (a * a * a)).sqrt();
    let mean_anomaly = n * t_peri;

    // Build the orbit in the reference frame (inertial for set01 inertial,
    // pfix for set01 pfix). The orbital elements define a rotation from the
    // perifocal frame into whichever reference frame `orbit_frame_name`
    // points to; our `init_from_mean_anomaly` computes that perifocal→reference
    // rotation internally via (RAAN, argp, inclination).
    let state_ref = init_from_mean_anomaly(
        init.semi_major_axis,
        init.eccentricity,
        init.inclination,
        init.ascending_node,
        init.arg_periapsis,
        mean_anomaly,
        mu_earth,
    );

    // Transform reference-frame state to inertial.
    let state_inertial = match init.reference_frame.as_str() {
        "Earth.inertial" => state_ref,
        "Earth.pfix" => {
            // JEOD `dyn_body_init_orbit.cc` lines 323-333:
            //   rel_state: orbit_frame wrt planet.inertial,
            //   T_parent_this == T_inertial_to_pfix (rotation pfix from inertial),
            //   Vector3::transform_transpose(T_parent_this, v) == T_inertial_to_pfix^T * v
            //     == T_pfix_to_inertial * v
            //   Applied to both position and velocity (no ω×r term).
            let t_inertial_pfix = compute_t_inertial_pfix_at_orbinit_epoch();
            let t_pfix_inertial = t_inertial_pfix.transpose();
            TranslationalState {
                position: t_pfix_inertial * state_ref.position,
                velocity: t_pfix_inertial * state_ref.velocity,
            }
        }
        other => panic!("{label}: unsupported reference_frame '{other}'"),
    };

    // Load JEOD's logged state at t=0 from CSV.
    let csv_path = test_data_path(csv_filename);
    assert!(
        csv_path.exists(),
        "{label}: JEOD reference CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
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

    let pos_err = (state_inertial.position - jeod.position).length();
    let vel_err = (state_inertial.velocity - jeod.velocity).length();

    println!(
        "  {label}: our pos=[{:.3}, {:.3}, {:.3}] m",
        state_inertial.position.x, state_inertial.position.y, state_inertial.position.z
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
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0001: ISS orbital elements in inertial frame
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0001_iss_inertial() {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    // RUN_0001: ISS, SmaEccIncAscnodeArgperTimeperi, reference=Earth.inertial.
    // No frame rotation required — our output is already in inertial.
    // Observed: pos=3.76e-9 m, vel=3.43e-12 m/s (5% above → listed).
    assert_orbinit_match(
        &jeod_root,
        "ISS",
        "trans_Orbit_inertial_body_set01",
        "orbinit_0001_orbinit.csv",
        "RUN_0001 (ISS inertial set01)",
        3.95e-9,
        3.61e-12,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0101: STS-114 orbital elements in inertial frame
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0101_sts_inertial() {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    // Observed: pos=1.04e-9 m, vel=1.83e-12 m/s (5% above → listed).
    assert_orbinit_match(
        &jeod_root,
        "STS_114",
        "trans_Orbit_inertial_body_set01",
        "orbinit_0101_orbinit.csv",
        "RUN_0101 (STS-114 inertial set01)",
        1.10e-9,
        1.93e-12,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// RUN_0201: ISS orbital elements in planet-fixed frame
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn tier3_orbinit_docker_run0201_iss_pfix() {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    // RUN_0201: ISS pfix set01. Requires RNP rotation at the SIM epoch.
    // Observed: pos=1.51e-5 m, vel=1.17e-8 m/s (5% above → listed).
    // The residual reflects tiny differences between our RNP series and
    // JEOD's over the ~11 000 km Earth rotation arm from 2005-07-28.
    assert_orbinit_match(
        &jeod_root,
        "ISS",
        "trans_Orbit_pfix_body_set01",
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
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    // Observed: pos=1.51e-5 m, vel=1.17e-8 m/s (5% above → listed).
    assert_orbinit_match(
        &jeod_root,
        "STS_114",
        "trans_Orbit_pfix_body_set01",
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
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    // RUN_0401 uses DynBodyInitTransState (direct Cartesian input in inertial).
    // The JEOD input.py sets position and velocity directly; initialization
    // should be a pass-through to the body state.
    let trans = jeod_test_data::orbital_init::load_trans_state(
        &jeod_root,
        "STS_114",
        "trans_TransState_inertial_body",
    );
    let expected = TranslationalState {
        position: DVec3::from_array(trans.position),
        velocity: DVec3::from_array(trans.velocity),
    };

    let csv_path = test_data_path("orbinit_0401_orbinit.csv");
    assert!(
        csv_path.exists(),
        "RUN_0401: JEOD reference CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );
    let records = load_orbinit_csv(&csv_path);
    assert!(!records.is_empty(), "RUN_0401: no records in CSV");
    let jeod = &records[0];
    assert_eq!(jeod.time, 0.0, "RUN_0401: expected t=0 row in CSV");

    let pos_err = (expected.position - jeod.position).length();
    let vel_err = (expected.velocity - jeod.velocity).length();
    println!(
        "  RUN_0401: our pos=[{:.3}, {:.3}, {:.3}] m",
        expected.position.x, expected.position.y, expected.position.z
    );
    println!(
        "  RUN_0401: JEOD pos=[{:.3}, {:.3}, {:.3}] m",
        jeod.position.x, jeod.position.y, jeod.position.z
    );
    println!("  RUN_0401: pos_err={pos_err:.6e} m  vel_err={vel_err:.6e} m/s");

    // Direct Cartesian: expected to be bit-exact (both read from same input).
    // The CSV has only 10-char precision for RUN_0401 inputs; allow 1 µm / µm/s.
    assert!(
        pos_err < 1.0e-6,
        "RUN_0401: position error {pos_err:.6e} m exceeds 1 µm tolerance"
    );
    assert!(
        vel_err < 1.0e-9,
        "RUN_0401: velocity error {vel_err:.6e} m/s exceeds 1 nm/s tolerance"
    );
}
