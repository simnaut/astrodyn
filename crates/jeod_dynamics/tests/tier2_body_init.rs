//! Tier 2: Validate body initialization from orbital elements against
//! ISS reference state from JEOD verification data.
//!
//! Requires JEOD_HOME (or JEOD_PATH) set to the JEOD source checkout.
//!
//! Tests exercise three JEOD element set parameterizations, all describing
//! the same ISS orbit at STS-114 MET 001:19:30:59.000:
//!
//! - set01: SmaEccIncAscnodeArgperTimeperi (time since periapsis)
//! - set02: SmaEccIncAscnodeArgperManom (mean anomaly, degrees)
//! - set10: SmaEccIncAscnodeArgperTanom (true anomaly, degrees)
//!
//! The expected state comes from `reference_inertial_trans_state.py`, which
//! contains the NASA JSC Flight Operations Directorate state vector.

use jeod_dynamics::{init_from_mean_anomaly, init_from_orbital_elements, TranslationalState};

/// Earth gravitational parameter (m^3/s^2), matching JEOD's value.
const EARTH_MU: f64 = 3.986_004_418e14;

/// Load the JEOD path, panicking with a clear message if not available.
fn jeod_root() -> std::path::PathBuf {
    let root = jeod_test_data::jeod_path();
    assert!(
        root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH \
         to the JEOD source checkout (e.g. ../jeod).",
        root.display()
    );
    root
}

/// Load the ISS inertial reference state (position + velocity in ECI).
fn load_iss_reference(root: &std::path::Path) -> TranslationalState {
    let ref_state = jeod_test_data::reference_state::load_reference_state(root, "ISS", "inertial");
    TranslationalState {
        position: ref_state.position,
        velocity: ref_state.velocity,
    }
}

/// Helper: print detailed comparison diagnostics.
fn print_comparison(
    label: &str,
    computed: &TranslationalState,
    expected: &TranslationalState,
    pos_err: f64,
    vel_err: f64,
) {
    println!("=== {} ===", label);
    println!(
        "  Computed pos: [{:>16.6}, {:>16.6}, {:>16.6}] m",
        computed.position.x, computed.position.y, computed.position.z
    );
    println!(
        "  Expected pos: [{:>16.6}, {:>16.6}, {:>16.6}] m",
        expected.position.x, expected.position.y, expected.position.z
    );
    println!(
        "  Pos error:    [{:>16.6}, {:>16.6}, {:>16.6}] m  (|e| = {:.6} m)",
        computed.position.x - expected.position.x,
        computed.position.y - expected.position.y,
        computed.position.z - expected.position.z,
        pos_err,
    );
    println!(
        "  Computed vel: [{:>16.9}, {:>16.9}, {:>16.9}] m/s",
        computed.velocity.x, computed.velocity.y, computed.velocity.z
    );
    println!(
        "  Expected vel: [{:>16.9}, {:>16.9}, {:>16.9}] m/s",
        expected.velocity.x, expected.velocity.y, expected.velocity.z
    );
    println!(
        "  Vel error:    [{:>16.9}, {:>16.9}, {:>16.9}] m/s  (|e| = {:.9} m/s)",
        computed.velocity.x - expected.velocity.x,
        computed.velocity.y - expected.velocity.y,
        computed.velocity.z - expected.velocity.z,
        vel_err,
    );
}

// =========================================================================
// Test 1: set01 — SmaEccIncAscnodeArgperTimeperi
//
// Computes mean anomaly from time_periapsis: M = n * t_peri
// where n = sqrt(mu / a^3). JEOD's time_periapsis means time SINCE
// periapsis passage (not time TO periapsis).
// =========================================================================

#[test]
fn iss_set01_time_periapsis() {
    let root = jeod_root();
    let init = jeod_test_data::orbital_init::load_orbital_init(
        &root,
        "ISS",
        "trans_Orbit_inertial_body_set01",
    );
    let expected = load_iss_reference(&root);

    // set01 provides time_periapsis — compute mean anomaly from it.
    // M = n * t_peri, where n = sqrt(mu / a^3).
    // JEOD convention: time_periapsis is time SINCE periapsis.
    let a = init.semi_major_axis;
    let n = (EARTH_MU / (a * a * a)).sqrt();
    let t_peri = init
        .time_periapsis
        .expect("ISS set01 must have time_periapsis");
    let mean_anomaly = n * t_peri;

    let computed = init_from_mean_anomaly(
        init.semi_major_axis,
        init.eccentricity,
        init.inclination,
        init.ascending_node,
        init.arg_periapsis,
        mean_anomaly,
        EARTH_MU,
    );

    let pos_err = (computed.position - expected.position).length();
    let vel_err = (computed.velocity - expected.velocity).length();

    print_comparison(
        "set01 (time_periapsis)",
        &computed,
        &expected,
        pos_err,
        vel_err,
    );

    assert!(
        pos_err < 1.0,
        "ISS set01 position error {:.6} m exceeds 1 m tolerance",
        pos_err
    );
    assert!(
        vel_err < 0.001,
        "ISS set01 velocity error {:.9} m/s exceeds 0.001 m/s tolerance",
        vel_err
    );
}

// =========================================================================
// Test 2: set02 — SmaEccIncAscnodeArgperManom
//
// Directly provides mean anomaly in degrees (parsed to radians).
// =========================================================================

#[test]
fn iss_set02_mean_anomaly() {
    let root = jeod_root();
    let init = jeod_test_data::orbital_init::load_orbital_init(
        &root,
        "ISS",
        "trans_Orbit_inertial_body_set02",
    );
    let expected = load_iss_reference(&root);

    let mean_anomaly = init.mean_anomaly.expect("ISS set02 must have mean_anomaly");

    let computed = init_from_mean_anomaly(
        init.semi_major_axis,
        init.eccentricity,
        init.inclination,
        init.ascending_node,
        init.arg_periapsis,
        mean_anomaly,
        EARTH_MU,
    );

    let pos_err = (computed.position - expected.position).length();
    let vel_err = (computed.velocity - expected.velocity).length();

    print_comparison(
        "set02 (mean_anomaly)",
        &computed,
        &expected,
        pos_err,
        vel_err,
    );

    assert!(
        pos_err < 1.0,
        "ISS set02 position error {:.6} m exceeds 1 m tolerance",
        pos_err
    );
    assert!(
        vel_err < 0.001,
        "ISS set02 velocity error {:.9} m/s exceeds 0.001 m/s tolerance",
        vel_err
    );
}

// =========================================================================
// Test 3: set10 — SmaEccIncTanomAscnodeArgper
//
// Directly provides true anomaly in degrees (parsed to radians).
// =========================================================================

#[test]
fn iss_set10_true_anomaly() {
    let root = jeod_root();
    let init = jeod_test_data::orbital_init::load_orbital_init(
        &root,
        "ISS",
        "trans_Orbit_inertial_body_set10",
    );
    let expected = load_iss_reference(&root);

    let true_anomaly = init.true_anomaly.expect("ISS set10 must have true_anomaly");

    let computed = init_from_orbital_elements(
        init.semi_major_axis,
        init.eccentricity,
        init.inclination,
        init.ascending_node,
        init.arg_periapsis,
        true_anomaly,
        EARTH_MU,
    );

    let pos_err = (computed.position - expected.position).length();
    let vel_err = (computed.velocity - expected.velocity).length();

    print_comparison(
        "set10 (true_anomaly)",
        &computed,
        &expected,
        pos_err,
        vel_err,
    );

    assert!(
        pos_err < 1.0,
        "ISS set10 position error {:.6} m exceeds 1 m tolerance",
        pos_err
    );
    assert!(
        vel_err < 0.001,
        "ISS set10 velocity error {:.9} m/s exceeds 0.001 m/s tolerance",
        vel_err
    );
}

// =========================================================================
// Test 4: Cross-consistency — all three element sets produce the same state
//
// Since set01, set02, and set10 all describe the same ISS orbit at the same
// epoch, the Cartesian states they produce should agree to high precision.
// =========================================================================

#[test]
fn iss_element_sets_cross_consistent() {
    let root = jeod_root();

    // set01: time_periapsis -> mean anomaly -> Cartesian
    let init01 = jeod_test_data::orbital_init::load_orbital_init(
        &root,
        "ISS",
        "trans_Orbit_inertial_body_set01",
    );
    let a = init01.semi_major_axis;
    let n = (EARTH_MU / (a * a * a)).sqrt();
    let t_peri = init01.time_periapsis.unwrap();
    let mean_anomaly_01 = n * t_peri;
    let state01 = init_from_mean_anomaly(
        init01.semi_major_axis,
        init01.eccentricity,
        init01.inclination,
        init01.ascending_node,
        init01.arg_periapsis,
        mean_anomaly_01,
        EARTH_MU,
    );

    // set02: mean anomaly (directly) -> Cartesian
    let init02 = jeod_test_data::orbital_init::load_orbital_init(
        &root,
        "ISS",
        "trans_Orbit_inertial_body_set02",
    );
    let state02 = init_from_mean_anomaly(
        init02.semi_major_axis,
        init02.eccentricity,
        init02.inclination,
        init02.ascending_node,
        init02.arg_periapsis,
        init02.mean_anomaly.unwrap(),
        EARTH_MU,
    );

    // set10: true anomaly (directly) -> Cartesian
    let init10 = jeod_test_data::orbital_init::load_orbital_init(
        &root,
        "ISS",
        "trans_Orbit_inertial_body_set10",
    );
    let state10 = init_from_orbital_elements(
        init10.semi_major_axis,
        init10.eccentricity,
        init10.inclination,
        init10.ascending_node,
        init10.arg_periapsis,
        init10.true_anomaly.unwrap(),
        EARTH_MU,
    );

    // Compare set01 vs set02
    let pos_err_01_02 = (state01.position - state02.position).length();
    let vel_err_01_02 = (state01.velocity - state02.velocity).length();
    println!(
        "set01 vs set02: pos_err = {:.6} m, vel_err = {:.9} m/s",
        pos_err_01_02, vel_err_01_02,
    );

    // Compare set01 vs set10
    let pos_err_01_10 = (state01.position - state10.position).length();
    let vel_err_01_10 = (state01.velocity - state10.velocity).length();
    println!(
        "set01 vs set10: pos_err = {:.6} m, vel_err = {:.9} m/s",
        pos_err_01_10, vel_err_01_10,
    );

    // Compare set02 vs set10
    let pos_err_02_10 = (state02.position - state10.position).length();
    let vel_err_02_10 = (state02.velocity - state10.velocity).length();
    println!(
        "set02 vs set10: pos_err = {:.6} m, vel_err = {:.9} m/s",
        pos_err_02_10, vel_err_02_10,
    );

    // All three should agree within sub-meter precision.
    // The time_periapsis -> mean_anomaly conversion may introduce a small
    // rounding difference, so we use 1 m / 0.001 m/s tolerance.
    assert!(
        pos_err_01_02 < 1.0,
        "set01 vs set02 position disagreement {:.6} m exceeds 1 m",
        pos_err_01_02
    );
    assert!(
        vel_err_01_02 < 0.001,
        "set01 vs set02 velocity disagreement {:.9} m/s exceeds 0.001 m/s",
        vel_err_01_02
    );
    assert!(
        pos_err_01_10 < 1.0,
        "set01 vs set10 position disagreement {:.6} m exceeds 1 m",
        pos_err_01_10
    );
    assert!(
        vel_err_01_10 < 0.001,
        "set01 vs set10 velocity disagreement {:.9} m/s exceeds 0.001 m/s",
        vel_err_01_10
    );
    assert!(
        pos_err_02_10 < 1.0,
        "set02 vs set10 position disagreement {:.6} m exceeds 1 m",
        pos_err_02_10
    );
    assert!(
        vel_err_02_10 < 0.001,
        "set02 vs set10 velocity disagreement {:.9} m/s exceeds 0.001 m/s",
        vel_err_02_10
    );
}
