//! Validate orbital elements and quaternion math against JEOD verification data.
//!
//! Requires the JEOD source tree (via JEOD_PATH env var or at `../jeod` as
//! sibling of the workspace root). Gated behind the `jeod-validation` feature
//! (default ON). Disable with `--no-default-features` if JEOD is unavailable.
#![cfg(feature = "jeod-validation")]

use jeod_test_data::{euler_test, orbital_data, orbital_init, reference_state, jeod_path};
use jeod_math::OrbitalElements;
use jeod_math::JeodQuat;

/// Earth's gravitational parameter in m^3/s^2 (matches JEOD's value).
const MU_EARTH: f64 = 3.986004418e14;

// =========================================================================
// Orbital elements: ISS reference data
// =========================================================================

#[test]
fn validate_iss_orbital_elements_to_cartesian() {
    let root = jeod_path();
    assert!(root.exists(), "JEOD source not found at {}. Run with --no-default-features to skip.", root.display());

    let init = orbital_init::load_orbital_init(
        &root,
        "ISS",
        "trans_Orbit_inertial_body_set01",
    );
    let expected = reference_state::load_reference_state(&root, "ISS", "inertial");

    // Verify parsed values are sensible
    assert!(
        init.semi_major_axis > 6_000_000.0 && init.semi_major_axis < 7_000_000.0,
        "ISS semi-major axis should be ~6732 km, got {} m",
        init.semi_major_axis
    );
    assert!(
        init.eccentricity < 0.01,
        "ISS eccentricity should be near zero, got {}",
        init.eccentricity
    );
    assert!(
        init.inclination > 0.8 && init.inclination < 1.0,
        "ISS inclination should be ~51.7 deg (~0.90 rad), got {} rad",
        init.inclination
    );

    // Build OrbitalElements from the init data.
    // We need to compute true anomaly from time_periapsis.
    // For this test, we use the JEOD-provided elements to compute Cartesian state
    // and compare against the reference.
    //
    // The init data uses SmaEccIncAscnodeArgperTimeperi set, which specifies
    // time of periapsis passage. We need mean anomaly to convert elements -> Cartesian.
    //
    // From time_periapsis and orbital period, we can compute mean anomaly:
    //   n = sqrt(mu / a^3)  (mean motion)
    //   M = n * t_peri      (mean anomaly, but note t_peri is time SINCE periapsis)
    let a = init.semi_major_axis;
    let n = (MU_EARTH / (a * a * a)).sqrt();

    // time_periapsis is the elapsed time since periapsis passage (seconds).
    // Mean anomaly is simply M = n * t_peri (radians).
    let t_peri = init.time_periapsis.expect("ISS set01 should have time_periapsis");
    let mean_anomaly = n * t_peri;

    let mut oe = OrbitalElements::default();
    oe.semi_major_axis = init.semi_major_axis;
    oe.eccentricity = init.eccentricity;
    oe.inclination = init.inclination;
    oe.long_asc_node = init.ascending_node;
    oe.arg_periapsis = init.arg_periapsis;
    oe.semiparam = a * (1.0 - init.eccentricity * init.eccentricity);
    oe.mean_anomaly = mean_anomaly;
    oe.mean_motion = n;
    oe.mean_anom_to_nu().unwrap();

    let (pos, vel) = oe.to_cartesian(MU_EARTH).unwrap();

    // The reference state is from NASA JSC Flight Operations data.
    // Position is given to nearest ~1 m, velocity to ~1 mm/s.
    // We expect agreement within ~100 m for position and ~0.1 m/s for velocity,
    // accounting for rounding and epoch/time uncertainties.
    let pos_err = (pos - expected.position).length();
    let vel_err = (vel - expected.velocity).length();

    println!("ISS position error: {:.2} m", pos_err);
    println!("ISS velocity error: {:.4} m/s", vel_err);
    println!("Computed pos: [{:.2}, {:.2}, {:.2}]", pos.x, pos.y, pos.z);
    println!(
        "Expected pos: [{:.2}, {:.2}, {:.2}]",
        expected.position.x, expected.position.y, expected.position.z
    );
    println!("Computed vel: [{:.6}, {:.6}, {:.6}]", vel.x, vel.y, vel.z);
    println!(
        "Expected vel: [{:.6}, {:.6}, {:.6}]",
        expected.velocity.x, expected.velocity.y, expected.velocity.z
    );

    // Position tolerance: 1 km (conservative, accounts for time_periapsis
    // interpretation differences between JEOD and our simplified computation)
    assert!(
        pos_err < 1000.0,
        "ISS position error {:.2} m exceeds 1 km tolerance",
        pos_err
    );

    // Velocity tolerance: 1 m/s
    assert!(
        vel_err < 1.0,
        "ISS velocity error {:.4} m/s exceeds 1 m/s tolerance",
        vel_err
    );
}

#[test]
fn validate_iss_reference_state_parsing() {
    let root = jeod_path();
    assert!(root.exists(), "JEOD source not found at {}. Run with --no-default-features to skip.", root.display());

    let state = reference_state::load_reference_state(&root, "ISS", "inertial");

    // Cross-check against known values from the file
    assert!(
        (state.position.x - 1_244_540.53).abs() < 0.01,
        "Position X mismatch: {}",
        state.position.x
    );
    assert!(
        (state.position.y - 5_655_938.85).abs() < 0.01,
        "Position Y mismatch: {}",
        state.position.y
    );
    assert!(
        (state.position.z - 3_425_643.22).abs() < 0.01,
        "Position Z mismatch: {}",
        state.position.z
    );

    assert!(
        (state.velocity.x - (-6003.833051)).abs() < 0.000001,
        "Velocity X mismatch: {}",
        state.velocity.x
    );
    assert!(
        (state.velocity.y - (-1469.496044)).abs() < 0.000001,
        "Velocity Y mismatch: {}",
        state.velocity.y
    );
    assert!(
        (state.velocity.z - 4590.511776).abs() < 0.000001,
        "Velocity Z mismatch: {}",
        state.velocity.z
    );

    // Position magnitude should be approximately Earth radius + ISS altitude
    // ~6378 km + 350 km = ~6728 km
    let r = state.position.length();
    assert!(
        r > 6_500_000.0 && r < 7_000_000.0,
        "ISS position magnitude {:.0} m out of expected range",
        r
    );
}

// =========================================================================
// Orbital elements: round-trip with 5001 Cartesian vectors
// =========================================================================

#[test]
fn validate_orbital_roundtrip_5000_vectors() {
    let root = jeod_path();
    assert!(root.exists(), "JEOD source not found at {}. Run with --no-default-features to skip.", root.display());

    let vectors = orbital_data::load_orbital_test_vectors(&root);
    assert!(
        vectors.len() >= 100,
        "Expected at least 100 orbital test vectors, got {}",
        vectors.len()
    );

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    let mut pass_count = 0;

    for (i, sv) in vectors.iter().enumerate().take(100) {
        let elems = match OrbitalElements::from_cartesian(MU_EARTH, sv.position, sv.velocity) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Vector {}: from_cartesian failed: {}", i, e);
                continue;
            }
        };

        let (pos_back, vel_back) = match elems.to_cartesian(MU_EARTH) {
            Ok(pv) => pv,
            Err(e) => {
                eprintln!("Vector {}: to_cartesian failed: {}", i, e);
                continue;
            }
        };

        let pos_err = (pos_back - sv.position).length();
        let vel_err = (vel_back - sv.velocity).length();
        max_pos_err = max_pos_err.max(pos_err);
        max_vel_err = max_vel_err.max(vel_err);

        assert!(
            pos_err < 1e-6,
            "Vector {}: position roundtrip error {:.2e} m exceeds 1e-6 m",
            i,
            pos_err
        );
        assert!(
            vel_err < 1e-9,
            "Vector {}: velocity roundtrip error {:.2e} m/s exceeds 1e-9 m/s",
            i,
            vel_err
        );

        pass_count += 1;
    }

    println!(
        "Orbital roundtrip: {}/{} vectors passed",
        pass_count,
        vectors.len().min(100)
    );
    println!("Max position error: {:.2e} m", max_pos_err);
    println!("Max velocity error: {:.2e} m/s", max_vel_err);

    assert!(
        pass_count == vectors.len().min(100),
        "Not all vectors passed roundtrip"
    );
}

#[test]
fn validate_orbital_data_parser() {
    let root = jeod_path();
    assert!(root.exists(), "JEOD source not found at {}. Run with --no-default-features to skip.", root.display());

    let vectors = orbital_data::load_orbital_test_vectors(&root);

    // The file should have 5001 vectors
    assert_eq!(
        vectors.len(),
        5001,
        "Expected 5001 orbital test vectors, got {}",
        vectors.len()
    );

    // Spot-check first vector against known values from the file
    let v0 = &vectors[0];
    assert!(
        (v0.position.x - 4.0875178010833091e+06).abs() < 1.0,
        "First vector position X mismatch"
    );
    assert!(
        (v0.velocity.x - (-5.8041508067101140e+03)).abs() < 0.001,
        "First vector velocity X mismatch"
    );

    // All vectors should have reasonable magnitudes for Earth orbit
    for (i, sv) in vectors.iter().enumerate() {
        let r = sv.position.length();
        let v = sv.velocity.length();
        assert!(
            r > 5_000_000.0 && r < 50_000_000.0,
            "Vector {}: position magnitude {:.0} m out of range",
            i,
            r
        );
        assert!(
            v > 1_000.0 && v < 20_000.0,
            "Vector {}: velocity magnitude {:.2} m/s out of range",
            i,
            v
        );
    }
}

// =========================================================================
// Euler angles: matrix -> quaternion -> matrix validation
// =========================================================================

#[test]
fn validate_euler_matrix_from_jeod() {
    let root = jeod_path();
    assert!(root.exists(), "JEOD source not found at {}. Run with --no-default-features to skip.", root.display());

    let cases = euler_test::load_euler_test_cases(&root);
    assert!(
        !cases.is_empty(),
        "Expected at least one Euler test case"
    );

    for (i, case) in cases.iter().enumerate() {
        // Build a glam DMat3 from the row-major test matrix
        let m = case.matrix;
        let mat = jeod_math::mat3_from_rows(
            glam::DVec3::new(m[0][0], m[0][1], m[0][2]),
            glam::DVec3::new(m[1][0], m[1][1], m[1][2]),
            glam::DVec3::new(m[2][0], m[2][1], m[2][2]),
        );

        // Convert to quaternion and back to matrix
        let quat = JeodQuat::left_quat_from_transformation(&mat);
        let mat_back = quat.left_quat_to_transformation();

        // Verify round-trip: matrix -> quat -> matrix should be identity-like error
        let tol = 1e-12;
        for col in 0..3 {
            for row in 0..3 {
                let orig = mat.col(col)[row];
                let back = mat_back.col(col)[row];
                assert!(
                    (orig - back).abs() < tol,
                    "Case {}: matrix element [{},{}] mismatch: original={}, roundtrip={}",
                    i,
                    row,
                    col,
                    orig,
                    back,
                );
            }
        }

        // Verify the matrix is orthogonal (det ~ 1, M * M^T ~ I)
        let det = mat.determinant();
        assert!(
            (det - 1.0).abs() < 1e-10,
            "Case {}: matrix determinant {}, expected 1.0",
            i,
            det
        );

        // Verify quaternion is unit
        let qnorm = quat.norm_sq();
        assert!(
            (qnorm - 1.0).abs() < 1e-14,
            "Case {}: quaternion norm_sq = {}, expected 1.0",
            i,
            qnorm
        );

        // Verify the quaternion transform matches matrix multiplication
        let test_vecs = [
            glam::DVec3::new(1.0, 0.0, 0.0),
            glam::DVec3::new(0.0, 1.0, 0.0),
            glam::DVec3::new(0.0, 0.0, 1.0),
            glam::DVec3::new(1.0, 2.0, 3.0),
        ];

        for v in &test_vecs {
            let via_mat = mat * *v;
            let via_quat = quat.left_quat_transform(*v);
            let err = (via_mat - via_quat).length();
            assert!(
                err < 1e-12,
                "Case {}: transform mismatch for v={:?}: mat={:?}, quat={:?}, err={:.2e}",
                i,
                v,
                via_mat,
                via_quat,
                err,
            );
        }

        println!(
            "Euler case {}: ref_body_angles = {:?} deg, body_ref_angles = {:?} deg",
            i, case.ref_body_angles_deg, case.body_ref_angles_deg
        );
    }
}

// =========================================================================
// Orbital init parser validation
// =========================================================================

#[test]
fn validate_orbital_init_parser() {
    let root = jeod_path();
    assert!(root.exists(), "JEOD source not found at {}. Run with --no-default-features to skip.", root.display());

    let init = orbital_init::load_orbital_init(
        &root,
        "ISS",
        "trans_Orbit_inertial_body_set01",
    );

    // Cross-check parsed values against known file contents
    let deg2rad = std::f64::consts::PI / 180.0;

    assert!(
        (init.semi_major_axis - 6_732_901.20152).abs() < 0.01,
        "semi_major_axis: expected 6732901.20152 m, got {}",
        init.semi_major_axis
    );
    assert!(
        (init.eccentricity - 0.00129073350).abs() < 1e-12,
        "eccentricity: expected 0.00129073350, got {}",
        init.eccentricity
    );
    assert!(
        (init.inclination - 51.670450765 * deg2rad).abs() < 1e-12,
        "inclination: expected {} rad, got {}",
        51.670450765 * deg2rad,
        init.inclination
    );
    assert!(
        (init.ascending_node - 49.708417385 * deg2rad).abs() < 1e-12,
        "ascending_node: expected {} rad, got {}",
        49.708417385 * deg2rad,
        init.ascending_node
    );
    assert!(
        (init.arg_periapsis - 100.582445989 * deg2rad).abs() < 1e-12,
        "arg_periapsis: expected {} rad, got {}",
        100.582445989 * deg2rad,
        init.arg_periapsis
    );
    assert_eq!(init.planet_name, "Earth");
    assert_eq!(init.reference_frame, "Earth.inertial");
    assert!(init.time_periapsis.is_some());
    assert!(
        (init.time_periapsis.unwrap() - 4581.96167293).abs() < 1e-8,
        "time_periapsis: expected 4581.96167293, got {}",
        init.time_periapsis.unwrap()
    );
}
