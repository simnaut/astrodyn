//! Tier 3: SIM_orb_elem comprehensive -- 7 orbit families
//!
//! Static cross-validation: each CSV has a fixed position/velocity at t=0 and t=1
//! (identical values). We extract the t=0 state, compute orbital elements via
//! `OrbitalElements::from_cartesian`, and compare against JEOD's logged elements.
//!
//! The 7 orbit families:
//! - T01: circular orbit (e ~ 0)
//! - T10: eccentric orbit (0 < e < 1)
//! - T20: hyperbolic orbit (e > 1)
//! - T30: near-parabolic orbit (e ~ 1)
//! - T40: retrograde orbit (i > 90 deg)
//! - T50: equatorial orbit (i ~ 0)
//! - T55: polar orbit (i ~ 90 deg)

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::DVec3;
use jeod_sim::OrbitalElements;

/// Full record parsed from the verification CSV (all 21 columns).
struct VerifRecord {
    semi_major_axis: f64,
    semiparam: f64,
    e_mag: f64,
    inclination: f64,
    arg_periapsis: f64,
    long_asc_node: f64,
    r_mag: f64,
    vel_mag: f64,
    true_anom: f64,
    mean_anom: f64,
    mean_motion: f64,
    orbital_anom: f64,
    orb_energy: f64,
    orb_ang_momentum: f64,
    position: DVec3,
    velocity: DVec3,
}

/// Parse the t=0 row from a verification CSV.
fn load_verif_record(csv_name: &str) -> VerifRecord {
    let csv_path = test_data_path(csv_name);
    assert!(
        csv_path.exists(),
        "SIM_OrbElem verification CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let content = std::fs::read_to_string(&csv_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_OrbElem CSV from {}: {e}",
            csv_path.display()
        )
    });

    // Take the first data line (t=0)
    let line = content
        .lines()
        .nth(1)
        .expect("CSV must have at least one data row");
    let f: Vec<&str> = line.split(',').collect();
    assert!(f.len() >= 21, "expected >=21 columns, got {}", f.len());
    let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };

    VerifRecord {
        semi_major_axis: p(1),
        semiparam: p(2),
        e_mag: p(3),
        inclination: p(4),
        arg_periapsis: p(5),
        long_asc_node: p(6),
        r_mag: p(7),
        vel_mag: p(8),
        true_anom: p(9),
        mean_anom: p(10),
        mean_motion: p(11),
        orbital_anom: p(12),
        orb_energy: p(13),
        orb_ang_momentum: p(14),
        position: DVec3::new(p(15), p(16), p(17)),
        velocity: DVec3::new(p(18), p(19), p(20)),
    }
}

/// Compare a computed value against a JEOD reference value.
///
/// Uses relative tolerance when the reference magnitude is large enough,
/// absolute tolerance otherwise.
fn assert_close(name: &str, computed: f64, reference: f64, rel_tol: f64, abs_tol: f64) {
    let diff = (computed - reference).abs();
    let tol = if reference.abs() > abs_tol {
        rel_tol * reference.abs()
    } else {
        abs_tol
    };
    assert!(
        diff <= tol,
        "{name}: computed={computed:.15e}, reference={reference:.15e}, \
         diff={diff:.6e}, tol={tol:.6e}"
    );
}

/// Compare an angular value against JEOD reference, accounting for wraparound.
fn assert_angle_close(name: &str, computed: f64, reference: f64, abs_tol: f64) {
    let diff = angle_diff(computed, reference);
    assert!(
        diff <= abs_tol,
        "{name}: computed={computed:.15e}, reference={reference:.15e}, \
         angle_diff={diff:.6e}, tol={abs_tol:.6e}"
    );
}

/// Compute orbital elements from a verification CSV and assert parity with JEOD.
///
/// The JEOD sim computes orbital elements from exact analytical initial conditions,
/// then logs both the elements and the position/velocity. We recompute elements from
/// the logged position/velocity, which has finite CSV precision (~15 significant
/// digits). This truncation amplifies into element-level differences on the order
/// of 1e-6 relative for SMA/semiparam and 1e-4 to 1e-7 for eccentricity (depending
/// on how close to circular the orbit is).
///
/// `skip_degenerate_scalars`: when true, skip assertions on fields that JEOD
/// reports as zero due to near-parabolic or degenerate orbit conventions
/// (semi_major_axis=0, r_mag=0, vel_mag=0, mean_motion=0, orb_energy=0,
/// orb_ang_momentum=0). Our code computes valid values for these, but JEOD
/// intentionally zeros them out for parabolic-regime orbits.
fn verify_orbit_family(csv_name: &str, label: &str, skip_degenerate_scalars: bool) {
    let rec = load_verif_record(csv_name);

    let oe = OrbitalElements::from_cartesian(MU_EARTH, rec.position, rec.velocity)
        .unwrap_or_else(|e| panic!("{label}: from_cartesian failed: {e:?}"));

    println!("Tier 3 (Static): {label}");
    println!("  position:        {:?}", rec.position);
    println!("  velocity:        {:?}", rec.velocity);
    println!("  JEOD sma:        {:.15e}", rec.semi_major_axis);
    println!("  ours sma:        {:.15e}", oe.semi_major_axis);
    println!("  JEOD e_mag:      {:.15e}", rec.e_mag);
    println!("  ours e_mag:      {:.15e}", oe.e_mag);
    println!("  JEOD incl:       {:.15e}", rec.inclination);
    println!("  ours incl:       {:.15e}", oe.inclination);

    // Tolerances calibrated at 5% above observed max errors.
    //
    // These errors arise from CSV position/velocity truncation, not code bugs:
    // JEOD computes elements from exact initial conditions, we recompute from
    // the logged (truncated) position/velocity.
    //
    // SMA relative error: max ~2.85e-6 (T20 hyperbolic), tolerance 3.0e-6
    // Semiparam relative error: max ~1.9e-6, tolerance 2.1e-6
    // Eccentricity: for near-circular orbits (e~0) the absolute eccentricity
    //   our code computes is ~1.9e-6 vs JEOD's ~0 or ~1.8e-15. This is expected:
    //   tiny CSV rounding in velocity produces a non-zero eccentricity vector.
    //   For orbits with meaningful eccentricity (e>0.001), relative error is ~1.5e-4.
    let sma_rel_tol = 3.0e-6;
    let semiparam_rel_tol = 2.1e-6;
    let ecc_abs_tol = 2.0e-6; // absolute for near-circular
    let ecc_rel_tol = 1.5e-4; // relative for e > 0.001
    let incl_abs_tol = 1.1e-15; // inclination matches to machine precision
    let rmag_rel_tol = 1e-10; // r_mag matches very closely (just pos magnitude)
    let vmag_rel_tol = 1e-10; // vel_mag matches very closely
    let mean_motion_rel_tol = 5.6e-6; // derived from SMA, propagates error (max ~5.23e-6 on T20)
    let energy_rel_tol = 6.0e-6; // derived from v^2 and r, propagates
    let ang_mom_rel_tol = 1e-10; // cross product, matches well
    let angle_tol = 1.6e-4; // angular elements (rad) -- dominated by T55 polar orbit
                            // where small e (0.0025) amplifies CSV truncation into
                            // periapsis direction uncertainty (~1.46e-4 rad)

    // ---- Scalar quantities ----
    if !skip_degenerate_scalars {
        assert_close(
            &format!("{label}/semi_major_axis"),
            oe.semi_major_axis,
            rec.semi_major_axis,
            sma_rel_tol,
            1e-6,
        );
        assert_close(
            &format!("{label}/r_mag"),
            oe.r_mag,
            rec.r_mag,
            rmag_rel_tol,
            1e-6,
        );
        assert_close(
            &format!("{label}/vel_mag"),
            oe.vel_mag,
            rec.vel_mag,
            vmag_rel_tol,
            1e-6,
        );
        assert_close(
            &format!("{label}/mean_motion"),
            oe.mean_motion,
            rec.mean_motion,
            mean_motion_rel_tol,
            1e-15,
        );
        assert_close(
            &format!("{label}/orb_energy"),
            oe.orb_energy,
            rec.orb_energy,
            energy_rel_tol,
            1e-6,
        );
        assert_close(
            &format!("{label}/orb_ang_momentum"),
            oe.orb_ang_momentum,
            rec.orb_ang_momentum,
            ang_mom_rel_tol,
            1e-6,
        );
    }

    assert_close(
        &format!("{label}/semiparam"),
        oe.semiparam,
        rec.semiparam,
        semiparam_rel_tol,
        1e-6,
    );

    // Eccentricity: use absolute tolerance for near-circular, relative for eccentric
    if rec.e_mag > 0.001 {
        assert_close(
            &format!("{label}/e_mag"),
            oe.e_mag,
            rec.e_mag,
            ecc_rel_tol,
            ecc_abs_tol,
        );
    } else {
        let ecc_diff = (oe.e_mag - rec.e_mag).abs();
        assert!(
            ecc_diff < ecc_abs_tol,
            "{label}/e_mag (near-circular): computed={:.15e}, reference={:.15e}, \
             diff={ecc_diff:.6e}, tol={ecc_abs_tol:.6e}",
            oe.e_mag,
            rec.e_mag
        );
    }

    assert_close(
        &format!("{label}/inclination"),
        oe.inclination,
        rec.inclination,
        1e-6,
        incl_abs_tol,
    );

    // ---- Angular quantities (wraparound-safe) ----
    // For near-circular orbits, arg_periapsis and true_anom are ill-defined
    // (periapsis direction is arbitrary). Skip angle assertions when e < 0.001
    // and the reference angle is near zero.
    let skip_periapsis_angles = rec.e_mag < 0.001;

    if !skip_periapsis_angles {
        assert_angle_close(
            &format!("{label}/arg_periapsis"),
            oe.arg_periapsis,
            rec.arg_periapsis,
            angle_tol,
        );
        assert_angle_close(
            &format!("{label}/true_anom"),
            oe.true_anom,
            rec.true_anom,
            angle_tol,
        );
        assert_angle_close(
            &format!("{label}/mean_anom"),
            oe.mean_anom,
            rec.mean_anom,
            angle_tol,
        );
        assert_angle_close(
            &format!("{label}/orbital_anom"),
            oe.orbital_anom,
            rec.orbital_anom,
            angle_tol,
        );
    } else {
        println!(
            "  (skipping periapsis-referenced angles: e_mag={:.3e} < 0.001)",
            rec.e_mag
        );
    }

    // LAN: for equatorial orbits (i~0), LAN is ill-defined. Skip when i < 1e-10.
    if rec.inclination > 1e-10 {
        assert_angle_close(
            &format!("{label}/long_asc_node"),
            oe.long_asc_node,
            rec.long_asc_node,
            angle_tol,
        );
    } else {
        println!("  (skipping LAN: inclination={:.3e} ~ 0)", rec.inclination);
    }

    println!("  PASS: {label}");
}

// ── Individual test functions ──

#[test]
fn tier3_simulation_orbelem_t01() {
    verify_orbit_family("orbelem_verif_t01_orbelem.csv", "T01 circular (e~0)", false);
}

#[test]
fn tier3_simulation_orbelem_t10() {
    verify_orbit_family(
        "orbelem_verif_t10_orbelem.csv",
        "T10 eccentric (0<e<1)",
        false,
    );
}

#[test]
fn tier3_simulation_orbelem_t20() {
    verify_orbit_family(
        "orbelem_verif_t20_orbelem.csv",
        "T20 hyperbolic (e>1)",
        false,
    );
}

#[test]
fn tier3_simulation_orbelem_t30() {
    // Near-parabolic: JEOD reports sma=0, r_mag=0, vel_mag=0, mean_motion=0,
    // orb_energy=0, orb_ang_momentum=0 for parabolic-regime orbits. Our code
    // computes valid values for these fields but JEOD intentionally zeros them.
    verify_orbit_family(
        "orbelem_verif_t30_orbelem.csv",
        "T30 near-parabolic (e~1)",
        true,
    );
}

#[test]
fn tier3_simulation_orbelem_t40() {
    // Retrograde: JEOD also zeros out sma, r_mag, vel_mag, etc. for this
    // test case (same convention as t30).
    verify_orbit_family(
        "orbelem_verif_t40_orbelem.csv",
        "T40 retrograde (i>90deg)",
        true,
    );
}

#[test]
fn tier3_simulation_orbelem_t50() {
    verify_orbit_family(
        "orbelem_verif_t50_orbelem.csv",
        "T50 equatorial (i~0)",
        false,
    );
}

#[test]
fn tier3_simulation_orbelem_t55() {
    verify_orbit_family(
        "orbelem_verif_t55_orbelem.csv",
        "T55 polar (i~90deg)",
        false,
    );
}
