//! Tier 3: Compare our RNP (precession, nutation, GAST rotation) matrices
//! element-by-element against JEOD's logged values at every timestep.
//!
//! This test pinpoints which component of the RNP pipeline causes trajectory
//! residuals by comparing each matrix (P, N, R, and the composed T_parent_this)
//! independently against JEOD SIM_dyncomp RUN_3A output.
//!
//! Epoch: 2007-11-20 00:00:00 UTC (same as Tier 3 trajectory tests).
//!
//! Requires: `test_data/dyncomp_run3a_Earth_RNP.csv` (generated via Docker).

use astrodyn_frames::nutation_j2000::nutation;
use astrodyn_frames::precession_j2000::precession_matrix;
use astrodyn_frames::rotation_j2000::{compute_t_parent_this, gast_rotation_matrix};
use astrodyn_time::epoch::{J2000_NOON_TJT, SECONDS_PER_DAY, TAI_TT_OFFSET};
use astrodyn_time::time_converter_ut1_gmst::ut1_to_gmst_days;
use glam::{DMat3, DVec3};
use std::path::Path;

// JEOD SIM_dyncomp epoch: 2007-11-20 00:00:00 UTC
const EPOCH_UTC_TJT: f64 = 14424.0;
const TAI_UTC_S: f64 = 32.0;
const TAI_TO_UT1_S: f64 = -32.469;

/// Compute time parameters at a given sim time.
///
/// Returns (tt_centuries, gmst_seconds).
fn time_params_at(sim_time_s: f64) -> (f64, f64) {
    let epoch_tai_tjt = EPOCH_UTC_TJT + TAI_UTC_S / SECONDS_PER_DAY;
    let tai_tjt = epoch_tai_tjt + sim_time_s / SECONDS_PER_DAY;
    let tt_tjt = tai_tjt + TAI_TT_OFFSET / SECONDS_PER_DAY;
    let tt_centuries = (tt_tjt - J2000_NOON_TJT) / 36525.0;

    let ut1_tjt = tai_tjt + TAI_TO_UT1_S / SECONDS_PER_DAY;
    let ut1_days = ut1_tjt - J2000_NOON_TJT;
    let gmst_days = ut1_to_gmst_days(ut1_days);
    let gmst_seconds = gmst_days * SECONDS_PER_DAY;

    (tt_centuries, gmst_seconds)
}

/// Parse a 3x3 matrix from CSV fields.
///
/// `cols` contains 9 column indices in row-major order:
/// [0][0], [0][1], [0][2], [1][0], [1][1], [1][2], [2][0], [2][1], [2][2]
///
/// JEOD stores matrices in row-major order. glam DMat3 is column-major,
/// so we transpose when constructing.
fn parse_matrix(fields: &[&str], cols: [usize; 9]) -> DMat3 {
    let p = |i: usize| -> f64 {
        let raw = fields[cols[i]].trim();
        raw.parse().unwrap_or_else(|e| {
            panic!(
                "parse_matrix: failed to parse field index {}  (col {}) value {:?}: {e}",
                i, cols[i], raw
            )
        })
    };
    // cols[0..9] are row-major: [0][0], [0][1], [0][2], [1][0], ...
    DMat3::from_cols(
        DVec3::new(p(0), p(3), p(6)), // column 0: T[0][0], T[1][0], T[2][0]
        DVec3::new(p(1), p(4), p(7)), // column 1: T[0][1], T[1][1], T[2][1]
        DVec3::new(p(2), p(5), p(8)), // column 2: T[0][2], T[1][2], T[2][2]
    )
}

/// Compute the maximum element-wise absolute difference between two 3x3 matrices.
fn max_matrix_error(a: &DMat3, b: &DMat3) -> f64 {
    let mut max_err = 0.0_f64;
    for col in 0..3 {
        for row in 0..3 {
            let err = (a.col(col)[row] - b.col(col)[row]).abs();
            max_err = max_err.max(err);
        }
    }
    max_err
}

/// JEOD RNP CSV record at one timestep. Mirrors the CSV column layout;
/// not every field is consumed by every assertion.
#[allow(dead_code)]
struct RnpRecord {
    time: f64,
    nutation_in_longitude: f64,
    nutation_in_obliquity: f64,
    equa_of_equi: f64,
    theta_gast: f64,
    t_parent_this: DMat3,
    r_matrix: DMat3,
    n_matrix: DMat3,
    p_matrix: DMat3,
}

fn load_rnp_csv(path: &Path) -> Vec<RnpRecord> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));

    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // skip header
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 41,
            "{}:{}: expected >= 41 CSV columns, got {} -- file may be truncated",
            path.display(),
            i + 1,
            f.len()
        );

        let pf = |idx: usize| -> f64 {
            let raw = f[idx].trim();
            raw.parse().unwrap_or_else(|e| {
                panic!(
                    "{}:{}: failed to parse column {} value {:?}: {e}",
                    path.display(),
                    i + 1,
                    idx,
                    raw
                )
            })
        };

        // T_parent_this: cols 5, 9, 13, 17, 21, 25, 29, 33, 37
        let t_cols = [5, 9, 13, 17, 21, 25, 29, 33, 37];
        // R (GAST rotation): cols 6, 10, 14, 18, 22, 26, 30, 34, 38
        let r_cols = [6, 10, 14, 18, 22, 26, 30, 34, 38];
        // N (nutation): cols 7, 11, 15, 19, 23, 27, 31, 35, 39
        let n_cols = [7, 11, 15, 19, 23, 27, 31, 35, 39];
        // P (precession): cols 8, 12, 16, 20, 24, 28, 32, 36, 40
        let p_cols = [8, 12, 16, 20, 24, 28, 32, 36, 40];

        records.push(RnpRecord {
            time: pf(0),
            nutation_in_longitude: pf(1),
            nutation_in_obliquity: pf(2),
            equa_of_equi: pf(3),
            theta_gast: pf(4),
            t_parent_this: parse_matrix(&f, t_cols),
            r_matrix: parse_matrix(&f, r_cols),
            n_matrix: parse_matrix(&f, n_cols),
            p_matrix: parse_matrix(&f, p_cols),
        });
    }
    records
}

#[test]
fn rnp_component_comparison() {
    let csv_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../astrodyn_verif_jeod/test_data/dyncomp_run3a_Earth_RNP.csv");
    assert!(
        csv_path.exists(),
        "JEOD RNP data not found at {}. \
         Generate with: docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let records = load_rnp_csv(&csv_path);
    assert!(
        records.len() >= 481,
        "Expected >= 481 RNP records (0 to 28800s at 60s), got {}",
        records.len()
    );

    eprintln!("Tier 3 RNP comparison: {} timesteps", records.len());
    eprintln!("  Comparing P (precession), N (nutation), R (GAST rotation), T (composed)");
    eprintln!();

    let mut max_p_err = 0.0_f64;
    let mut max_n_err = 0.0_f64;
    let mut max_r_err = 0.0_f64;
    let mut max_t_err = 0.0_f64;
    let mut max_equa_err = 0.0_f64;
    let mut max_theta_gast_err = 0.0_f64;

    let mut worst_p_time = 0.0_f64;
    let mut worst_n_time = 0.0_f64;
    let mut worst_r_time = 0.0_f64;
    let mut worst_t_time = 0.0_f64;

    for rec in &records {
        let (tt_centuries, gmst_seconds) = time_params_at(rec.time);

        // 1. Precession matrix
        let our_p = precession_matrix(tt_centuries);
        let p_err = max_matrix_error(&our_p, &rec.p_matrix);
        if p_err > max_p_err {
            max_p_err = p_err;
            worst_p_time = rec.time;
        }

        // 2. Nutation matrix + equation of equinoxes
        let nut = nutation(tt_centuries);
        let n_err = max_matrix_error(&nut.rotation, &rec.n_matrix);
        if n_err > max_n_err {
            max_n_err = n_err;
            worst_n_time = rec.time;
        }

        let equa_err = (nut.equa_of_equi - rec.equa_of_equi).abs();
        max_equa_err = max_equa_err.max(equa_err);

        // 3. GAST rotation matrix
        let our_r = gast_rotation_matrix(gmst_seconds, nut.equa_of_equi);
        let r_err = max_matrix_error(&our_r, &rec.r_matrix);
        if r_err > max_r_err {
            max_r_err = r_err;
            worst_r_time = rec.time;
        }

        // Compute theta_gast for comparison (GAST in radians)
        let theta_gast_ours =
            ((gmst_seconds + nut.equa_of_equi) / 240.0) * std::f64::consts::PI / 180.0;
        let temp = theta_gast_ours / (2.0 * std::f64::consts::PI);
        let theta_normalized = (temp - temp.floor()) * 2.0 * std::f64::consts::PI;
        let delta = (theta_normalized - rec.theta_gast + std::f64::consts::PI)
            .rem_euclid(2.0 * std::f64::consts::PI)
            - std::f64::consts::PI;
        let theta_gast_err = delta.abs();
        max_theta_gast_err = max_theta_gast_err.max(theta_gast_err);

        // 4. Composed T_parent_this
        let our_t = compute_t_parent_this(gmst_seconds, tt_centuries);
        let t_err = max_matrix_error(&our_t, &rec.t_parent_this);
        if t_err > max_t_err {
            max_t_err = t_err;
            worst_t_time = rec.time;
        }

        // Hourly diagnostics
        if rec.time == 0.0 || (rec.time % 3600.0).abs() < 0.1 {
            eprintln!(
                "  t={:6.0}s ({:.1}h): P_err={:.6e}  N_err={:.6e}  R_err={:.6e}  T_err={:.6e}  equa_err={:.6e}s  theta_err={:.6e}rad",
                rec.time,
                rec.time / 3600.0,
                p_err,
                n_err,
                r_err,
                t_err,
                equa_err,
                theta_gast_err,
            );
        }
    }

    eprintln!();
    eprintln!(
        "  === Max errors across all {} timesteps ===",
        records.len()
    );
    eprintln!(
        "  Precession (P):        {:.4e}  (worst at t={:.0}s)",
        max_p_err, worst_p_time
    );
    eprintln!(
        "  Nutation (N):          {:.4e}  (worst at t={:.0}s)",
        max_n_err, worst_n_time
    );
    eprintln!(
        "  GAST rotation (R):     {:.4e}  (worst at t={:.0}s)",
        max_r_err, worst_r_time
    );
    eprintln!(
        "  Composed T_parent_this:{:.4e}  (worst at t={:.0}s)",
        max_t_err, worst_t_time
    );
    eprintln!("  Equa of equinoxes:     {:.4e} s", max_equa_err);
    eprintln!("  Theta GAST:            {:.4e} rad", max_theta_gast_err);
    eprintln!();

    // Tolerances: these are element-wise matrix errors.
    // Precession and nutation should match to near machine precision since
    // they depend only on tt_centuries (which changes negligibly over 8 hours).
    // GAST rotation depends on GMST + equa_of_equi and is more sensitive.
    // T_parent_this compounds all three.
    //
    // Any error > 1e-10 in P or N indicates a formula difference.
    // Any error > 1e-10 in R (beyond P/N contribution) indicates a GMST or
    // theta_gast computation difference.

    assert!(
        max_p_err < 2.05e-18,
        "Precession matrix error {:.4e} exceeds 2.05e-18 (worst at t={:.0}s). \
         This indicates a formula difference in precession_matrix().",
        max_p_err,
        worst_p_time,
    );

    assert!(
        max_n_err < 1.63e-18,
        "Nutation matrix error {:.4e} exceeds 1.63e-18 (worst at t={:.0}s). \
         This indicates a formula difference in nutation().",
        max_n_err,
        worst_n_time,
    );

    assert!(
        max_r_err < 2.054e-11,
        "GAST rotation matrix error {:.4e} exceeds 2.054e-11 (worst at t={:.0}s). \
         This indicates a GMST or theta_gast computation difference.",
        max_r_err,
        worst_r_time,
    );

    assert!(
        max_t_err < 2.053e-11,
        "T_parent_this error {:.4e} exceeds 2.053e-11 (worst at t={:.0}s). \
         This indicates a composition error in compute_t_parent_this().",
        max_t_err,
        worst_t_time,
    );

    assert!(
        max_equa_err < 2.233e-14,
        "Equation of equinoxes error {:.4e}s exceeds 2.233e-14s.",
        max_equa_err,
    );

    assert!(
        max_theta_gast_err < 2.101e-11,
        "Theta GAST error {:.4e} rad exceeds 2.101e-11 rad.",
        max_theta_gast_err,
    );

    eprintln!("  All RNP components match JEOD within 1e-10 element-wise.");
}
