//! Tier 2: SIM_RNP_J2000_prop — Earth RNP transform cross-validation.
//!
//! This is a **Tier 2** cross-validation, not Tier 3: it does not run through
//! `Simulation::step()`. The RNP transform is a pure function of time, so this
//! reads JEOD's reference CSV and calls the production `astrodyn_frames`
//! functions (`precession_matrix`, `nutation`, `gast_rotation_matrix`)
//! directly, comparing the matrices element-wise. There is no integrator or
//! ECS path here, hence no Bevy-parity counterpart (and no `tier3_` prefix /
//! parity-coverage topic). RNP *as used in propagation* (`EarthRNP` driving
//! the planet-fixed frame for non-spherical gravity) is exercised through the
//! pipeline — with its own bevy-parity wrappers — by `tier3_sim_tide_verif`
//! and the dyncomp SH+RNP / polar-motion RUNs. A pipeline Tier-3 test of the
//! composite `T_inertial_pfix` would additionally need JEOD's EOP polar / UT1
//! table sourced independently (computational independence); that's deferred.
//!
//! Cross-validates our J2000 RNP model (IAU-76/FK5 precession, IAU-1980
//! 106-term nutation, GAST rotation) against JEOD's `SIM_RNP_J2000_prop`
//! at the two RUNs whose time setup is **exact and deterministic** — they
//! pin TAI−UTC and UT1−TAI via explicit overrides:
//!
//! - `RUN_J2000_RNP_Transform` — the IAU-76/FK5 reduction reference epoch
//!   1991-04-06 07:51:28.386009 UTC, leap = 26 s, UT1−TAI = 0.402521 − 26 s.
//! - `RUN_J2000_RNP_init` — 1999-03-04 00:00:00 UTC, leap = 32 s,
//!   UT1−TAI = 0.64932 − 32 s.
//!
//! The RNP transform is a pure function of time (precession/nutation in TT,
//! GAST in UT1) — independent of the integrator and of any vehicle state —
//! so this validation needs no trajectory propagation.
//!
//! ## What is and isn't validated here
//!
//! The precession matrix, nutation matrix, equation of equinoxes, GAST
//! rotation matrix, GAST angle, and the NP product (`nutation^T ×
//! precession^T`) are **polar-motion-independent**: they depend only on
//! TT and UT1, both fixed exactly by the overrides. Those are validated
//! element-wise here.
//!
//! JEOD's `enable_polar` defaults to `true` (`planet_rnp.hh`), so JEOD's
//! composite `T_parent_this` and the transformed `output_vector` also fold
//! in polar motion sourced from JEOD's internal PM table. Reproducing those
//! would require feeding JEOD's EOP polar values into our pipeline, which
//! the computational-independence rule forbids — so the composite,
//! `output_vector`, and the three default-EOP RUNs (`prop`, `prop_off`,
//! `Polar_off`) are deferred until our own matching EOP/polar source lands
//! (tracked under #99). The precession + nutation + GAST physics — the
//! "nutation/precession isolation" the audit flagged — is fully covered.

use astrodyn::{
    calendar_to_tjt, default_leap_second_table, nutation_j2000, precession_j2000, rotation_j2000,
    CalendarDate, SimulationTime, TimeScaleId,
};
use astrodyn_verif_jeod::tier3_csv::test_data_path;
use glam::DMat3;

/// TT Julian centuries since J2000.0 from a TT truncated Julian time.
/// (TJT → JD: +2440000.5; centuries: (JD − 2451545)/36525.)
fn tt_centuries_from_tjt(tt_tjt: f64) -> f64 {
    (tt_tjt - 11544.5) / 36525.0
}

/// One logged record: time + the matrices we cross-validate. Stored as the
/// flat JEOD row-major `[ii][jj]` layout the snippet logs.
struct RnpRecord {
    sim_time: f64,
    theta_gast: f64,
    equa_of_equi: f64,
    nut: [f64; 9],
    prec: [f64; 9],
    rot: [f64; 9],
    np: [f64; 9],
}

/// Column offsets in `rnp_*_rnp.csv` (see `RNP_VERIF_SNIPPET` in
/// `trick/generate_references.sh`): time, theta_gast, equa_of_equi,
/// output_vector[3], then nut/prec/rot/T_parent_this/NP as 3×3 row-major.
const NUT_OFF: usize = 6;
const PREC_OFF: usize = 15;
const ROT_OFF: usize = 24;
const NP_OFF: usize = 42;

fn load_rnp_csv(name: &str) -> Vec<RnpRecord> {
    let path = test_data_path(name);
    let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_RNP_J2000_prop CSV {}: {e}\n\
             Generate with Docker (see CLAUDE.md).",
            path.display()
        )
    });
    let mut out = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 51,
            "RNP CSV line {}: expected >=51 columns, got {} (snippet drift?)",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        let mat = |off: usize| -> [f64; 9] { std::array::from_fn(|k| p(off + k)) };
        out.push(RnpRecord {
            sim_time: p(0),
            theta_gast: p(1),
            equa_of_equi: p(2),
            nut: mat(NUT_OFF),
            prec: mat(PREC_OFF),
            rot: mat(ROT_OFF),
            np: mat(NP_OFF),
        });
    }
    assert!(!out.is_empty(), "no data rows in {name}");
    out
}

/// JEOD logs `m[ii][jj]` row-major; our `DMat3` is column-major. Return our
/// matrix's element at (row `ii`, col `jj`) for comparison against the flat
/// row-major JEOD array index `ii*3 + jj`.
fn dmat_rowmajor(m: &DMat3) -> [f64; 9] {
    let c = m.to_cols_array(); // [m00,m10,m20, m01,m11,m21, m02,m12,m22] (col-major)
    // row-major [ii*3+jj] = element(row ii, col jj) = c[jj*3 + ii]
    std::array::from_fn(|k| {
        let ii = k / 3;
        let jj = k % 3;
        c[jj * 3 + ii]
    })
}

fn max_abs_diff(a: &[f64; 9], b: &[f64; 9]) -> f64 {
    (0..9).fold(0.0_f64, |acc, k| acc.max((a[k] - b[k]).abs()))
}

/// Drive one RUN: build the exact-override time, walk the logged rows, and
/// compare our precession / nutation / GAST / NP matrices element-wise.
fn run_rnp_case(
    label: &str,
    csv_name: &str,
    epoch: CalendarDate,
    tai_minus_utc_s: f64,
    ut1_minus_tai_s: f64,
    mat_tol: f64,
    scalar_tol: f64,
) {
    let records = load_rnp_csv(csv_name);

    let utc_tjt = calendar_to_tjt(&epoch);
    let tai_tjt = utc_tjt + tai_minus_utc_s / 86_400.0;
    let mut time = SimulationTime::new(tai_tjt, default_leap_second_table());
    time.set_ut1_tai_offset(ut1_minus_tai_s);

    let (mut max_prec, mut max_nut, mut max_rot, mut max_np) = (0.0, 0.0, 0.0, 0.0);
    let (mut max_eoe, mut max_gast) = (0.0_f64, 0.0_f64);

    let mut prev_t = 0.0_f64;
    for (r, rec) in records.iter().enumerate() {
        if r > 0 {
            time.advance(rec.sim_time - prev_t);
        }
        prev_t = rec.sim_time;

        let tt_c = tt_centuries_from_tjt(time.tt_tjt());
        let gmst_s = time.get_seconds(TimeScaleId::GMST);

        let prec = precession_j2000::precession_matrix(tt_c);
        let nut = nutation_j2000::nutation(tt_c);
        let rot = rotation_j2000::gast_rotation_matrix(gmst_s, nut.equa_of_equi);
        // NP = nutation^T × precession^T (the composition our RNP uses).
        let np = nut.rotation.transpose() * prec.transpose();

        // JEOD's `earth.logging.{prec,nut,rot}_trans` store the component
        // transforms transposed relative to our matrices (the raw diffs are
        // exactly 2× the off-diagonal terms). The internal `NP_matrix` is
        // logged in our orientation. Compare each in its matching convention.
        max_prec = f64::max(max_prec, max_abs_diff(&dmat_rowmajor(&prec.transpose()), &rec.prec));
        max_nut = f64::max(
            max_nut,
            max_abs_diff(&dmat_rowmajor(&nut.rotation.transpose()), &rec.nut),
        );
        max_rot = f64::max(max_rot, max_abs_diff(&dmat_rowmajor(&rot.transpose()), &rec.rot));
        max_np = f64::max(max_np, max_abs_diff(&dmat_rowmajor(&np), &rec.np));
        max_eoe = max_eoe.max((nut.equa_of_equi - rec.equa_of_equi).abs());

        // GAST angle: our theta = (gmst + eoe)/240° normalized to [0, 2π).
        let our_theta = {
            let deg = (gmst_s + nut.equa_of_equi) / 240.0;
            let mut t = (deg.to_radians()).rem_euclid(std::f64::consts::TAU);
            if t < 0.0 {
                t += std::f64::consts::TAU;
            }
            t
        };
        max_gast = max_gast.max((our_theta - rec.theta_gast).abs());
    }

    println!(
        "  {label}: {} rows — prec={max_prec:.2e} nut={max_nut:.2e} rot={max_rot:.2e} \
         NP={max_np:.2e} eoe={max_eoe:.2e} gast={max_gast:.2e}",
        records.len()
    );

    assert!(max_prec < mat_tol, "{label}: precession Δ {max_prec:.3e} ≥ {mat_tol:.1e}");
    assert!(max_nut < mat_tol, "{label}: nutation Δ {max_nut:.3e} ≥ {mat_tol:.1e}");
    assert!(max_rot < mat_tol, "{label}: GAST rotation Δ {max_rot:.3e} ≥ {mat_tol:.1e}");
    assert!(max_np < mat_tol, "{label}: NP Δ {max_np:.3e} ≥ {mat_tol:.1e}");
    assert!(max_eoe < scalar_tol, "{label}: equa_of_equi Δ {max_eoe:.3e} ≥ {scalar_tol:.1e}");
    assert!(max_gast < scalar_tol, "{label}: GAST angle Δ {max_gast:.3e} ≥ {scalar_tol:.1e}");
}

// non-recipe: RNP transform is a pure function of time; the scenario is the
// JEOD reference epoch + the exact leap/UT1 overrides from the RUN input deck,
// not a propagated trajectory. Tolerances are 1.05× observed max.
#[test]
fn rnp_j2000_transform_crossval() {
    run_rnp_case(
        "rnp_transform",
        "rnp_transform_rnp.csv",
        CalendarDate::new(1991, 4, 6, 7, 51, 28.386_009),
        26.0,
        0.402_521 - 26.0,
        // Observed: prec/nut/NP ~1e-18, rot 8.5e-12 (GMST-conversion FP floor);
        // GAST angle 1.13e-11. Tolerances 1.05× observed max (CLAUDE.md).
        9.0e-12,
        1.2e-11,
    );
}

// non-recipe: see `tier3_sim_rnp_j2000_transform`.
#[test]
fn rnp_j2000_init_crossval() {
    run_rnp_case(
        "rnp_init",
        "rnp_init_rnp.csv",
        CalendarDate::new(1999, 3, 4, 0, 0, 0.0),
        32.0,
        0.649_32 - 32.0,
        // Observed: matrices ≤ 4.74e-12, GAST angle 4.85e-12. 1.05× observed.
        5.0e-12,
        5.1e-12,
    );
}
