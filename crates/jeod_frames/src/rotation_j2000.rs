//! J2000 GAST rotation and full RNP composition.
//!
//! Faithful port of JEOD's `rotation_j2000.cc` (IAU-76/FK5).
//! Computes the inertial-to-planet-fixed rotation matrix from time scales.
//!
//! Reference: Mulcahy & Bond, "The RNP Routine for the Standard Epoch J2000",
//! NASA JSC-24574, September 1990.

// Constants ported verbatim from JEOD — suppress excessive precision warnings.
#![allow(clippy::excessive_precision)]

use crate::nutation_j2000::nutation;
use crate::precession_j2000::precession_matrix;
use glam::{DMat3, DVec3};
use jeod_math::matrix3x3_product_transpose_transpose;
use std::f64::consts::PI;

const DEG_TO_RAD: f64 = PI / 180.0;

/// Compute the GAST (Greenwich Apparent Sidereal Time) rotation matrix.
///
/// Port of JEOD `rotation_j2000.cc` (full RNP branch).
///
/// # Arguments
/// * `gmst_seconds` — GMST in seconds since J2000.0 (from jeod_time)
/// * `equa_of_equi` — equation of equinoxes in seconds of sidereal time (from nutation)
pub fn gast_rotation_matrix(gmst_seconds: f64, equa_of_equi: f64) -> DMat3 {
    // Convert GMST (sidereal seconds) + equation of equinoxes to radians
    // 240 sidereal seconds = 1 degree
    let theta_gast = ((gmst_seconds + equa_of_equi) / 240.0) * DEG_TO_RAD;

    // Normalize to [0, 2π]
    let temp = theta_gast / (2.0 * PI);
    let mut theta = (temp - temp.floor()) * 2.0 * PI;
    if theta < 0.0 {
        theta += 2.0 * PI;
    }

    let (sin_ang, cos_ang) = theta.sin_cos();

    DMat3::from_cols(
        DVec3::new(cos_ang, sin_ang, 0.0),
        DVec3::new(-sin_ang, cos_ang, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    )
}

/// Compute the polar motion rotation matrix from IERS x_p, y_p values.
///
/// Port of JEOD `polar_motion_j2000.cc:152-167`. Uses exact trigonometric
/// functions (not small-angle approximation, per JEOD issue #713).
///
/// # Arguments
/// * `xp` — polar motion x_p in radians (from IERS EOP data)
/// * `yp` — polar motion y_p in radians (from IERS EOP data)
pub fn polar_motion_matrix(xp: f64, yp: f64) -> DMat3 {
    let (sin_xp, cos_xp) = xp.sin_cos();
    let (sin_yp, cos_yp) = yp.sin_cos();

    // JEOD polar_motion_j2000.cc:152-167 (exact trigonometric form)
    DMat3::from_cols(
        DVec3::new(cos_xp, sin_xp * sin_yp, sin_xp * cos_yp),
        DVec3::new(0.0, cos_yp, -sin_yp),
        DVec3::new(-sin_xp, cos_xp * sin_yp, cos_xp * cos_yp),
    )
}

/// Compute the full inertial-to-planet-fixed rotation matrix (T_parent_this).
///
/// Composition: T_parent_this = polar^T × rotation^T × NP
/// where NP = nutation^T × precession^T.
///
/// When `polar_motion` is `None`, polar motion is omitted (equivalent to
/// JEOD's `enable_polar = false`).
///
/// # Arguments
/// * `gmst_seconds` — GMST seconds since J2000.0
/// * `tt_centuries` — TT Julian centuries since J2000.0
/// * `polar_motion` — optional (xp, yp) in radians from IERS EOP data
pub fn compute_t_parent_this(gmst_seconds: f64, tt_centuries: f64) -> DMat3 {
    compute_t_parent_this_with_polar(gmst_seconds, tt_centuries, None)
}

/// Compute the full inertial-to-planet-fixed rotation matrix with optional
/// polar motion.
///
/// Composition (JEOD `planet_rnp.cc:propagate_rnp()`):
/// - With polar motion: T_parent_this = polar^T × rotation^T × NP
/// - Without: T_parent_this = rotation^T × NP
///
/// # Arguments
/// * `gmst_seconds` — GMST seconds since J2000.0
/// * `tt_centuries` — TT Julian centuries since J2000.0
/// * `polar` — optional (xp, yp) in radians; `None` disables polar motion
pub fn compute_t_parent_this_with_polar(
    gmst_seconds: f64,
    tt_centuries: f64,
    polar: Option<(f64, f64)>,
) -> DMat3 {
    let prec = precession_matrix(tt_centuries);
    let nut = nutation(tt_centuries);
    let rot = gast_rotation_matrix(gmst_seconds, nut.equa_of_equi);

    // NP = nutation^T × precession^T
    let np = matrix3x3_product_transpose_transpose(&nut.rotation, &prec);

    // JEOD planet_rnp.cc:224-240:
    // scratch_matrix = polar^T × rotation^T (or just rotation^T if no polar)
    // T_parent_this = scratch_matrix × NP
    let scratch = if let Some((xp, yp)) = polar {
        let pm = polar_motion_matrix(xp, yp);
        pm.transpose() * rot.transpose()
    } else {
        rot.transpose()
    };

    scratch * np
}

/// Convenience: compute T_parent_this from simulation time parameters.
///
/// # Arguments
/// * `gmst_seconds` — GMST seconds since J2000 (from `time_gmst.seconds`)
/// * `tt_tjt` — TT truncated Julian time (from `time_tt.trunc_julian_time`)
pub fn compute_t_parent_this_from_tjt(gmst_seconds: f64, tt_tjt: f64) -> DMat3 {
    compute_t_parent_this_from_tjt_with_polar(gmst_seconds, tt_tjt, None)
}

/// Convenience: compute T_parent_this from simulation time parameters with
/// optional polar motion.
///
/// # Arguments
/// * `gmst_seconds` — GMST seconds since J2000 (from `time_gmst.seconds`)
/// * `tt_tjt` — TT truncated Julian time (from `time_tt.trunc_julian_time`)
/// * `polar` — optional (xp, yp) in radians; `None` disables polar motion
pub fn compute_t_parent_this_from_tjt_with_polar(
    gmst_seconds: f64,
    tt_tjt: f64,
    polar: Option<(f64, f64)>,
) -> DMat3 {
    // Convert TT TJT to Julian centuries since J2000
    // TJT → JD: jd = tjt + 40000 + 2400000.5 = tjt + 2440000.5
    // Centuries: (jd - 2451545.0) / 36525.0 = (tjt + 2440000.5 - 2451545.0) / 36525.0
    //         = (tjt - 11544.5) / 36525.0
    let tt_centuries = (tt_tjt - 11544.5) / 36525.0;
    compute_t_parent_this_with_polar(gmst_seconds, tt_centuries, polar)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_t_parent_this_at_nonzero_epoch() {
        // TT centuries = 0.1 (~3.65 years from J2000)
        let tt_centuries = 0.1;
        // GMST at some time: ~1e6 sidereal seconds
        let gmst_seconds = 1_000_000.0;
        let t = compute_t_parent_this(gmst_seconds, tt_centuries);

        // Matrix should be orthogonal: T * T^T = I
        let product = t * t.transpose();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product.col(j)[i] - expected).abs() < 1e-12,
                    "T * T^T [{},{}] = {}, expected {}",
                    i,
                    j,
                    product.col(j)[i],
                    expected
                );
            }
        }

        // Should differ significantly from identity (not at J2000)
        let diff_from_identity = (t - glam::DMat3::IDENTITY).col(0).length()
            + (t - glam::DMat3::IDENTITY).col(1).length()
            + (t - glam::DMat3::IDENTITY).col(2).length();
        assert!(
            diff_from_identity > 0.01,
            "RNP at non-J2000 should differ from identity, diff = {}",
            diff_from_identity
        );

        // Determinant should be +1 (proper rotation)
        let det = t.determinant();
        assert!(
            (det - 1.0).abs() < 1e-12,
            "determinant should be 1.0, got {}",
            det
        );
    }

    #[test]
    fn polar_motion_identity_at_zero() {
        let pm = polar_motion_matrix(0.0, 0.0);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (pm.col(j)[i] - expected).abs() < 1e-15,
                    "PM(0,0) [{},{}] = {}, expected {}",
                    i,
                    j,
                    pm.col(j)[i],
                    expected
                );
            }
        }
    }

    #[test]
    fn polar_motion_is_orthogonal() {
        // Typical polar motion values: ~0.3 arcsec
        let arcsec = 4.848_136_811_095_36e-6;
        let xp = 0.3 * arcsec;
        let yp = 0.2 * arcsec;
        let pm = polar_motion_matrix(xp, yp);

        // Orthogonality: PM * PM^T = I
        let product = pm * pm.transpose();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product.col(j)[i] - expected).abs() < 1e-14,
                    "PM*PM^T [{},{}] = {}",
                    i,
                    j,
                    product.col(j)[i]
                );
            }
        }

        // Determinant = +1
        let det = pm.determinant();
        assert!((det - 1.0).abs() < 1e-14, "det(PM) = {}", det);

        // Should differ very slightly from identity (arcsecond-level values)
        let diff = (pm - DMat3::IDENTITY).col(0).length();
        assert!(diff > 1e-10 && diff < 1e-4, "PM diff from I = {}", diff);
    }

    #[test]
    fn rnp_with_polar_differs_from_without() {
        let gmst = 1_000_000.0;
        let tt_c = 0.1;
        let arcsec = 4.848_136_811_095_36e-6;
        let xp = 0.3 * arcsec;
        let yp = 0.2 * arcsec;

        let without = compute_t_parent_this_with_polar(gmst, tt_c, None);
        let with = compute_t_parent_this_with_polar(gmst, tt_c, Some((xp, yp)));

        // Both should be orthogonal
        let det_w = with.determinant();
        assert!((det_w - 1.0).abs() < 1e-12);

        // They should differ by a small amount (arcsecond-level)
        let diff = (with - without).col(0).length()
            + (with - without).col(1).length()
            + (with - without).col(2).length();
        assert!(diff > 1e-10 && diff < 1e-4, "polar motion diff = {}", diff);

        // None should equal zero polar motion
        let with_zero = compute_t_parent_this_with_polar(gmst, tt_c, Some((0.0, 0.0)));
        let diff_zero = (without - with_zero).col(0).length()
            + (without - with_zero).col(1).length()
            + (without - with_zero).col(2).length();
        assert!(diff_zero < 1e-15, "None != Some(0,0): diff = {}", diff_zero);
    }

    #[test]
    fn gast_rotation_is_orthogonal() {
        let r = gast_rotation_matrix(1000.0, 0.5);
        let rrt = r * r.transpose();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (rrt.col(j)[i] - expected).abs() < 1e-14,
                    "R*R^T [{}][{}] = {}",
                    i,
                    j,
                    rrt.col(j)[i]
                );
            }
        }
    }
}
