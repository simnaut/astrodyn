//! J2000 GAST rotation and full RNP composition.
//!
//! Faithful port of JEOD's `rotation_j2000.cc` (IAU-76/FK5).
//! Computes the inertial-to-planet-fixed rotation matrix from time scales.
//!
//! Reference: Mulcihy & Bond, "The RNP Routine for the Standard Epoch J2000",
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
    let frac = theta_gast / (2.0 * PI);
    let mut theta = (frac - frac.floor()) * 2.0 * PI;
    if theta < 0.0 {
        theta += 2.0 * PI;
    }

    let (s, c) = theta.sin_cos();

    DMat3::from_cols(
        DVec3::new(c, s, 0.0),
        DVec3::new(-s, c, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    )
}

/// Compute the full inertial-to-planet-fixed rotation matrix (T_parent_this).
///
/// Composition: T_parent_this = rotation^T × nutation^T × precession^T
/// (No polar motion — disabled in JEOD SIM_dyncomp.)
///
/// # Arguments
/// * `gmst_seconds` — GMST seconds since J2000.0
/// * `tt_centuries` — TT Julian centuries since J2000.0
pub fn compute_t_parent_this(gmst_seconds: f64, tt_centuries: f64) -> DMat3 {
    let prec = precession_matrix(tt_centuries);
    let nut = nutation(tt_centuries);
    let rot = gast_rotation_matrix(gmst_seconds, nut.equa_of_equi);

    // Matrix3x3::product_transpose_transpose(nutation, precession, NP_matrix)
    let np = matrix3x3_product_transpose_transpose(&nut.matrix, &prec);

    // scratch_matrix = rotation^T (no polar motion)
    // T_parent_this = scratch_matrix × NP
    rot.transpose() * np
}

/// Convenience: compute T_parent_this from simulation time parameters.
///
/// # Arguments
/// * `gmst_seconds` — GMST seconds since J2000 (from `time_gmst.seconds`)
/// * `tt_tjt` — TT truncated Julian time (from `time_tt.trunc_julian_time`)
pub fn compute_t_parent_this_from_tjt(gmst_seconds: f64, tt_tjt: f64) -> DMat3 {
    // Convert TT TJT to Julian centuries since J2000
    // TJT → JD: jd = tjt + 40000 + 2400000.5 = tjt + 2440000.5
    // Centuries: (jd - 2451545.0) / 36525.0 = (tjt + 2440000.5 - 2451545.0) / 36525.0
    //         = (tjt - 11544.5) / 36525.0
    let tt_centuries = (tt_tjt - 11544.5) / 36525.0;
    compute_t_parent_this(gmst_seconds, tt_centuries)
}

#[cfg(test)]
mod tests {
    use super::*;

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
                    i, j, rrt.col(j)[i]
                );
            }
        }
    }
}
