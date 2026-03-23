//! J2000 Precession matrix computation.
//!
//! Faithful port of JEOD's `precession_j2000.cc` (IAU-76/FK5).
//!
//! Reference: Mulcihy & Bond, "The RNP Routine for the Standard Epoch J2000",
//! NASA JSC-24574, September 1990.

// Constants ported verbatim from JEOD — suppress excessive precision warnings.
#![allow(clippy::excessive_precision)]

use glam::{DMat3, DVec3};
use std::f64::consts::PI;

const DEG_TO_RAD: f64 = PI / 180.0;
const ARCSEC_TO_RAD: f64 = DEG_TO_RAD / 3600.0;

/// Compute the precession matrix.
///
/// Port of JEOD `precession_j2000.cc`.
///
/// # Arguments
/// * `t` — Julian centuries since J2000.0 TT
///
/// # Returns
/// 3x3 precession rotation matrix (stored as JEOD convention: row-major in the
/// `rotation` array, representing the transformation from J2000 mean equator
/// to mean equator of date).
pub fn precession_matrix(t: f64) -> DMat3 {
    let t2 = t * t;
    let t3 = t2 * t;

    // Precession parameters in arcseconds (Mulcihy & Bond, JSC-24574)
    let zeta_as = 2306.2181 * t + 0.30188 * t2 + 0.017998 * t3;
    let theta_as = 2004.3109 * t - 0.42665 * t2 - 0.041833 * t3;
    let z_as = 2306.2181 * t + 1.09468 * t2 + 0.018203 * t3;

    let zeta = zeta_as * ARCSEC_TO_RAD;
    let theta = theta_as * ARCSEC_TO_RAD;
    let z = z_as * ARCSEC_TO_RAD;

    let (s_zeta, c_zeta) = zeta.sin_cos();
    let (s_theta, c_theta) = theta.sin_cos();
    let (s_z, c_z) = z.sin_cos();

    // Precession matrix: rot_z(zeta) * rot_y(-theta) * rot_z(z)
    // Stored in JEOD convention (see precession_j2000.cc)
    DMat3::from_cols(
        DVec3::new(
            c_theta * c_z * c_zeta - s_z * s_zeta,
            -s_zeta * c_theta * c_z - c_zeta * s_z,
            -s_theta * c_z,
        ),
        DVec3::new(
            s_z * c_theta * c_zeta + s_zeta * c_z,
            -s_z * s_zeta * c_theta + c_z * c_zeta,
            -s_theta * s_z,
        ),
        DVec3::new(
            c_zeta * s_theta,
            -s_zeta * s_theta,
            c_theta,
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precession_at_j2000_is_identity() {
        let p = precession_matrix(0.0);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (p.col(j)[i] - expected).abs() < 1e-15,
                    "precession[{}][{}] = {}, expected {}",
                    i, j, p.col(j)[i], expected
                );
            }
        }
    }
}
