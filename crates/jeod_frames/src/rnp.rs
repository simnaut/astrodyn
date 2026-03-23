//! Earth Rotation-Nutation-Precession (RNP) model.
//!
//! Faithful port of JEOD's RNPJ2000 pipeline (IAU-76/FK5).
//! Computes the inertial-to-planet-fixed rotation matrix from time scales.
//!
//! Reference: Mulcihy & Bond, "The RNP Routine for the Standard Epoch J2000",
//! NASA JSC-24574, September 1990.

// Constants ported verbatim from JEOD — suppress excessive precision warnings.
#![allow(clippy::excessive_precision)]

use crate::rnp_data::*;
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

/// Nutation computation results.
pub struct NutationResult {
    /// 3x3 nutation rotation matrix.
    pub matrix: DMat3,
    /// Equation of equinoxes in seconds of sidereal time.
    pub equa_of_equi: f64,
}

/// Compute the nutation matrix and equation of equinoxes.
///
/// Port of JEOD `nutation_j2000.cc` with IAU 1980 model (106 terms).
///
/// # Arguments
/// * `t` — Julian centuries since J2000.0 TT
pub fn nutation(t: f64) -> NutationResult {
    let t2 = t * t;
    let t3 = t2 * t;

    // Fundamental arguments in degrees (Mulcihy & Bond, JSC-24574)
    let l = 134.9629813888889 + 477198.8673980555 * t + 0.008697222222222223 * t2
        + 0.00001777777777777778 * t3;
    let m = 357.5277233333333 + 35999.05034 * t - 0.00016027777777777778 * t2
        - 0.000003333333333333333 * t3;
    let f = 93.27191027777778 + 483202.0175380555 * t - 0.0036825 * t2
        + 0.000003055555555555555 * t3;
    let d = 297.8503630555556 + 445267.11148 * t - 0.001914166666666667 * t2
        + 0.0000052777777777777778 * t3;
    let omega = 125.0445222222222 - 1934.136260833333 * t + 0.00207083333333333 * t2
        + 0.000002222222222222222 * t3;

    // Sum nutation series (106 terms)
    let mut nutation_in_longitude = 0.0_f64; // units: 1e-4 arcseconds
    let mut nutation_in_obliquity = 0.0_f64; // units: 1e-4 arcseconds

    for i in 0..NUM_NUTATION_COEFFS {
        let api = (L_COEFFS[i] * l + M_COEFFS[i] * m + F_COEFFS[i] * f
            + D_COEFFS[i] * d + OMEGA_COEFFS[i] * omega)
            * DEG_TO_RAD;

        nutation_in_longitude += (LONG_COEFFS[i] + LONG_T_COEFFS[i] * t) * api.sin();
        nutation_in_obliquity += (OBLIQ_COEFFS[i] + OBLIQ_T_COEFFS[i] * t) * api.cos();
    }

    // Mean obliquity of the ecliptic (degrees → radians)
    let epsilon_bar = (23.43929111111111 - 0.01300416666666667 * t
        - 0.00000016388888889 * t2 + 0.00000050361111111 * t3)
        * DEG_TO_RAD;

    // Convert nutation in obliquity: 1e-4 arcsec → arcsec → degrees → radians
    let nutation_in_obliquity_rad = (nutation_in_obliquity / 10000.0) * ARCSEC_TO_RAD;
    let epsilon = epsilon_bar + nutation_in_obliquity_rad;

    // Nutation in longitude: 1e-4 arcsec → arcsec (for equation of equinoxes)
    let nutation_in_longitude_as = nutation_in_longitude / 10000.0;

    // Equation of equinoxes (in seconds of sidereal time)
    let c_eps = epsilon.cos();
    let equa_of_equi = (nutation_in_longitude_as * c_eps) / 15.0;

    // Nutation in longitude: arcsec → radians
    let nutation_in_longitude_rad = nutation_in_longitude_as * ARCSEC_TO_RAD;

    // Build nutation matrix
    let c_long = nutation_in_longitude_rad.cos();
    let s_long = nutation_in_longitude_rad.sin();
    let s_eps = epsilon.sin();
    let c_eps_bar = epsilon_bar.cos();
    let s_eps_bar = epsilon_bar.sin();

    let matrix = DMat3::from_cols(
        DVec3::new(
            c_long,
            -c_eps_bar * s_long,
            -s_eps_bar * s_long,
        ),
        DVec3::new(
            c_eps * s_long,
            c_eps * c_long * c_eps_bar + s_eps * s_eps_bar,
            c_eps * c_long * s_eps_bar - s_eps * c_eps_bar,
        ),
        DVec3::new(
            s_eps * s_long,
            s_eps * c_long * c_eps_bar - c_eps * s_eps_bar,
            s_eps * s_eps_bar * c_long + c_eps * c_eps_bar,
        ),
    );

    NutationResult { matrix, equa_of_equi }
}

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

    // NP = nutation^T × precession^T
    let np = nut.matrix.transpose() * prec.transpose();

    // T_parent_this = rotation^T × NP
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

    #[test]
    fn nutation_matrix_is_near_identity_at_j2000() {
        let n = nutation(0.0);
        // Nutation is small — matrix should be near identity
        for i in 0..3 {
            let diag = n.matrix.col(i)[i];
            assert!(
                (diag - 1.0).abs() < 1e-4,
                "nutation diagonal [{}] = {}",
                i, diag
            );
        }
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
                    i, j, rrt.col(j)[i]
                );
            }
        }
    }
}
