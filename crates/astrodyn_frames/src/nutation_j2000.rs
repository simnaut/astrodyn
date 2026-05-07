//! J2000 Nutation matrix and equation of equinoxes computation.
//!
//! Faithful port of JEOD's `nutation_j2000.cc` with IAU 1980 model (106 terms).
//!
//! Reference: Mulcahy & Bond, "The RNP Routine for the Standard Epoch J2000",
//! NASA JSC-24574, September 1990.

// Constants ported verbatim from JEOD — suppress excessive precision warnings.
#![allow(clippy::excessive_precision)]

use crate::data_nutation_j2000::*;
use glam::{DMat3, DVec3};
use std::f64::consts::PI;

const DEG_TO_RAD: f64 = PI / 180.0;
const ARCSEC_TO_RAD: f64 = DEG_TO_RAD / 3600.0;

/// Nutation computation results.
pub struct NutationResult {
    /// 3x3 nutation rotation matrix.
    pub rotation: DMat3,
    /// Equation of equinoxes in seconds of sidereal time.
    pub equa_of_equi: f64,
}

/// Compute the nutation matrix and equation of equinoxes.
///
/// Port of JEOD `nutation_j2000.cc` with IAU 1980 model (106 terms).
///
/// # Arguments
/// * `time` — Julian centuries since J2000.0 TT
pub fn nutation(time: f64) -> NutationResult {
    let time2 = time * time;
    let time3 = time2 * time;

    // Fundamental arguments in degrees (Mulcahy & Bond, JSC-24574)
    let l = 134.9629813888889
        + 477198.8673980555 * time
        + 0.008697222222222223 * time2
        + 0.00001777777777777778 * time3;
    let m = 357.5277233333333 + 35999.05034 * time
        - 0.00016027777777777778 * time2
        - 0.000003333333333333333 * time3;
    let f = 93.27191027777778 + 483202.0175380555 * time - 0.0036825 * time2
        + 0.000003055555555555555 * time3;
    let d = 297.8503630555556 + 445267.11148 * time - 0.001914166666666667 * time2
        + 0.0000052777777777777778 * time3;
    let omega = 125.0445222222222 - 1934.136260833333 * time
        + 0.00207083333333333 * time2
        + 0.000002222222222222222 * time3;

    // Sum nutation series (106 terms)
    let mut nutation_in_longitude = 0.0_f64; // units: 1e-4 arcseconds
    let mut nutation_in_obliquity = 0.0_f64; // units: 1e-4 arcseconds

    for i in 0..NUM_NUTATION_COEFFS {
        let api = (L_COEFFS[i] * l
            + M_COEFFS[i] * m
            + F_COEFFS[i] * f
            + D_COEFFS[i] * d
            + OMEGA_COEFFS[i] * omega)
            * DEG_TO_RAD;

        nutation_in_longitude += (LONG_COEFFS[i] + LONG_T_COEFFS[i] * time) * api.sin();
        nutation_in_obliquity += (OBLIQ_COEFFS[i] + OBLIQ_T_COEFFS[i] * time) * api.cos();
    }

    // Mean obliquity of the ecliptic (degrees -> radians)
    let epsilon_bar =
        (23.43929111111111 - 0.01300416666666667 * time - 0.00000016388888889 * time2
            + 0.00000050361111111 * time3)
            * DEG_TO_RAD;

    // Convert nutation in obliquity: 1e-4 arcsec -> arcsec -> degrees -> radians
    let nutation_in_obliquity_rad = (nutation_in_obliquity / 10000.0) * ARCSEC_TO_RAD;
    let epsilon = epsilon_bar + nutation_in_obliquity_rad;

    // Nutation in longitude: 1e-4 arcsec -> arcsec (for equation of equinoxes)
    let nutation_in_longitude_as = nutation_in_longitude / 10000.0;

    // Equation of equinoxes (in seconds of sidereal time)
    let c_eps = epsilon.cos();
    let equa_of_equi = (nutation_in_longitude_as * c_eps) / 15.0;

    // Nutation in longitude: arcsec -> radians
    let nutation_in_longitude_rad = nutation_in_longitude_as * ARCSEC_TO_RAD;

    // Build nutation matrix
    let c_long = nutation_in_longitude_rad.cos();
    let s_long = nutation_in_longitude_rad.sin();
    let s_eps = epsilon.sin();
    let c_eps_bar = epsilon_bar.cos();
    let s_eps_bar = epsilon_bar.sin();

    let rotation = DMat3::from_cols(
        DVec3::new(c_long, -c_eps_bar * s_long, -s_eps_bar * s_long),
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

    NutationResult {
        rotation,
        equa_of_equi,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nutation_matrix_is_near_identity_at_j2000() {
        let n = nutation(0.0);
        // Nutation is small — matrix should be near identity
        for i in 0..3 {
            let diag = n.rotation.col(i)[i];
            assert!(
                (diag - 1.0).abs() < 1e-4,
                "nutation diagonal [{}] = {}",
                i,
                diag
            );
        }
    }
}
