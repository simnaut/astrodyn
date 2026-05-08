//! Moon rotation model (IAU 2009 simplified).
//!
//! Computes the inertial-to-Moon-body-fixed transformation matrix using the
//! IAU recommended pole orientation and prime meridian angle. This is a
//! simplified model without the full libration terms; it provides the mean
//! Earth/rotation axis orientation accurate to ~0.1° for trajectory-level
//! gravity evaluation.
//!
//! For higher fidelity, use DE421/DE430 libration data via ANISE BPC kernels.
//!
//! Reference: Archinal et al. (2011), "Report of the IAU Working Group on
//! Cartographic Coordinates and Rotational Elements: 2009", Celestial
//! Mechanics and Dynamical Astronomy, 109(2), 101-135.

use glam::DMat3;
use std::f64::consts::PI;

const DEG2RAD: f64 = PI / 180.0;
const SECONDS_PER_DAY: f64 = 86400.0;
const DAYS_PER_CENTURY: f64 = 36525.0;

/// Compute IAU Moon body-fixed rotation matrix from TDB seconds since J2000.
///
/// Returns the 3x3 rotation matrix mapping vectors from the Moon-centered
/// J2000 inertial frame to the Moon mean-Earth/rotation-axis (ME) frame.
///
/// The IAU model defines:
/// - Right ascension of pole: α₀ = 269.9949° + 0.0031°T
/// - Declination of pole: δ₀ = 66.5392° + 0.0130°T
/// - Prime meridian: W = 38.3213° + 13.17635815°d + libration terms
///
/// where T = Julian centuries since J2000, d = days since J2000.
pub fn compute_moon_rotation(tdb_seconds_since_j2000: f64) -> DMat3 {
    let d = tdb_seconds_since_j2000 / SECONDS_PER_DAY;
    let t = d / DAYS_PER_CENTURY;

    // IAU 2009 fundamental arguments (degrees)
    let e1 = (125.045 - 0.052_992_0 * d) * DEG2RAD;
    let e2 = (250.089 - 0.105_984_0 * d) * DEG2RAD;
    let e3 = (260.008 + 13.012_009_0 * d) * DEG2RAD;
    let e4 = (176.625 + 13.340_716_0 * d) * DEG2RAD;
    let e6 = (311.589 + 26.724_071_0 * d) * DEG2RAD;
    let e7 = (134.963 + 13.064_993_0 * d) * DEG2RAD;
    let e10 = (15.134 - 0.155_280_0 * d) * DEG2RAD;
    let e13 = (25.053 + 12.151_058_0 * d) * DEG2RAD;

    // Right ascension of pole (degrees)
    let alpha0 = 269.9949 + 0.0031 * t - 3.8787 * e1.sin() - 0.1204 * e2.sin() + 0.0700 * e3.sin()
        - 0.0172 * e4.sin()
        + 0.0072 * e6.sin()
        - 0.0052 * e10.sin()
        + 0.0043 * e13.sin();

    // Declination of pole (degrees)
    let delta0 = 66.5392 + 0.0130 * t + 1.5419 * e1.cos() + 0.0239 * e2.cos() - 0.0278 * e3.cos()
        + 0.0068 * e4.cos()
        - 0.0029 * e6.cos()
        + 0.0009 * e7.cos()
        + 0.0008 * e10.cos()
        - 0.0009 * e13.cos();

    // Prime meridian angle (degrees)
    let w = 38.3213 + 13.176_358_15 * d + 3.5610 * e1.sin() + 0.1208 * e2.sin() - 0.0642 * e3.sin()
        + 0.0158 * e4.sin()
        + 0.0252 * e5_sin(d)
        - 0.0066 * e6.sin()
        - 0.0047 * e7.sin()
        - 0.0046 * e8_sin(d)
        + 0.0028 * e9_sin(d)
        + 0.0052 * e10.sin()
        + 0.0040 * e11_sin(d)
        + 0.0019 * e12_sin(d)
        - 0.0044 * e13.sin();

    // Convert to radians
    let alpha = alpha0 * DEG2RAD;
    let delta = delta0 * DEG2RAD;
    let w_rad = w * DEG2RAD;

    // Build rotation matrix: R = Rz(-W) * Rx(-(90-δ)) * Rz(-(90+α))
    // This is the standard IAU pole/prime-meridian to body-fixed transformation.
    let ca = (90.0_f64.to_radians() + alpha).cos();
    let sa = (90.0_f64.to_radians() + alpha).sin();
    let cd = (90.0_f64.to_radians() - delta).cos();
    let sd = (90.0_f64.to_radians() - delta).sin();
    let cw = w_rad.cos();
    let sw = w_rad.sin();

    // Rz(-(90+α))
    let rz1 = DMat3::from_cols_array(&[ca, sa, 0.0, -sa, ca, 0.0, 0.0, 0.0, 1.0]);
    // Rx(-(90-δ))
    let rx = DMat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, cd, sd, 0.0, -sd, cd]);
    // Rz(-W)
    let rz2 = DMat3::from_cols_array(&[cw, sw, 0.0, -sw, cw, 0.0, 0.0, 0.0, 1.0]);

    rz2 * rx * rz1
}

/// Typed sibling of [`compute_moon_rotation`] returning the rotation
/// already wrapped as
/// `FrameTransform<RootInertial, PlanetFixed<Moon>>`. Pinned to
/// Moon by definition — this kernel is the IAU Moon rotation formula.
pub fn compute_moon_rotation_typed(
    tdb_seconds_since_j2000: f64,
) -> astrodyn_quantities::frame_transform::FrameTransform<
    astrodyn_quantities::frame::RootInertial,
    astrodyn_quantities::frame::PlanetFixed<astrodyn_quantities::frame::Moon>,
> {
    let mat = compute_moon_rotation(tdb_seconds_since_j2000);
    astrodyn_quantities::frame_transform::FrameTransform::from_matrix(mat)
}

// Helper functions for the remaining E arguments not stored as variables
fn e5_sin(d: f64) -> f64 {
    ((358.473 + 0.985_600_3 * d) * DEG2RAD).sin()
}
fn e8_sin(d: f64) -> f64 {
    ((326.325 + 0.985_600_3 * d) * DEG2RAD).sin() // ~ E5 shifted
}
fn e9_sin(d: f64) -> f64 {
    ((85.0 + 0.003_2 * d) * DEG2RAD).sin()
}
fn e11_sin(d: f64) -> f64 {
    ((119.743 + 0.008_6 * d) * DEG2RAD).sin()
}
fn e12_sin(d: f64) -> f64 {
    ((239.961 + 0.183_6 * d) * DEG2RAD).sin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moon_rotation_valid_at_j2000() {
        let mat = compute_moon_rotation(0.0);
        let det = mat.determinant();
        assert!((det - 1.0).abs() < 1e-12, "det should be 1, got {det}");
        // M * M^T = I
        let mt = mat.transpose();
        let product = mat * mt;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product.col(j)[i] - expected).abs() < 1e-12,
                    "M*M^T[{i}][{j}] = {}, expected {expected}",
                    product.col(j)[i]
                );
            }
        }
    }

    #[test]
    fn moon_rotation_changes_over_day() {
        let mat0 = compute_moon_rotation(0.0);
        let mat1 = compute_moon_rotation(86400.0);
        let diff = (mat1.col(0) - mat0.col(0)).length();
        // Moon rotates ~13.2°/day → significant rotation matrix change
        assert!(
            diff > 0.1,
            "Moon should rotate noticeably over 1 day, diff={diff}"
        );
    }

    #[test]
    fn moon_sidereal_month() {
        // Moon sidereal period ≈ 27.322 days
        let period_s = 27.322 * 86400.0;
        let mat0 = compute_moon_rotation(0.0);
        let mat1 = compute_moon_rotation(period_s);
        let diff = (mat1.col(0) - mat0.col(0)).length();
        // After one sidereal month, rotation should nearly repeat
        assert!(
            diff < 0.05,
            "Moon rotation should nearly repeat after 1 sidereal month, diff={diff}"
        );
    }
}
