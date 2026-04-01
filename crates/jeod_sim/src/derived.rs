//! Per-body derived state functions.
//!
//! Post-integration observational quantities computed from the integrated state.
//! Each function is pure (no side effects) and takes explicit parameters so that
//! any ECS adapter can call it from a system function.

use glam::{DMat3, DVec3};

use crate::{EulerSequence, GeodeticState, LvlhFrame, OrbitalElements, RotationalState};
use jeod_math::OrbitalError;

/// Compute orbital elements from translational state.
///
/// Delegates to [`OrbitalElements::from_cartesian`].
pub fn compute_orbital_elements(
    mu: f64,
    position: DVec3,
    velocity: DVec3,
) -> Result<OrbitalElements, OrbitalError> {
    OrbitalElements::from_cartesian(mu, position, velocity)
}

/// Compute Euler angles from body attitude.
///
/// Converts the left-transformation quaternion to a rotation matrix, then
/// decomposes it into Euler angles using the given sequence.
pub fn compute_body_euler_angles(rot: &RotationalState, sequence: EulerSequence) -> [f64; 3] {
    let t_parent_body = rot.quaternion.left_quat_to_transformation();
    jeod_math::compute_euler_angles_from_matrix(&t_parent_body, sequence)
}

/// Compute the LVLH (Local Vertical Local Horizontal) frame from translational state.
///
/// Delegates to [`jeod_math::compute_lvlh_frame`].
pub fn compute_body_lvlh_frame(position: DVec3, velocity: DVec3) -> LvlhFrame {
    jeod_math::compute_lvlh_frame(position, velocity)
}

/// Compute geodetic state (latitude, longitude, altitude) from inertial position.
///
/// Rotates the inertial position into the planet-fixed frame using the given
/// transformation matrix, then converts to geodetic coordinates on the reference
/// ellipsoid defined by `r_eq` and `r_pol`.
pub fn compute_body_geodetic(
    position: DVec3,
    t_inertial_pfix: &DMat3,
    r_eq: f64,
    r_pol: f64,
) -> GeodeticState {
    // Rotate inertial position to planet-fixed frame
    let pos_pfix = *t_inertial_pfix * position;
    jeod_math::cartesian_to_geodetic(pos_pfix, r_eq, r_pol)
}

/// Compute the solar beta angle (angle between orbit plane and Sun direction).
///
/// Computes the orbital angular momentum vector `h = r × v`, then delegates
/// to [`jeod_math::solar_beta_angle`].
///
/// # Panics
///
/// Panics if the orbital angular momentum `h = r × v` is zero (degenerate
/// orbit) or if the Sun position coincides with the body position.
pub fn compute_body_solar_beta(position: DVec3, velocity: DVec3, sun_position: DVec3) -> f64 {
    let h = position.cross(velocity);
    let rel_sun = sun_position - position;

    assert!(
        h.length_squared() > 0.0,
        "compute_body_solar_beta: orbital angular momentum is zero; \
         solar beta angle is undefined"
    );
    assert!(
        rel_sun.length_squared() > 0.0,
        "compute_body_solar_beta: sun_position coincides with position; \
         solar beta angle is undefined"
    );

    let sun_dir = rel_sun.normalize();
    jeod_math::solar_beta_angle(h, sun_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `compute_body_geodetic` correctly applies the inertial-to-
    /// planet-fixed rotation before computing geodetic coordinates.
    ///
    /// Uses a 90-degree rotation about Z so a position on the inertial +X axis
    /// maps to the planet-fixed +Y axis, producing longitude ≈ π/2.
    /// A transpose/sign error in the rotation would yield longitude ≈ −π/2.
    #[test]
    fn geodetic_with_rotation() {
        use std::f64::consts::FRAC_PI_2;

        const R_EQ: f64 = 6_378_137.0;
        const R_POL: f64 = R_EQ * (1.0 - 1.0 / 298.257_223_563); // JEOD: r_eq * (1 - flat_coeff)

        // 90° rotation about Z: maps +X_inertial → +Y_pfix
        //
        // Row-major rotation matrix for +90° about Z:
        //   [ cos  -sin  0 ]     [ 0  -1  0 ]
        //   [ sin   cos  0 ]  =  [ 1   0  0 ]
        //   [  0     0   1 ]     [ 0   0  1 ]
        //
        // glam DMat3::from_cols takes column-major, so transpose the rows.
        let t_inertial_pfix = DMat3::from_cols(
            DVec3::new(0.0, 1.0, 0.0),  // col 0
            DVec3::new(-1.0, 0.0, 0.0), // col 1
            DVec3::new(0.0, 0.0, 1.0),  // col 2
        );

        // ISS-altitude position along inertial +X axis
        let pos_inertial = DVec3::new(R_EQ + 408_000.0, 0.0, 0.0);

        let geo = compute_body_geodetic(pos_inertial, &t_inertial_pfix, R_EQ, R_POL);

        // After rotation, planet-fixed position is along +Y → longitude ≈ π/2
        assert!(
            (geo.longitude - FRAC_PI_2).abs() < 1e-10,
            "expected longitude ≈ π/2, got {}",
            geo.longitude
        );
        // Latitude should be ≈ 0 (equatorial)
        assert!(
            geo.latitude.abs() < 1e-10,
            "expected latitude ≈ 0, got {}",
            geo.latitude
        );
        // Altitude should be ≈ 408 km
        assert!(
            (geo.altitude - 408_000.0).abs() < 1.0,
            "expected altitude ≈ 408000 m, got {}",
            geo.altitude
        );

        // Verify against direct computation to lock down the convention
        let pos_pfix = t_inertial_pfix * pos_inertial;
        let expected = jeod_math::cartesian_to_geodetic(pos_pfix, R_EQ, R_POL);
        assert_eq!(geo, expected);
    }
}
