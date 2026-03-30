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
