//! Per-body derived state functions.
//!
//! Post-integration observational quantities computed from the integrated state.
//! Each function is pure (no side effects) and takes explicit parameters so that
//! any ECS adapter can call it from a system function.

use glam::{DMat3, DQuat, DVec3};

use crate::{EulerSequence, GeodeticState, LvlhFrame, OrbitalElements, RotationalState};
use jeod_math::OrbitalError;

/// Relative state between two bodies in the inertial frame.
///
/// Position/velocity are of `subject` relative to `reference`, expressed in
/// the inertial frame. The quaternion is the relative attitude (reference-to-
/// subject), and angular velocity is of `subject` relative to `reference`,
/// expressed in the subject body frame.
#[derive(Debug, Clone)]
pub struct RelativeState {
    /// Position of subject relative to reference (inertial frame, m).
    pub position: DVec3,
    /// Velocity of subject relative to reference (inertial frame, m/s).
    pub velocity: DVec3,
    /// Relative quaternion: reference body frame → subject body frame.
    pub quaternion: DQuat,
    /// Angular velocity of subject relative to reference (subject body frame, rad/s).
    pub ang_vel: DVec3,
}

/// Relative state expressed in the LVLH frame of the reference vehicle.
#[derive(Debug, Clone)]
pub struct LvlhRelativeState {
    /// Position of subject relative to reference (LVLH frame, m).
    pub position: DVec3,
    /// Velocity of subject relative to reference (LVLH frame, m/s).
    pub velocity: DVec3,
}

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

/// Compute the relative state between two bodies.
///
/// Returns the state of `subject` relative to `reference`. If either body
/// has no rotational state, the quaternion and angular velocity are set to
/// identity/zero.
///
/// Mirrors JEOD's `compute_relative_state` for dynamic bodies where both
/// states are in the same inertial frame.
pub fn compute_relative_state(
    ref_trans: &crate::TranslationalState,
    ref_rot: Option<&RotationalState>,
    subj_trans: &crate::TranslationalState,
    subj_rot: Option<&RotationalState>,
) -> RelativeState {
    let position = subj_trans.position - ref_trans.position;
    let velocity = subj_trans.velocity - ref_trans.velocity;

    // Relative quaternion: q_rel = q_subj * q_ref^{-1}
    // This gives the rotation from reference body frame to subject body frame.
    let (quaternion, ang_vel) = match (ref_rot, subj_rot) {
        (Some(r_ref), Some(r_subj)) => {
            // Convert JEOD scalar-first to glam DQuat for composition
            let q_ref = r_ref.quaternion.to_glam();
            let q_subj = r_subj.quaternion.to_glam();
            // q_rel = q_subj * conj(q_ref)
            let q_rel = q_subj * q_ref.conjugate();
            // Relative angular velocity in subject body frame
            let rel_ang_vel = r_subj.ang_vel_body - r_ref.ang_vel_body;
            (q_rel, rel_ang_vel)
        }
        _ => (DQuat::IDENTITY, DVec3::ZERO),
    };

    RelativeState {
        position,
        velocity,
        quaternion,
        ang_vel,
    }
}

/// Compute relative state expressed in the LVLH frame of the reference vehicle.
///
/// Takes the inertial relative position/velocity and rotates them into the
/// LVLH frame of the reference vehicle. The LVLH frame is computed from the
/// reference vehicle's position and velocity.
pub fn compute_lvlh_relative_state(
    ref_pos: DVec3,
    ref_vel: DVec3,
    subj_pos: DVec3,
    subj_vel: DVec3,
) -> LvlhRelativeState {
    let lvlh = compute_body_lvlh_frame(ref_pos, ref_vel);

    // Relative state in inertial frame
    let rel_pos_inertial = subj_pos - ref_pos;
    let rel_vel_inertial = subj_vel - ref_vel;

    // Rotate into LVLH frame using the T_parent_this matrix
    // T_parent_this transforms from parent (inertial) to this (LVLH)
    let pos_lvlh = lvlh.t_parent_this * rel_pos_inertial;
    let vel_lvlh = lvlh.t_parent_this * rel_vel_inertial;

    LvlhRelativeState {
        position: pos_lvlh,
        velocity: vel_lvlh,
    }
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
