//! LVLH (Local Vertical Local Horizontal) frame computation.
//!
//! Faithful port of JEOD's `lvlh_frame.cc:247-285`.
//!
//! The LVLH frame is defined relative to a planet-centered inertial frame:
//! - Z-hat = -r̂ (nadir, toward planet center)
//! - Y-hat = -ĥ (negative orbit normal)
//! - X-hat = Y-hat × Z-hat (approximately along velocity for circular orbits)
//!
//! The frame origin is co-located and co-moving with the vehicle.

use glam::{DMat3, DVec3};

/// LVLH frame state: rotation matrix, angular velocity, and origin position/velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LvlhFrame {
    /// Transformation matrix from parent (inertial) frame to LVLH frame.
    /// Rows are the LVLH frame axes expressed in the inertial frame.
    pub t_parent_this: DMat3,
    /// Angular velocity of the LVLH frame in the LVLH frame (rad/s).
    pub ang_vel_this: DVec3,
    /// Position of the LVLH frame origin (same as vehicle) in the inertial frame (m).
    pub position: DVec3,
    /// Velocity of the LVLH frame origin (same as vehicle) in the inertial frame (m/s).
    pub velocity: DVec3,
}

/// Compute the LVLH frame from position and velocity in a planet-centered inertial frame.
///
/// Port of JEOD `LvlhFrame::compute_lvlh_frame()` (lvlh_frame.cc:247-285).
///
/// # Arguments
/// * `position` - Vehicle position in planet-centered inertial frame (m)
/// * `velocity` - Vehicle velocity in planet-centered inertial frame (m/s)
///
/// # Panics
/// Panics if position or angular momentum magnitude is zero.
pub fn compute_lvlh_frame(position: DVec3, velocity: DVec3) -> LvlhFrame {
    // Compute angular momentum vector: h = r × v
    let angmom = position.cross(velocity);
    let hmag = angmom.length();
    let rmagsq = position.length_squared();
    let rmag = rmagsq.sqrt();

    assert!(rmag > 0.0, "compute_lvlh_frame: position magnitude is zero");
    assert!(
        hmag > 0.0,
        "compute_lvlh_frame: angular momentum is zero (radial trajectory)"
    );

    // Orbital angular velocity magnitude
    let wmag = hmag / rmagsq;

    // LVLH frame axes (rows of T_parent_this):
    //   z_hat = -r/rmag (nadir, toward planet center)
    //   y_hat = -h/hmag (negative orbit normal)
    //   x_hat = y_hat × z_hat (forward, approximately along velocity)
    let z_hat = -position / rmag;
    let y_hat = -angmom / hmag;
    let x_hat = y_hat.cross(z_hat).normalize();

    // T_parent_this is a row-major transformation matrix where rows are frame axes
    // In glam DMat3 (column-major), we construct from columns then transpose,
    // or equivalently construct from rows using from_cols on the transposed data.
    //
    // JEOD stores T_parent_this[0] = x_hat, T_parent_this[1] = y_hat, T_parent_this[2] = z_hat
    // These are rows of the transformation matrix. In glam's column-major DMat3:
    let t_parent_this = DMat3::from_cols(
        DVec3::new(x_hat.x, y_hat.x, z_hat.x),
        DVec3::new(x_hat.y, y_hat.y, z_hat.y),
        DVec3::new(x_hat.z, y_hat.z, z_hat.z),
    );

    // Angular velocity of the LVLH frame in the LVLH frame:
    // ω = [0, -wmag, 0] (rotation about the -Y axis at orbital rate)
    let ang_vel_this = DVec3::new(0.0, -wmag, 0.0);

    LvlhFrame {
        t_parent_this,
        ang_vel_this,
        position,
        velocity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EARTH_MU: f64 = 3.986_004_418e14; // m^3/s^2

    #[test]
    fn circular_equatorial_orbit() {
        // Circular orbit at 400 km altitude, equatorial
        let r = 6_778_137.0; // r_eq + 400 km
        let v = (EARTH_MU / r).sqrt(); // circular velocity

        let position = DVec3::new(r, 0.0, 0.0);
        let velocity = DVec3::new(0.0, v, 0.0);

        let lvlh = compute_lvlh_frame(position, velocity);

        // Z-hat should point toward planet center (nadir): -r/|r| = [-1, 0, 0]
        let z_hat = DVec3::new(
            lvlh.t_parent_this.col(0).z,
            lvlh.t_parent_this.col(1).z,
            lvlh.t_parent_this.col(2).z,
        );
        assert!((z_hat.x + 1.0).abs() < 1e-14, "Z-hat x: {}", z_hat.x);
        assert!(z_hat.y.abs() < 1e-14, "Z-hat y: {}", z_hat.y);
        assert!(z_hat.z.abs() < 1e-14, "Z-hat z: {}", z_hat.z);

        // Y-hat should be negative orbit normal: -(r×v)/|r×v| = [0, 0, -1]
        let y_hat = DVec3::new(
            lvlh.t_parent_this.col(0).y,
            lvlh.t_parent_this.col(1).y,
            lvlh.t_parent_this.col(2).y,
        );
        assert!(y_hat.x.abs() < 1e-14, "Y-hat x: {}", y_hat.x);
        assert!(y_hat.y.abs() < 1e-14, "Y-hat y: {}", y_hat.y);
        assert!((y_hat.z + 1.0).abs() < 1e-14, "Y-hat z: {}", y_hat.z);

        // X-hat should be along velocity direction: [0, 1, 0]
        let x_hat = DVec3::new(
            lvlh.t_parent_this.col(0).x,
            lvlh.t_parent_this.col(1).x,
            lvlh.t_parent_this.col(2).x,
        );
        assert!(x_hat.x.abs() < 1e-14, "X-hat x: {}", x_hat.x);
        assert!((x_hat.y - 1.0).abs() < 1e-14, "X-hat y: {}", x_hat.y);
        assert!(x_hat.z.abs() < 1e-14, "X-hat z: {}", x_hat.z);

        // Angular velocity should be [0, -n, 0] where n = orbital rate
        let n = (EARTH_MU / (r * r * r)).sqrt();
        assert!(lvlh.ang_vel_this.x.abs() < 1e-14);
        assert!((lvlh.ang_vel_this.y + n).abs() / n < 1e-10);
        assert!(lvlh.ang_vel_this.z.abs() < 1e-14);
    }

    #[test]
    fn transformation_is_orthonormal() {
        let position = DVec3::new(5e6, 3e6, 4e6);
        let velocity = DVec3::new(-2000.0, 5000.0, 3000.0);

        let lvlh = compute_lvlh_frame(position, velocity);
        let t = lvlh.t_parent_this;

        // T * T^T should be identity (orthogonal matrix)
        let product = t * t.transpose();
        let diff = product - DMat3::IDENTITY;
        assert!(diff.x_axis.length() < 1e-14);
        assert!(diff.y_axis.length() < 1e-14);
        assert!(diff.z_axis.length() < 1e-14);

        // Determinant should be +1 (proper rotation)
        assert!((t.determinant() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn inclined_orbit() {
        // ISS-like inclined orbit
        let r = 6_778_137.0;
        let v = (EARTH_MU / r).sqrt();
        let inc = 51.6_f64.to_radians();

        let position = DVec3::new(r, 0.0, 0.0);
        let velocity = DVec3::new(0.0, v * inc.cos(), v * inc.sin());

        let lvlh = compute_lvlh_frame(position, velocity);

        // Z-hat should still point toward center: [-1, 0, 0]
        let z_hat = DVec3::new(
            lvlh.t_parent_this.col(0).z,
            lvlh.t_parent_this.col(1).z,
            lvlh.t_parent_this.col(2).z,
        );
        assert!((z_hat.x + 1.0).abs() < 1e-14);
        assert!(z_hat.y.abs() < 1e-14);
        assert!(z_hat.z.abs() < 1e-14);

        // Angular velocity magnitude should match orbital rate
        let n = (EARTH_MU / (r * r * r)).sqrt();
        assert!((lvlh.ang_vel_this.length() - n).abs() / n < 1e-10);

        // Origin matches vehicle
        assert_eq!(lvlh.position, position);
        assert_eq!(lvlh.velocity, velocity);
    }

    #[test]
    fn half_orbit_frame_rotates() {
        // At position [r, 0, 0], LVLH Z = [-1, 0, 0]
        // At position [0, r, 0], LVLH Z = [0, -1, 0]
        let r = 6_778_137.0;
        let v = (EARTH_MU / r).sqrt();

        let pos2 = DVec3::new(0.0, r, 0.0);
        let vel2 = DVec3::new(-v, 0.0, 0.0);
        let lvlh2 = compute_lvlh_frame(pos2, vel2);

        let z_hat = DVec3::new(
            lvlh2.t_parent_this.col(0).z,
            lvlh2.t_parent_this.col(1).z,
            lvlh2.t_parent_this.col(2).z,
        );
        assert!(z_hat.x.abs() < 1e-14);
        assert!((z_hat.y + 1.0).abs() < 1e-14);
        assert!(z_hat.z.abs() < 1e-14);
    }
}
