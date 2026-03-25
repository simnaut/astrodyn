//! Solar beta angle computation.
//!
//! The solar beta angle is the angle between the orbital plane and the
//! Sun direction vector. It determines the eclipse geometry and thermal
//! environment of a satellite.
//!
//! β = π/2 - acos(ĥ · ŝ)
//!
//! where ĥ is the orbit normal unit vector and ŝ is the Sun direction unit vector.
//! When β = 0, the Sun is in the orbital plane.
//! When β = ±90°, the Sun is perpendicular to the orbital plane.

use glam::DVec3;

/// Compute the solar beta angle.
///
/// # Arguments
/// * `orbit_ang_momentum` - Orbital angular momentum vector (r × v), does not need to be unit
/// * `sun_direction` - Direction vector toward the Sun, does not need to be unit
///
/// # Returns
/// Solar beta angle in radians, in range [-π/2, π/2].
/// Positive when the Sun is on the same side as the angular momentum vector.
pub fn solar_beta_angle(orbit_ang_momentum: DVec3, sun_direction: DVec3) -> f64 {
    let h_hat = orbit_ang_momentum.normalize();
    let s_hat = sun_direction.normalize();

    // β = π/2 - acos(h_hat · s_hat)
    // Equivalently: β = asin(h_hat · s_hat)
    h_hat.dot(s_hat).asin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn sun_in_orbit_plane() {
        // Sun direction lies in the orbital plane (perpendicular to h)
        let h = DVec3::new(0.0, 0.0, 1.0); // orbit normal = +Z
        let sun = DVec3::new(1.0, 0.0, 0.0); // Sun in +X (in plane)
        let beta = solar_beta_angle(h, sun);
        assert!(beta.abs() < 1e-15);
    }

    #[test]
    fn sun_perpendicular_to_orbit_plane() {
        // Sun direction is along the orbit normal
        let h = DVec3::new(0.0, 0.0, 1.0);
        let sun = DVec3::new(0.0, 0.0, 1.0);
        let beta = solar_beta_angle(h, sun);
        assert!((beta - PI / 2.0).abs() < 1e-15);
    }

    #[test]
    fn sun_opposite_to_orbit_normal() {
        let h = DVec3::new(0.0, 0.0, 1.0);
        let sun = DVec3::new(0.0, 0.0, -1.0);
        let beta = solar_beta_angle(h, sun);
        assert!((beta + PI / 2.0).abs() < 1e-15);
    }

    #[test]
    fn forty_five_degree_beta() {
        let h = DVec3::new(0.0, 0.0, 1.0);
        let sun = DVec3::new(1.0, 0.0, 1.0); // 45 degrees from plane
        let beta = solar_beta_angle(h, sun);
        assert!((beta - PI / 4.0).abs() < 1e-14);
    }

    #[test]
    fn unnormalized_inputs() {
        // Inputs don't need to be unit vectors
        let h = DVec3::new(0.0, 0.0, 1e10);
        let sun = DVec3::new(1e8, 0.0, 0.0);
        let beta = solar_beta_angle(h, sun);
        assert!(beta.abs() < 1e-14);
    }

    #[test]
    fn iss_like_orbit() {
        // ISS orbit normal (inclined ~51.6 degrees from ecliptic)
        let inc = 51.6_f64.to_radians();
        let h = DVec3::new(0.0, -inc.sin(), inc.cos());
        // Sun approximately in ecliptic plane along +X
        let sun = DVec3::new(1.0, 0.0, 0.0);
        let beta = solar_beta_angle(h, sun);
        // Beta should be small (sun nearly in orbit plane at equinox)
        assert!(beta.abs() < 1e-14);
    }
}
