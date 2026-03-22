use glam::{DMat3, DVec3};
use jeod_dynamics::GravityAcceleration;

use crate::source::{GravityModel, GravitySource};

/// Compute point-mass gravitational acceleration, gradient, and potential.
/// Position is the body's position relative to the gravity source center.
pub fn compute_point_mass_gravity(mu: f64, position: DVec3) -> GravityAcceleration {
    let r_sq = position.length_squared();
    debug_assert!(
        r_sq > 0.0,
        "position must be non-zero (body is at gravity source center)"
    );
    let r_mag = r_sq.sqrt();
    let r_3rd = r_sq * r_mag;

    // Acceleration: a = -mu/r^3 * r
    let accel = position * (-mu / r_3rd);

    // Potential: V = -mu/r
    let potential = -mu / r_mag;

    // Gradient tensor: G[i][j] = mu/r^5 * (3*r[i]*r[j] - delta_ij*r^2)
    let r_5th = r_3rd * r_sq;
    let mu_r5 = mu / r_5th;
    // Build as columns for glam's column-major DMat3
    let gradient = DMat3::from_cols(
        DVec3::new(
            mu_r5 * (3.0 * position.x * position.x - r_sq),
            mu_r5 * 3.0 * position.y * position.x,
            mu_r5 * 3.0 * position.z * position.x,
        ),
        DVec3::new(
            mu_r5 * 3.0 * position.x * position.y,
            mu_r5 * (3.0 * position.y * position.y - r_sq),
            mu_r5 * 3.0 * position.z * position.y,
        ),
        DVec3::new(
            mu_r5 * 3.0 * position.x * position.z,
            mu_r5 * 3.0 * position.y * position.z,
            mu_r5 * (3.0 * position.z * position.z - r_sq),
        ),
    );

    GravityAcceleration {
        accel,
        gradient,
        potential,
    }
}

/// Dispatch gravity computation based on model type.
pub fn compute_gravity(source: &GravitySource, position: DVec3) -> GravityAcceleration {
    match &source.model {
        GravityModel::PointMass => compute_point_mass_gravity(source.mu, position),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EARTH_MU: f64 = 3.986_004_418e14; // m^3/s^2
    const EARTH_RADIUS: f64 = 6_378_137.0; // m

    #[test]
    fn earth_surface_gravity_magnitude() {
        // At Earth's surface along x-axis
        let pos = DVec3::new(EARTH_RADIUS, 0.0, 0.0);
        let result = compute_point_mass_gravity(EARTH_MU, pos);
        let accel_mag = result.accel.length();
        // Should be approximately 9.80 m/s^2
        assert!(
            (accel_mag - 9.80).abs() < 0.1,
            "Earth surface gravity: expected ~9.80 m/s^2, got {} m/s^2",
            accel_mag
        );
    }

    #[test]
    fn inverse_square_law() {
        let mu = 1e14;
        let pos1 = DVec3::new(1e7, 0.0, 0.0);
        let pos2 = DVec3::new(2e7, 0.0, 0.0); // double distance

        let result1 = compute_point_mass_gravity(mu, pos1);
        let result2 = compute_point_mass_gravity(mu, pos2);

        let a1 = result1.accel.length();
        let a2 = result2.accel.length();

        // Double distance -> accel reduces by factor 4
        let ratio = a1 / a2;
        assert!(
            (ratio - 4.0).abs() < 1e-14,
            "Inverse-square ratio: expected 4.0, got {}",
            ratio
        );
    }

    #[test]
    fn gradient_symmetry() {
        // Test with an off-axis position so all gradient components are non-zero
        let mu = 3.986e14;
        let pos = DVec3::new(4e6, 3e6, 5e6);
        let result = compute_point_mass_gravity(mu, pos);
        let g = result.gradient;

        // G[i][j] == G[j][i]
        // glam: col(j)[i] is element (i,j) in row-col notation
        let tol = 1e-10;
        assert!(
            (g.col(1)[0] - g.col(0)[1]).abs() < tol,
            "G[0][1]={} != G[1][0]={}",
            g.col(1)[0],
            g.col(0)[1]
        );
        assert!(
            (g.col(2)[0] - g.col(0)[2]).abs() < tol,
            "G[0][2]={} != G[2][0]={}",
            g.col(2)[0],
            g.col(0)[2]
        );
        assert!(
            (g.col(2)[1] - g.col(1)[2]).abs() < tol,
            "G[1][2]={} != G[2][1]={}",
            g.col(2)[1],
            g.col(1)[2]
        );
    }

    #[test]
    fn gradient_trace_is_zero() {
        // Laplace's equation: trace of gravity gradient = 0 outside the body
        let mu = 3.986e14;
        let pos = DVec3::new(4e6, 3e6, 5e6);
        let result = compute_point_mass_gravity(mu, pos);
        let g = result.gradient;

        // trace = G[0][0] + G[1][1] + G[2][2]
        let trace = g.col(0)[0] + g.col(1)[1] + g.col(2)[2];
        assert!(
            trace.abs() < 1e-10,
            "Gradient trace should be ~0 (Laplace), got {}",
            trace
        );
    }

    #[test]
    fn potential_at_known_distance() {
        let mu = 3.986e14;
        let r = 7e6;
        let pos = DVec3::new(r, 0.0, 0.0);
        let result = compute_point_mass_gravity(mu, pos);

        let expected_potential = -mu / r;
        assert!(
            (result.potential - expected_potential).abs() < 1e-6,
            "Potential: expected {}, got {}",
            expected_potential,
            result.potential
        );
    }

    #[test]
    fn acceleration_direction_opposite_to_position() {
        let mu = 1e14;
        let pos = DVec3::new(3e6, 4e6, 5e6);
        let result = compute_point_mass_gravity(mu, pos);

        // Acceleration should point opposite to position (toward the source)
        let accel_dir = result.accel.normalize();
        let pos_dir = pos.normalize();

        // accel_dir should be -pos_dir
        let sum = accel_dir + pos_dir;
        assert!(
            sum.length() < 1e-14,
            "Acceleration should be opposite to position: accel_dir={:?}, pos_dir={:?}",
            accel_dir,
            pos_dir
        );
    }

    #[test]
    fn compute_gravity_dispatches_point_mass() {
        let source = GravitySource {
            mu: EARTH_MU,
            model: GravityModel::PointMass,
        };
        let pos = DVec3::new(EARTH_RADIUS, 0.0, 0.0);
        let result = compute_gravity(&source, pos);
        let direct = compute_point_mass_gravity(EARTH_MU, pos);

        assert_eq!(result.accel, direct.accel);
        assert_eq!(result.potential, direct.potential);
        assert_eq!(result.gradient, direct.gradient);
    }

    #[test]
    fn earth_surface_gravity_off_axis() {
        // Test at a 45-degree position to verify off-axis consistency
        let r = EARTH_RADIUS;
        let component = r / 3.0_f64.sqrt();
        let pos = DVec3::new(component, component, component);
        let result = compute_point_mass_gravity(EARTH_MU, pos);
        let accel_mag = result.accel.length();
        assert!(
            (accel_mag - 9.80).abs() < 0.1,
            "Off-axis Earth surface gravity: expected ~9.80, got {}",
            accel_mag
        );
    }

    #[test]
    fn gradient_at_earth_surface() {
        // The gradient magnitude at Earth surface should be on the order of
        // mu/r^3 ~ 3.986e14 / (6.378e6)^3 ~ 1.5e-6 /s^2
        let pos = DVec3::new(EARTH_RADIUS, 0.0, 0.0);
        let result = compute_point_mass_gravity(EARTH_MU, pos);
        let g = result.gradient;

        // For position along x-axis: G[0][0] = mu/r^5 * (3*x^2 - r^2) = mu/r^5 * 2*r^2 = 2*mu/r^3
        let expected_g00 = 2.0 * EARTH_MU / (EARTH_RADIUS * EARTH_RADIUS * EARTH_RADIUS);
        assert!(
            (g.col(0)[0] - expected_g00).abs() / expected_g00.abs() < 1e-10,
            "G[0][0]: expected {}, got {}",
            expected_g00,
            g.col(0)[0]
        );

        // Off-diagonal should be zero for on-axis position
        assert!(
            g.col(1)[0].abs() < 1e-20,
            "G[0][1] should be 0 for on-axis, got {}",
            g.col(1)[0]
        );
    }
}
