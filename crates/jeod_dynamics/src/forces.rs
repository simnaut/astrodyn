use glam::{DMat3, DVec3};

/// Gravitational acceleration, gradient tensor, and potential for a body.
///
/// Computed by the gravity subsystem and consumed by the dynamics integrator.
/// All quantities are expressed in the integration frame (typically J2000 ECI).
///
/// # Sign conventions
/// - `grav_accel`: gravitational acceleration in m/s^2. Points toward the
///   attracting body (negative radial direction for a single point mass).
/// - `grav_grad`: gravity gradient tensor in 1/s^2. Symmetric 3x3 matrix;
///   trace is zero outside the attracting body (Laplace's equation).
/// - `grav_pot`: gravitational potential in m^2/s^2. Convention: **+mu/r**
///   for point mass (positive, matching JEOD `gravity_controls.cc`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GravityAcceleration {
    /// Gravitational acceleration in m/s^2, in integration frame.
    pub grav_accel: DVec3,
    /// Gravity gradient tensor in 1/s^2. Symmetric; trace = 0 outside body.
    pub grav_grad: DMat3,
    /// Gravitational potential in m^2/s^2. Convention: +mu/r for point mass (JEOD).
    pub grav_pot: f64,
}

impl Default for GravityAcceleration {
    fn default() -> Self {
        Self {
            grav_accel: DVec3::ZERO,
            grav_grad: DMat3::ZERO,
            grav_pot: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TotalForce {
    pub force: DVec3,  // N, in integration frame
    pub torque: DVec3, // N*m, in body frame
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FrameDerivatives {
    pub trans_accel: DVec3, // m/s^2
    pub rot_accel: DVec3,   // rad/s^2
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicsConfig {
    pub translational_dynamics: bool,
    pub rotational_dynamics: bool,
    pub three_dof: bool,
}

impl Default for DynamicsConfig {
    fn default() -> Self {
        Self {
            translational_dynamics: true,
            rotational_dynamics: false,
            three_dof: true,
        }
    }
}

// JEOD_INV: DB.18 — inverse_mass used for F=ma (precomputed)
pub fn compute_translational_acceleration(force: DVec3, mass: f64) -> DVec3 {
    assert!(mass > 0.0, "mass must be positive for F=ma, got {}", mass);
    force / mass
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_gravity_acceleration() {
        let ga = GravityAcceleration::default();
        assert_eq!(ga.grav_accel, DVec3::ZERO);
        assert_eq!(ga.grav_grad, DMat3::ZERO);
        assert_eq!(ga.grav_pot, 0.0);
    }

    #[test]
    fn default_total_force() {
        let tf = TotalForce::default();
        assert_eq!(tf.force, DVec3::ZERO);
        assert_eq!(tf.torque, DVec3::ZERO);
    }

    #[test]
    fn default_frame_derivatives() {
        let fd = FrameDerivatives::default();
        assert_eq!(fd.trans_accel, DVec3::ZERO);
        assert_eq!(fd.rot_accel, DVec3::ZERO);
    }

    #[test]
    fn default_dynamics_config() {
        let dc = DynamicsConfig::default();
        assert!(dc.translational_dynamics);
        assert!(!dc.rotational_dynamics);
        assert!(dc.three_dof);
    }

    #[test]
    fn translational_acceleration_basic() {
        let force = DVec3::new(10.0, 20.0, 30.0);
        let mass = 5.0;
        let accel = compute_translational_acceleration(force, mass);
        assert_eq!(accel, DVec3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn translational_acceleration_unit_mass() {
        let force = DVec3::new(3.0, -1.0, 7.0);
        let accel = compute_translational_acceleration(force, 1.0);
        assert_eq!(accel, force);
    }
}
