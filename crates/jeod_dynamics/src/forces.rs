use glam::{DMat3, DVec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GravityAcceleration {
    pub accel: DVec3,    // m/s^2, in integration frame
    pub gradient: DMat3, // 1/s^2, tidal gradient tensor
    pub potential: f64,  // m^2/s^2
}

impl Default for GravityAcceleration {
    fn default() -> Self {
        Self {
            accel: DVec3::ZERO,
            gradient: DMat3::ZERO,
            potential: 0.0,
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
    pub translational: bool,
    pub rotational: bool,
    pub three_dof: bool,
}

impl Default for DynamicsConfig {
    fn default() -> Self {
        Self {
            translational: true,
            rotational: false,
            three_dof: true,
        }
    }
}

pub fn compute_translational_acceleration(force: DVec3, mass: f64) -> DVec3 {
    force / mass
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_gravity_acceleration() {
        let ga = GravityAcceleration::default();
        assert_eq!(ga.accel, DVec3::ZERO);
        assert_eq!(ga.gradient, DMat3::ZERO);
        assert_eq!(ga.potential, 0.0);
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
        assert!(dc.translational);
        assert!(!dc.rotational);
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
