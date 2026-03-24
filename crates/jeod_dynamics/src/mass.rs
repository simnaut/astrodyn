use glam::{DMat3, DVec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MassProperties {
    pub mass: f64,              // kg
    pub inertia: DMat3,         // kg*m^2, in body frame
    pub inverse_inertia: DMat3, // precomputed I^-1
    pub position: DVec3,  // m, in structural frame
}

impl MassProperties {
    /// Create mass properties for a point mass (unit sphere inertia: I = m * I_{3x3}).
    ///
    /// Phase 1 placeholder. When rotational dynamics are added in Phase 2,
    /// callers must specify the actual inertia tensor for their geometry.
    pub fn new(mass: f64) -> Self {
        assert!(mass > 0.0, "mass must be positive, got {mass}");
        Self {
            mass,
            inertia: DMat3::IDENTITY * mass,
            inverse_inertia: DMat3::IDENTITY / mass,
            position: DVec3::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_mass_inertia() {
        let mp = MassProperties::new(10.0);
        assert_eq!(mp.mass, 10.0);
        assert_eq!(mp.inertia, DMat3::IDENTITY * 10.0);
        assert_eq!(mp.inverse_inertia, DMat3::IDENTITY / 10.0);
        assert_eq!(mp.position, DVec3::ZERO);
    }

    #[test]
    fn inertia_times_inverse_is_identity() {
        let mp = MassProperties::new(42.0);
        let product = mp.inertia * mp.inverse_inertia;
        let diff = product - DMat3::IDENTITY;
        // Check all 9 elements are near zero
        assert!(diff.x_axis.length() < 1e-12);
        assert!(diff.y_axis.length() < 1e-12);
        assert!(diff.z_axis.length() < 1e-12);
    }
}
