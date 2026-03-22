use glam::{DMat3, DVec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MassProperties {
    pub mass: f64,              // kg
    pub inertia: DMat3,         // kg*m^2, in body frame
    pub inertia_inverse: DMat3, // precomputed I^-1
    pub center_of_mass: DVec3,  // m, in structural frame
}

impl MassProperties {
    pub fn new(mass: f64) -> Self {
        // Simple point mass: identity-scaled inertia (placeholder for Phase 1)
        Self {
            mass,
            inertia: DMat3::IDENTITY * mass,
            inertia_inverse: DMat3::IDENTITY / mass,
            center_of_mass: DVec3::ZERO,
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
        assert_eq!(mp.inertia_inverse, DMat3::IDENTITY / 10.0);
        assert_eq!(mp.center_of_mass, DVec3::ZERO);
    }

    #[test]
    fn inertia_times_inverse_is_identity() {
        let mp = MassProperties::new(42.0);
        let product = mp.inertia * mp.inertia_inverse;
        let diff = product - DMat3::IDENTITY;
        // Check all 9 elements are near zero
        assert!(diff.x_axis.length() < 1e-12);
        assert!(diff.y_axis.length() < 1e-12);
        assert!(diff.z_axis.length() < 1e-12);
    }
}
