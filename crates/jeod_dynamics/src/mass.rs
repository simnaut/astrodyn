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
    /// **Warning:** The placeholder inertia `I = m * I_{3x3}` is only valid for
    /// translational dynamics. It will produce **wrong results** for rotational
    /// dynamics because real spacecraft have non-spherical inertia tensors with
    /// distinct principal moments (I_xx != I_yy != I_zz) and potentially
    /// non-zero products of inertia. When rotational dynamics are enabled,
    /// callers must specify the actual inertia tensor for their geometry.
    // JEOD_INV: MA.02 — mass > 0 for meaningful dynamics
    pub fn new(mass: f64) -> Self {
        assert!(mass > 0.0, "mass must be positive, got {mass}");
        Self {
            mass,
            inertia: DMat3::IDENTITY * mass,
            inverse_inertia: DMat3::IDENTITY / mass,
            position: DVec3::ZERO,
        }
    }

    /// Create mass properties with explicit inertia tensor and center-of-mass position.
    ///
    /// The inertia tensor is about the body frame axes through the center of mass.
    /// The position is the center of mass in the structural frame.
    // JEOD_INV: MA.02 — mass > 0 for meaningful dynamics
    // JEOD_INV: MA.05 — JEOD computes inverse inertia only for root bodies; we compute for all (structural divergence)
    // JEOD_INV: DB.23 — compute_inverse_inertia enabled (always computed here)
    // JEOD_INV: MA.04 — inverse_inertia consistent with inertia (computed from inertia)
    pub fn with_inertia(mass: f64, inertia: DMat3, position: DVec3) -> Self {
        assert!(mass > 0.0, "mass must be positive, got {mass}");
        let det = inertia.determinant();
        assert!(
            det.abs() > 1e-30,
            "inertia tensor is singular or near-singular (det={det:.2e}); \
             inverse will produce inf/NaN"
        );
        let inverse_inertia = inertia.inverse();
        Self {
            mass,
            inertia,
            inverse_inertia,
            position,
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
