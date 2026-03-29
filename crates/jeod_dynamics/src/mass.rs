use glam::{DMat3, DVec3};

/// Default tolerance for [`MassProperties::validate_consistency`].
///
/// Checks that `I * I^-1` is within this tolerance of the identity matrix.
/// Matches the precision expected from `DMat3::inverse()` for typical
/// spacecraft inertia tensors (principal moments ~1–10000 kg*m^2).
pub const INERTIA_CONSISTENCY_TOL: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MassProperties {
    pub mass: f64,              // kg
    pub inverse_mass: f64,      // 1/kg, precomputed (matches JEOD MassPointState.inverse_mass)
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
            inverse_mass: 1.0 / mass,
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
            inverse_mass: 1.0 / mass,
            inertia,
            inverse_inertia,
            position,
        }
    }

    /// Recompute `inverse_mass` and `inverse_inertia` from `mass` and `inertia`.
    ///
    /// Port of the recomputation logic in JEOD's `MassBody::update_mass_properties()`
    /// (`mass_update.cc` lines 62-68, 118-124). JEOD runs this every timestep
    /// at the dynamics rate to pick up runtime mass changes (fuel burn, staging,
    /// attach/detach).
    ///
    /// Call this after modifying `mass` or `inertia` directly on the struct.
    /// Constructors (`new`, `with_inertia`) call this implicitly.
    ///
    /// # Panics
    /// Panics if `mass <= 0` or `inertia` is singular.
    // JEOD_INV: MA.04 — inverse_inertia consistent with inertia (recomputed from inertia)
    // JEOD_INV: MA.07 — derived quantities recomputed after mutation
    pub fn recompute_derived(&mut self) {
        assert!(self.mass > 0.0, "mass must be positive, got {}", self.mass);
        self.inverse_mass = 1.0 / self.mass;

        let det = self.inertia.determinant();
        assert!(
            det.abs() > 1e-30,
            "inertia tensor is singular or near-singular (det={det:.2e}); \
             inverse will produce inf/NaN"
        );
        self.inverse_inertia = self.inertia.inverse();
    }

    /// Validate that `inertia` and `inverse_inertia` are consistent.
    ///
    /// In JEOD, `inverse_inertia` is always recomputed from `inertia` (via
    /// `compute_inverse_inertia()`), so they are guaranteed consistent. In ECS
    /// both fields are public, so external code could set them independently.
    /// This method checks that `I * I^-1 ≈ identity` to the given tolerance.
    ///
    /// # Panics
    /// Panics if `I * I^-1` deviates from identity by more than `tol`.
    // JEOD_INV: DB.19 — inverse_inertia used for Euler equation (validated I*I^-1 ≈ identity)
    // JEOD_INV: MA.04 — inverse_inertia consistent with inertia
    pub fn validate_consistency(&self, tol: f64) {
        let product = self.inertia * self.inverse_inertia;
        assert!(
            (product - DMat3::IDENTITY).abs_diff_eq(DMat3::ZERO, tol),
            "MassProperties: inertia and inverse_inertia are inconsistent \
             (I * I^-1 != identity to {tol:.0e}). In JEOD, inverse_inertia \
             is always recomputed from inertia. Use MassProperties::with_inertia() \
             which computes the inverse automatically."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_mass_inertia() {
        let mp = MassProperties::new(10.0);
        assert_eq!(mp.mass, 10.0);
        assert_eq!(mp.inverse_mass, 0.1);
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

    #[test]
    fn validate_consistency_passes_for_consistent() {
        let mp = MassProperties::with_inertia(
            10.0,
            DMat3::from_diagonal(DVec3::new(100.0, 200.0, 300.0)),
            DVec3::ZERO,
        );
        mp.validate_consistency(1e-6); // should not panic
    }

    #[test]
    #[should_panic(expected = "inconsistent")]
    fn validate_consistency_fails_for_wrong_inverse() {
        let mut mp = MassProperties::with_inertia(
            10.0,
            DMat3::from_diagonal(DVec3::new(100.0, 200.0, 300.0)),
            DVec3::ZERO,
        );
        // Corrupt the inverse
        mp.inverse_inertia = DMat3::IDENTITY;
        mp.validate_consistency(1e-6);
    }

    #[test]
    fn recompute_derived_after_mass_change() {
        let mut mp = MassProperties::new(10.0);
        assert_eq!(mp.inverse_mass, 0.1);

        // Simulate fuel burn: mass decreases
        mp.mass = 8.0;
        // inverse_mass is now stale (still 0.1)
        assert_eq!(mp.inverse_mass, 0.1);

        mp.recompute_derived();
        assert!((mp.inverse_mass - 0.125).abs() < 1e-15);
        assert!((mp.mass * mp.inverse_mass - 1.0).abs() < 1e-15);
    }

    #[test]
    fn recompute_derived_after_inertia_change() {
        let mut mp = MassProperties::with_inertia(
            10.0,
            DMat3::from_diagonal(DVec3::new(100.0, 200.0, 300.0)),
            DVec3::ZERO,
        );

        // Change inertia (e.g., fuel redistribution)
        mp.inertia = DMat3::from_diagonal(DVec3::new(50.0, 100.0, 150.0));
        // inverse_inertia is now stale
        mp.recompute_derived();

        // Verify consistency
        mp.validate_consistency(1e-6);
        assert!((mp.inverse_mass - 0.1).abs() < 1e-15);
    }
}
