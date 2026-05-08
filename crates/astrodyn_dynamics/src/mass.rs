//! [`MassProperties`] and the typed sibling [`MassPropertiesTyped`] —
//! mass, inertia tensor, and CoM offset for a rigid body.
//!
//! Ports
//! [`models/dynamics/mass/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/dynamics/mass/)
//! from JEOD v5.4.0. Inertia is stored about the body-frame axes
//! through the centre of mass; composing child masses into a parent
//! applies the parallel-axis (Steiner) theorem.

use core::marker::PhantomData;

use astrodyn_quantities::aliases::{InertiaTensor, Position};
use astrodyn_quantities::frame::{BodyFrame, StructuralFrame, Vehicle};
use glam::{DMat3, DVec3};
use uom::si::f64::Mass;
use uom::si::mass::kilogram;

/// Default tolerance for [`MassProperties::validate_consistency`].
///
/// Checks that `I * I^-1` is within this tolerance of the identity matrix.
/// Matches the precision expected from `DMat3::inverse()` for typical
/// spacecraft inertia tensors (principal moments ~1–10000 kg*m^2).
pub const INERTIA_CONSISTENCY_TOL: f64 = 1e-6;

/// Rigid-body mass / inertia / CoM-offset block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MassProperties {
    /// Total mass in kg.
    pub mass: f64,
    /// Pre-computed inverse mass (`1/mass`, in `1/kg`). Mirrors JEOD's
    /// `MassPointState::inverse_mass` so the inner loop is a multiply.
    pub inverse_mass: f64,
    /// Inertia tensor about the body-frame axes through the centre of
    /// mass (kg·m²).
    pub inertia: DMat3,
    /// Pre-computed inverse inertia tensor.
    pub inverse_inertia: DMat3,
    /// Centre-of-mass position relative to the structural-frame
    /// origin, in metres.
    pub position: DVec3,
    /// Rotation matrix from the structural frame to the body frame,
    /// matching JEOD `MassPointState::T_parent_this` for the
    /// composite-body point. Defaults to `IDENTITY` (struct = body), which
    /// is the right answer for any body whose `pt_orientation` was set to
    /// identity in JEOD's `Modified_data/mass/*.py`. Bodies with a
    /// non-identity orientation (e.g. SIM_Apollo's CM/LES/DM/Ascent
    /// modules each declare a 180° eigen-rotation about Z) must set this
    /// explicitly, otherwise the attach-algorithm conversion of
    /// struct-frame quantities to inertial picks up the wrong rotation.
    pub t_parent_this: DMat3,
    /// Set to `true` after mutating `mass` or `inertia` to trigger
    /// recomputation of `inverse_mass` and `inverse_inertia` on the next
    /// call to [`Self::recompute_derived`]. Constructors leave this `false`
    /// (derived quantities are already computed).
    pub dirty: bool,
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
            t_parent_this: DMat3::IDENTITY,
            dirty: false,
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
            t_parent_this: DMat3::IDENTITY,
            dirty: false,
        }
    }

    /// Builder: set the struct→body rotation. See the [`Self::t_parent_this`]
    /// field doc-comment for when this is needed.
    pub fn with_t_parent_this(mut self, t_parent_this: DMat3) -> Self {
        self.t_parent_this = t_parent_this;
        self
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
    // JEOD_INV: MA.03 — inverse_mass consistent with mass (recomputed as 1/mass)
    // JEOD_INV: MA.04 — inverse_inertia consistent with inertia (recomputed from inertia)
    // JEOD_INV: MA.07 — derived quantities recomputed after mutation
    pub fn recompute_derived(&mut self) {
        if !self.dirty {
            return;
        }
        self.dirty = false;

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
             when constructing, or call MassProperties::recompute_derived() after \
             mutating mass/inertia."
        );
    }
}

/// Typed sibling of [`MassProperties`] parameterized by a vehicle marker
/// `V`. Mass becomes a `uom::si::f64::Mass`, inertia is wrapped in
/// [`InertiaTensor<BodyFrame<V>>`], and the center-of-mass position
/// carries the `StructuralFrame<V>` phantom tag.
///
/// `inverse_mass` and `inverse_inertia` remain untyped (`f64` and
/// `DMat3`): they are the integrator-hot-path caches for `F = m·a`
/// resolution. The dimension is `1/M` and `1/(M·L²)` respectively —
/// `astrodyn_quantities` does not expose a typed `Inverse<Mass>` alias and
/// adding one would be churn for no enforced invariant beyond the f64
/// path's own unit consistency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MassPropertiesTyped<V: Vehicle> {
    /// Total mass.
    pub mass: Mass,
    /// Precomputed `1 / mass` in `kg⁻¹` (caller maintains consistency
    /// via [`Self::recompute_derived`]).
    pub inverse_mass: f64,
    /// Inertia tensor about the body-frame axes through the center of mass.
    pub inertia: InertiaTensor<BodyFrame<V>>,
    /// Precomputed `inertia⁻¹`. Maintained consistent with `inertia` via
    /// [`Self::recompute_derived`].
    pub inverse_inertia: DMat3,
    /// Center of mass in the structural frame.
    pub center_of_mass: Position<StructuralFrame<V>>,
    /// See [`MassProperties::dirty`].
    pub dirty: bool,
    _v: PhantomData<V>,
}

impl<V: Vehicle> MassPropertiesTyped<V> {
    /// Point-mass constructor with placeholder spherical inertia
    /// (`I = m · I_{3×3}`) — see [`MassProperties::new`] for the same
    /// caveat about translational-only validity.
    // JEOD_INV: MA.02 — mass > 0 for meaningful dynamics
    pub fn new(mass: Mass) -> Self {
        let m = mass.get::<kilogram>();
        assert!(m > 0.0, "mass must be positive, got {m}");
        Self {
            mass,
            inverse_mass: 1.0 / m,
            inertia: InertiaTensor::<BodyFrame<V>>::from_dmat3_unchecked(DMat3::IDENTITY * m),
            inverse_inertia: DMat3::IDENTITY / m,
            center_of_mass: Position::<StructuralFrame<V>>::zero(),
            dirty: false,
            _v: PhantomData,
        }
    }

    /// Constructor with explicit inertia and center-of-mass position.
    // JEOD_INV: MA.02 — mass > 0 for meaningful dynamics
    // JEOD_INV: MA.04 — inverse_inertia computed from inertia
    // JEOD_INV: DB.23 — inverse_inertia always computed
    pub fn with_inertia(
        mass: Mass,
        inertia: InertiaTensor<BodyFrame<V>>,
        center_of_mass: Position<StructuralFrame<V>>,
    ) -> Self {
        let m = mass.get::<kilogram>();
        assert!(m > 0.0, "mass must be positive, got {m}");
        let inertia_dmat = inertia.as_dmat3();
        let det = inertia_dmat.determinant();
        assert!(
            det.abs() > 1e-30,
            "inertia tensor is singular or near-singular (det={det:.2e}); \
             inverse will produce inf/NaN"
        );
        Self {
            mass,
            inverse_mass: 1.0 / m,
            inertia,
            inverse_inertia: inertia_dmat.inverse(),
            center_of_mass,
            dirty: false,
            _v: PhantomData,
        }
    }

    /// Drop the phantoms and emit the untyped storage form. Numeric
    /// values (kg, kg·m², m) are preserved exactly.
    pub fn to_untyped(&self) -> MassProperties {
        MassProperties {
            mass: self.mass.get::<kilogram>(),
            inverse_mass: self.inverse_mass,
            inertia: self.inertia.as_dmat3(),
            inverse_inertia: self.inverse_inertia,
            position: self.center_of_mass.raw_si(),
            t_parent_this: DMat3::IDENTITY,
            dirty: self.dirty,
        }
    }

    /// Wrap an untyped [`MassProperties`] as typed for vehicle `V`.
    /// **The caller asserts** the inertia is in `BodyFrame<V>` and the
    /// position is in `StructuralFrame<V>`.
    pub fn from_untyped_unchecked(mp: &MassProperties) -> Self {
        Self {
            mass: Mass::new::<kilogram>(mp.mass),
            inverse_mass: mp.inverse_mass,
            inertia: InertiaTensor::<BodyFrame<V>>::from_dmat3_unchecked(mp.inertia),
            inverse_inertia: mp.inverse_inertia,
            center_of_mass: Position::<StructuralFrame<V>>::from_raw_si(mp.position),
            dirty: mp.dirty,
            _v: PhantomData,
        }
    }

    /// Recompute `inverse_mass` and `inverse_inertia`.
    // JEOD_INV: MA.03 — inverse_mass = 1/mass (recomputed)
    // JEOD_INV: MA.04 — inverse_inertia consistent with inertia (recomputed)
    // JEOD_INV: MA.07 — derived quantities recomputed after mutation
    pub fn recompute_derived(&mut self) {
        if !self.dirty {
            return;
        }
        self.dirty = false;
        let m = self.mass.get::<kilogram>();
        assert!(m > 0.0, "mass must be positive, got {m}");
        self.inverse_mass = 1.0 / m;
        let inertia_dmat = self.inertia.as_dmat3();
        let det = inertia_dmat.determinant();
        assert!(
            det.abs() > 1e-30,
            "inertia tensor is singular or near-singular (det={det:.2e}); \
             inverse will produce inf/NaN"
        );
        self.inverse_inertia = inertia_dmat.inverse();
    }

    /// JEOD MA.04 invariant check: `inertia · inverse_inertia ≈ I`.
    // JEOD_INV: MA.04 — inverse_inertia consistent with inertia
    pub fn validate_consistency(&self, tol: f64) {
        let product = self.inertia.as_dmat3() * self.inverse_inertia;
        assert!(
            (product - DMat3::IDENTITY).abs_diff_eq(DMat3::ZERO, tol),
            "MassPropertiesTyped: inertia and inverse_inertia inconsistent \
             (I·I⁻¹ != identity to {tol:.0e})"
        );
    }
}

impl<V: Vehicle> From<MassProperties> for MassPropertiesTyped<V> {
    /// Lift an untyped [`MassProperties`] into the typed form,
    /// asserting the caller's vehicle phantom `V`. Documented
    /// test/scenario / BodyAction-payload boundary.
    #[inline]
    fn from(mp: MassProperties) -> Self {
        Self::from_untyped_unchecked(&mp)
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
        mp.dirty = true;
        // inverse_mass is now stale (still 0.1)
        assert_eq!(mp.inverse_mass, 0.1);

        mp.recompute_derived();
        assert!((mp.inverse_mass - 0.125).abs() < 1e-15);
        assert!((mp.mass * mp.inverse_mass - 1.0).abs() < 1e-15);
        assert!(!mp.dirty);
    }

    #[test]
    fn recompute_derived_skips_when_clean() {
        let mut mp = MassProperties::new(10.0);
        assert!(!mp.dirty);
        // recompute_derived is a no-op when clean
        mp.recompute_derived();
        assert_eq!(mp.inverse_mass, 0.1);
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
        mp.dirty = true;
        // inverse_inertia is now stale
        mp.recompute_derived();

        // Verify consistency
        mp.validate_consistency(1e-6);
        assert!((mp.inverse_mass - 0.1).abs() < 1e-15);
    }

    // ---- typed MassPropertiesTyped<V> ----------------------------------

    #[test]
    fn typed_point_mass_round_trips_to_untyped() {
        use astrodyn_quantities::frame::TestVehicle;

        let typed = MassPropertiesTyped::<TestVehicle>::new(Mass::new::<kilogram>(10.0));
        let untyped = typed.to_untyped();

        assert_eq!(untyped.mass, 10.0);
        assert_eq!(untyped.inverse_mass, 0.1);
        assert_eq!(untyped.inertia, DMat3::IDENTITY * 10.0);
        assert_eq!(untyped.inverse_inertia, DMat3::IDENTITY / 10.0);
        assert_eq!(untyped.position, DVec3::ZERO);
    }

    #[test]
    fn typed_with_inertia_matches_untyped() {
        use astrodyn_quantities::frame::TestVehicle;

        let m = 5.0;
        let i = DMat3::from_diagonal(DVec3::new(50.0, 60.0, 70.0));
        let pos = DVec3::new(0.1, 0.2, 0.3);

        let typed = MassPropertiesTyped::<TestVehicle>::with_inertia(
            Mass::new::<kilogram>(m),
            InertiaTensor::<BodyFrame<TestVehicle>>::from_dmat3_unchecked(i),
            Position::<StructuralFrame<TestVehicle>>::from_raw_si(pos),
        );
        let untyped = MassProperties::with_inertia(m, i, pos);

        assert_eq!(typed.to_untyped(), untyped);
    }

    #[test]
    fn typed_round_trip_through_from_untyped_unchecked() {
        use astrodyn_quantities::frame::TestVehicle;

        let untyped = MassProperties::with_inertia(
            7.5,
            DMat3::from_diagonal(DVec3::new(40.0, 50.0, 60.0)),
            DVec3::new(0.0, 0.05, 0.0),
        );
        let typed = MassPropertiesTyped::<TestVehicle>::from_untyped_unchecked(&untyped);
        let back = typed.to_untyped();

        assert_eq!(back, untyped);
    }

    #[test]
    fn typed_validate_consistency_passes() {
        use astrodyn_quantities::frame::TestVehicle;

        let typed = MassPropertiesTyped::<TestVehicle>::with_inertia(
            Mass::new::<kilogram>(10.0),
            InertiaTensor::<BodyFrame<TestVehicle>>::from_dmat3_unchecked(DMat3::from_diagonal(
                DVec3::new(100.0, 200.0, 300.0),
            )),
            Position::<StructuralFrame<TestVehicle>>::zero(),
        );
        typed.validate_consistency(1e-6);
    }
}
