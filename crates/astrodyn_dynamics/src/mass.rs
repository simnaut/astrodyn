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

/// Lower bound on mass (kg) accepted by the typed and untyped constructors.
///
/// Rationale: the point-mass placeholder inertia `I = m·I_{3×3}` has
/// determinant `m³`, and `DMat3::inverse()` divides each cofactor by that
/// determinant. The cofactors of a diagonal `m·I_{3×3}` are themselves
/// `O(m²)`, so the full inverse formula propagates `m³` through both
/// numerator and denominator before a final division. Once `m³` underflows
/// to subnormal range (`< f64::MIN_POSITIVE ≈ 2.2e-308`), the intermediate
/// products lose precision catastrophically and the cached `inverse_inertia`
/// becomes non-finite even though `1/m` would still round to a finite f64.
/// A floor of `1e-100` keeps `m³ ≥ 1e-300`, well above the subnormal
/// threshold, while remaining far below any realistic spacecraft mass
/// (a 1 g cubesat is `1e-3 kg`).
///
/// The inertia-tensor singularity check shared by `with_inertia` /
/// `recompute_derived` is scale-invariant — it rejects only matrices
/// that produce a non-finite inverse, so the placeholder inertia at
/// `m = MIN_SAFE_MASS_KG` (`det = 1e-300`, well-conditioned but tiny)
/// is accepted by every construction path, including the raw→typed
/// bridge and `recompute_derived`.
pub const MIN_SAFE_MASS_KG: f64 = 1e-100;

/// Upper bound on mass (kg) accepted by the typed and untyped constructors.
///
/// Rationale: symmetric to [`MIN_SAFE_MASS_KG`]. The point-mass placeholder
/// determinant `m³` must stay finite, i.e. below `f64::MAX ≈ 1.8e308`. A
/// ceiling of `1e100` keeps `m³ ≤ 1e300`, well below overflow, while
/// remaining far above any plausible simulated body (the Sun is `~2e30 kg`).
/// The bound is numerical, not physical: `with_inertia` uses the same
/// general 3×3 inverse, so its cofactors scale with the inertia tensor's
/// own magnitudes and the same guard catches mass-driven overflow at the
/// point-mass placeholder path.
pub const SAFE_MAX_MASS_KG: f64 = 1e100;

/// Invert `inertia` and assert the inverse is finite and well-defined.
///
/// Used by every site that recomputes `inverse_inertia` from `inertia`
/// — both untyped and typed `with_inertia`/`recompute_derived` —
/// so all four entry points apply the same singularity check.
///
/// The check is **scale-invariant**: instead of comparing the
/// determinant against an absolute threshold (which rejects
/// well-conditioned but small-magnitude tensors, e.g. the placeholder
/// `m·I_{3×3}` at `m = MIN_SAFE_MASS_KG` where `det = m³ = 1e-300`),
/// we accept the inertia iff
///   1. the determinant is finite and non-zero (rejects the
///      overflow case where `glam::inverse()` divides finite
///      cofactors by `det = +inf` and silently returns the all-zero
///      matrix, which would pass the post-`is_finite()` check but
///      be physically useless — `I · I⁻¹ = 0`, not the identity), and
///   2. the inverse itself is finite (rejects genuinely singular
///      matrices with `det = 0`, near-zero subnormals, or linearly
///      dependent columns — `inverse()` returns `inf`/`NaN` entries
///      in those cases).
///
/// The determinant is included in the diagnostic message for debugging.
#[inline]
fn checked_inertia_inverse(inertia: DMat3) -> DMat3 {
    let det = inertia.determinant();
    assert!(
        det.is_finite() && det != 0.0,
        "inertia tensor has a non-finite or zero determinant \
         (det={det:.2e}); the inverse would be all zeros (det=±inf, \
         where finite cofactors divided by ±inf round to 0) or contain \
         inf/NaN entries (det=0). Supply a non-singular inertia tensor \
         whose entries stay within the f64 dynamic range — \
         e.g. diag(1e103, 1e103, 1e103) overflows `det = m³` to +inf \
         even though every entry is a normal-range f64."
    );
    let inverse = inertia.inverse();
    assert!(
        inverse.is_finite(),
        "inertia tensor is singular or ill-conditioned \
         (det={det:.2e}); inverse contains inf/NaN entries. \
         Supply a non-singular inertia tensor."
    );
    inverse
}

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
    ///
    /// `inverse_inertia` is computed via the general 3×3 inverse
    /// (`(I·m).inverse()`) rather than the element-wise reciprocal
    /// (`I/m`). This is the same formula [`Self::with_inertia`] and
    /// [`Self::recompute_derived`] use, so all three sites agree
    /// byte-for-byte — sub-ULP divergence between constructors integrates
    /// to multi-kilometre drift on long-arc rotational-dynamics runs.
    ///
    /// # Panics
    /// Panics if `mass` is `NaN`, infinite, zero, negative, or outside
    /// the safe range `[MIN_SAFE_MASS_KG, SAFE_MAX_MASS_KG]`. The
    /// placeholder inertia `m·I_{3×3}` is always non-singular within
    /// that range, so the inertia-inverse guard cannot panic from this
    /// entry point — only the mass guard can.
    // JEOD_INV: MA.02 — mass > 0, finite, and within
    // [MIN_SAFE_MASS_KG, SAFE_MAX_MASS_KG] for the general 3×3 inverse
    // to remain finite (see `MIN_SAFE_MASS_KG` doc-comment).
    // JEOD_INV: MA.04 — inverse_inertia computed from inertia via the same
    // general 3×3 inverse used by `with_inertia` (byte-identical across
    // constructors).
    pub fn new(mass: f64) -> Self {
        assert!(
            (MIN_SAFE_MASS_KG..=SAFE_MAX_MASS_KG).contains(&mass),
            "MassProperties: mass {mass} kg out of safe range \
             [{MIN_SAFE_MASS_KG:.0e}, {SAFE_MAX_MASS_KG:.0e}] kg. \
             This is a hard numerical limit shared by every constructor \
             (`new`, `with_inertia`, `recompute_derived`): the inverse-mass \
             cache `1/mass` and the inertia-inverse cofactors all derive \
             from `mass`, so the guard applies regardless of how the \
             inertia tensor is supplied. Rescale your physical model into \
             this range, or contact maintainers if a real scenario \
             genuinely requires masses outside it."
        );
        let inertia = DMat3::IDENTITY * mass;
        Self {
            mass,
            inverse_mass: 1.0 / mass,
            inertia,
            inverse_inertia: checked_inertia_inverse(inertia),
            position: DVec3::ZERO,
            t_parent_this: DMat3::IDENTITY,
            dirty: false,
        }
    }

    /// Create mass properties with explicit inertia tensor and center-of-mass position.
    ///
    /// The inertia tensor is about the body frame axes through the center of mass.
    /// The position is the center of mass in the structural frame.
    ///
    /// # Panics
    /// Panics (with a diagnostic that names the broken assumption) if:
    /// - `mass` is `NaN`, infinite, zero, or negative;
    /// - `mass` is outside `[MIN_SAFE_MASS_KG, SAFE_MAX_MASS_KG]`;
    /// - `inertia` has a non-finite or zero determinant (overflowed
    ///   `±inf` from entry magnitudes, `NaN` from non-finite inputs,
    ///   or `0` from a singular matrix);
    /// - `inertia.inverse()` produces `inf`/`NaN` entries (numerically
    ///   singular even though `det != 0`).
    // JEOD_INV: MA.02 — mass > 0, finite, and within
    // [MIN_SAFE_MASS_KG, SAFE_MAX_MASS_KG] (same bound as `new` so
    // `inverse_mass = 1/mass` stays finite and the inertia inverse is
    // guarded by the explicit singularity check below).
    // JEOD_INV: MA.05 — JEOD computes inverse inertia only for root bodies; we compute for all (structural divergence)
    // JEOD_INV: DB.23 — compute_inverse_inertia enabled (always computed here)
    // JEOD_INV: MA.04 — inverse_inertia consistent with inertia (computed from inertia)
    pub fn with_inertia(mass: f64, inertia: DMat3, position: DVec3) -> Self {
        assert!(
            (MIN_SAFE_MASS_KG..=SAFE_MAX_MASS_KG).contains(&mass),
            "MassProperties: mass {mass} kg out of safe range \
             [{MIN_SAFE_MASS_KG:.0e}, {SAFE_MAX_MASS_KG:.0e}] kg"
        );
        let inverse_inertia = checked_inertia_inverse(inertia);
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
    /// Panics (with a diagnostic that names the broken assumption) if any
    /// of the following hold after the in-place mutation:
    /// - `mass` is `NaN`, infinite, zero, or negative;
    /// - `mass < MIN_SAFE_MASS_KG` (`1e-100 kg`) — the placeholder
    ///   inertia's `m³` term underflows;
    /// - `mass > SAFE_MAX_MASS_KG` (`1e100 kg`) — the placeholder
    ///   inertia's `m³` term overflows;
    /// - `inertia` has a non-finite determinant (i.e. `±inf` from
    ///   per-entry overflow, or `NaN` from non-finite inputs);
    /// - `inertia` has a zero determinant (singular matrix);
    /// - `inertia.inverse()` returns a matrix with `inf`/`NaN`
    ///   entries (numerically singular even though `det != 0`).
    // JEOD_INV: MA.03 — inverse_mass consistent with mass (recomputed as 1/mass)
    // JEOD_INV: MA.04 — inverse_inertia consistent with inertia (recomputed from inertia)
    // JEOD_INV: MA.07 — derived quantities recomputed after mutation
    pub fn recompute_derived(&mut self) {
        if !self.dirty {
            return;
        }
        self.dirty = false;

        assert!(
            (MIN_SAFE_MASS_KG..=SAFE_MAX_MASS_KG).contains(&self.mass),
            "MassProperties: mass {} kg out of safe range \
             [{:.0e}, {:.0e}] kg",
            self.mass,
            MIN_SAFE_MASS_KG,
            SAFE_MAX_MASS_KG,
        );
        self.inverse_mass = 1.0 / self.mass;
        self.inverse_inertia = checked_inertia_inverse(self.inertia);
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
    /// Structure-to-body rotation. Same semantic as
    /// [`MassProperties::t_parent_this`]; carried on the typed sibling
    /// because Apollo (and any vehicle whose `pt_orientation` is not
    /// the identity) sets a non-identity value, and a round-trip that
    /// silently rewrote it to identity caused launch-stack composite
    /// COMs to drift in `tier3_sim_apollo_trajectory`.
    pub t_parent_this: DMat3,
    /// See [`MassProperties::dirty`].
    pub dirty: bool,
    _v: PhantomData<V>,
}

impl<V: Vehicle> MassPropertiesTyped<V> {
    /// Point-mass constructor with placeholder spherical inertia
    /// (`I = m · I_{3×3}`) — see [`MassProperties::new`] for the same
    /// caveat about translational-only validity.
    ///
    /// `inverse_inertia` is computed via the general 3×3 inverse
    /// (`(I·m).inverse()`) rather than the algebraically-equivalent
    /// element-wise reciprocal (`I/m`). The general inverse is the same
    /// formula [`Self::with_inertia`] and [`Self::recompute_derived`] use,
    /// so a point mass built through this constructor agrees byte-for-byte
    /// with one rebuilt through the raw→typed bridge or any
    /// `recompute_derived` round-trip. The two formulas differ by a few
    /// ULPs on the diagonal and produce non-zero (~1e-25) off-diagonal
    /// residues from adjugate cancellations; sub-ULP divergence here
    /// integrates to multi-kilometre position error on long-arc
    /// rotational-dynamics runs.
    ///
    /// # Panics
    /// Panics if `mass` is `NaN`, infinite, zero, negative, or outside
    /// the safe range `[MIN_SAFE_MASS_KG, SAFE_MAX_MASS_KG]`. The
    /// placeholder inertia `m·I_{3×3}` is always non-singular within
    /// that range, so the inertia-inverse guard cannot panic from this
    /// entry point — only the mass guard can.
    // JEOD_INV: MA.02 — mass > 0, finite, and within
    // [MIN_SAFE_MASS_KG, SAFE_MAX_MASS_KG] (mirrors the untyped
    // `MassProperties::new` guard, so both construction paths reject the
    // same set of inputs and the rebuilt typed sibling never carries
    // non-finite cache values).
    // JEOD_INV: MA.04 — inverse_inertia computed from inertia via the same
    // general 3×3 inverse used by `with_inertia` (byte-identical across
    // constructors).
    pub fn new(mass: Mass) -> Self {
        let m = mass.get::<kilogram>();
        assert!(
            (MIN_SAFE_MASS_KG..=SAFE_MAX_MASS_KG).contains(&m),
            "MassPropertiesTyped: mass {m} kg out of safe range \
             [{MIN_SAFE_MASS_KG:.0e}, {SAFE_MAX_MASS_KG:.0e}] kg. \
             This is a hard numerical limit shared by every constructor \
             (`new`, `with_inertia`, `recompute_derived`): the inverse-mass \
             cache `1/mass` and the inertia-inverse cofactors all derive \
             from `mass`, so the guard applies regardless of how the \
             inertia tensor is supplied. Rescale your physical model into \
             this range, or contact maintainers if a real scenario \
             genuinely requires masses outside it."
        );
        let inertia_dmat = DMat3::IDENTITY * m;
        Self {
            mass,
            inverse_mass: 1.0 / m,
            inertia: InertiaTensor::<BodyFrame<V>>::from_dmat3_unchecked(inertia_dmat),
            inverse_inertia: checked_inertia_inverse(inertia_dmat),
            center_of_mass: Position::<StructuralFrame<V>>::zero(),
            t_parent_this: DMat3::IDENTITY,
            dirty: false,
            _v: PhantomData,
        }
    }

    /// Constructor with explicit inertia and center-of-mass position.
    ///
    /// # Panics
    /// Panics (with a diagnostic that names the broken assumption) if:
    /// - `mass` is `NaN`, infinite, zero, or negative;
    /// - `mass` is outside `[MIN_SAFE_MASS_KG, SAFE_MAX_MASS_KG]`;
    /// - `inertia` has a non-finite or zero determinant (overflowed
    ///   `±inf` from entry magnitudes, `NaN` from non-finite inputs,
    ///   or `0` from a singular matrix);
    /// - `inertia.inverse()` produces `inf`/`NaN` entries (numerically
    ///   singular even though `det != 0`).
    // JEOD_INV: MA.02 — mass > 0, finite, and within
    // [MIN_SAFE_MASS_KG, SAFE_MAX_MASS_KG] (same bound as `new`).
    // JEOD_INV: MA.04 — inverse_inertia computed from inertia
    // JEOD_INV: DB.23 — inverse_inertia always computed
    pub fn with_inertia(
        mass: Mass,
        inertia: InertiaTensor<BodyFrame<V>>,
        center_of_mass: Position<StructuralFrame<V>>,
    ) -> Self {
        let m = mass.get::<kilogram>();
        assert!(
            (MIN_SAFE_MASS_KG..=SAFE_MAX_MASS_KG).contains(&m),
            "MassPropertiesTyped: mass {m} kg out of safe range \
             [{MIN_SAFE_MASS_KG:.0e}, {SAFE_MAX_MASS_KG:.0e}] kg"
        );
        let inverse_inertia = checked_inertia_inverse(inertia.as_dmat3());
        Self {
            mass,
            inverse_mass: 1.0 / m,
            inertia,
            inverse_inertia,
            center_of_mass,
            t_parent_this: DMat3::IDENTITY,
            dirty: false,
            _v: PhantomData,
        }
    }

    /// Set the structure-to-body rotation (mirrors
    /// [`MassProperties::with_t_parent_this`]).
    pub fn with_t_parent_this(mut self, t_parent_this: DMat3) -> Self {
        self.t_parent_this = t_parent_this;
        self
    }

    /// Recompute `inverse_mass` and `inverse_inertia`.
    ///
    /// Call after mutating `mass` or `inertia` directly. No-op when
    /// `dirty == false`. Mirrors [`MassProperties::recompute_derived`].
    ///
    /// # Panics
    /// Panics (with a diagnostic that names the broken assumption) if any
    /// of the following hold after the in-place mutation:
    /// - `mass` is `NaN`, infinite, zero, or negative;
    /// - `mass < MIN_SAFE_MASS_KG` (`1e-100 kg`) — the placeholder
    ///   inertia's `m³` term underflows;
    /// - `mass > SAFE_MAX_MASS_KG` (`1e100 kg`) — the placeholder
    ///   inertia's `m³` term overflows;
    /// - `inertia` has a non-finite determinant (i.e. `±inf` from
    ///   per-entry overflow, or `NaN` from non-finite inputs);
    /// - `inertia` has a zero determinant (singular matrix);
    /// - `inertia.inverse()` returns a matrix with `inf`/`NaN`
    ///   entries (numerically singular even though `det != 0`).
    // JEOD_INV: MA.03 — inverse_mass = 1/mass (recomputed)
    // JEOD_INV: MA.04 — inverse_inertia consistent with inertia (recomputed)
    // JEOD_INV: MA.07 — derived quantities recomputed after mutation
    pub fn recompute_derived(&mut self) {
        if !self.dirty {
            return;
        }
        self.dirty = false;
        let m = self.mass.get::<kilogram>();
        assert!(
            (MIN_SAFE_MASS_KG..=SAFE_MAX_MASS_KG).contains(&m),
            "MassPropertiesTyped: mass {m} kg out of safe range \
             [{MIN_SAFE_MASS_KG:.0e}, {SAFE_MAX_MASS_KG:.0e}] kg"
        );
        self.inverse_mass = 1.0 / m;
        self.inverse_inertia = checked_inertia_inverse(self.inertia.as_dmat3());
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

    /// Drop the phantoms and emit the untyped storage form, preserving
    /// every field verbatim (including the cache fields `inverse_mass`,
    /// `inverse_inertia`, and `dirty`, plus `t_parent_this` — the field
    /// whose silent drop was the Apollo regression in #393).
    #[inline]
    pub fn to_untyped(&self) -> MassProperties {
        MassProperties {
            mass: self.mass.get::<kilogram>(),
            inverse_mass: self.inverse_mass,
            inertia: self.inertia.as_dmat3(),
            inverse_inertia: self.inverse_inertia,
            position: self.center_of_mass.raw_si(),
            t_parent_this: self.t_parent_this,
            dirty: self.dirty,
        }
    }

    /// Wrap an untyped [`MassProperties`] as typed. **The caller
    /// asserts** body-frame inertia, structural-frame center of mass,
    /// and consistency between `inverse_mass`/`inverse_inertia` and
    /// `mass`/`inertia` — the latter is the same contract the untyped
    /// struct exposes, since both fields are public there too.
    #[inline]
    pub fn from_untyped_unchecked(s: &MassProperties) -> Self {
        Self {
            mass: Mass::new::<kilogram>(s.mass),
            inverse_mass: s.inverse_mass,
            inertia: InertiaTensor::<BodyFrame<V>>::from_dmat3_unchecked(s.inertia),
            inverse_inertia: s.inverse_inertia,
            center_of_mass: Position::<StructuralFrame<V>>::from_raw_si(s.position),
            t_parent_this: s.t_parent_this,
            dirty: s.dirty,
            _v: PhantomData,
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
        assert_eq!(mp.inverse_mass, 0.1);
        assert_eq!(mp.inertia, DMat3::IDENTITY * 10.0);
        // `new` uses the same general 3×3 inverse as `with_inertia` (see
        // doc-comment on `MassProperties::new`); diagonal entries agree
        // with `IDENTITY / m` to ~1 ULP but are not bit-identical.
        assert_eq!(mp.inverse_inertia, (DMat3::IDENTITY * 10.0).inverse());
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

        assert_eq!(typed.mass.get::<kilogram>(), 10.0);
        assert_eq!(typed.inverse_mass, 0.1);
        assert_eq!(typed.inertia.as_dmat3(), DMat3::IDENTITY * 10.0);
        // `MassPropertiesTyped::new` uses the same general 3×3 inverse as
        // `with_inertia` and the raw→typed bridge — sub-ULP equivalent to
        // `IDENTITY / m` on the diagonal but byte-identical across the
        // three construction paths (closes the ULP-drift gap in #459).
        assert_eq!(typed.inverse_inertia, (DMat3::IDENTITY * 10.0).inverse());
        assert_eq!(typed.center_of_mass.raw_si(), DVec3::ZERO);
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

        assert_eq!(typed.mass.get::<kilogram>(), untyped.mass);
        assert_eq!(typed.inverse_mass, untyped.inverse_mass);
        assert_eq!(typed.inertia.as_dmat3(), untyped.inertia);
        assert_eq!(typed.inverse_inertia, untyped.inverse_inertia);
        assert_eq!(typed.center_of_mass.raw_si(), untyped.position);
        assert_eq!(typed.t_parent_this, untyped.t_parent_this);
        assert_eq!(typed.dirty, untyped.dirty);
    }

    // ---- safe-range guards (#506) -------------------------------------
    //
    // Both untyped and typed point-mass constructors reject masses
    // outside `[MIN_SAFE_MASS_KG, SAFE_MAX_MASS_KG]` because the general
    // 3×3 inverse of the placeholder inertia `m·I_{3×3}` involves `m³`
    // and goes non-finite at the f64 dynamic-range extremes. Same guard
    // applies to `with_inertia` and `recompute_derived`. NaN/infinity
    // and zero/negative masses are rejected by the same composite
    // `is_finite() && in [min, max]` predicate.

    #[test]
    #[should_panic(expected = "out of safe range")]
    fn untyped_new_panics_on_zero_mass() {
        let _ = MassProperties::new(0.0);
    }

    #[test]
    #[should_panic(expected = "out of safe range")]
    fn untyped_new_panics_on_negative_mass() {
        let _ = MassProperties::new(-1.0);
    }

    #[test]
    #[should_panic(expected = "out of safe range")]
    fn untyped_new_panics_on_nan_mass() {
        let _ = MassProperties::new(f64::NAN);
    }

    #[test]
    #[should_panic(expected = "out of safe range")]
    fn untyped_new_panics_on_infinite_mass() {
        let _ = MassProperties::new(f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "out of safe range")]
    fn untyped_new_panics_on_mass_below_safe_floor_cubic_underflow() {
        // `1e-150` is itself a normal-range f64 (well above
        // `f64::MIN_POSITIVE ≈ 2.2e-308`) — the failure mode is that
        // it is below `MIN_SAFE_MASS_KG = 1e-100`, and the placeholder
        // inertia's determinant `m³ = 1e-450` underflows to
        // subnormal/zero, which the inertia-inverse guard would reject
        // downstream. The safe-range guard rejects it up-front before
        // the cache fields are touched.
        let _ = MassProperties::new(1e-150);
    }

    #[test]
    #[should_panic(expected = "out of safe range")]
    fn untyped_new_panics_on_huge_mass() {
        // 1e150 > SAFE_MAX_MASS_KG (1e100); `m³` here is 1e450
        // which overflows to +inf.
        let _ = MassProperties::new(1e150);
    }

    #[test]
    fn untyped_new_accepts_safe_extremes() {
        // Both endpoints inclusive — finite caches.
        let lo = MassProperties::new(MIN_SAFE_MASS_KG);
        let hi = MassProperties::new(SAFE_MAX_MASS_KG);
        assert!(lo.inverse_mass.is_finite());
        assert!(hi.inverse_mass.is_finite());
        for v in [
            lo.inverse_inertia.x_axis,
            lo.inverse_inertia.y_axis,
            lo.inverse_inertia.z_axis,
            hi.inverse_inertia.x_axis,
            hi.inverse_inertia.y_axis,
            hi.inverse_inertia.z_axis,
        ] {
            assert!(v.x.is_finite() && v.y.is_finite() && v.z.is_finite());
        }
    }

    #[test]
    #[should_panic(expected = "out of safe range")]
    fn untyped_with_inertia_panics_on_zero_mass() {
        let _ = MassProperties::with_inertia(0.0, DMat3::IDENTITY, DVec3::ZERO);
    }

    #[test]
    fn safe_extremes_round_trip_through_with_inertia_and_recompute() {
        // The mass-range guard and the inertia-singularity guard must
        // agree on what they accept: any mass in
        // `[MIN_SAFE_MASS_KG, SAFE_MAX_MASS_KG]` paired with the
        // placeholder inertia `m·I_{3×3}` (well-conditioned at every
        // mass) must round-trip through both `with_inertia` and a
        // `recompute_derived` cycle without the inertia-inverse check
        // rejecting it. At `m = MIN_SAFE_MASS_KG`, `det = 1e-300` — a
        // normal-range f64 with a finite inverse, so the
        // scale-invariant `is_finite()` check passes even though an
        // absolute threshold like `det > 1e-30` would not.
        for &m in &[MIN_SAFE_MASS_KG, 1.0_f64, SAFE_MAX_MASS_KG] {
            let inertia = DMat3::IDENTITY * m;
            let via_with_inertia = MassProperties::with_inertia(m, inertia, DVec3::ZERO);
            assert!(via_with_inertia.inverse_inertia.is_finite());

            let mut mp = MassProperties::new(m);
            mp.dirty = true;
            mp.recompute_derived();
            assert!(mp.inverse_inertia.is_finite());
        }
    }

    #[test]
    #[should_panic(expected = "non-finite or zero determinant")]
    fn singular_inertia_rejected_by_with_inertia() {
        // Zero matrix has det = 0; the pre-inverse determinant guard
        // catches this without ever calling `inverse()`.
        let _ = MassProperties::with_inertia(1.0, DMat3::ZERO, DVec3::ZERO);
    }

    #[test]
    #[should_panic(expected = "non-finite or zero determinant")]
    fn det_overflow_inertia_rejected_by_with_inertia() {
        // `diag(1e103, 1e103, 1e103)` has every entry within the f64
        // dynamic range, but `det = 1e309` overflows to `+inf`.
        // `glam::DMat3::inverse()` then divides each finite cofactor by
        // `+inf` and returns the all-zero matrix, which would pass a
        // post-inverse `is_finite()` check despite being physically
        // useless (`I · I⁻¹ = 0`, not the identity). The pre-inverse
        // determinant guard rejects it before that pathological
        // all-zero "inverse" can be cached.
        let _ = MassProperties::with_inertia(
            1.0,
            DMat3::from_diagonal(DVec3::new(1e103, 1e103, 1e103)),
            DVec3::ZERO,
        );
    }

    #[test]
    #[should_panic(expected = "out of safe range")]
    fn typed_new_panics_on_zero_mass() {
        use astrodyn_quantities::frame::TestVehicle;
        let _ = MassPropertiesTyped::<TestVehicle>::new(Mass::new::<kilogram>(0.0));
    }

    #[test]
    #[should_panic(expected = "out of safe range")]
    fn typed_new_panics_on_nan_mass() {
        use astrodyn_quantities::frame::TestVehicle;
        let _ = MassPropertiesTyped::<TestVehicle>::new(Mass::new::<kilogram>(f64::NAN));
    }

    #[test]
    #[should_panic(expected = "out of safe range")]
    fn typed_with_inertia_panics_on_huge_mass() {
        use astrodyn_quantities::frame::TestVehicle;
        let _ = MassPropertiesTyped::<TestVehicle>::with_inertia(
            Mass::new::<kilogram>(1e200),
            InertiaTensor::<BodyFrame<TestVehicle>>::from_dmat3_unchecked(DMat3::IDENTITY),
            Position::<StructuralFrame<TestVehicle>>::zero(),
        );
    }

    #[test]
    #[should_panic(expected = "out of safe range")]
    fn untyped_recompute_derived_panics_on_out_of_range_mass() {
        let mut mp = MassProperties::new(10.0);
        mp.mass = 1e200; // outside safe range
        mp.dirty = true;
        mp.recompute_derived();
    }

    #[test]
    #[should_panic(expected = "out of safe range")]
    fn typed_recompute_derived_panics_on_out_of_range_mass() {
        use astrodyn_quantities::frame::TestVehicle;
        let mut mp = MassPropertiesTyped::<TestVehicle>::new(Mass::new::<kilogram>(10.0));
        mp.mass = Mass::new::<kilogram>(1e200);
        mp.dirty = true;
        mp.recompute_derived();
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

    // ---- proptest round-trips (#398) ----------------------------------
    //
    // Apollo regression class: the typed sibling silently dropped
    // `t_parent_this` in #393. These property tests assert verbatim
    // field-level round-trip equality so any future field added on one
    // side without updating the other fails CI immediately.

    use astrodyn_quantities::frame::TestVehicle;
    use proptest::prelude::*;

    fn arb_finite_bounded() -> impl Strategy<Value = f64> {
        prop_oneof![
            (1.0e-9_f64..1.0e9_f64),
            (1.0e-9_f64..1.0e9_f64).prop_map(|x| -x),
        ]
    }

    fn arb_dvec3() -> impl Strategy<Value = DVec3> {
        (
            arb_finite_bounded(),
            arb_finite_bounded(),
            arb_finite_bounded(),
        )
            .prop_map(|(x, y, z)| DVec3::new(x, y, z))
    }

    fn arb_dmat3_full_rank() -> impl Strategy<Value = DMat3> {
        // Build a diagonal matrix with strictly positive principal
        // moments (ensures non-singular and `inverse()` is well-defined),
        // then conjugate by a small rotation so off-diagonal terms can
        // arise without risking degeneracy.
        (
            (1.0_f64..1.0e6_f64),
            (1.0_f64..1.0e6_f64),
            (1.0_f64..1.0e6_f64),
            (-1.0_f64..1.0_f64),
            (-1.0_f64..1.0_f64),
            (-1.0_f64..1.0_f64),
        )
            .prop_map(|(ix, iy, iz, ax, ay, az)| {
                let diag = DMat3::from_diagonal(DVec3::new(ix, iy, iz));
                let axis = DVec3::new(ax, ay, az);
                let rot = if axis.length_squared() > 1.0e-6 {
                    let angle = 0.1; // small bounded rotation; off-diagonal magnitude ~10%
                    glam::DMat3::from_axis_angle(axis.normalize(), angle)
                } else {
                    DMat3::IDENTITY
                };
                rot.transpose() * diag * rot
            })
    }

    fn arb_arbitrary_dmat3() -> impl Strategy<Value = DMat3> {
        // 9 independent finite scalars — used for `t_parent_this` and
        // `inverse_inertia` (both are stored verbatim and compared
        // verbatim, so no positive-definiteness constraint is needed
        // for round-trip purposes).
        (
            arb_finite_bounded(),
            arb_finite_bounded(),
            arb_finite_bounded(),
            arb_finite_bounded(),
            arb_finite_bounded(),
            arb_finite_bounded(),
            arb_finite_bounded(),
            arb_finite_bounded(),
            arb_finite_bounded(),
        )
            .prop_map(|(a, b, c, d, e, f, g, h, i)| {
                DMat3::from_cols(
                    DVec3::new(a, b, c),
                    DVec3::new(d, e, f),
                    DVec3::new(g, h, i),
                )
            })
    }

    fn arb_mass_properties() -> impl Strategy<Value = MassProperties> {
        // Generate self-consistent caches per the plan: inverse_mass =
        // 1/mass, inverse_inertia = inertia.inverse(), dirty = false.
        // `t_parent_this` is independent (and the regression-class
        // field — generate it as an arbitrary DMat3 so the round-trip
        // sees a non-identity value).
        (
            (1.0e-3_f64..1.0e6_f64),
            arb_dmat3_full_rank(),
            arb_dvec3(),
            arb_arbitrary_dmat3(),
        )
            .prop_map(|(mass, inertia, position, t_parent_this)| MassProperties {
                mass,
                inverse_mass: 1.0 / mass,
                inertia,
                inverse_inertia: inertia.inverse(),
                position,
                t_parent_this,
                dirty: false,
            })
    }

    proptest! {
        #[test]
        fn round_trip_mass_properties_untyped_typed_untyped(orig in arb_mass_properties()) {
            let typed = MassPropertiesTyped::<TestVehicle>::from_untyped_unchecked(&orig);
            prop_assert_eq!(typed.to_untyped(), orig);
        }

        // Asserted via the untyped projection — `MassPropertiesTyped`'s
        // derived `PartialEq` requires `TestVehicle: PartialEq`, which
        // it isn't. Catches dropped/added fields equally well.
        #[test]
        fn round_trip_mass_properties_typed_untyped_typed(orig in arb_mass_properties()) {
            let typed = MassPropertiesTyped::<TestVehicle>::from_untyped_unchecked(&orig);
            let lifted = MassPropertiesTyped::<TestVehicle>::from_untyped_unchecked(&typed.to_untyped());
            prop_assert_eq!(lifted.to_untyped(), typed.to_untyped());
        }
    }
}
