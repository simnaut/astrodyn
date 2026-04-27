use glam::DMat3;
use glam::DVec3;
use jeod_dynamics::GravityAcceleration;
use jeod_quantities::aliases::HarmonicDegree;

use crate::gravity_source::{GravityModel, GravitySource};
use log::warn;

#[derive(Debug, Clone)]
pub struct GravityControl<SourceId = String> {
    pub source_name: SourceId,
    pub gradient: bool,
    /// If true, use only point-mass (spherical) gravity for this source,
    /// ignoring any spherical harmonics data. Matches JEOD's `spherical` flag
    /// on `GravityControls`.
    pub spherical: bool,
    /// Non-spherical degree to use. Must be <= source degree.
    /// Ignored when `spherical` is true.
    pub degree: usize,
    /// Non-spherical order to use. Must be <= degree and <= source order.
    /// Ignored when `spherical` is true.
    pub order: usize,
    /// If true, exclude point-mass (n=0,1) terms.
    pub perturbing_only: bool,
    /// Degree for gradient computation. Must be <= degree.
    pub gradient_degree: usize,
    /// Order for gradient computation. Must be <= order and <= gradient_degree.
    pub gradient_order: usize,
    /// If true, compute gravity as differential acceleration: the acceleration
    /// of the vehicle toward this source minus the acceleration of the
    /// integration frame origin toward this source. This is the correct
    /// treatment for third-body perturbations (e.g., Sun/Moon when integrating
    /// in an Earth-centered frame).
    ///
    /// Matches JEOD's `GravityIntegFrame::is_third_body` flag. In JEOD, this
    /// is set automatically based on whether the source's inertial frame is a
    /// progeny of the integration frame. Here it is set explicitly per control.
    // JEOD_INV: GV.14 — third-body vs direct gravity classification (set explicitly; JEOD derives from frame tree ancestry)
    pub differential: bool,
    /// If true, use Battin's method for improved numerical accuracy in
    /// third-body (differential) gravity computation. Only meaningful when
    /// `differential` is also true. Off by default in JEOD.
    ///
    /// Battin's method reformulates the differential acceleration to avoid
    /// catastrophic cancellation when the vehicle is close to the integration
    /// frame origin relative to the third-body source distance.
    ///
    /// JEOD ref: `gravity_controls.cc:317-331`.
    pub battin_method: bool,
    /// If true, apply post-Newtonian relativistic correction for this source.
    /// Requires source velocity in `GravitySourceEntry`. Only significant for
    /// Mercury-like orbits near massive bodies.
    pub relativistic: bool,
}

impl<SourceId> GravityControl<SourceId> {
    /// Create a spherical (point-mass only) gravity control.
    ///
    /// Uses only µ/r² acceleration. Any spherical harmonics data on the source
    /// is ignored. For gravity with J2+ harmonics, use [`new_nonspherical`](Self::new_nonspherical).
    pub fn new_spherical(source_name: SourceId, gradient: bool) -> Self {
        Self {
            source_name,
            gradient,
            spherical: true,
            degree: 0,
            order: 0,
            perturbing_only: false,
            gradient_degree: 0,
            gradient_order: 0,
            differential: false,
            battin_method: false,
            relativistic: false,
        }
    }

    /// Create a non-spherical (spherical harmonics) gravity control.
    ///
    /// Evaluates the source's spherical harmonics coefficients up to the given
    /// `degree` and `order`. The source must have a `SphericalHarmonics` model
    /// and the gravity source entry must provide a planet-fixed rotation matrix.
    pub fn new_nonspherical(
        source_name: SourceId,
        degree: usize,
        order: usize,
        gradient: bool,
    ) -> Self {
        Self {
            source_name,
            gradient,
            spherical: false,
            degree,
            order,
            perturbing_only: false,
            gradient_degree: 0,
            gradient_order: 0,
            differential: false,
            battin_method: false,
            relativistic: false,
        }
    }

    /// Create a spherical (point-mass) gravity control for a third-body source.
    ///
    /// Third-body sources use differential acceleration: the acceleration of
    /// the vehicle toward this source minus the acceleration of the integration
    /// frame origin toward this source.
    pub fn new_third_body(source_name: SourceId) -> Self {
        Self {
            source_name,
            gradient: false,
            spherical: true,
            degree: 0,
            order: 0,
            perturbing_only: false,
            gradient_degree: 0,
            gradient_order: 0,
            differential: true,
            battin_method: false,
            relativistic: false,
        }
    }

    /// Validate this control against its gravity source, matching JEOD's
    /// `SphericalHarmonicsGravityControls::check_validity()`.
    ///
    /// # Panics
    /// - `spherical` is false but source is `PointMass`
    /// - degree > source degree
    /// - order > source order
    /// - order > degree
    ///
    /// # Notes
    /// - If `spherical` is false and `degree` is 0, `spherical` is auto-corrected
    ///   to true with a warning (matches JEOD's non-fatal auto-correction).
    /// - Invalid `gradient_degree` and `gradient_order` values do not panic;
    ///   they are clamped to valid ranges and a warning is logged.
    // JEOD_INV: GV.03 — check_validity() called on degree/order mutation
    pub fn check_validity(&mut self, source: &GravitySource) {
        if self.spherical {
            return;
        }

        // JEOD_INV: GV.07 — degree=0 with spherical=false auto-corrects to spherical
        // JEOD spherical_harmonics_gravity_controls.cc:334-346:
        // degree=0 with spherical=false is auto-corrected to spherical=true
        // via MessageHandler::error (non-fatal).
        if self.degree == 0 {
            warn!(
                "Non-spherical gravity requested but degree is 0; \
                 setting spherical=true (matches JEOD auto-correction)."
            );
            self.spherical = true;
            return;
        }

        match &source.model {
            GravityModel::SphericalHarmonics(ref data) => {
                // JEOD_INV: GV.04 — degree <= source degree
                // JEOD_INV: GV.19 — source-side degree/order clamp (same check, catalogued separately)
                assert!(
                    self.degree <= data.degree,
                    "Gravity field degree requested ({}) is greater than max gravity field degree ({}).",
                    self.degree, data.degree
                );
                // JEOD_INV: GV.05 — order <= source order
                assert!(
                    self.order <= data.order,
                    "Gravity field order requested ({}) is greater than max gravity field order ({}).",
                    self.order, data.order
                );
            }
            GravityModel::PointMass => {
                panic!(
                    "Non-spherical gravity (spherical=false) is only supported for \
                     SphericalHarmonics gravity models. Set spherical=true for \
                     point-mass gravity sources."
                );
            }
        }

        // JEOD_INV: GV.06 — requested spherical-harmonics order must not exceed requested degree
        assert!(
            self.order <= self.degree,
            "Gravity field order ({}) is greater than gravity field degree ({}).",
            self.order,
            self.degree
        );

        // Gradient validation: JEOD spherical_harmonics_gravity_controls.cc:395-454
        // uses MessageHandler::error (non-fatal) and auto-corrects invalid values.
        if self.gradient {
            // JEOD_INV: GV.08 — gradient_degree <= degree (clamped)
            if self.gradient_degree > self.degree {
                warn!(
                    "Gravity gradient degree ({}) > gravity degree ({}); clamping.",
                    self.gradient_degree, self.degree
                );
                self.gradient_degree = self.degree;
            }
            // JEOD_INV: GV.09 — gradient_degree != 1 (reset to 0)
            if self.gradient_degree == 1 {
                warn!("Gravity gradient degree must not equal 1; resetting to 0.");
                self.gradient_degree = 0;
            }
            // JEOD_INV: GV.10 — gradient_order <= gradient_degree (clamped)
            if self.gradient_order > self.gradient_degree {
                warn!(
                    "Gravity gradient order ({}) > gradient degree ({}); clamping.",
                    self.gradient_order, self.gradient_degree
                );
                self.gradient_order = self.gradient_degree;
            }
            // JEOD_INV: GV.11 — gradient_order <= order (clamped)
            if self.gradient_order > self.order {
                warn!(
                    "Gravity gradient order ({}) > gravity order ({}); clamping.",
                    self.gradient_order, self.order
                );
                self.gradient_order = self.order;
            }
        }
    }

    /// Returns true if this control's *configuration* selects non-spherical
    /// (spherical-harmonics) computation, i.e. `spherical` is false and
    /// `degree > 0`.
    ///
    /// This is **purely config-based** — it does not consult the source.
    /// For the runtime question "will this control actually compute SH
    /// terms against `source`?", use [`Self::requires_planet_fixed_rotation`].
    /// Examples where `is_nonspherical()` returns true but the runtime
    /// path collapses to spherical:
    /// - source is `GravityModel::PointMass` (no SH data → effective degree = 0)
    /// - the configured degree exceeds the source degree (clamped down to 0)
    /// - the configured degree is 1 (Gottlieb returns zero perturbation for
    ///   degree < 2, so the per-step clamp collapses it to 0)
    pub fn is_nonspherical(&self) -> bool {
        !self.spherical && self.degree > 0
    }

    /// Returns true if evaluating this control against `source` will require
    /// the planet-fixed rotation matrix (i.e., the runtime path is genuinely
    /// non-spherical after clamping). Mirrors the gate inside `evaluate`'s
    /// dispatch.
    ///
    /// Use this for pre-flight checks (e.g., "does this source need a
    /// rotation model wired up?") instead of [`Self::is_nonspherical`],
    /// which is config-only and would over-approximate the requirement.
    pub fn requires_planet_fixed_rotation(&self, source: &GravitySource) -> bool {
        self.effective_orders(source).0 > 0
    }

    /// Compute the effective `(degree, order, gradient_degree, gradient_order)`
    /// quadruple after clamping to the source's bounds. Does not mutate `self`.
    ///
    /// This is the per-step variant of [`Self::check_validity`]: where
    /// `check_validity` is a startup gate that mutates `self` and panics on
    /// out-of-range degree/order (matching JEOD's "fatal" classification
    /// for GV.04 / GV.05 / GV.06 at initialization), `effective_orders`
    /// is the runtime path used by [`Self::evaluate_inner`] on every
    /// step. It clamps gracefully instead of panicking so a control that
    /// was constructed outside the validation pipeline (or mutated
    /// mid-mission) doesn't crash deep inside the spherical-harmonics
    /// kernel.
    ///
    /// Returns `(0, 0, 0, 0)` for spherical controls, point-mass sources,
    /// or any case where the request collapses to point-mass gravity
    /// (e.g., `spherical=false` against a `GravityModel::PointMass`).
    /// Callers see the spherical branch in [`Self::evaluate_inner`] when
    /// the returned degree is 0.
    // JEOD_INV: GV.04 — degree clamped to source degree (per-step path)
    // JEOD_INV: GV.05 — order clamped to source order
    // JEOD_INV: GV.06 — order clamped to degree
    // JEOD_INV: GV.08 — gradient_degree clamped to degree
    // JEOD_INV: GV.09 — gradient_degree=1 collapses to 0
    // JEOD_INV: GV.10 — gradient_order clamped to gradient_degree
    // JEOD_INV: GV.11 — gradient_order clamped to order
    fn effective_orders(&self, source: &GravitySource) -> (usize, usize, usize, usize) {
        if self.spherical {
            return (0, 0, 0, 0);
        }
        let (src_degree, src_order) = match &source.model {
            GravityModel::SphericalHarmonics(data) => (data.degree, data.order),
            // JEOD_INV: GV.07 — non-spherical against point-mass collapses
            // to spherical at startup; here we mirror by returning the
            // zero quadruple so `evaluate_inner` takes the spherical
            // branch.
            GravityModel::PointMass => return (0, 0, 0, 0),
        };
        let mut degree = self.degree.min(src_degree);
        // Gottlieb early-returns zero perturbation for degree < 2 (see
        // `calc_nonspherical_with_scratch`), so a configured degree of 1
        // produces no SH contribution. Collapse to 0 here so the runtime
        // predicate `eff_degree > 0` aligns with "the SH path actually does
        // work," and so the planet-fixed rotation matrix is not required
        // for a control whose computation degenerates to point-mass anyway.
        if degree == 1 {
            degree = 0;
        }
        // GV.06: order ≤ degree; GV.05: order ≤ source order
        let order = self.order.min(src_order).min(degree);
        // GV.08: gradient_degree ≤ degree
        let mut gradient_degree = self.gradient_degree.min(degree);
        // GV.09: gradient_degree of exactly 1 is meaningless; collapse to 0
        if gradient_degree == 1 {
            gradient_degree = 0;
        }
        // GV.10: gradient_order ≤ gradient_degree; GV.11: gradient_order ≤ order
        let gradient_order = self.gradient_order.min(gradient_degree).min(order);
        (degree, order, gradient_degree, gradient_order)
    }

    /// Evaluate this gravity control for a single source at the given position.
    ///
    /// Dispatches to spherical (point-mass) or non-spherical (spherical harmonics)
    /// gravity computation based on this control's configuration. For non-spherical
    /// gravity, `t_inertial_pfix` must be `Some` (matching JEOD's requirement that
    /// the planet-fixed frame is subscribed for non-spherical gravity).
    ///
    /// # Arguments
    /// - `source`: the gravity source (mu + model data)
    /// - `position`: body position relative to source center, in inertial frame
    /// - `t_inertial_pfix`: inertial-to-planet-fixed rotation (required for non-spherical)
    ///
    /// # Panics
    /// Panics if non-spherical gravity is requested but `t_inertial_pfix` is `None`.
    // JEOD_INV: GV.13 — gravity source must have inertial frame (planet-fixed rotation required for non-spherical)
    // JEOD_INV: GV.17 — active nonspherical controls subscribe to planet-fixed frame
    pub fn evaluate(
        &self,
        source: &GravitySource,
        position: DVec3,
        t_inertial_pfix: Option<&DMat3>,
        delta_c20: f64,
        has_delta_coeffs: bool,
    ) -> GravityAcceleration {
        self.evaluate_inner(
            source,
            position,
            t_inertial_pfix,
            self.gradient,
            self.gradient_degree,
            self.gradient_order,
            delta_c20,
            has_delta_coeffs,
        )
    }

    /// Like [`evaluate`](Self::evaluate), but passes `compute_gradient=false`
    /// regardless of this control's `gradient` flag.
    ///
    /// This skips the spherical-harmonics gradient tensor computation (the
    /// expensive part). Point-mass acceleration, potential, and point-mass
    /// gradient are still computed internally by `gravitation()` but the
    /// caller typically reads only `.grav_accel`.
    ///
    /// Use this in hot loops (e.g., RK4 inner stages) where only the
    /// gravitational acceleration vector is needed.
    pub fn evaluate_accel_only(
        &self,
        source: &GravitySource,
        position: DVec3,
        t_inertial_pfix: Option<&DMat3>,
        delta_c20: f64,
        has_delta_coeffs: bool,
    ) -> GravityAcceleration {
        self.evaluate_inner(
            source,
            position,
            t_inertial_pfix,
            false,
            0,
            0,
            delta_c20,
            has_delta_coeffs,
        )
    }

    /// Shared dispatch for [`evaluate`] and [`evaluate_accel_only`].
    ///
    /// All four spherical-harmonic ordinals (`degree`, `order`,
    /// `gradient_degree`, `gradient_order`) are clamped to the source's
    /// bounds via [`Self::effective_orders`] before reaching the
    /// `gravitation` kernel. This makes the per-step path safe against
    /// controls that were constructed outside the validation pipeline
    /// or mutated mid-mission — the kernel never sees an out-of-range
    /// request that would panic deep inside the spherical-harmonics
    /// recurrence. For already-validated controls (the common case),
    /// the clamp is a no-op and Tier 3 baselines stay bit-identical.
    // JEOD_INV: GV.13 — gravity source must have inertial frame (planet-fixed rotation required for non-spherical)
    // JEOD_INV: GV.17 — active nonspherical controls subscribe to planet-fixed frame
    #[allow(clippy::too_many_arguments)]
    fn evaluate_inner(
        &self,
        source: &GravitySource,
        position: DVec3,
        t_inertial_pfix: Option<&DMat3>,
        compute_gradient: bool,
        gradient_degree_request: usize,
        gradient_order_request: usize,
        delta_c20: f64,
        has_delta_coeffs: bool,
    ) -> GravityAcceleration {
        let (eff_degree, eff_order, mut eff_grad_degree, mut eff_grad_order) =
            self.effective_orders(source);
        // The caller may pass an explicit `gradient_degree_request` /
        // `gradient_order_request` — `evaluate_accel_only` passes 0/0 to
        // skip the gradient. Honor the caller's request *capped* by the
        // already-clamped ordinals.
        eff_grad_degree = eff_grad_degree.min(gradient_degree_request);
        eff_grad_order = eff_grad_order
            .min(gradient_order_request)
            .min(eff_grad_degree);

        if eff_degree > 0 {
            // Non-spherical path: requires the planet-fixed rotation.
            let rot = t_inertial_pfix.unwrap_or_else(|| {
                panic!(
                    "Non-spherical gravity (degree={}/order={}) requires planet-fixed \
                     rotation matrix. In JEOD, the planet-fixed frame is always \
                     subscribed for non-spherical gravity.",
                    eff_degree, eff_order
                )
            });
            crate::gravitation(
                source,
                position,
                rot,
                eff_degree,
                eff_order,
                self.perturbing_only,
                compute_gradient,
                eff_grad_degree,
                eff_grad_order,
                delta_c20,
                has_delta_coeffs,
            )
        } else {
            crate::gravitation(
                source,
                position,
                &DMat3::IDENTITY,
                0,
                0,
                self.perturbing_only,
                compute_gradient,
                eff_grad_degree,
                eff_grad_order,
                0.0,   // point-mass: no SH coefficients to modify
                false, // point-mass: no delta coefficients
            )
        }
    }
}

/// Typed sibling of [`GravityControl<SourceId>`].
///
/// Field-for-field parity with the untyped form, except the four
/// spherical-harmonic ordinals (`degree`, `order`, `gradient_degree`,
/// `gradient_order`) carry the [`HarmonicDegree`] newtype so the
/// compiler distinguishes them from angular `Angle` or dimensionless
/// `Ratio`.
///
/// Cross-field invariants like `degree <= source.degree` (JEOD
/// `GV.03`–`GV.11`) remain runtime-checked via
/// [`GravityControlTyped::check_validity`] (which delegates to the
/// untyped [`GravityControl::check_validity`]) — the type system
/// can prove ordinals are distinct kinds, not that one specific
/// ordinal is bounded by another's runtime value.
#[derive(Debug, Clone)]
pub struct GravityControlTyped<SourceId = String> {
    pub source_name: SourceId,
    pub gradient: bool,
    pub spherical: bool,
    pub degree: HarmonicDegree,
    pub order: HarmonicDegree,
    pub perturbing_only: bool,
    pub gradient_degree: HarmonicDegree,
    pub gradient_order: HarmonicDegree,
    pub differential: bool,
    pub battin_method: bool,
    pub relativistic: bool,
}

impl<SourceId> GravityControlTyped<SourceId> {
    /// Spherical (point-mass) typed control.
    pub fn new_spherical(source_name: SourceId, gradient: bool) -> Self {
        Self {
            source_name,
            gradient,
            spherical: true,
            degree: HarmonicDegree::default(),
            order: HarmonicDegree::default(),
            perturbing_only: false,
            gradient_degree: HarmonicDegree::default(),
            gradient_order: HarmonicDegree::default(),
            differential: false,
            battin_method: false,
            relativistic: false,
        }
    }

    /// Non-spherical (spherical-harmonics) typed control.
    pub fn new_nonspherical(
        source_name: SourceId,
        degree: HarmonicDegree,
        order: HarmonicDegree,
        gradient: bool,
    ) -> Self {
        Self {
            source_name,
            gradient,
            spherical: false,
            degree,
            order,
            perturbing_only: false,
            gradient_degree: HarmonicDegree::default(),
            gradient_order: HarmonicDegree::default(),
            differential: false,
            battin_method: false,
            relativistic: false,
        }
    }

    /// Third-body (point-mass + differential) typed control.
    pub fn new_third_body(source_name: SourceId) -> Self {
        Self {
            source_name,
            gradient: false,
            spherical: true,
            degree: HarmonicDegree::default(),
            order: HarmonicDegree::default(),
            perturbing_only: false,
            gradient_degree: HarmonicDegree::default(),
            gradient_order: HarmonicDegree::default(),
            differential: true,
            battin_method: false,
            relativistic: false,
        }
    }
}

impl<SourceId: Clone> GravityControlTyped<SourceId> {
    /// Validate this typed control against its gravity source.
    ///
    /// Delegates to [`GravityControl::check_validity`] on the untyped
    /// projection — runtime-checked invariants (`GV.03`–`GV.11`)
    /// stay in the canonical f64 path. Mutations the validator
    /// performs (e.g., auto-correcting `degree == 0` to
    /// `spherical = true`, clamping out-of-range gradient_degree /
    /// gradient_order) are reflected back into `self` via the
    /// `HarmonicDegree` newtypes.
    // JEOD_INV: GV.03 — check_validity() called on degree/order mutation
    pub fn check_validity(&mut self, source: &GravitySource) {
        let mut untyped = self.to_untyped();
        untyped.check_validity(source);
        // Reflect any auto-corrections back into the typed surface.
        self.spherical = untyped.spherical;
        self.degree = HarmonicDegree::from(untyped.degree);
        self.order = HarmonicDegree::from(untyped.order);
        self.gradient_degree = HarmonicDegree::from(untyped.gradient_degree);
        self.gradient_order = HarmonicDegree::from(untyped.gradient_order);
    }

    /// Drop the [`HarmonicDegree`] newtypes and emit the untyped
    /// storage form. Cross-field invariants (GV.03–GV.11) remain
    /// runtime-checked via the resulting
    /// [`GravityControl::check_validity`].
    pub fn to_untyped(&self) -> GravityControl<SourceId> {
        GravityControl {
            source_name: self.source_name.clone(),
            gradient: self.gradient,
            spherical: self.spherical,
            degree: self.degree.get(),
            order: self.order.get(),
            perturbing_only: self.perturbing_only,
            gradient_degree: self.gradient_degree.get(),
            gradient_order: self.gradient_order.get(),
            differential: self.differential,
            battin_method: self.battin_method,
            relativistic: self.relativistic,
        }
    }

    /// Wrap an untyped [`GravityControl`] as typed. Lossless conversion.
    pub fn from_untyped_unchecked(c: &GravityControl<SourceId>) -> Self {
        Self {
            source_name: c.source_name.clone(),
            gradient: c.gradient,
            spherical: c.spherical,
            degree: HarmonicDegree::from(c.degree),
            order: HarmonicDegree::from(c.order),
            perturbing_only: c.perturbing_only,
            gradient_degree: HarmonicDegree::from(c.gradient_degree),
            gradient_order: HarmonicDegree::from(c.gradient_order),
            differential: c.differential,
            battin_method: c.battin_method,
            relativistic: c.relativistic,
        }
    }
}

impl<SourceId: Default> Default for GravityControlTyped<SourceId> {
    fn default() -> Self {
        Self::new_spherical(SourceId::default(), false)
    }
}

impl<SourceId: Default> Default for GravityControl<SourceId> {
    fn default() -> Self {
        Self::new_spherical(SourceId::default(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spherical_harmonics_gravity_source::SphericalHarmonicsData;

    /// Build an in-memory spherical-harmonics source with all zero
    /// coefficients (degenerate, but sufficient for exercising the
    /// degree/order clamping logic — the actual numeric values don't
    /// matter; we're checking that the kernel runs without panicking
    /// and returns a finite acceleration). The coefficient arrays
    /// follow the triangular `cnm[n].len() == n + 1` shape that
    /// `SphericalHarmonicsData::new` expects.
    fn dummy_sh_source(degree: usize, order: usize) -> GravitySource {
        let mu = 3.986_004_415e14;
        let radius = 6_378_137.0;
        let cnm: Vec<Vec<f64>> = (0..=degree).map(|n| vec![0.0_f64; n + 1]).collect();
        let snm: Vec<Vec<f64>> = (0..=degree).map(|n| vec![0.0_f64; n + 1]).collect();
        let data = SphericalHarmonicsData::new(degree, order, radius, mu, cnm, snm, true, 0.0);
        GravitySource {
            mu,
            model: GravityModel::SphericalHarmonics(Box::new(data)),
        }
    }

    /// `effective_orders` for a spherical control returns all zeros
    /// regardless of the source's degree.
    #[test]
    fn effective_orders_spherical_returns_zeros() {
        let src = dummy_sh_source(8, 8);
        let ctrl = GravityControl::<usize>::new_spherical(0, false);
        assert_eq!(ctrl.effective_orders(&src), (0, 0, 0, 0));
    }

    /// `effective_orders` against a `PointMass` source collapses any
    /// non-spherical request to zeros (mirrors GV.07 startup auto-correct).
    #[test]
    fn effective_orders_against_point_mass_collapses_to_zero() {
        let src = GravitySource {
            mu: 3.986_004_415e14,
            model: GravityModel::PointMass,
        };
        let ctrl = GravityControl::<usize> {
            spherical: false,
            degree: 8,
            order: 8,
            ..GravityControl::new_spherical(0, false)
        };
        assert_eq!(ctrl.effective_orders(&src), (0, 0, 0, 0));
    }

    /// Out-of-range degree/order are clamped down to the source's bounds
    /// and to each other (GV.04, GV.05, GV.06).
    #[test]
    fn effective_orders_clamps_degree_order_to_source() {
        let src = dummy_sh_source(8, 8);
        let ctrl = GravityControl::<usize> {
            spherical: false,
            degree: 100,
            order: 100,
            ..GravityControl::new_spherical(0, false)
        };
        assert_eq!(ctrl.effective_orders(&src), (8, 8, 0, 0));
    }

    /// Order > degree clamps order down to degree (GV.06).
    #[test]
    fn effective_orders_clamps_order_to_degree() {
        let src = dummy_sh_source(8, 8);
        let ctrl = GravityControl::<usize> {
            spherical: false,
            degree: 4,
            order: 8,
            ..GravityControl::new_spherical(0, false)
        };
        assert_eq!(ctrl.effective_orders(&src), (4, 4, 0, 0));
    }

    /// Gradient ordinals are clamped: `gradient_degree=1` collapses to
    /// 0 (GV.09), `gradient_order ≤ gradient_degree` (GV.10), and
    /// `gradient_order ≤ order` (GV.11).
    #[test]
    fn effective_orders_clamps_gradient_ordinals() {
        let src = dummy_sh_source(8, 6);
        let ctrl = GravityControl::<usize> {
            spherical: false,
            degree: 8,
            order: 4,
            gradient: true,
            gradient_degree: 1, // → collapses to 0
            gradient_order: 5,  // > gradient_degree, > order → clamped
            ..GravityControl::new_spherical(0, false)
        };
        // After clamping: degree=8, order=4 (≤ src.order=6), gradient_degree=0, gradient_order=0.
        assert_eq!(ctrl.effective_orders(&src), (8, 4, 0, 0));
    }

    /// Regression test for H4: a control with out-of-range
    /// `gradient_degree` constructed without going through `check_validity`
    /// previously panicked deep inside `gravitation`. After the fix,
    /// `evaluate_inner` clamps gracefully and returns a finite acceleration.
    #[test]
    fn evaluate_does_not_panic_on_out_of_range_gradient_degree() {
        let src = dummy_sh_source(4, 4);
        let ctrl = GravityControl::<usize> {
            spherical: false,
            degree: 4,
            order: 4,
            gradient: true,
            gradient_degree: 100, // wildly out of range
            gradient_order: 100,
            ..GravityControl::new_spherical(0, false)
        };
        let pos = DVec3::new(7_000_000.0, 0.0, 0.0);
        let rot = DMat3::IDENTITY;
        // No panic; the kernel sees clamped (gradient_degree, gradient_order)=(4, 4).
        let result = ctrl.evaluate(&src, pos, Some(&rot), 0.0, false);
        assert!(result.grav_accel.is_finite());
    }

    /// Regression test for H4: a control with out-of-range `degree` /
    /// `order` against a real spherical-harmonics source no longer
    /// panics deep in `calc_nonspherical`.
    #[test]
    fn evaluate_does_not_panic_on_out_of_range_degree() {
        let src = dummy_sh_source(4, 4);
        let ctrl = GravityControl::<usize> {
            spherical: false,
            degree: 100,
            order: 100,
            ..GravityControl::new_spherical(0, false)
        };
        let pos = DVec3::new(7_000_000.0, 0.0, 0.0);
        let rot = DMat3::IDENTITY;
        let result = ctrl.evaluate(&src, pos, Some(&rot), 0.0, false);
        assert!(result.grav_accel.is_finite());
    }

    /// `degree == 1` is meaningless for spherical harmonics: Gottlieb
    /// returns zero perturbation for `degree < 2`. `effective_orders`
    /// collapses such a control to all-zeros so the runtime path takes
    /// the spherical branch and does not require the planet-fixed
    /// rotation matrix.
    #[test]
    fn effective_orders_collapses_degree_one_to_zero() {
        let src = dummy_sh_source(8, 8);
        let ctrl = GravityControl::<usize> {
            spherical: false,
            degree: 1,
            order: 1,
            ..GravityControl::new_spherical(0, false)
        };
        assert_eq!(ctrl.effective_orders(&src), (0, 0, 0, 0));
    }

    /// `requires_planet_fixed_rotation` returns false when the runtime
    /// path collapses to spherical (`PointMass` source, `degree=0`,
    /// `degree=1`), even if `is_nonspherical()` is true. Pre-flight
    /// checks must use this method, not `is_nonspherical()`.
    #[test]
    fn requires_rotation_false_when_runtime_collapses() {
        let sh = dummy_sh_source(8, 8);
        let pm = GravitySource {
            mu: 3.986_004_415e14,
            model: GravityModel::PointMass,
        };
        let degree_one = GravityControl::<usize> {
            spherical: false,
            degree: 1,
            order: 1,
            ..GravityControl::new_spherical(0, false)
        };
        let against_pm = GravityControl::<usize> {
            spherical: false,
            degree: 8,
            order: 8,
            ..GravityControl::new_spherical(0, false)
        };
        assert!(degree_one.is_nonspherical()); // config says yes
        assert!(!degree_one.requires_planet_fixed_rotation(&sh)); // runtime says no
        assert!(against_pm.is_nonspherical()); // config says yes
        assert!(!against_pm.requires_planet_fixed_rotation(&pm)); // runtime says no

        let real_sh = GravityControl::<usize> {
            spherical: false,
            degree: 4,
            order: 4,
            ..GravityControl::new_spherical(0, false)
        };
        assert!(real_sh.requires_planet_fixed_rotation(&sh));
    }

    /// Regression test for the degree-1 panic (review feedback on PR #182):
    /// a non-spherical control with `degree=1` against a SH source no
    /// longer panics when `t_inertial_pfix` is `None`, because
    /// `effective_orders` collapses degree=1 to 0 (Gottlieb produces zero
    /// perturbation for degree < 2 anyway).
    #[test]
    fn evaluate_degree_one_does_not_require_rotation() {
        let src = dummy_sh_source(8, 8);
        let ctrl = GravityControl::<usize> {
            spherical: false,
            degree: 1,
            order: 1,
            ..GravityControl::new_spherical(0, false)
        };
        let pos = DVec3::new(7_000_000.0, 0.0, 0.0);
        // No rotation matrix supplied; would have panicked previously.
        let result = ctrl.evaluate(&src, pos, None, 0.0, false);
        // Should match point-mass gravity from the source (the SH branch
        // contributes zero for degree < 2 and `perturbing_only` is false).
        assert!(result.grav_accel.is_finite());
        assert!(result.grav_accel.length() > 0.0);
    }
}
