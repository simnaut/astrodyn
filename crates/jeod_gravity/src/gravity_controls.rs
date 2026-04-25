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

    /// Returns true if this control requires non-spherical (spherical harmonics)
    /// computation, i.e. `spherical` is false and degree > 0.
    pub fn is_nonspherical(&self) -> bool {
        !self.spherical && self.degree > 0
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
    // JEOD_INV: GV.13 — gravity source must have inertial frame (planet-fixed rotation required for non-spherical)
    // JEOD_INV: GV.17 — active nonspherical controls subscribe to planet-fixed frame
    #[allow(clippy::too_many_arguments)]
    fn evaluate_inner(
        &self,
        source: &GravitySource,
        position: DVec3,
        t_inertial_pfix: Option<&DMat3>,
        compute_gradient: bool,
        gradient_degree: usize,
        gradient_order: usize,
        delta_c20: f64,
        has_delta_coeffs: bool,
    ) -> GravityAcceleration {
        if self.is_nonspherical() {
            let rot = t_inertial_pfix.unwrap_or_else(|| {
                panic!(
                    "Non-spherical gravity (degree={}/order={}) requires planet-fixed \
                     rotation matrix. In JEOD, the planet-fixed frame is always \
                     subscribed for non-spherical gravity.",
                    self.degree, self.order
                )
            });
            crate::gravitation(
                source,
                position,
                rot,
                self.degree,
                self.order,
                self.perturbing_only,
                compute_gradient,
                gradient_degree,
                gradient_order,
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
                gradient_degree,
                gradient_order,
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
