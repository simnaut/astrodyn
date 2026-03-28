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
}

impl<SourceId> GravityControl<SourceId> {
    /// Create a spherical (point-mass) gravity control.
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
        }
    }

    /// Create a non-spherical (spherical harmonics) gravity control.
    pub fn new_nonspherical(source_name: SourceId, degree: usize, order: usize, gradient: bool) -> Self {
        Self {
            source_name,
            gradient,
            spherical: false,
            degree,
            order,
            perturbing_only: false,
            gradient_degree: 0,
            gradient_order: 0,
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
    pub fn check_validity(&mut self, source: &GravitySource) {
        if self.spherical {
            return;
        }

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
                assert!(
                    self.degree <= data.degree,
                    "Gravity field degree requested ({}) is greater than max gravity field degree ({}).",
                    self.degree, data.degree
                );
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

        assert!(
            self.order <= self.degree,
            "Gravity field order ({}) is greater than gravity field degree ({}).",
            self.order, self.degree
        );

        // Gradient validation: JEOD spherical_harmonics_gravity_controls.cc:395-454
        // uses MessageHandler::error (non-fatal) and auto-corrects invalid values.
        if self.gradient {
            if self.gradient_degree > self.degree {
                warn!(
                    "Gravity gradient degree ({}) > gravity degree ({}); clamping.",
                    self.gradient_degree, self.degree
                );
                self.gradient_degree = self.degree;
            }
            if self.gradient_degree == 1 {
                warn!(
                    "Gravity gradient degree must not equal 1; resetting to 0."
                );
                self.gradient_degree = 0;
            }
            if self.gradient_order > self.gradient_degree {
                warn!(
                    "Gravity gradient order ({}) > gradient degree ({}); clamping.",
                    self.gradient_order, self.gradient_degree
                );
                self.gradient_order = self.gradient_degree;
            }
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
}

impl<SourceId: Default> Default for GravityControl<SourceId> {
    fn default() -> Self {
        Self::new_spherical(SourceId::default(), false)
    }
}
