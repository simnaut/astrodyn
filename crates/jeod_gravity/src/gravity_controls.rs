use crate::gravity_source::GravitySource;

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
    /// - degree > source degree
    /// - order > source order
    /// - order > degree
    /// - gradient_degree > degree
    /// - gradient_order > gradient_degree
    /// - gradient_order > order
    pub fn check_validity(&self, source: &GravitySource) {
        if self.spherical {
            return;
        }

        assert!(
            self.degree > 0,
            "Non-spherical gravity (spherical=false) requires degree > 0. \
             Set spherical=true for point-mass gravity."
        );

        if let crate::gravity_source::GravityModel::SphericalHarmonics(ref data) = source.model {
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

        assert!(
            self.order <= self.degree,
            "Gravity field order ({}) is greater than gravity field degree ({}).",
            self.order, self.degree
        );

        if self.gradient {
            if self.gradient_degree > self.degree {
                panic!(
                    "Gravity gradient degree ({}) is greater than gravity degree ({}).",
                    self.gradient_degree, self.degree
                );
            }
            assert!(
                self.gradient_degree != 1,
                "Gravity gradient degree must not equal 1."
            );
            if self.gradient_order > self.gradient_degree {
                panic!(
                    "Gravity gradient order ({}) is greater than gravity gradient degree ({}).",
                    self.gradient_order, self.gradient_degree
                );
            }
            if self.gradient_order > self.order {
                panic!(
                    "Gravity gradient order ({}) is greater than gravity order ({}).",
                    self.gradient_order, self.order
                );
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
