#[derive(Debug, Clone)]
pub struct GravityControl<SourceId = String> {
    pub source_id: SourceId,
    pub gradient: bool,
    /// Max degree for this evaluation (None = use source max).
    pub degree: Option<usize>,
    /// Max order for this evaluation (None = use source max).
    pub order: Option<usize>,
    /// If true, exclude point-mass (n=0,1) terms.
    pub perturbing_only: bool,
    /// Max degree for gradient computation.
    pub gradient_degree: Option<usize>,
    /// Max order for gradient computation.
    pub gradient_order: Option<usize>,
}

impl<SourceId> GravityControl<SourceId> {
    /// Create a gravity control with default truncation settings.
    pub fn new(source_id: SourceId, gradient: bool) -> Self {
        Self {
            source_id,
            gradient,
            degree: None,
            order: None,
            perturbing_only: false,
            gradient_degree: None,
            gradient_order: None,
        }
    }
}

impl<SourceId: Default> Default for GravityControl<SourceId> {
    fn default() -> Self {
        Self::new(SourceId::default(), false)
    }
}
