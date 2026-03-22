#[derive(Debug, Clone, Default)]
pub struct GravityControls<SourceId = String> {
    pub controls: Vec<GravityControl<SourceId>>,
}

#[derive(Debug, Clone)]
pub struct GravityControl<SourceId = String> {
    pub source_id: SourceId,
    pub compute_gradient: bool,
}
