use crate::gravity_controls::GravityControl;

#[derive(Debug, Clone, Default)]
pub struct GravityControls<SourceId = String> {
    pub controls: Vec<GravityControl<SourceId>>,
}
