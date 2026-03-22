use glam::DVec3;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TranslationalState {
    pub position: DVec3, // m, in integration frame
    pub velocity: DVec3, // m/s, in integration frame
}
