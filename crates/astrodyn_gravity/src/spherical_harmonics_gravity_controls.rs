//! Multi-source [`GravityControls`] aggregator — the per-vehicle list
//! of [`GravityControl`] entries the gravity pipeline iterates each
//! step.
//!
//! Mirrors the `GravityControls` collection inside JEOD's
//! [`spherical_harmonics_gravity_controls.cc`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/gravity/src/spherical_harmonics_gravity_controls.cc)
//! from JEOD v5.4.0.

use crate::gravity_controls::GravityControl;

/// Ordered list of [`GravityControl`] entries — one per gravity source
/// active for the vehicle. The pipeline accumulates each source's
/// contribution into the vehicle's net gravity acceleration in
/// iteration order.
#[derive(Debug, Clone, Default)]
pub struct GravityControls<SourceId = String> {
    /// Per-source controls. Order is significant for differential /
    /// third-body sources whose contributions partially cancel.
    pub controls: Vec<GravityControl<SourceId>>,
}
