//! Canonical pipeline stage ordering shared by every JEOD adapter.
//!
//! Adapters that schedule per-stage system calls (the Bevy plugin, the
//! standalone runner, batch tools) all consult [`PIPELINE_ORDER`] for the
//! authoritative sequence so the order cannot drift between adapters.

/// Pipeline stages mirroring JEOD's init/update ordering.
///
/// JEOD's simulation loop runs these stages in strict order every timestep.
/// ECS adapters must schedule their systems to respect this ordering.
/// The `astrodyn_runner::Simulation` struct runs them internally in `step()`.
///
/// Ordering dependencies:
/// - **Time** must be current before frame transforms (GMST, TT for RNP).
/// - **Ephemeris** (planet rotations) must be current before spherical-harmonic gravity.
/// - **Gravity gradient** must be available before gravity torque.
/// - **All forces** must be collected before integration.
/// - **Gravity** is precomputed in Environment but recomputed per RK4 stage during Integration.
// JEOD_INV: DM.04 — system ordering mirrors JEOD init/update pipeline
// JEOD_INV: DM.13 — ephemeris updated before gravity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    /// Advance TAI, UTC, TDB, GMST, etc.
    TimeUpdate,
    /// Update planet positions and rotations (DE4xx ephemeris, RNP).
    EphemerisUpdate,
    /// Compute gravity and atmosphere for each body.
    Environment,
    /// Compute interactions: aerodynamic drag, SRP, gravity torque.
    Interaction,
    /// Collect all forces/torques, resolve frame transforms.
    ForceCollection,
    /// Integrate translational and rotational state (RK4).
    Integration,
    /// Compute derived quantities: orbital elements, Euler angles.
    DerivedState,
}

/// Canonical pipeline execution order.
pub const PIPELINE_ORDER: &[PipelineStage] = &[
    PipelineStage::TimeUpdate,
    PipelineStage::EphemerisUpdate,
    PipelineStage::Environment,
    PipelineStage::Interaction,
    PipelineStage::ForceCollection,
    PipelineStage::Integration,
    PipelineStage::DerivedState,
];
