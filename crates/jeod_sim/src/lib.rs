//! ECS-agnostic orchestration layer for JEOD physics.
//!
//! This crate sits between the pure physics layer (`jeod_*` crates) and
//! framework-specific adapters (e.g., `bevy_jeod_*`). It provides two APIs:
//!
//! ## Per-body functions (primary API for ECS integration)
//!
//! Composable, borrow-based functions that any ECS adapter can call from
//! its system functions. The ECS world remains the single source of truth.
//!
//! - [`accumulate_gravity`] — gravity accumulation across sources
//! - [`evaluate_atmosphere`] — atmosphere evaluation pipeline
//! - [`collect_and_resolve_forces`] — force/torque collection with frame transforms
//! - [`integrate_body`] — RK4 integration with 6-DOF/3-DOF routing
//! - [`validate_body`] — JEOD invariant checking
//!
//! ## Simulation runner (for non-ECS use)
//!
//! A standalone [`Simulation`] struct for batch propagation, scripting, and
//! tests. Owns all state and runs the pipeline internally. ECS adapters
//! should **not** use this — use the per-body functions instead.
//!
//! ## Pipeline ordering
//!
//! See [`PipelineStage`] and [`PIPELINE_ORDER`] for the canonical stage
//! execution order that any adapter must respect.

pub mod atmosphere;
pub mod forces;
pub mod gravity;
pub mod integration;
pub mod interactions;
pub mod pipeline;
pub mod simulation;
pub mod validation;

// Re-exports for convenience
pub use atmosphere::{evaluate_atmosphere, AtmosphereConfig, AtmosphereModel};
pub use forces::collect_and_resolve_forces;
pub use gravity::accumulate_gravity;
pub use integration::integrate_body;
pub use interactions::{compute_drag, compute_gravity_torque, compute_spherical_srp};
pub use pipeline::{PipelineStage, PIPELINE_ORDER};
pub use simulation::{GravitySourceEntry, SimBody, Simulation};
pub use validation::{validate_body, ValidationError};
