//! ECS-agnostic orchestration layer for JEOD physics.
//!
//! This crate is the **single dependency** for ECS adapters. It re-exports all
//! types from the `jeod_*` physics crates that an adapter needs, plus
//! orchestration functions that compose them into pipeline stages.
//!
//! ## Per-body functions (primary API for ECS integration)
//!
//! Composable, borrow-based functions that any ECS adapter can call from
//! its system functions. The ECS world remains the single source of truth.
//!
//! - [`accumulate_gravity`] — gravity accumulation across sources
//! - [`evaluate_atmosphere`] — atmosphere evaluation pipeline
//! - [`compute_drag`] — aerodynamic drag with frame transform
//! - [`compute_gravity_torque`] — gravity gradient torque with quaternion conversion
//! - [`compute_spherical_srp`] — spherical solar radiation pressure
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

// ── Orchestration functions ──
pub use atmosphere::{evaluate_atmosphere, AtmosphereConfig, AtmosphereModel};
pub use forces::collect_and_resolve_forces;
pub use gravity::accumulate_gravity;
pub use integration::integrate_body;
pub use interactions::{compute_drag, compute_gravity_torque, compute_spherical_srp};
pub use pipeline::{PipelineStage, PIPELINE_ORDER};
pub use simulation::{GravitySourceEntry, SimBody, Simulation};
pub use validation::{validate_body, ValidationError};

// ── Re-exports from jeod_* crates ──
// ECS adapters depend only on jeod_sim — these re-exports provide all the
// types needed for component definitions, system parameters, and resources.

// jeod_dynamics: state types, force types, mass, config, frame utilities
pub use jeod_dynamics::{
    compute_t_inertial_struct, DynamicsConfig, ForceContributions, FrameDerivatives,
    GravityAcceleration, MassProperties, RotationalState, SixDofState, TotalForce,
    TranslationalState, INERTIA_CONSISTENCY_TOL,
};

// jeod_gravity: source definitions and controls
pub use jeod_gravity::{GravityControl, GravityControls, GravityModel, GravitySource};

// jeod_atmosphere: state output
pub use jeod_atmosphere::AtmosphereState;

// jeod_interactions: config, result types, and computation functions
pub use jeod_interactions::{
    compute_flat_plate_srp_thermal, compute_shadow_fraction, solar_flux_at_distance,
    AerodynamicForce, DragConfig, FlatPlate, FlatPlateParams, FlatPlateSrpResult, FlatPlateThermal,
    RadiationForce, SrpConfig, SOLAR_RADIUS,
};

// jeod_frames: reference frame state
pub use jeod_frames::RefFrameState;

// jeod_time: simulation time and leap seconds
pub use jeod_time::{leap_second::default_leap_second_table, SimulationTime};

// jeod_time: planet rotation (used by ephemeris stage)
pub use jeod_frames::rotation_j2000::compute_t_parent_this_from_tjt;

// jeod_ephemeris: ephemeris data
pub use jeod_ephemeris::Ephemeris;

// jeod_planet: planet shape
pub use jeod_planet::PlanetShape;

// jeod_math: quaternion type (used in RotationalState)
pub use jeod_math::JeodQuat;
