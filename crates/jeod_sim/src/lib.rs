// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
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
//! - [`collect_and_resolve_forces`] — force/torque collection with frame transforms
//! - [`integrate_body`] — RK4 integration with 6-DOF/3-DOF routing
//! - [`validate_body`] — JEOD invariant checking
//! - [`compute_orbital_elements`] — orbital elements from translational state
//! - [`compute_body_euler_angles`] — Euler angles from body attitude
//! - [`compute_body_lvlh_frame`] — LVLH frame from translational state
//! - [`compute_body_geodetic`] — geodetic coordinates from inertial position
//! - [`compute_body_solar_beta`] — solar beta angle
//!
//! ## Standalone runner
//!
//! For batch propagation and Tier 3 tests, see the `jeod_runner` crate which
//! provides a standalone `Simulation` struct that owns all state and drives the
//! pipeline. ECS adapters should **not** use `jeod_runner` — use the per-body
//! functions from this crate instead.
//!
//! ## Pipeline ordering
//!
//! See [`PipelineStage`] and [`PIPELINE_ORDER`] for the canonical stage
//! execution order that any adapter must respect.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod atmosphere;
pub mod attach;
pub mod body_action;
pub mod derived;
pub mod forces;
pub mod frame_orchestration;
pub mod gravity;
pub mod integrable;
pub mod integration;
pub mod interactions;
pub mod kinematic_propagation;
pub mod pipeline;
pub mod planet_config;
pub mod recipes;
pub mod rotation_model;
pub mod simulation_builder;
pub mod source_frames;
pub mod source_state;
pub mod sources;
pub mod validation;
pub mod vehicle_builder;
pub mod vehicle_config;
pub mod wrench;

// ── Orchestration functions ──
pub use atmosphere::{
    evaluate_atmosphere, evaluate_atmosphere_typed, evaluate_body_atmosphere,
    evaluate_body_atmosphere_typed, run_atmosphere_stage, AtmosphereBodyInputs, AtmosphereConfig,
    AtmosphereModel,
};
pub use attach::{
    stage_attach_combine, stage_detach_capture, StageAttachInputs, StageAttachOutputs,
};
pub use body_action::{BodyAction, LvlhAngularVelocityFrame, OrbitalElementSet};
pub use derived::{
    compute_body_euler_angles, compute_body_euler_angles_typed, compute_body_geodetic,
    compute_body_geodetic_typed, compute_body_lvlh_frame, compute_body_lvlh_frame_typed,
    compute_body_solar_beta, compute_body_solar_beta_typed, compute_lvlh_relative_state,
    compute_lvlh_relative_state_typed, compute_orbital_elements, compute_orbital_elements_typed,
    compute_relative_state, LvlhRelativeState, RelativeState, RelativeTranslation,
};
pub use forces::{collect_and_resolve_forces, collect_and_resolve_forces_typed};
pub use frame_orchestration::{
    compute_relative_state_typed, evaluate_and_apply_frame_switch, frame_origin,
    frame_origin_typed, sync_pfix_rotation, FrameSwitchTargetMissing,
};
pub use gravity::{
    accumulate_gravity, accumulate_gravity_typed, accumulate_relativistic_corrections,
    accumulate_relativistic_corrections_typed, evaluate_body_gravity, evaluate_body_gravity_typed,
    run_gravity_stage, GravityBodyInputs, ResolvedRelativisticSource, ResolvedSource,
};
pub use integrable::IntegrableObject;
pub use integration::{
    integrate_bodies_contact_coupled, integrate_body, integrate_body_coupled, integrate_body_typed,
    reset_integrators, CoupledBodyInput, CoupledIntegScratch, CoupledStageEval,
};
pub use interactions::{
    compute_cannonball_srp, compute_cannonball_srp_typed, compute_drag, compute_drag_typed,
    compute_gravity_torque, compute_gravity_torque_typed, evaluate_contact_pair,
    evaluate_ground_contact_pair, ContactPairEval, FlatPlateStageInputs, FlatPlateState,
    GroundContactPairEval, ThermalIntegrationOrder,
};
pub use jeod_dynamics::kinematic_joint::{
    evaluate as evaluate_joint_kinematics, evaluate_closure as evaluate_closure_kinematics,
    evaluate_model as evaluate_joint_kinematics_model,
    evaluate_multi_dof as evaluate_multi_dof_kinematics,
    evaluate_sinusoidal as evaluate_sinusoidal_kinematics, ClosureJointKinematicsSpec,
    JointKinematicsModel, JointKinematicsSpec, MultiDofJointKinematicsSpec, SingleDofKinematics,
    SinusoidalJointKinematicsSpec, AXIS_NORM_TOL, MAX_MULTI_DOF_AXES,
};
pub use jeod_dynamics::{
    Abm4State, GaussJacksonConfig, GaussJacksonState, IntegratorResult, IntegratorType,
};
pub use kinematic_propagation::{propagate_state_via_storage, KinematicEdge, KinematicNodeState};
pub use pipeline::{PipelineStage, PIPELINE_ORDER};
pub use planet_config::{PlanetConfig, EARTH, MARS, MOON, SUN};
pub use rotation_model::RotationModel;
pub use simulation_builder::{MassTreeAttachment, SimulationBuilder};
pub use source_frames::SourceFrameIds;
pub use source_state::{
    set_source_position, set_source_state, source_frame_id, source_pfix_rotation, source_position,
};
pub use sources::GravitySourceEntry;
pub use validation::{validate_body, ValidationError};
pub use vehicle_builder::{
    BuildState, HasIntegrator, NeedsMass, NeedsState, Ready, VehicleBuilder,
};
pub use vehicle_config::{
    DerivedStateConfig, EarthLightingConfig, FrameSwitchConfig, GeodeticConfig, ShadowBody,
    SrpModel, SwitchSense, VehicleConfig,
};
pub use wrench::{aggregate_wrenches_via_storage, edge_geometry_from_composites, EdgeGeometry};

// ── Re-exports from jeod_* crates ──
// ECS adapters depend only on jeod_sim — these re-exports provide all the
// types needed for component definitions, system parameters, and resources.

// jeod_dynamics: state types, force types, mass, config, frame utilities
pub use jeod_dynamics::{
    combine_states_at_attach, compute_frame_derivatives, compute_kinematic_child_state,
    compute_kinematic_child_state_dquat, compute_kinematic_child_state_typed,
    compute_node_composite, compute_t_inertial_struct, compute_translational_derivatives,
    derive_frame_attached_state, derive_kinematic_child_from_states,
    finalize_child_in_parent_frame, propagate_forward, recompute_composites_via_storage,
    shift_wrench_to_parent, shift_wrench_to_parent_typed, AttachCombineInputs,
    AttachCombineOutputs, DetachedSubtreeState, DynamicsConfig, ForceContributions,
    FrameAttachInputs, FrameDerivatives, GravityAcceleration, KinematicChildInputs,
    KinematicChildOutputs, MassBodyId, MassNodeOutputs, MassNodeView, MassPointState,
    MassProperties, MassStorage, MassTree, RotationalState, SixDofState, TotalForce,
    TranslationalState, Wrench, INERTIA_CONSISTENCY_TOL,
};

// jeod_dynamics typed siblings (used by Bevy components after #172 H1
// migration so the ECS storage layer carries frame phantoms rather
// than re-lifting raw DVec3 every step).
pub use jeod_dynamics::forces::{FrameDerivativesTyped, GravityAccelerationTyped, TotalForceTyped};
pub use jeod_dynamics::mass::MassPropertiesTyped;
pub use jeod_dynamics::rotational::RotationalStateTyped;
pub use jeod_dynamics::state::TranslationalStateTyped;

// jeod_gravity: source definitions, controls, and tides
pub use jeod_gravity::tides::{
    compute_delta_c20, compute_delta_c20_typed, TidalBody, TidalConfig, TidalConfigTyped, EARTH_K2,
};
pub use jeod_gravity::{GravityControl, GravityControls, GravityModel, GravitySource};

// jeod_atmosphere: state output and model types
pub use jeod_atmosphere::exponential::ExponentialAtmosphere;
pub use jeod_atmosphere::met::{self as met_atmosphere, GeoIndexType, MetAtmosphere};
pub use jeod_atmosphere::AtmosphereState;

// jeod_interactions: config, result types, and computation functions
pub use jeod_interactions::{
    compute_contact_force, compute_contact_force_from_geometry, compute_contact_geometry,
    compute_earth_lighting, compute_earth_lighting_typed, compute_flat_plate_srp_thermal,
    compute_flat_plate_srp_thermal_conduction, compute_ground_contact_geometry,
    compute_shadow_fraction, solar_flux_at_distance, AerodynamicForce, AerodynamicForceTyped,
    ContactFacet, ContactForce, ContactGeometry, ContactMaterial, ContactShape, DragConfig,
    DragConfigTyped, EarthLightingState, FlatPlate, FlatPlateParams, FlatPlateSrpResult,
    FlatPlateThermal, GroundFacet, LightingBody, LightingParams, Phase, RadiationForce,
    SphericalTerrain, Terrain, ThermalConductionMatrix, SOLAR_RADIUS, SPEED_OF_LIGHT,
};

// jeod_frames: reference frame state and arena-based frame tree.
// `FrameTree`/`FrameId`/`FrameNode`/`RefFrameKind` are re-exported here so
// every consumer of `jeod_sim` (`jeod_runner`, `bevy_jeod`, mission crates)
// can build and walk the frame hierarchy without depending on `jeod_frames`
// directly. Issue #71 — without this re-export, frame-tree orchestration
// is bottled up inside `jeod_runner::Simulation` and unavailable to the
// Bevy adapter.
//
// `RefFrameRot`/`RefFrameTrans` are the per-link state structs callers
// need when constructing or reading frame nodes; `RefFrameStateTyped` is
// the typed sibling for Phase C of issue #71.
pub use jeod_frames::{
    common_ancestor as frame_common_ancestor, compose_to_ancestor as frame_compose_to_ancestor,
    compute_relative_state as frame_compute_relative_state_via_storage, FrameId, FrameNode,
    FrameStorage, FrameTree, RefFrameKind, RefFrameRot, RefFrameState, RefFrameStateTyped,
    RefFrameTrans,
};

// jeod_time: simulation time, leap seconds, epoch constants, and time scale network
pub use jeod_time::{
    epoch::{J2000_TT_JD, J2000_TT_TJT, SECONDS_PER_DAY},
    leap_second::{default_leap_second_table, LeapSecondTable},
    time_utc::{calendar_to_tjt, tjt_to_calendar, CalendarDate},
    DynamicTime, MissionElapsedTime, SimulationTime, TimeManager, TimeScaleId, UserDefinedEpoch,
    TAI_GPS_OFFSET,
};

// jeod_time: planet rotation (used by ephemeris stage)
pub use jeod_frames::rotation_j2000::{
    compute_t_parent_this_from_tjt, compute_t_parent_this_from_tjt_with_polar, polar_motion_matrix,
};
pub use jeod_frames::rotation_mars;
pub use jeod_frames::rotation_moon;

// jeod_ephemeris: ephemeris data
pub use jeod_ephemeris::{Ephemeris, EphemerisBody};

// jeod_gravity: binary coefficient loading. The JEOD `.cc` source-file
// parser (`load_from_jeod_cc`, `load_mu_from_jeod_cc`) lives in the
// dev/test crate `jeod_test_data::jeod_cc` — production gravity does
// not parse JEOD source.
pub use jeod_gravity::coefficients;
pub use jeod_gravity::relativistic;

// jeod_planet: planet shape
pub use jeod_planet::PlanetShape;

// jeod_quantities: typed-quantity foundation. ECS adapters (e.g. the
// `bevy_jeod` root crate) consume these types via `jeod_sim` to
// preserve the "single dependency" invariant.
pub use jeod_quantities::aliases::{
    Acceleration, AngularAcceleration, AngularMomentum, AngularVelocity, Force, Jerk, Position,
    Torque, Velocity,
};
pub use jeod_quantities::diagnostics::{CompatibleVehiclePair, CompatibleVehicles};
pub use jeod_quantities::dims::{GravParam, MassFlowRate, SpecificAngMom, SpecificEnergy};
pub use jeod_quantities::ext::{Array3Ext, F64Ext, Vec3Ext};
pub use jeod_quantities::frame::{
    BodyFrame, Earth, Ecef, Frame, IntegrationFrame, Lvlh, Mars, Moon, Ned, Planet, PlanetFixed,
    PlanetInertial, RootInertial, SelfPlanet, SelfRef, StructuralFrame, Sun, Vehicle,
};
pub use jeod_quantities::integ_origin::IntegOrigin;
// Macros that mint downstream `Vehicle`/`Planet` markers. Re-exported so
// mission crates depending only on `jeod_sim` don't need a direct
// `jeod_quantities` line in their `Cargo.toml`. The macro body resolves
// `$crate` to `jeod_quantities` regardless of where the macro is
// invoked from, so the sealed-trait bound is satisfied transparently.
pub use jeod_quantities::body_attitude::BodyAttitude;
pub use jeod_quantities::frame_transform::FrameTransform;
pub use jeod_quantities::inertia::InertiaTensor;
pub use jeod_quantities::qty3::Qty3;
pub use jeod_quantities::quat::{LeftTransform, NormalizedQuat, ScalarFirst};
pub use jeod_quantities::{define_planet, define_vehicle};

// uom scalar quantities used directly by the Bevy adapter for typed
// component fields (`Angle` for Euler angles, `Ratio` for tidal ΔC20).
pub use uom::si::f64::{Angle, Ratio};

/// Convenience constructor — wrap a raw f64 (radians) as a typed
/// [`Angle`].
///
/// Mirrors `jeod_quantities::ext::F64Ext::rad(f)` but exists at the
/// `jeod_sim` boundary so ECS adapters don't need to import
/// `jeod_quantities` (or `uom::si::angle::radian`) directly.
#[inline]
pub fn radians(value: f64) -> Angle {
    Angle::new::<uom::si::angle::radian>(value)
}

/// Convenience constructor — wrap a raw f64 (dimensionless) as a typed
/// [`Ratio`].
#[inline]
pub fn dimensionless(value: f64) -> Ratio {
    Ratio::new::<uom::si::ratio::ratio>(value)
}

// jeod_math: quaternion type (used in RotationalState)
pub use jeod_math::JeodQuat;

// jeod_math: derived state types
pub use jeod_math::{
    cartesian_to_geodetic_typed, compute_euler_angles_from_matrix_typed, compute_lvlh_frame_typed,
    geodetic_to_cartesian_typed, solar_beta_angle_typed, EulerSequence, GeodeticState,
    GeodeticStateTyped, LvlhFrame, OrbitalElements, OrbitalError,
};
