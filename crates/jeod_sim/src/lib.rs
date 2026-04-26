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

pub mod atmosphere;
pub mod derived;
pub mod forces;
pub mod gravity;
pub mod integrable;
pub mod integration;
pub mod interactions;
pub mod pipeline;
pub mod planet_config;
pub mod recipes;
pub mod rotation_model;
pub mod simulation_builder;
pub mod sources;
pub mod validation;
pub mod vehicle_builder;
pub mod vehicle_config;

// ── Orchestration functions ──
pub use atmosphere::{
    evaluate_atmosphere, evaluate_atmosphere_typed, AtmosphereConfig, AtmosphereModel,
};
pub use derived::{
    compute_body_euler_angles, compute_body_geodetic, compute_body_lvlh_frame,
    compute_body_solar_beta, compute_lvlh_relative_state, compute_orbital_elements,
    compute_orbital_elements_typed, compute_relative_state, LvlhRelativeState, RelativeState,
};
pub use forces::{collect_and_resolve_forces, collect_and_resolve_forces_typed};
pub use gravity::{
    accumulate_gravity, accumulate_gravity_typed, accumulate_relativistic_corrections,
    accumulate_relativistic_corrections_typed, ResolvedRelativisticSource, ResolvedSource,
};
pub use integrable::IntegrableObject;
pub use integration::{
    integrate_bodies_contact_coupled, integrate_body, integrate_body_coupled, integrate_body_typed,
    CoupledBodyInput, CoupledIntegScratch, CoupledStageEval,
};
pub use interactions::{
    compute_cannonball_srp, compute_cannonball_srp_typed, compute_drag, compute_drag_typed,
    compute_gravity_torque, compute_gravity_torque_typed, evaluate_contact_pair, ContactPairEval,
    FlatPlateStageInputs, FlatPlateStageInputsTyped, FlatPlateState, ThermalIntegrationOrder,
};
pub use jeod_dynamics::{
    Abm4State, GaussJacksonConfig, GaussJacksonState, IntegratorResult, IntegratorType,
};
pub use pipeline::{PipelineStage, PIPELINE_ORDER};
pub use planet_config::{PlanetConfig, EARTH, MARS, MOON, SUN};
pub use rotation_model::RotationModel;
pub use simulation_builder::{MassTreeAttachment, SimulationBuilder};
pub use sources::GravitySourceEntry;
pub use validation::{validate_body, ValidationError};
pub use vehicle_builder::{
    BuildState, HasIntegrator, NeedsMass, NeedsState, Ready, VehicleBuilder,
};
pub use vehicle_config::{
    DerivedStateConfig, EarthLightingConfig, FrameSwitchConfig, GeodeticConfig, ShadowBody,
    SrpModel, SwitchSense, VehicleConfig,
};

// ── Re-exports from jeod_* crates ──
// ECS adapters depend only on jeod_sim — these re-exports provide all the
// types needed for component definitions, system parameters, and resources.

// jeod_dynamics: state types, force types, mass, config, frame utilities
pub use jeod_dynamics::{
    compute_t_inertial_struct, DynamicsConfig, ForceContributions, FrameDerivatives,
    GravityAcceleration, MassBodyId, MassProperties, MassTree, RotationalState, SixDofState,
    TotalForce, TranslationalState, INERTIA_CONSISTENCY_TOL,
};

// jeod_gravity: source definitions, controls, and tides
pub use jeod_gravity::tides::{compute_delta_c20, TidalBody, TidalConfig, EARTH_K2};
pub use jeod_gravity::{GravityControl, GravityControls, GravityModel, GravitySource};

// jeod_atmosphere: state output and model types
pub use jeod_atmosphere::exponential::ExponentialAtmosphere;
pub use jeod_atmosphere::met::{self as met_atmosphere, GeoIndexType, MetAtmosphere};
pub use jeod_atmosphere::AtmosphereState;

// jeod_interactions: config, result types, and computation functions
pub use jeod_interactions::{
    compute_contact_force, compute_contact_force_from_geometry, compute_contact_geometry,
    compute_earth_lighting, compute_flat_plate_srp_thermal,
    compute_flat_plate_srp_thermal_conduction, compute_shadow_fraction, solar_flux_at_distance,
    AerodynamicForce, ContactFacet, ContactForce, ContactGeometry, ContactMaterial, ContactShape,
    DragConfig, EarthLightingState, FlatPlate, FlatPlateParams, FlatPlateSrpResult,
    FlatPlateThermal, LightingBody, LightingParams, RadiationForce, ThermalConductionMatrix,
    SOLAR_RADIUS, SPEED_OF_LIGHT,
};

// jeod_frames: reference frame state
pub use jeod_frames::RefFrameState;

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

// jeod_gravity: coefficient loading (for test/data infrastructure)
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
pub use jeod_quantities::frame::{
    BodyFrame, Earth, Ecef, Frame, Inertial, Lvlh, Mars, Moon, Ned, Planet, PlanetFixed, SelfRef,
    StructuralFrame, Sun, Vehicle,
};
pub use jeod_quantities::frame_transform::FrameTransform;
pub use jeod_quantities::inertia::InertiaTensor;
pub use jeod_quantities::qty3::Qty3;

// jeod_math: quaternion type (used in RotationalState)
pub use jeod_math::JeodQuat;

// jeod_math: derived state types
#[allow(deprecated)]
pub use jeod_math::{
    cartesian_to_geodetic, compute_euler_angles_from_matrix, compute_lvlh_frame,
    geodetic_to_cartesian, solar_beta_angle,
};
pub use jeod_math::{
    cartesian_to_geodetic_typed, compute_euler_angles_from_matrix_typed, compute_lvlh_frame_typed,
    geodetic_to_cartesian_typed, solar_beta_angle_typed, EulerSequence, GeodeticState,
    GeodeticStateTyped, LvlhFrame, OrbitalElements, OrbitalError,
};
