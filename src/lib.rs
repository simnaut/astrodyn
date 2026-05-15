// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! ECS-agnostic orchestration layer for JEOD physics.
//!
//! This crate is the **single dependency** for ECS adapters and mission
//! crates. It re-exports the types from the `astrodyn_*` physics crates that
//! such consumers need, plus orchestration functions that compose them
//! into pipeline stages.
//!
//! ## Per-body functions (primary API for ECS integration)
//!
//! Composable, borrow-based functions that any ECS adapter can call from
//! its system functions. The ECS world remains the single source of truth.
//!
//! - [`accumulate_gravity`] — gravity accumulation across sources
//! - [`evaluate_atmosphere`] — atmosphere evaluation pipeline
//! - [`compute_ballistic_drag`] — aerodynamic drag with frame transform
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
//! For batch propagation and Tier 3 tests, see the `astrodyn_runner` crate which
//! provides a standalone `Simulation` struct that owns all state and drives the
//! pipeline. ECS adapters should **not** use `astrodyn_runner` — use the per-body
//! functions from this crate instead. `astrodyn_runner` is a parallel non-Bevy
//! consumer that, like `astrodyn_bevy` and mission code, depends on `astrodyn`
//! and only `astrodyn` for physics.
//!
//! ## Re-export discipline
//!
//! Every `pub use astrodyn_*::...` re-export is justified by an active mission-
//! crate, `astrodyn_bevy` adapter, or `astrodyn_runner` consumer — every
//! non-verification consumer of the pipeline routes through this crate. The
//! contract is intentionally tight: a rename in one of the underlying physics
//! crates only ripples to mission code if the affected type is one the
//! mission API genuinely owns. The criteria are spelled out at the head of
//! the re-export block in [`lib.rs`][self].
//!
//! ## Pipeline ordering
//!
//! See [`PipelineStage`] and [`PIPELINE_ORDER`] for the canonical stage
//! execution order that any adapter must respect.
//!
//! ## Quick start
//!
//! Compose a vehicle through the typestate [`VehicleBuilder`] using only
//! gateway re-exports — mission code never reaches into the physics
//! crates directly:
//!
//! ```
//! use astrodyn::{
//!     recipes::{earth, orbital_elements, vehicle},
//!     F64Ext, GravityControl, GravityGradient, VehicleBuilder,
//! };
//!
//! let mu = earth::point_mass().source.mu.m3_per_s2();
//! let cfg = VehicleBuilder::new()
//!     .from_orbital_elements(orbital_elements::iss(), mu)
//!     .three_dof_point_mass(vehicle::iss_mass())
//!     .rk4()
//!     .gravity(GravityControl::new_spherical(0_usize, GravityGradient::Skip))
//!     .build();
//! # let _ = cfg;
//! ```
//!
//! `cfg` (a [`VehicleConfig`]) is then handed to either
//! `astrodyn_bevy::VehicleConfigBevyExt::spawn_bevy` or
//! `astrodyn_runner::Simulation::add_vehicle` — both consumers share the
//! same configuration shape so a mission can swap between them without
//! reauthoring its setup code.

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
pub mod integrator;
pub mod interactions;
pub mod kinematic_propagation;
pub mod pipeline;
pub mod planet_config;
pub mod recipes;
pub mod rotation_model;
pub mod simulation_builder;
pub mod source_frames;
pub mod source_handle;
pub mod source_state;
pub mod sources;
// Internal bridge between typed `*Typed<F>` shapes and the underlying raw
// `DVec3`/`DQuat`/`DMat3` shapes that the kernel functions still consume.
// Adapters (`astrodyn_bevy`, `astrodyn_runner`, the verif crates) need the
// symbols `pub` so they can translate at the kernel boundary, but mission
// code shouldn't see them — the type-system promise is that mission code
// never reaches raw `DVec3`/`DQuat`. Hidden from rustdoc to keep that
// promise visible on docs.rs while preserving the adapter call sites.
#[doc(hidden)] // allowed: structural adapter boundary — see preceding rationale block
pub mod typed_bridge;
pub mod validation;
pub mod vehicle_builder;
pub mod vehicle_config;
pub mod wrench;

// ── Orchestration functions ──
pub use astrodyn_dynamics::kinematic_joint::{
    evaluate as evaluate_joint_kinematics, evaluate_closure as evaluate_closure_kinematics,
    evaluate_multi_dof as evaluate_multi_dof_kinematics,
    evaluate_sinusoidal as evaluate_sinusoidal_kinematics, ClosureJointKinematicsSpec,
    JointKinematicsModel, JointKinematicsSpec, MultiDofJointKinematicsSpec, SingleDofKinematics,
    SinusoidalJointKinematicsSpec, AXIS_NORM_TOL, MAX_MULTI_DOF_AXES,
};
pub use atmosphere::{
    evaluate_atmosphere, evaluate_atmosphere_typed, evaluate_body_atmosphere_typed,
    run_atmosphere_stage, AtmosphereBodyInputs, AtmosphereConfig, AtmosphereModel,
};
pub use attach::{
    stage_attach_combine, stage_detach_capture, CrossIntegFrameStateShift, StageAttachInputs,
    StageAttachOutputs,
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
    accumulate_relativistic_corrections_typed, evaluate_body_gravity_typed, run_gravity_stage,
    GravityBodyInputs, ResolvedRelativisticSource, ResolvedSource,
};
pub use integrable::IntegrableObject;
pub use integration::{
    integrate_bodies_contact_coupled, integrate_bodies_contact_coupled_typed, integrate_body,
    integrate_body_coupled, integrate_body_coupled_typed, integrate_body_typed, reset_integrators,
    CoupledBodyInput, CoupledBodyInputTyped, CoupledIntegScratch, CoupledStageEval,
};
pub use integrator::{Abm4State, GaussJacksonConfig, GaussJacksonState, IntegratorType};
pub use interactions::{
    evaluate_contact_pair, evaluate_ground_contact_pair, ContactPairEval, FlatPlateStageInputs,
    FlatPlateState, GroundContactPairEval, ThermalIntegrationOrder,
};
pub use kinematic_propagation::{propagate_state_via_storage, KinematicEdge, KinematicNodeState};
pub use pipeline::{PipelineStage, PIPELINE_ORDER};
pub use planet_config::{PlanetConfig, EARTH, MARS, MOON, SUN};
pub use rotation_model::RotationModel;
pub use simulation_builder::{MassTreeAttachment, SimulationBuilder};
pub use source_frames::SourceFrameIds;
pub use source_handle::SourceHandle;
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

// ── Re-exports from astrodyn_* crates ──
//
// Curation criteria:
//
// 1. Every workspace consumer that *uses* the pipeline — mission crates,
//    `astrodyn_bevy`, `astrodyn_runner` — depends on `astrodyn` (and
//    only `astrodyn`, plus `bevy` for the Bevy adapter). Verification
//    crates (`astrodyn_verif_jeod`, `astrodyn_verif_parity`) are the
//    explicit exception: they reach physics crates directly because
//    they need internals to construct test fixtures.
// 2. Therefore every public type, function, or module a non-verification
//    consumer reaches must be reachable through here.
// 3. Items that no consumer reaches are dropped — every entry below
//    earns its place by an active consumer.
//
// A CI lint (`scripts/check_no_bypass_deps.sh`) enforces criterion #1
// structurally by failing the build if `astrodyn_runner` or
// `astrodyn_bevy` declare any direct `astrodyn_*` physics-crate dep.

// astrodyn_dynamics: state types, force types, mass, config, frame utilities
pub use astrodyn_dynamics::{
    abm4_translational_step, combine_states_at_attach, compute_frame_derivatives,
    compute_kinematic_child_state, compute_left_quat_deriv_typed,
    compute_rotational_acceleration_typed, compute_t_inertial_struct,
    compute_translational_derivatives, derive_frame_attached_state, propagate_forward,
    recompute_composites_via_storage, shift_wrench_to_parent, AttachCombineInputs,
    DetachedSubtreeState, DynamicsConfig, FrameAttachInputs, FrameDerivatives, GravityAcceleration,
    MassBodyId, MassNodeOutputs, MassNodeView, MassPointState, MassProperties, MassStorage,
    MassTree, RotationalState, SixDofState, SixDofStateTyped, TotalForce, TranslationalState,
    Wrench, MAX_SAFE_MASS_KG, MIN_SAFE_MASS_KG,
};

// astrodyn_dynamics::body_init: typed orbital-element initializer
// consumed by mission examples (e.g. `examples/kepler_orbit.rs`).
// `init_from_mean_anomaly` and `init_rot_from_lvlh` cover the JEOD
// orbital-element / LVLH initialization paths used by mission init code
// and JEOD parity tests.
pub use astrodyn_dynamics::body_init::{
    init_from_mean_anomaly, init_from_orbital_elements_typed, init_rot_from_lvlh,
};

// astrodyn_dynamics::kinematic_propagation: input struct paired with the
// already-exposed `compute_kinematic_child_state`.
pub use astrodyn_dynamics::kinematic_propagation::KinematicChildInputs;

// astrodyn_dynamics typed siblings: ECS components built on the typed
// state primitives so storage carries frame phantoms rather than
// re-lifting raw `DVec3` every step.
pub use astrodyn_dynamics::forces::{
    FrameDerivativesTyped, GravityAccelerationTyped, TotalForceTyped,
};
pub use astrodyn_dynamics::mass::MassPropertiesTyped;
pub use astrodyn_dynamics::rotational::RotationalStateTyped;
pub use astrodyn_dynamics::state::TranslationalStateTyped;

// astrodyn_gravity: source definitions, controls, and tides
pub use astrodyn_gravity::tides::{
    compute_delta_c20, compute_delta_c20_typed, TidalBody, TidalConfig, TidalConfigTyped, EARTH_K2,
};

// astrodyn_gravity: test fixtures (e.g. `fixtures::load_ggm05c()`)
// consumed by integration tests that need a representative gravity model.
pub use astrodyn_gravity::fixtures as gravity_fixtures;
pub use astrodyn_gravity::{
    GravityControl, GravityControls, GravityGradient, GravityModel, GravitySource,
};

// astrodyn_atmosphere: state output and model types
pub use astrodyn_atmosphere::exponential::ExponentialAtmosphere;
pub use astrodyn_atmosphere::met::{GeoIndexType, MetAtmosphere};
pub use astrodyn_atmosphere::AtmosphereState;

// astrodyn_interactions: config, result types, and computation functions
pub use astrodyn_interactions::{
    compute_ballistic_drag, compute_ballistic_drag_typed, compute_cannonball_srp,
    compute_cannonball_srp_typed, compute_earth_lighting, compute_earth_lighting_typed,
    compute_flat_plate_srp_thermal, compute_gravity_torque, compute_gravity_torque_typed,
    compute_shadow_fraction, solar_flux_at_distance, AerodynamicForce, ContactFacet,
    ContactMaterial, DragConfig, DragConfigTyped, EarthLightingState, FlatPlate, FlatPlateParams,
    FlatPlateSrpResult, FlatPlateThermal, GroundFacet, LightingBody, LightingParams, Phase,
    RadiationForce, SphericalTerrain, Terrain, SOLAR_RADIUS,
};

// astrodyn_frames: reference frame state and arena-based frame tree.
// `FrameStorage` (trait) plus the per-link state structs (`RefFrameRot`,
// `RefFrameTrans`, `RefFrameState`) are needed by mission code that
// constructs or reads frame nodes; `frame_compute_relative_state_via_storage`
// drives cross-frame state queries. `FrameTree` (concrete arena) is the
// non-Bevy storage implementation — used by `astrodyn_runner`'s
// `Simulation` state container.
pub use astrodyn_frames::{
    compute_relative_state as frame_compute_relative_state_via_storage, FrameId, FrameStorage,
    FrameTree, RefFrameKind, RefFrameRot, RefFrameState, RefFrameTrans,
};

// astrodyn_time: simulation-time + leap-second + epoch surface that the
// Bevy adapter and mission code consume through `SimulationTime` /
// `default_leap_second_table()`. `TimeManager` + `TimeScaleId` are the
// multi-scale time API; `time_utc::{calendar_to_tjt, tjt_to_calendar,
// CalendarDate}` are the civil-time conversion helpers used by JEOD
// orbital-init and time-verification cross-validation.
pub use astrodyn_time::{
    epoch::{J2000_TAI_TJT, J2000_TT_JD, J2000_TT_TJT, SECONDS_PER_DAY, TAI_TT_OFFSET},
    leap_second::default_leap_second_table,
    time_converter_ut1_gmst::ut1_to_gmst_seconds,
    time_utc::{calendar_to_tjt, tjt_to_calendar, CalendarDate},
    SimulationTime, TimeManager, TimeScaleId,
};

// astrodyn_frames: planet rotation (used by ephemeris stage and mission
// code that sets up Mars/Moon planetary configurations). The
// `nutation_j2000`, `precession_j2000`, and `rotation_j2000` modules
// expose the lower-level primitives consumed by code that splits the
// Earth RNP composition across cache-refresh boundaries.
pub use astrodyn_frames::rotation_j2000::{
    compute_t_parent_this_from_tjt, compute_t_parent_this_from_tjt_with_polar,
    compute_t_parent_this_from_tjt_with_polar_typed,
};
pub use astrodyn_frames::{
    nutation_j2000, precession_j2000, rotation_j2000, rotation_mars, rotation_moon,
};

// astrodyn_ephemeris: ephemeris data
pub use astrodyn_ephemeris::{
    assets as ephemeris_assets, Ephemeris, EphemerisBody, EphemerisError, Epoch,
};

// astrodyn_gravity: relativistic-correction submodule consumed by mission
// code that builds relativistic-source lists. The JEOD `.cc`
// source-file parser (`load_from_jeod_cc`, `load_mu_from_jeod_cc`)
// lives in the dev/test crate `astrodyn_gravity::jeod_cc` — production
// gravity does not parse JEOD source.
pub use astrodyn_gravity::relativistic;

// astrodyn_planet: planet shape
pub use astrodyn_planet::PlanetShape;

// astrodyn_quantities: typed-quantity foundation. ECS adapters (e.g. the
// `astrodyn_bevy` root crate) consume these types via `astrodyn` to
// preserve the "single dependency" invariant.
pub use astrodyn_quantities::aliases::{
    Acceleration, AngularAcceleration, AngularVelocity, Force, InertiaTensor, Position, Torque,
    Velocity,
};
pub use astrodyn_quantities::diagnostics::CompatibleVehiclePair;
pub use astrodyn_quantities::dims::GravParam;
pub use astrodyn_quantities::ext::{Array3Ext, F64Ext, Vec3Ext};
pub use astrodyn_quantities::frame::{
    BodyFrame, Earth, Ecef, Frame, IntegrationFrame, Lvlh, Mars, MassNode, Moon, Ned, Planet,
    PlanetFixed, PlanetInertial, RootInertial, SelfPlanet, SelfRef, StructuralFrame, Sun, Vehicle,
};
pub use astrodyn_quantities::integ_origin::IntegOrigin;
// Macros that mint downstream `Vehicle`/`Planet` markers. Re-exported so
// mission crates depending only on `astrodyn` don't need a direct
// `astrodyn_quantities` line in their `Cargo.toml`. The macro body resolves
// `$crate` to `astrodyn_quantities` regardless of where the macro is
// invoked from, so the sealed-trait bound is satisfied transparently.
pub use astrodyn_quantities::body_attitude::BodyAttitude;
pub use astrodyn_quantities::frame_transform::FrameTransform;
pub use astrodyn_quantities::qty3::Qty3;
pub use astrodyn_quantities::{define_planet, define_vehicle};

// uom scalar quantities used directly by the Bevy adapter for typed
// component fields (`Angle` for Euler angles, `Ratio` for tidal ΔC20).
pub use uom::si::f64::{Angle, Mass, Ratio};

/// Re-export of [`uom::si::mass::kilogram`] for typed-quantity callers
/// that need the unit token directly (`Mass::new::<kilogram>(420_000.0)`,
/// `mass.get::<kilogram>()`). Adapter crates that consume astrodyn but
/// not uom (the production-path layer rule) reach through this alias.
pub use uom::si::mass::kilogram;

/// Convenience constructor — wrap a raw f64 (radians) as a typed
/// [`Angle`].
///
/// Mirrors `astrodyn_quantities::ext::F64Ext::rad(f)` but exists at the
/// `astrodyn` boundary so ECS adapters don't need to import
/// `astrodyn_quantities` (or `uom::si::angle::radian`) directly.
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

// astrodyn_math: quaternion type (used in RotationalState)
pub use astrodyn_math::JeodQuat;

// astrodyn_math: derived state types
pub use astrodyn_math::{EulerSequence, GeodeticState, LvlhFrame, OrbitalElements};

// astrodyn_math::euler_angles: typed-quantity Euler-angle helpers
// paired with the already-exposed `EulerSequence`.
pub use astrodyn_math::euler_angles::{
    compute_euler_angles_from_matrix_typed, compute_matrix_from_euler_angles_typed,
    compute_quaternion_from_euler_angles_typed,
};
