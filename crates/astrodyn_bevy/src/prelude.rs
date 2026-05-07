// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Mission-crate prelude: `use astrodyn_bevy::prelude::*;`.
//!
//! Brings into scope the typed Bevy Components, the [`AstrodynPlugin`] and
//! [`AstrodynSet`] schedule sets, the [`VehicleConfigBevyExt`] terminal that
//! materializes a [`astrodyn::VehicleConfig`] onto a Bevy entity, and the
//! [`F64Ext`] / [`Vec3Ext`] / [`Array3Ext`] facade traits so mission code
//! can write `400.0.km()` and `DVec3::new(...).m_at::<RootInertial>()`.
//!
//! Pair with [`crate::recipes`] (`use astrodyn_bevy::recipes::*;`) for the
//! scenario-composition catalogue (`earth::point_mass()`,
//! `orbital_elements::iss()`, `vehicle::iss_mass()`, …).
//!
//! ```
//! use bevy::prelude::*;
//! use astrodyn_bevy::prelude::*;
//! use astrodyn_bevy::recipes::*;
//!
//! let cfg = VehicleBuilder::new()
//!     .from_orbital_elements(orbital_elements::iss(), earth::point_mass().source.mu.m3_per_s2())
//!     .three_dof_point_mass(vehicle::iss_mass())
//!     .rk4()
//!     .gravity(GravityControl::new_spherical(0_usize, false))
//!     .build();
//! assert!(cfg.mass.is_some());
//! ```

pub use crate::{
    Abm4StateC, AerodynamicForceC, AstrodynPlugin, AstrodynSet, AtmosphericStateC,
    BodyActionCommandsExt, BodyActionEvent, BodyActionsR, BodyFrameMarker, ClosureJointKinematicsC,
    DetachedSubtreeStateC, DynamicsConfigC, FrameAngVelC, FrameDerivativesC, FrameEntityC,
    FrameRotC, FrameTransC, GaussJacksonStateC, GravityAccelerationC, GravityControlsC,
    GravitySourceC, GravityTorqueC, InertialFrameMarker, IntegrationFrameMarker, IntegratorTypeC,
    JointKinematicsC, MassPropertiesC, MultiDofJointKinematicsC, PfixFrameEntityC,
    PlanetFixedFrameMarker, PlanetFixedRotationC, RadiationForceC, RootFrameEntityR,
    RotationalStateC, ScenarioHandles, SimulationBuilderBevyExt, SimulationTimeR,
    SinusoidalJointKinematicsC, SourceInertialPositionC, SourceInertialVelocityC,
    StructuralTransformC, TotalForceC, TranslationalStateC, VehicleConfigBevyExt,
};
// ECS-native frame-tree mission-code surface.
// `RelativeFrameState` is the mission-facing API for cross-frame
// state queries. `FrameOrigin` is the specialized "origin of frame F
// in an ancestor frame" SystemParam — typed
// `(Position<RootInertial>, Velocity<RootInertial>)` for the common
// root-inertial form. Both live in the dedicated `frame_param` module
// so their `SystemParam` imports are explicit at the use site, but
// the prelude re-exports them for the "use astrodyn_bevy::prelude::*"
// path.
pub use crate::frame_param::{FrameOrigin, RelativeFrameState};

// All `astrodyn_quantities` re-exports come through `astrodyn` so the
// `astrodyn_bevy` root package keeps its single dependency on `astrodyn`
// (per CLAUDE.md "Three-Layer Architecture": the root package depends
// only on `astrodyn` + `bevy`).
pub use astrodyn::{
    Array3Ext, BodyAction, BodyFrame, ClosureJointKinematicsSpec, Earth, Ecef, F64Ext, Frame,
    FrameTransform, GravityControl, JeodQuat, JointKinematicsModel, JointKinematicsSpec, Lvlh,
    Mars, Moon, MultiDofJointKinematicsSpec, Ned, OrbitalElementSet, Planet, PlanetFixed, Qty3,
    RootInertial, SelfPlanet, SelfRef, SingleDofKinematics, SinusoidalJointKinematicsSpec,
    StructuralFrame, Sun, Vec3Ext, Vehicle, VehicleBuilder, VehicleConfig, AXIS_NORM_TOL,
    MAX_MULTI_DOF_AXES,
};
// Mission-crate macros for defining additional `Vehicle` / `Planet`
// markers. Re-exported so `use astrodyn_bevy::prelude::*;` brings them into
// scope alongside the typed-quantity API.
pub use astrodyn::{define_planet, define_vehicle};
