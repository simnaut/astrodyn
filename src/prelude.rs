//! Mission-crate prelude: `use bevy_jeod::prelude::*;`.
//!
//! Brings into scope the typed Bevy Components, the [`JeodPlugin`] and
//! [`JeodSet`] schedule sets, the [`VehicleConfigBevyExt`] terminal that
//! materializes a [`jeod_sim::VehicleConfig`] onto a Bevy entity, and the
//! [`F64Ext`] / [`Vec3Ext`] / [`Array3Ext`] facade traits so mission code
//! can write `400.0.km()` and `DVec3::new(...).m_at::<RootInertial>()`.
//!
//! Pair with [`crate::recipes`] (`use bevy_jeod::recipes::*;`) for the
//! scenario-composition catalogue (`earth::point_mass()`,
//! `orbital_elements::iss()`, `vehicle::iss_mass()`, …).
//!
//! ```
//! use bevy::prelude::*;
//! use bevy_jeod::prelude::*;
//! use bevy_jeod::recipes::*;
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
    Abm4StateC, AerodynamicForceC, AtmosphericStateC, BodyFrameMarker, DynamicsConfigC,
    FrameAngVelC, FrameDerivativesC, FrameEntityC, FrameRotC, FrameTransC, GaussJacksonStateC,
    GravityAccelerationC, GravityControlsC, GravitySourceC, GravityTorqueC, InertialFrameMarker,
    IntegrationFrameMarker, IntegratorTypeC, JeodPlugin, JeodSet, MassPropertiesC,
    PfixFrameEntityC, PlanetFixedFrameMarker, PlanetFixedRotationC, RadiationForceC,
    RootFrameEntityR, RotationalStateC, SimulationTimeR, SourceInertialPositionC,
    SourceInertialVelocityC, StructuralTransformC, TotalForceC, TranslationalStateC,
    VehicleConfigBevyExt,
};
// Issue #277 / #278 — ECS-native frame-tree mission-code surface.
// `RelativeFrameState` (issue #277) is the mission-facing replacement
// for `FrameTreeR.compute_relative_state`. `FrameOrigin` (issue #278)
// is the specialized "origin of frame F in an ancestor frame"
// SystemParam — typed `(Position<RootInertial>, Velocity<RootInertial>)`
// for the common root-inertial form, matching `frame_origin_typed::<RootInertial>`.
// Both live in the dedicated `frame_param` module so their
// `SystemParam` imports are explicit at the use site, but the prelude
// re-exports them for the "use bevy_jeod::prelude::*" path.
pub use crate::frame_param::{FrameOrigin, RelativeFrameState};

// All `jeod_quantities` re-exports come through `jeod_sim` so the
// `bevy_jeod` root package keeps its single dependency on `jeod_sim`
// (per CLAUDE.md "Three-Layer Architecture": the root package depends
// only on `jeod_sim` + `bevy`).
pub use jeod_sim::{
    Array3Ext, BodyFrame, Ecef, F64Ext, Frame, FrameTransform, GravityControl, JeodQuat, Lvlh, Ned,
    Planet, PlanetFixed, Qty3, RootInertial, SelfPlanet, SelfRef, StructuralFrame, Vec3Ext,
    Vehicle, VehicleBuilder, VehicleConfig,
};
// Mission-crate macros for defining additional `Vehicle` / `Planet`
// markers. Re-exported so `use bevy_jeod::prelude::*;` brings them into
// scope alongside the typed-quantity API.
pub use jeod_sim::{define_planet, define_vehicle};
