//! Surface interactions: aerodynamics, radiation pressure, contact, lighting, torques.
//!
//! Pure-Rust port of JEOD's `models/interactions/` plus the surface-model and
//! shadow geometry that those interactions depend on. Each module produces a
//! force, torque, or environmental scalar at a vehicle position; the
//! orchestration that sums these into a body's `astrodyn_dynamics::TotalForce`
//! lives in `astrodyn`.
//!
//! ## Public surface
//!
//! - **Aerodynamic drag**: [`aero_drag`] (port of JEOD
//!   `models/interactions/aerodynamics/`) provides scalar-Cd drag against a
//!   ballistic-coefficient body. [`flat_plate_aero`] adds the per-facet
//!   panel-method drag/lift used for articulated spacecraft.
//! - **Radiation pressure**: [`radiation_pressure`] ports JEOD
//!   `models/interactions/radiation_pressure/` — solar flux against a
//!   surface model with absorption / specular / diffuse coefficients,
//!   shadow-corrected via the [`shadow`] module's umbra/penumbra geometry
//!   and via [`compute_earth_lighting`] for Earth-albedo and Earth-IR
//!   contributions ([`EarthLightingState`], [`LightingBody`],
//!   [`LightingParams`]).
//! - **Gravity-gradient torque**: [`compute_gravity_torque`] and its typed
//!   sibling [`compute_gravity_torque_typed`] port JEOD
//!   `models/interactions/gravity_torque/` — the cross product of the
//!   inertia tensor and the gravity-gradient tensor projected through the
//!   body attitude.
//! - **Surface model**: [`SurfaceFacet`], [`ArticulatedFacet`],
//!   [`ArticulationState`], [`SurfaceShape`] in [`surface_model`] are the
//!   per-facet geometry inputs that aero, SRP, contact, and thermal share.
//! - **Contact**: [`compute_contact_force`],
//!   [`compute_contact_force_from_geometry`], [`compute_contact_geometry`],
//!   [`ContactFacet`], [`ContactForce`], [`ContactGeometry`],
//!   [`ContactMaterial`], [`ContactShape`] for collision/contact response.
//! - **Thermal**: [`compute_thermal_power_balance`],
//!   [`ThermalEnvironment`], [`ThermalFacet`], [`ThermalPowerBalance`] —
//!   per-facet power balance for thermal-rider models.
//!
//! JEOD source: `models/interactions/` and surrounding utilities. Pure Rust,
//! zero Bevy dependency.
//!
//! ## Example
//!
//! Compute ballistic aerodynamic drag for a 1 m² Cd=2.2 plate moving at
//! 7.5 km/s through a thin LEO atmosphere:
//!
//! ```
//! use astrodyn_atmosphere::AtmosphereState;
//! use astrodyn_interactions::{compute_ballistic_drag, DragConfig};
//! use astrodyn_quantities::frame::SelfPlanet;
//! use glam::{DMat3, DVec3};
//!
//! let cfg = DragConfig { cd: 2.2, area: 1.0, constant_density: None, ..Default::default() };
//! // 1e-12 kg/m^3 is roughly 400 km altitude.
//! let atmos = AtmosphereState::<SelfPlanet>::from_raw(1.0e-12, 0.0, 0.0, DVec3::ZERO);
//! // Velocity along +X at 7.5 km/s; identity rotation (inertial == body).
//! let v = DVec3::new(7_500.0, 0.0, 0.0);
//! let force = compute_ballistic_drag(&cfg, &atmos, v, &DMat3::IDENTITY);
//!
//! // Drag opposes motion (-X) and is small but non-zero.
//! assert!(force.force.x < 0.0);
//! assert!(force.force.length() > 0.0 && force.force.length() < 1.0);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod aero_drag;
pub mod contact;
pub mod earth_lighting;
pub mod flat_plate_aero;
pub mod forces;
pub mod gravity_torque;
pub mod radiation_pressure;
pub mod shadow;
pub mod surface_model;
pub mod thermal_rider;

pub use aero_drag::*;
pub use contact::{
    compute_contact_force, compute_contact_force_from_geometry, compute_contact_geometry,
    compute_ground_contact_geometry, ContactFacet, ContactForce, ContactGeometry, ContactMaterial,
    ContactShape, GroundFacet, Phase, SphericalTerrain, Terrain,
};
pub use earth_lighting::{
    compute_earth_lighting, compute_earth_lighting_typed, EarthLightingState, LightingBody,
    LightingParams,
};
pub use flat_plate_aero::*;
pub use forces::{collect_and_resolve_forces, collect_and_resolve_forces_typed};
pub use gravity_torque::{compute_gravity_torque, compute_gravity_torque_typed};
pub use radiation_pressure::*;
pub use shadow::*;
pub use surface_model::{ArticulatedFacet, ArticulationState, SurfaceFacet, SurfaceShape};
pub use thermal_rider::{
    compute_thermal_power_balance, ThermalEnvironment, ThermalFacet, ThermalPowerBalance,
};
