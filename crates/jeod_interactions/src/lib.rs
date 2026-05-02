//! Surface interactions: aerodynamics, radiation pressure, contact, lighting, torques.
//!
//! Pure-Rust port of JEOD's `models/interactions/` plus the surface-model and
//! shadow geometry that those interactions depend on. Each module produces a
//! force, torque, or environmental scalar at a vehicle position; the
//! orchestration that sums these into a body's `jeod_dynamics::TotalForce`
//! lives in `jeod_sim`.
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

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod aero_drag;
pub mod contact;
pub mod earth_lighting;
pub mod flat_plate_aero;
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
pub use gravity_torque::{compute_gravity_torque, compute_gravity_torque_typed};
pub use radiation_pressure::*;
pub use shadow::*;
pub use surface_model::{ArticulatedFacet, ArticulationState, SurfaceFacet, SurfaceShape};
pub use thermal_rider::{
    compute_thermal_power_balance, ThermalEnvironment, ThermalFacet, ThermalPowerBalance,
};
