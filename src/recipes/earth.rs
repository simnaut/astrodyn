//! Earth gravity-source recipes.
//!
//! Each function returns a fully-populated [`GravitySourceEntry`] ready
//! for [`SimulationBuilder::add_source`](crate::SimulationBuilder::add_source).
//!
//! ```
//! use astrodyn::recipes::earth;
//! let earth = earth::point_mass();
//! assert_eq!(earth.source.mu, 3.986_004_415e14);
//! ```
//!
//! Recipes here are JEOD-source-independent — they describe Earth via
//! constants and binary fixtures the Rust port owns. The [`ggm05c`],
//! [`ggm02c`], and [`gemt1`] high-fidelity spherical-harmonics variants
//! load coefficient blobs that are embedded into the published crate
//! (via `include_bytes!` in `astrodyn_gravity`), so they work without a
//! JEOD checkout.

use crate::sources::GravitySourceEntry;
use crate::EARTH;
use astrodyn_gravity::fixtures;

/// Earth as a point-mass central body (no spherical harmonics).
///
/// Includes the JEOD `EarthRNP` rotation model so the simulation
/// updates `t_inertial_pfix` from time each step.
pub fn point_mass() -> GravitySourceEntry {
    GravitySourceEntry::central_body(&EARTH)
}

/// Earth with the GGM05C spherical-harmonics gravity field
/// (degree=order=360).
///
/// JEOD-equivalent of `models/environment/gravity/data/src/earth_GGM05C.cc`.
/// Coefficient bytes are embedded at compile time via `include_bytes!`,
/// so this recipe works without a JEOD checkout and from the published
/// `.crate`.
pub fn ggm05c() -> GravitySourceEntry {
    GravitySourceEntry::central_body_sh(&EARTH, fixtures::load_ggm05c())
}

/// Earth with the GGM02C spherical-harmonics gravity field
/// (degree=order=200).
///
/// JEOD-equivalent of `models/environment/gravity/data/src/earth_GGM02C.cc`.
pub fn ggm02c() -> GravitySourceEntry {
    GravitySourceEntry::central_body_sh(&EARTH, fixtures::load_ggm02c())
}

/// Earth with the GEM-T1 spherical-harmonics gravity field
/// (degree=order=36).
///
/// JEOD-equivalent of `models/environment/gravity/data/src/earth_GEMT1.cc`.
/// Used by JEOD's `SIM_7_time_reversal` verification scenario.
pub fn gemt1() -> GravitySourceEntry {
    GravitySourceEntry::central_body_sh(&EARTH, fixtures::load_gemt1())
}

/// Earth as a point-mass third-body perturbation source at the given
/// inertial position. No rotation model.
pub fn third_body(
    position: astrodyn_quantities::aliases::Position<astrodyn_quantities::frame::RootInertial>,
) -> GravitySourceEntry {
    GravitySourceEntry::third_body(&EARTH, position)
}
