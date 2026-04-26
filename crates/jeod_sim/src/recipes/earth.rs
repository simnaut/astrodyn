//! Earth gravity-source recipes.
//!
//! Each function returns a fully-populated [`GravitySourceEntry`] ready
//! for [`SimulationBuilder::add_source`](crate::SimulationBuilder::add_source).
//!
//! ```
//! use jeod_sim::recipes::earth;
//! let earth = earth::point_mass();
//! assert_eq!(earth.source.mu, 3.986_004_415e14);
//! ```
//!
//! Recipes here are JEOD-source-independent — they describe Earth via
//! constants the Rust port owns. High-fidelity spherical-harmonics
//! gravity that requires loading JEOD coefficient files lives in
//! [`verification::reference_data`](super::verification::reference_data),
//! whose use is appropriate only for cross-validation against JEOD.
//! Mission code that needs an SH source supplies its own
//! [`SphericalHarmonicsData`](jeod_gravity::SphericalHarmonicsData) and
//! constructs the entry manually via
//! [`GravitySourceEntry::central_body_sh`].

use crate::sources::GravitySourceEntry;
use crate::EARTH;

/// Earth as a point-mass central body (no spherical harmonics).
///
/// Includes the JEOD `EarthRNP` rotation model so the simulation
/// updates `t_inertial_pfix` from time each step.
pub fn point_mass() -> GravitySourceEntry {
    GravitySourceEntry::central_body(&EARTH)
}

/// Earth as a point-mass third-body perturbation source at the given
/// inertial position. No rotation model.
pub fn third_body(position: glam::DVec3) -> GravitySourceEntry {
    GravitySourceEntry::third_body(&EARTH, position)
}
