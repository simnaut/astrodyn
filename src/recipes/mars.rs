//! Mars gravity-source recipes.
//!
//! ```
//! use astrodyn::recipes::mars;
//! let m = mars::point_mass();
//! assert!(m.source.mu > 4.28e13 && m.source.mu < 4.29e13);
//! ```
//!
//! [`mro110b2`] returns high-fidelity spherical-harmonics Mars gravity
//! backed by a coefficient blob embedded into the published crate (via
//! `include_bytes!` in `astrodyn_gravity`), so it works without a JEOD
//! checkout.

use crate::sources::GravitySourceEntry;
use crate::MARS;
use astrodyn_gravity::fixtures;

/// Mars as a point-mass central body with the IAU rotation model.
pub fn point_mass() -> GravitySourceEntry {
    GravitySourceEntry::central_body(&MARS)
}

/// Mars with the MRO110B2 spherical-harmonics gravity field
/// (degree=order=110).
///
/// JEOD-equivalent of `models/environment/gravity/data/src/mars_MRO110B2.cc`.
pub fn mro110b2() -> GravitySourceEntry {
    GravitySourceEntry::central_body_sh(&MARS, fixtures::load_mars_mro110b2())
}

/// Mars as a third-body perturbation source at the given inertial position.
pub fn third_body(
    position: astrodyn_quantities::aliases::Position<astrodyn_quantities::frame::RootInertial>,
) -> GravitySourceEntry {
    GravitySourceEntry::third_body(&MARS, position)
}
