//! Mars gravity-source recipes.
//!
//! ```
//! use astrodyn::recipes::mars;
//! let m = mars::point_mass();
//! assert!(m.source.mu > 4.28e13 && m.source.mu < 4.29e13);
//! ```
//!
//! Spherical-harmonics gravity for Mars (MRO110B2 et al.) requires
//! loading JEOD coefficient files; that loader lives in
//! `astrodyn_verif_jeod::verification::reference_data`.

use crate::sources::GravitySourceEntry;
use crate::MARS;

/// Mars as a point-mass central body with the IAU rotation model.
pub fn point_mass() -> GravitySourceEntry {
    GravitySourceEntry::central_body(&MARS)
}

/// Mars as a third-body perturbation source at the given inertial position.
pub fn third_body(
    position: astrodyn_quantities::aliases::Position<astrodyn_quantities::frame::RootInertial>,
) -> GravitySourceEntry {
    GravitySourceEntry::third_body(&MARS, position)
}
