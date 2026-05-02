//! Moon gravity-source recipes.
//!
//! ```
//! use jeod_sim::recipes::moon;
//! let m = moon::point_mass();
//! assert!(m.source.mu > 4.9e12 && m.source.mu < 4.91e12);
//! ```
//!
//! Spherical-harmonics gravity for the Moon (LP150Q et al.) requires
//! loading JEOD coefficient files; the loader lives in
//! [`verification::reference_data`](super::verification::reference_data).
//! Mission code that needs an SH source supplies its own data and
//! builds the entry via [`GravitySourceEntry::central_body_sh`].

use crate::sources::GravitySourceEntry;
use crate::MOON;

/// Moon as a point-mass body with the JEOD IAU rotation model.
pub fn point_mass() -> GravitySourceEntry {
    GravitySourceEntry::central_body(&MOON)
}

/// Moon as a third-body perturbation source (point-mass, no rotation)
/// at the given inertial position.
pub fn third_body(
    position: jeod_quantities::aliases::Position<jeod_quantities::frame::RootInertial>,
) -> GravitySourceEntry {
    GravitySourceEntry::third_body(&MOON, position)
}
