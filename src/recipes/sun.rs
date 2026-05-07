//! Sun gravity-source recipes.
//!
//! ```
//! use astrodyn::recipes::sun;
//! let s = sun::point_mass();
//! assert!(s.source.mu > 1.3e20);
//! ```

use crate::sources::GravitySourceEntry;
use crate::SUN;

/// Sun as a point-mass body. No rotation model — the JEOD `RotationModel::None`
/// keeps the source's planet-fixed transform absent (Sun is treated as a point
/// emitter for SRP and a third-body gravity source).
pub fn point_mass() -> GravitySourceEntry {
    GravitySourceEntry::central_body(&SUN)
}

/// Sun as a third-body perturbation source at the given inertial position.
pub fn third_body(
    position: astrodyn_quantities::aliases::Position<astrodyn_quantities::frame::RootInertial>,
) -> GravitySourceEntry {
    GravitySourceEntry::third_body(&SUN, position)
}
