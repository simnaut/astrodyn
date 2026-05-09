//! Moon gravity-source recipes.
//!
//! ```
//! use astrodyn::recipes::moon;
//! let m = moon::point_mass();
//! assert!(m.source.mu > 4.9e12 && m.source.mu < 4.91e12);
//! ```
//!
//! [`lp150q`] and [`grail150`] return high-fidelity spherical-harmonics
//! Moon gravity backed by coefficient blobs embedded into the published
//! crate (via `include_bytes!` in `astrodyn_gravity`), so they work
//! without a JEOD checkout.

use crate::sources::GravitySourceEntry;
use crate::MOON;
use astrodyn_gravity::fixtures;

/// Moon as a point-mass body with the JEOD IAU rotation model.
pub fn point_mass() -> GravitySourceEntry {
    GravitySourceEntry::central_body(&MOON)
}

/// Moon with the LP150Q (Lunar Prospector) spherical-harmonics gravity
/// field (degree=order=150).
///
/// JEOD-equivalent of `models/environment/gravity/data/src/moon_LP150Q.cc`.
/// Used by JEOD's `SIM_Earth_Moon` verification scenario.
pub fn lp150q() -> GravitySourceEntry {
    GravitySourceEntry::central_body_sh(&MOON, fixtures::load_moon_lp150q())
}

/// Moon with the GRAIL150 spherical-harmonics gravity field
/// (degree=order=150).
///
/// JEOD-equivalent of `models/environment/gravity/data/src/moon_GRAIL150.cc`.
/// GRAIL is the newer JEOD default for the Moon and is used by
/// SIM_dyncomp's third-body Moon source as well as the gravity-gradient
/// torque rigs.
pub fn grail150() -> GravitySourceEntry {
    GravitySourceEntry::central_body_sh(&MOON, fixtures::load_moon_grail150())
}

/// Moon as a third-body perturbation source (point-mass, no rotation)
/// at the given inertial position.
pub fn third_body(
    position: astrodyn_quantities::aliases::Position<astrodyn_quantities::frame::RootInertial>,
) -> GravitySourceEntry {
    GravitySourceEntry::third_body(&MOON, position)
}
