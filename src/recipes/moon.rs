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

/// Moon with GRAIL150 spherical-harmonics gravity **and** the DE421 BPC
/// libration rotation model (per-step `t_inertial_pfix` updates from the
/// committed `moon_pa_de421_*.bpc` kernel).
///
/// Equivalent to [`grail150`] with `rotation_model` overridden from
/// `MoonIAU` (analytic IAU-2009 mean orientation) to
/// [`MoonDE421`](crate::RotationModel::MoonDE421) (high-fidelity
/// libration). The initial `t_inertial_pfix` is left as identity — the
/// per-step ephemeris stage rewrites it from BPC data before any gravity
/// evaluation. Mission code is responsible for wiring the BPC-loaded
/// ephemeris on the simulation builder
/// ([`recipes::ephemeris::de421_with_moon_pa`](super::ephemeris::de421_with_moon_pa)
/// — DE440 once that asset lands).
///
/// Used by `astrodyn_verif_nesc::cc8` and any future Moon-central
/// scenario that needs sub-arcsecond libration accuracy. The truncation
/// degree/order is selected at the call site via
/// [`GravityControl::new_nonspherical`](astrodyn_gravity::GravityControl::new_nonspherical),
/// so the same recipe serves both 8×8 (NESC CC8) and 150×150
/// (high-fidelity research) callers from one fixture.
pub fn grail150_with_libration() -> GravitySourceEntry {
    let mut entry = GravitySourceEntry::central_body_sh(&MOON, fixtures::load_moon_grail150());
    entry.rotation_model = crate::RotationModel::MoonDE421;
    entry
}

/// Moon as a third-body perturbation source (point-mass, no rotation)
/// at the given inertial position.
pub fn third_body(
    position: astrodyn_quantities::aliases::Position<astrodyn_quantities::frame::RootInertial>,
) -> GravitySourceEntry {
    GravitySourceEntry::third_body(&MOON, position)
}
