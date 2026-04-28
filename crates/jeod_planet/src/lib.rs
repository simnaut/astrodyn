//! Planetary shape and standard preset bodies.
//!
//! Pure-Rust port of JEOD's `models/environment/planet/` — the per-body
//! reference-ellipsoid parameters (gravitational parameter, equatorial and
//! polar radii, flattening) consumed by gravity, geodetic, atmospheric, and
//! frame-rotation code.
//!
//! ## Public surface
//!
//! - [`PlanetShape`] (re-exported from [`planet`]) — the JEOD `Planet`
//!   struct equivalent. Stores `name`, `mu` (m^3/s^2), `r_eq` (m),
//!   `r_pol` (m), and `flat_coeff` along with derived helpers
//!   (`flat_inv`, `e_ellipsoid`).
//! - [`presets`] — canonical body constants matching JEOD source data
//!   files. Earth uses WGS84 geometry from `planet/data/src/earth.cc`
//!   together with the GGM05C gravitational parameter
//!   `mu = 398_600.441_50e9 m^3/s^2` (which differs from IERS 2010 by
//!   3e6 m^3/s^2; we follow JEOD's value to keep cross-validation faithful).
//!   Additional presets cover the other bodies the JEOD verification sims
//!   exercise.
//!
//! ## Role in the pipeline
//!
//! `PlanetShape` is the shared parameter block consumed by
//! `jeod_math::geodetic` for ellipsoidal coordinate conversions, by
//! `jeod_gravity` when the gravity model needs the reference radius, and
//! by `jeod_frames` for body-fixed rotation models. Pure Rust, zero Bevy
//! dependency.

#![forbid(unsafe_code)]

pub use jeod_quantities::prelude::*;

pub mod planet;
pub mod presets;

pub use planet::*;
pub use presets::*;
