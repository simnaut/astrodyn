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
//! `astrodyn_math::geodetic` for ellipsoidal coordinate conversions, by
//! `astrodyn_gravity` when the gravity model needs the reference radius, and
//! by `astrodyn_frames` for body-fixed rotation models. Pure Rust, zero Bevy
//! dependency.
//!
//! ## Example
//!
//! ```
//! use astrodyn_planet::{EARTH, MOON};
//!
//! // WGS84 equatorial radius, GGM05C gravitational parameter.
//! assert!((EARTH.r_eq - 6_378_137.0).abs() < 1.0);
//! assert!((EARTH.mu - 3.986_004_415e14).abs() < 1e6);
//!
//! // Moon is much smaller and lighter than Earth.
//! assert!(MOON.r_eq < EARTH.r_eq);
//! assert!(MOON.mu < EARTH.mu);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub use astrodyn_quantities::prelude::*;

pub mod geodetic_verif;
pub mod planet;
pub mod presets;

pub use planet::*;
pub use presets::*;
