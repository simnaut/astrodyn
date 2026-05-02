//! Gravity: spherical, spherical harmonics (Gottlieb), tides, and relativistic terms.
//!
//! Pure-Rust port of JEOD's `models/environment/gravity/`. The crate produces
//! gravitational acceleration, the gravity-gradient tensor, and the
//! gravitational potential at a body's position, given a configured set of
//! gravity sources.
//!
//! ## Public surface
//!
//! - **Spherical (point-mass) gravity**: [`calc_spherical`], [`gravitation`],
//!   and [`gravitation_with_scratch`] from [`compute`]. Output is a
//!   `jeod_dynamics::GravityAcceleration` containing acceleration, gradient,
//!   and potential — gradient and potential are filled even for the point-
//!   mass case so downstream consumers (gravity-gradient torque, energy
//!   diagnostics) have what they need.
//! - **Spherical harmonics**: [`calc_nonspherical`],
//!   [`calc_nonspherical_typed`], [`calc_nonspherical_with_scratch`], and
//!   [`GottliebScratch`] from [`spherical_harmonics_calc_nonspherical`].
//!   This is the ported Gottlieb algorithm from JEOD
//!   `models/environment/gravity/src/spherical_harmonics_calc_nonspherical.cc`
//!   — a numerically stable normalized Legendre recursion that scales to
//!   high degree and order without the underflow/overflow problems of the
//!   classical formulation.
//! - **Configuration types**: [`GravitySource`], [`GravityModel`],
//!   gravity-controls re-exports from [`gravity_controls`] and
//!   [`spherical_harmonics_gravity_controls`], and the
//!   [`SphericalHarmonicsData`] coefficient container.
//! - **Tides and relativistic corrections**: [`tides`] and [`relativistic`]
//!   carry the small post-Newtonian and luni-solar tide terms.
//!
//! JEOD coefficient data lives in
//! `models/environment/gravity/data/include/earth_GGM05C.hh` (and similar
//! per-body files); [`coefficients`] contains the parsing logic that turns
//! the C++ array headers into `Vec<Vec<f64>>` C and S coefficient tables.
//! Coefficients are normalized as in JEOD; the recursion expects normalized
//! input. Pure Rust, zero Bevy dependency.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod coefficients;
pub mod compute;
pub mod gravity_controls;
pub mod gravity_source;
pub mod relativistic;
pub mod spherical_harmonics_calc_nonspherical;
pub mod spherical_harmonics_gravity_controls;
pub mod spherical_harmonics_gravity_source;
pub mod tides;

pub use compute::{calc_spherical, gravitation, gravitation_with_scratch};
pub use gravity_controls::*;
pub use gravity_source::*;
pub use spherical_harmonics_calc_nonspherical::{
    calc_nonspherical, calc_nonspherical_typed, calc_nonspherical_with_scratch, GottliebScratch,
};
pub use spherical_harmonics_gravity_controls::*;
pub use spherical_harmonics_gravity_source::SphericalHarmonicsData;
