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
//!   `astrodyn_dynamics::GravityAcceleration` containing acceleration, gradient,
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
//!
//! ## Example
//!
//! Point-mass gravity at the equator of a body with Earth's `mu` should
//! point inward (toward the body's centre):
//!
//! ```
//! use astrodyn_gravity::calc_spherical;
//! use glam::DVec3;
//!
//! let mu = 3.986_004_415e14; // Earth GGM05C, m^3/s^2
//! let r = DVec3::new(6_778_137.0, 0.0, 0.0); // 400 km altitude on +X axis
//!
//! let g = calc_spherical(mu, r);
//! // Acceleration points back toward the origin (-X) at ~8.7 m/s^2.
//! assert!(g.grav_accel.x < 0.0);
//! assert!((g.grav_accel.length() - mu / r.length().powi(2)).abs() < 1e-6);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod accumulate;
pub mod coefficients;
pub mod compute;
pub mod data;
pub mod fixtures;
pub mod gravity_controls;
pub mod gravity_source;
pub mod jeod_cc;
pub mod relativistic;
pub mod spherical_harmonics_calc_nonspherical;
pub mod spherical_harmonics_gravity_controls;
pub mod spherical_harmonics_gravity_source;
pub mod tides;
pub mod verif;

pub use accumulate::{
    accumulate_gravity, accumulate_gravity_typed, accumulate_relativistic_corrections,
    accumulate_relativistic_corrections_typed, evaluate_body_gravity_typed, run_gravity_stage,
    GravityBodyInputs, ResolvedRelativisticSource, ResolvedSource,
};
pub use compute::{calc_spherical, gravitation, gravitation_with_scratch, GravityKernelOutput};
pub use gravity_controls::*;
pub use gravity_source::*;
pub use spherical_harmonics_calc_nonspherical::{
    calc_nonspherical, calc_nonspherical_typed, calc_nonspherical_with_scratch, GottliebScratch,
};
pub use spherical_harmonics_gravity_controls::*;
pub use spherical_harmonics_gravity_source::SphericalHarmonicsData;
