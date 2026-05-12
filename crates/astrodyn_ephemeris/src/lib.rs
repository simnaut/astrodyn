//! Planetary ephemerides backed by JPL DE-series SPK files.
//!
//! Pure-Rust replacement for JEOD's `models/environment/ephemerides/` DE4xx
//! reader. Where JEOD links a hand-rolled binary loader to JPL DE405/DE421
//! kernels, this crate delegates the file format and Chebyshev evaluation to
//! the `anise` crate (a Rust SPICE/NAIF reimplementation) and exposes a
//! thin, frame-tagged API on top.
//!
//! ## Public surface
//!
//! - [`Ephemeris`] — owns an `anise::Almanac` and answers position/velocity
//!   queries from `.bsp` files (e.g., `de421.bsp`, `de440.bsp`). All vectors
//!   are returned in J2000 ICRF, in meters and m/s, wrapped as
//!   `Position<RootInertial>` / `Velocity<RootInertial>` from `astrodyn_quantities`.
//! - [`EphemerisBody`] — the body-identifier enum that maps JEOD's
//!   `EphemerisBody` constants to `anise`'s NAIF integer IDs (Sun, Moon,
//!   Earth, the planets and barycenters needed by Tier 3 tests).
//! - [`EphemerisError`] — fail-loudly error type for missing files,
//!   unsupported bodies, or out-of-range epochs.
//!
//! ## Role in the pipeline
//!
//! The ephemeris populates third-body positions for gravity (Sun and Moon
//! perturbations on Earth orbits, planet-on-planet perturbations on
//! interplanetary trajectories) and for radiation-pressure / shadow geometry
//! in `astrodyn_interactions`. JEOD source: `models/environment/ephemerides/`.
//! Pure Rust, zero Bevy dependency.
//!
//! ## Example
//!
//! Resolve DE421 via [`data::load`] (in-workspace `assets/` during dev,
//! cached fetch from the `kernels-v1` GitHub Release for downstream
//! consumers), then query the Sun's position relative to the
//! solar-system barycenter at J2000.0 (TDB Julian Day `2_451_545.0`):
//!
//! ```ignore
//! use astrodyn_ephemeris::{data, Ephemeris, EphemerisBody};
//!
//! let bytes = data::load(&data::DE421)?;
//! let eph = Ephemeris::from_bsp_bytes(&bytes)?;
//! let (pos, _vel) = eph.get_state_typed(
//!     EphemerisBody::Sun,
//!     EphemerisBody::SolarSystemBarycenter,
//!     2_451_545.0,
//! )?;
//! // The Sun is close to the barycenter (~1 solar radius offset).
//! assert!(pos.raw_si().length() < 2.0e9);
//! # Ok::<(), astrodyn_ephemeris::EphemerisError>(())
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub use astrodyn_quantities::prelude::*;

pub mod assets;
pub mod bodies;
pub mod data;
pub mod ephemeris;

pub use bodies::EphemerisBody;
pub use ephemeris::{Ephemeris, EphemerisError};
