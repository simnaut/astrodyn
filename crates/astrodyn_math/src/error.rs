//! Error type for the orbital-element kernels in
//! [`crate::orbital_elements`].
//!
//! Mirrors the failure cases JEOD's
//! [`models/utils/orbital_elements/src/orbital_elements.cc`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/orbital_elements/src/orbital_elements.cc)
//! reports through `MessageHandler::fail()`.

/// Errors emitted by the orbital-element conversions.
#[derive(Debug, thiserror::Error)]
pub enum OrbitalError {
    /// Newton iteration on Kepler's equation failed to converge within
    /// the configured iteration cap. The carried value is the iteration
    /// count actually attempted.
    #[error("Kepler equation failed to converge after {0} iterations")]
    KeplerConvergence(usize),
    /// Gravitational parameter `mu` was non-positive — JEOD requires
    /// `mu > 0` for any orbital computation.
    #[error("Invalid gravitational parameter: {0}")]
    InvalidMu(f64),
    /// Position or velocity vector had effectively zero magnitude, so
    /// the orbital frame cannot be built.
    #[error("Degenerate orbit: zero position or velocity magnitude")]
    DegenerateOrbit,
}
