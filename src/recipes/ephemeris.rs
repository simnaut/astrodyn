//! Planetary ephemeris recipes backed by the bundled JPL kernels.
//!
//! Each recipe wraps an embedded SPK / BPC byte blob from
//! [`astrodyn_ephemeris::data`] and returns an [`Ephemeris`] ready to
//! plug into a [`SimulationBuilder`](crate::SimulationBuilder) via
//! `.ephemeris(...)`. Because the kernels are pulled in with
//! `include_bytes!`, these recipes work identically inside the
//! workspace and from the published `.crate` — no filesystem lookups,
//! no `JEOD_HOME`.
//!
//! ```ignore
//! use astrodyn::recipes::ephemeris;
//! let eph = ephemeris::de421()?;
//! # Ok::<(), astrodyn::EphemerisError>(())
//! ```

use crate::{Ephemeris, EphemerisError};

/// JPL DE421 planetary ephemeris (Sun, Moon, planets, 1900–2050).
///
/// Equivalent to `Ephemeris::from_bsp("de421.bsp")` against the JEOD-
/// vendored kernel, but the bytes are embedded at compile time.
pub fn de421() -> Result<Ephemeris, EphemerisError> {
    Ephemeris::from_bsp_bytes(astrodyn_ephemeris::data::DE421_BSP)
}

/// DE421 ephemeris plus the Moon principal-axes orientation kernel.
///
/// Use this when the simulation needs the Moon's body-fixed attitude
/// (libration) — e.g., lunar-fixed frames, lunar-surface targeting, or
/// torque computations against the Moon. The plain [`de421`] recipe
/// suffices when only Moon position/velocity are needed.
pub fn de421_with_moon_pa() -> Result<Ephemeris, EphemerisError> {
    let mut eph = de421()?;
    eph.load_bpc_bytes(astrodyn_ephemeris::data::MOON_PA_BPC)?;
    Ok(eph)
}
