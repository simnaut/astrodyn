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

/// JPL DE440 planetary ephemeris (Sun, Moon, planets, 1849–2150).
///
/// Required by the NASA NESC GN&C Lunar Check Cases (NESC-RP-23-01853);
/// CC8 in particular pins its reference trajectory to this generation.
/// We ship the `de440s` short-subset (~32 MB) — sufficient for the
/// 2026-class epochs we currently target and two orders of magnitude
/// smaller than the full DE440 archive.
///
/// Equivalent to `Ephemeris::from_bsp("de440s.bsp")` against the NAIF-
/// vendored kernel, but the bytes are embedded at compile time.
pub fn de440() -> Result<Ephemeris, EphemerisError> {
    Ephemeris::from_bsp_bytes(astrodyn_ephemeris::data::DE440_BSP)
}

/// DE440 ephemeris plus the Moon principal-axes orientation kernel.
///
/// Use this when the simulation needs the Moon's body-fixed attitude
/// (libration) — e.g., lunar-fixed frames, lunar-surface targeting, or
/// torque computations against the Moon. The plain [`de440`] recipe
/// suffices when only Moon position/velocity are needed.
///
/// The bundled BPC kernel is `moon_pa_de421_1900-2050.bpc` (the same
/// kernel [`de421_with_moon_pa`] uses). Mixing a DE440 BSP with a DE421
/// BPC introduces a small inconsistency in the Moon's libration model
/// (sub-arcsecond level over a few-day propagation), which is acceptable
/// for the NESC CC8 NRHO use case but may be tightened later by
/// switching to the DE440-aligned `moon_pa_de440_*.bpc` kernel.
pub fn de440_with_moon_pa() -> Result<Ephemeris, EphemerisError> {
    let mut eph = de440()?;
    eph.load_bpc_bytes(astrodyn_ephemeris::data::MOON_PA_BPC)?;
    Ok(eph)
}
