//! Planetary ephemeris recipes backed by JPL kernel assets.
//!
//! Each recipe pulls a kernel via [`astrodyn_ephemeris::data::load`] and
//! returns an [`Ephemeris`] ready to plug into a
//! [`SimulationBuilder`](crate::SimulationBuilder) via `.ephemeris(...)`.
//! Kernels resolve from the in-workspace `assets/` dir during dev/test
//! and from the project's `kernels-v1` GitHub Release (cached in
//! `~/.cache/astrodyn-ephemeris/`) for downstream consumers. See the
//! [`astrodyn_ephemeris::data`] module docs for the full lookup order
//! and offline-build instructions.
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
/// vendored kernel.
pub fn de421() -> Result<Ephemeris, EphemerisError> {
    let bytes = astrodyn_ephemeris::data::load(&astrodyn_ephemeris::data::DE421)?;
    Ephemeris::from_bsp_bytes(&bytes)
}

/// DE421 ephemeris plus the Moon principal-axes orientation kernel.
///
/// Use this when the simulation needs the Moon's body-fixed attitude
/// (libration) — e.g., lunar-fixed frames, lunar-surface targeting, or
/// torque computations against the Moon. The plain [`de421`] recipe
/// suffices when only Moon position/velocity are needed.
pub fn de421_with_moon_pa() -> Result<Ephemeris, EphemerisError> {
    let mut eph = de421()?;
    let bpc = astrodyn_ephemeris::data::load(&astrodyn_ephemeris::data::MOON_PA)?;
    eph.load_bpc_bytes(&bpc)?;
    Ok(eph)
}

/// JPL DE440 planetary ephemeris (Sun, Moon, planets, 1849–2150).
///
/// Required by the NASA NESC GN&C Lunar Check Cases (NESC-RP-23-01853);
/// CC8 in particular pins its reference trajectory to this generation.
/// We ship the `de440s` short-subset (~32 MB) — sufficient for the
/// 2026-class epochs we currently target and two orders of magnitude
/// smaller than the full DE440 archive.
pub fn de440() -> Result<Ephemeris, EphemerisError> {
    let bytes = astrodyn_ephemeris::data::load(&astrodyn_ephemeris::data::DE440)?;
    Ephemeris::from_bsp_bytes(&bytes)
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
    let bpc = astrodyn_ephemeris::data::load(&astrodyn_ephemeris::data::MOON_PA)?;
    eph.load_bpc_bytes(&bpc)?;
    Ok(eph)
}
