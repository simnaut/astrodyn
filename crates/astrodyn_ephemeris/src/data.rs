//! Embedded planetary ephemeris and orientation kernels.
//!
//! Each constant is the raw bytes of one of the binary fixtures committed
//! under `crates/astrodyn_ephemeris/assets/`. The bytes are pulled in at
//! compile time via [`include_bytes!`], so consumers of the published
//! `.crate` get the exact same kernels as in-workspace builds with no
//! filesystem lookups, environment variables, or `JEOD_HOME` checkout.
//!
//! Pair these with [`Ephemeris::from_bsp_bytes`] /
//! [`Ephemeris::load_bpc_bytes`](crate::Ephemeris::load_bpc_bytes), or use
//! the higher-level mission-facing recipes in
//! `astrodyn::recipes::ephemeris`.
//!
//! [`Ephemeris::from_bsp_bytes`]: crate::Ephemeris::from_bsp_bytes

/// JPL DE421 planetary ephemeris (~17 MB).
///
/// Covers Sun, Moon, planets, and Earth–Moon barycenter from 1900 to 2050
/// in J2000 ICRF. The same `.bsp` JEOD's SIM_dyncomp Tier 3 baselines
/// were generated against.
pub const DE421_BSP: &[u8] = include_bytes!("../assets/de421.bsp");

/// Moon principal-axes orientation kernel (~1.7 MB).
///
/// Covers 1900–2050 and is required by consumers that need the Moon's
/// physical orientation (libration); the SPK alone gives positions but
/// not body-fixed attitude.
pub const MOON_PA_BPC: &[u8] = include_bytes!("../assets/moon_pa_de421_1900-2050.bpc");
