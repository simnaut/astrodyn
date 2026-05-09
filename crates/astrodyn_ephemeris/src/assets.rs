//! Path resolvers for the bundled ephemeris fixtures.
//!
//! `astrodyn_ephemeris/assets/` holds the binary kernels that the
//! workspace's tests, examples, and downstream mission crates load by
//! default:
//!
//! - `de421.bsp` (~17 MB) — production JPL DE421 ephemeris.
//! - `moon_pa_de421_1900-2050.bpc` (~1.7 MB) — lunar PA orientation kernel.
//!
//! These are also embedded as `&'static [u8]` constants in
//! [`crate::data`] (via `include_bytes!`) and ship inside the published
//! `.crate`. New code should generally prefer
//! [`Ephemeris::from_bsp_bytes`](crate::Ephemeris::from_bsp_bytes) over a
//! path-based load — the bytes path works identically inside the
//! workspace and from the published crate, with no `CARGO_MANIFEST_DIR`
//! filesystem lookup.
//!
//! The path resolvers in this module remain useful when a caller needs a
//! `Path` (e.g. shelling a kernel out to another tool, or passing it to
//! an SPK loader the consumer manages itself). They resolve via
//! `CARGO_MANIFEST_DIR` so they only work for in-workspace builds — they
//! will *not* find the assets when called from a downstream consumer of
//! the published crate.
//!
//! [`Ephemeris::from_bsp_bytes`]: crate::Ephemeris::from_bsp_bytes

use std::path::PathBuf;

/// Absolute path to the bundled `de421.bsp` (in-workspace builds only).
///
/// Resolves via `CARGO_MANIFEST_DIR`. For builds that go through the
/// published crate, prefer
/// [`Ephemeris::from_bsp_bytes(crate::data::DE421_BSP)`](crate::Ephemeris::from_bsp_bytes)
/// or the mission-facing `astrodyn::recipes::ephemeris::de421()`.
pub fn de421_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/de421.bsp")
}

/// Absolute path to the bundled `moon_pa_de421_1900-2050.bpc`
/// (in-workspace builds only).
///
/// See [`de421_path`] for the published-crate caveat.
pub fn moon_pa_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/moon_pa_de421_1900-2050.bpc")
}
