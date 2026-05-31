//! Path resolvers for the in-workspace kernel fixtures.
//!
//! `crates/astrodyn_ephemeris/assets/` holds the committed binary kernels
//! that workspace tests, examples, and dev tooling use:
//!
//! - `de421.bsp` (~17 MB) — JPL DE421 ephemeris.
//! - `de440.bsp` (~31 MB) — JPL DE440 short-subset ephemeris.
//! - `moon_pa_de421_1900-2050.bpc` (~1.7 MB) — lunar PA orientation kernel.
//! - `moon_fk_de421.epa` (~345 B) — lunar PA→ME frame-offset kernel
//!   (converted from NAIF `moon_080317.tf`).
//! - `pck11.pca` (~45 KB) — IAU planetary-constants kernel (converted from
//!   NAIF `pck00011.tpc` + `gm_de440.tpc`).
//!
//! These files are *not* bundled into the published `.crate`. Downstream
//! consumers obtain them on demand via [`crate::data::load`], which
//! fetches and caches them from the project's `kernels-v1` GitHub
//! Release (or from a directory set in `$ASTRODYN_EPHEMERIS_KERNELS_DIR`).
//!
//! New code should prefer [`crate::data::load`] over the path resolvers
//! here — the [`load`](crate::data::load) path works identically inside
//! the workspace and from the published crate.
//!
//! The path resolvers in this module remain useful when a caller needs
//! a [`PathBuf`] (e.g. shelling a kernel out to another tool, or passing
//! it to an SPK loader the consumer manages itself). They resolve via
//! `CARGO_MANIFEST_DIR` so they only work for in-workspace builds — they
//! will *not* find the assets when called from a downstream consumer of
//! the published crate.

use std::path::PathBuf;

/// Absolute path to the committed `de421.bsp` (in-workspace builds only).
///
/// Resolves via `CARGO_MANIFEST_DIR`. For builds that go through the
/// published crate, prefer `data::load(&data::DE421)` or the mission-
/// facing `astrodyn::recipes::ephemeris::de421()`.
pub fn de421_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/de421.bsp")
}

/// Absolute path to the committed `moon_pa_de421_1900-2050.bpc`
/// (in-workspace builds only).
///
/// See [`de421_path`] for the published-crate caveat.
pub fn moon_pa_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/moon_pa_de421_1900-2050.bpc")
}

/// Absolute path to the committed `moon_fk_de421.epa` — the Moon PA→ME frame
/// kernel (Euler parameters) converted from NAIF `moon_080317.tf` via
/// `cargo xtask generate-orientation-kernels` (in-workspace builds only).
///
/// See [`de421_path`] for the published-crate caveat.
pub fn moon_fk_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/moon_fk_de421.epa")
}

/// Absolute path to the committed `pck11.pca` — the IAU planetary-constants
/// kernel (pole/PM orientation), converted from NAIF `pck00011.tpc` +
/// `gm_de440.tpc` via `cargo xtask generate-orientation-kernels` (in-workspace
/// builds only).
///
/// See [`de421_path`] for the published-crate caveat.
pub fn pck_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/pck11.pca")
}
