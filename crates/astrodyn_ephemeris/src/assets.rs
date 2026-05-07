//! Path resolvers for the bundled ephemeris fixtures.
//!
//! `astrodyn_ephemeris/assets/` holds the binary kernels that the
//! workspace's tests, examples, and downstream mission crates load by
//! default:
//!
//! - `de421.bsp` (16.8 MB) — production JPL DE421 ephemeris.
//! - `moon_pa_de421_1900-2050.bpc` (1.7 MB) — lunar PA orientation kernel.
//!
//! These are excluded from the published `.crate` (see `Cargo.toml`'s
//! `exclude`) since they are large; downstream users that build against
//! the published crate must provide their own paths to
//! [`Ephemeris::from_bsp`](crate::Ephemeris::from_bsp).
//!
//! In-workspace consumers call these resolvers to find the committed
//! fixtures via `CARGO_MANIFEST_DIR`. Each returns a path relative to
//! the workspace root that the consumer can pass to `Ephemeris::from_bsp`.

use std::path::PathBuf;

/// Absolute path to the bundled `de421.bsp`.
///
/// Resolves via `CARGO_MANIFEST_DIR`, so this works for any in-workspace
/// crate (the assets sit at a stable path inside the `astrodyn_ephemeris`
/// crate). Downstream consumers building against the published crate
/// will not have this file (it is `exclude`d from publishes); they
/// should ship their own `.bsp` and pass the path explicitly to
/// [`Ephemeris::from_bsp`](crate::Ephemeris::from_bsp).
pub fn de421_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/de421.bsp")
}

/// Absolute path to the bundled `moon_pa_de421_1900-2050.bpc`.
///
/// See [`de421_path`] for the publishing caveat.
pub fn moon_pa_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/moon_pa_de421_1900-2050.bpc")
}
