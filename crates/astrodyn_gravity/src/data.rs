//! Embedded planetary gravity-coefficient blobs.
//!
//! Each `*_BIN` constant is the raw bytes of one of the `.bin` fixtures
//! committed under `crates/astrodyn_gravity/test_data/gravity/`. The bytes are
//! pulled in at compile time via [`include_bytes!`], so consumers of the
//! published `.crate` get the exact same data as in-workspace builds without
//! any filesystem lookups, environment variables, or `JEOD_HOME` checkout.
//!
//! The high-level API for loading a [`SphericalHarmonicsData`] from one of
//! these blobs lives in [`crate::fixtures`]; downstream code generally
//! prefers the `fixtures::load_*` helpers (or the mission-facing recipes
//! in `astrodyn::recipes::{earth, moon, mars}`) to working with the byte
//! constants directly.
//!
//! [`SphericalHarmonicsData`]: crate::SphericalHarmonicsData

/// GGM05C Earth gravity coefficients (degree=order=360).
pub const GGM05C_BIN: &[u8] = include_bytes!("../test_data/gravity/ggm05c.bin");

/// GGM02C Earth gravity coefficients (degree=order=200).
pub const GGM02C_BIN: &[u8] = include_bytes!("../test_data/gravity/ggm02c.bin");

/// GEM-T1 Earth gravity coefficients (degree=order=36).
pub const GEMT1_BIN: &[u8] = include_bytes!("../test_data/gravity/gemt1.bin");

/// LP150Q Moon gravity coefficients (degree=order=150).
pub const MOON_LP150Q_BIN: &[u8] = include_bytes!("../test_data/gravity/moon_lp150q.bin");

/// GRAIL150 Moon gravity coefficients (degree=order=150).
pub const MOON_GRAIL150_BIN: &[u8] = include_bytes!("../test_data/gravity/moon_grail150.bin");

/// MRO110B2 Mars gravity coefficients (degree=order=110).
pub const MARS_MRO110B2_BIN: &[u8] = include_bytes!("../test_data/gravity/mars_mro110b2.bin");

/// Sun point-mass record encoded as a degree-1 zero-coefficient SH so the
/// uniform binary loader works. Only `mu` and `radius` are physically
/// meaningful; do not evaluate this as a non-spherical model.
pub const SUN_SPHERICAL_BIN: &[u8] = include_bytes!("../test_data/gravity/sun_spherical.bin");
