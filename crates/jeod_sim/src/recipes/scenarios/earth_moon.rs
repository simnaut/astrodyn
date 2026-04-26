//! Earth-Moon multi-body scenario.
//!
//! Convenience wrapper that re-exports
//! [`clementine_lunar`](super::clementine_lunar::clementine_lunar) as a
//! generic Earth-Moon trans-lunar-coast scenario. The
//! `crates/jeod_runner/examples/earth_moon.rs` example uses this entry
//! point. Phase 7 will add a dedicated `tier3_earth_moon` verification
//! case.

use crate::SimulationBuilder;

/// Trans-lunar coast with Earth, Moon, and Sun. See
/// [`clementine_lunar`](super::clementine_lunar::clementine_lunar) for
/// the underlying setup.
pub fn earth_moon_translunar() -> SimulationBuilder {
    super::clementine_lunar::clementine_lunar()
}
