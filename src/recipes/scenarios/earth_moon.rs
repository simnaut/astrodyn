//! Earth-Moon multi-body scenario.
//!
//! Convenience wrapper that re-exports
//! [`clementine_lunar`](super::clementine_lunar::clementine_lunar) as a
//! generic Earth-Moon trans-lunar-coast scenario. The
//! `crates/astrodyn_runner/examples/earth_moon.rs` example uses this entry
//! point. Phase 7 will add a dedicated `tier3_earth_moon` verification
//! case.

use crate::SimulationBuilder;

/// Trans-lunar coast with Earth, Moon, and Sun. See
/// [`clementine_lunar`](super::clementine_lunar::clementine_lunar) for
/// the underlying setup.
///
/// ```
/// use astrodyn::recipes::scenarios::earth_moon;
/// let sb = earth_moon::earth_moon_translunar();
/// assert_eq!(sb.sources.len(), 3);
/// ```
pub fn earth_moon_translunar() -> SimulationBuilder {
    super::clementine_lunar::clementine_lunar()
}
