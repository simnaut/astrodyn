//! First-class catalog of pre-composed reference missions.
//!
//! [`Mission`] replaces the bare `recipes::scenarios::*()` fn pointers with
//! a discoverable type-driven catalog. Each constructor returns a `Mission`
//! carrying scenario metadata; [`into_builder()`](Mission::into_builder)
//! materializes the underlying `SimulationBuilder` and
//! [`with_time()`](Mission::with_time) overrides the epoch without
//! reconstructing the scenario.
//!
//! # Example
//! ```
//! use jeod_sim::recipes::Mission;
//! let sb = Mission::iss_leo().into_builder();
//! ```
//!
//! Override the epoch:
//! ```
//! use jeod_sim::recipes::{epoch, Mission};
//! let sb = Mission::iss_leo().with_time(epoch::clementine_1994()).into_builder();
//! ```

use crate::recipes::scenarios;
use crate::SimulationBuilder;
use crate::SimulationTime;

/// A reference mission scenario.
///
/// Construct one via [`Mission::iss_leo`] / [`Mission::apollo_translunar`] /
/// etc., optionally compose with [`with_time`](Self::with_time), then
/// finalize with [`into_builder`](Self::into_builder).
pub struct Mission {
    description: &'static str,
    builder_fn: fn() -> SimulationBuilder,
    time_override: Option<SimulationTime>,
}

impl Mission {
    /// 3-DOF point-mass ISS-like LEO with Earth point-mass gravity.
    /// Epoch J2000, 60 s timestep.
    pub fn iss_leo() -> Self {
        Self {
            description: "ISS-class LEO; 3-DOF point-mass Earth gravity at J2000.",
            builder_fn: scenarios::iss_leo::iss_leo,
            time_override: None,
        }
    }

    /// ISS-like LEO with atmospheric drag (MET solar-mean atmosphere) and
    /// 6-DOF identity attitude. 60 s timestep.
    pub fn iss_leo_drag() -> Self {
        Self {
            description: "ISS-class LEO with MET solar-mean drag; 6-DOF identity attitude.",
            builder_fn: scenarios::iss_leo::iss_leo_drag,
            time_override: None,
        }
    }

    /// Apollo translunar trajectory with multi-body Earth + Moon + Sun.
    pub fn apollo_translunar() -> Self {
        Self {
            description: "Apollo translunar trajectory; Earth + Moon + Sun gravity.",
            builder_fn: scenarios::apollo::apollo_translunar,
            time_override: None,
        }
    }

    /// Earth-Moon translunar reference orbit using DE421 ephemerides.
    pub fn earth_moon_translunar() -> Self {
        Self {
            description: "Earth-Moon translunar trajectory; DE421 ephemeris.",
            builder_fn: scenarios::earth_moon::earth_moon_translunar,
            time_override: None,
        }
    }

    /// Geostationary equatorial orbit at sidereal rate.
    pub fn geo() -> Self {
        Self {
            description: "Geostationary equatorial orbit at sidereal rate.",
            builder_fn: scenarios::geostationary::geo,
            time_override: None,
        }
    }

    /// Clementine lunar mission setup at the 1994 launch epoch.
    pub fn clementine_lunar() -> Self {
        Self {
            description: "Clementine lunar mission; 1994 launch epoch.",
            builder_fn: scenarios::clementine_lunar::clementine_lunar,
            time_override: None,
        }
    }

    /// Mars-centered orbit with Sun perturbation; Dawn 2009 epoch.
    pub fn mars_orbit() -> Self {
        Self {
            description: "Mars-centered orbit; Sun perturbation, Dawn 2009 epoch.",
            builder_fn: scenarios::mars_orbit::mars_orbit,
            time_override: None,
        }
    }

    /// Heliocentric Mercury orbit with PPN relativistic corrections.
    pub fn mercury_relativistic() -> Self {
        Self {
            description: "Mercury heliocentric orbit; PPN relativistic corrections.",
            builder_fn: scenarios::mercury::mercury_relativistic,
            time_override: None,
        }
    }

    /// Human-readable scenario description.
    pub fn description(&self) -> &'static str {
        self.description
    }

    /// Override the simulation epoch (e.g. to start the mission at a
    /// non-default time). The override is applied to the
    /// [`SimulationBuilder`] inside [`into_builder`](Self::into_builder).
    pub fn with_time(mut self, time: SimulationTime) -> Self {
        self.time_override = Some(time);
        self
    }

    /// Materialize the underlying [`SimulationBuilder`].
    pub fn into_builder(self) -> SimulationBuilder {
        let mut sb = (self.builder_fn)();
        if let Some(time) = self.time_override {
            sb.time = time;
        }
        sb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipes::epoch;

    #[test]
    fn each_constructor_returns_a_buildable_scenario() {
        // Smoke: every scenario must materialize without panicking.
        let _ = Mission::iss_leo().into_builder();
        let _ = Mission::iss_leo_drag().into_builder();
        let _ = Mission::apollo_translunar().into_builder();
        let _ = Mission::earth_moon_translunar().into_builder();
        let _ = Mission::geo().into_builder();
        let _ = Mission::clementine_lunar().into_builder();
        let _ = Mission::mars_orbit().into_builder();
        let _ = Mission::mercury_relativistic().into_builder();
    }

    #[test]
    fn description_is_non_empty() {
        assert!(!Mission::iss_leo().description().is_empty());
        assert!(!Mission::apollo_translunar().description().is_empty());
    }

    #[test]
    fn with_time_overrides_simulation_epoch() {
        let custom = epoch::clementine_1994();
        let sb = Mission::iss_leo().with_time(custom).into_builder();
        let default = Mission::iss_leo().into_builder();
        // Custom override must differ from the default J2000 epoch.
        assert_ne!(sb.time.tai_tjt_at_epoch, default.time.tai_tjt_at_epoch);
    }
}
