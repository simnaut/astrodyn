//! Apollo trans-lunar-injection scenario.
//!
//! ```
//! use jeod_sim::recipes::scenarios;
//! let sb = scenarios::apollo_translunar();
//! assert_eq!(sb.sources.len(), 3);
//! ```

use glam::DVec3;
use jeod_gravity::GravityControl;

use crate::recipes::{constants, earth, epoch, moon, sun, vehicle};
use crate::vehicle_builder::VehicleBuilder;
use crate::SimulationBuilder;

/// CSM in a trans-lunar coast: Earth point-mass central, Moon and Sun
/// as third bodies. Mass-tree-aware so mission code can attach an
/// S-IVB stage (see `examples/apollo.rs`). Step size 60 s, J2000 epoch.
///
/// The example then layers stage separation and an impulsive trans-
/// lunar Δv burn on top of this scenario as inline maneuvers — those
/// are mission-code-specific and not baked into the scenario.
pub fn apollo_translunar() -> SimulationBuilder {
    let mut sb = SimulationBuilder::new(epoch::j2000(), 60.0);
    let earth_idx = sb.add_source("Earth", earth::point_mass());
    // Moon and Sun positions are overwritten each step by the
    // ephemeris stage; the seed values here are immaterial.
    let moon_idx = sb.add_source("Moon", moon::third_body(DVec3::new(3.85e8, 0.0, 0.0)));
    let sun_idx = sb.add_source("Sun", sun::third_body(DVec3::new(1.496e11, 0.0, 0.0)));
    sb = sb.sun(sun_idx).moon(moon_idx);

    // CSM in a 200 km parking orbit, equatorial.
    let r = 6_578_137.0; // Earth radius + 200 km
    let v = (constants::mu_ggm05c().value / r).sqrt();
    use jeod_dynamics::TranslationalState;
    let csm_state = TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        velocity: DVec3::new(0.0, v, 0.0),
    };

    let csm = VehicleBuilder::new()
        .with_state(csm_state)
        .three_dof_point_mass(vehicle::apollo_csm_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(earth_idx, false))
        .gravity(GravityControl::new_third_body(moon_idx))
        .gravity(GravityControl::new_third_body(sun_idx))
        .build();
    sb.add_body(csm);
    sb
}
