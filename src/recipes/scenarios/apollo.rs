//! Apollo trans-lunar-injection scenario.
//!
//! ```
//! use astrodyn::recipes::scenarios::apollo;
//! use astrodyn::recipes::Mission;
//! let sb = Mission::apollo_translunar().into_builder();
//! assert_eq!(sb.sources.len(), 3);
//! assert_eq!(apollo::EARTH_IDX, 0);
//! assert_eq!(apollo::MOON_IDX, 1);
//! ```

use astrodyn_gravity::{GravityControl, GravityRole};
use astrodyn_quantities::ext::Vec3Ext;
use glam::DVec3;

use crate::recipes::{constants, earth, epoch, moon, sun, vehicle};
use crate::vehicle_builder::VehicleBuilder;
use crate::SimulationBuilder;

/// Source index of Earth in the [`apollo_translunar`] scenario.
pub const EARTH_IDX: usize = 0;
/// Source index of the Moon in the [`apollo_translunar`] scenario.
pub const MOON_IDX: usize = 1;
/// Source index of the Sun in the [`apollo_translunar`] scenario.
pub const SUN_IDX: usize = 2;

/// CSM in a trans-lunar coast: Earth point-mass central, Moon and Sun
/// as third bodies. Mass-tree-aware so mission code can attach an
/// S-IVB stage (see `crates/astrodyn_runner/examples/apollo.rs`). Step
/// size 60 s, J2000 epoch.
///
/// Source indices are exposed as
/// [`EARTH_IDX`] / [`MOON_IDX`] / [`SUN_IDX`] so mission code that
/// extends the scenario (e.g., wires DE421 ephemeris on Moon and Sun)
/// doesn't depend on hard-coded magic numbers.
///
/// The example then layers stage separation and an impulsive
/// trans-lunar Δv burn on top of this scenario as inline maneuvers —
/// those are mission-code-specific and not baked into the scenario.
///
/// ```
/// use astrodyn::recipes::scenarios::apollo;
/// let sb = apollo::apollo_translunar();
/// // Earth + Moon + Sun gravity sources.
/// assert_eq!(sb.sources.len(), 3);
/// assert_eq!(sb.bodies.len(), 1);
/// ```
pub fn apollo_translunar() -> SimulationBuilder {
    let mut sb = SimulationBuilder::new(epoch::j2000(), 60.0);
    let earth_idx = sb.add_source("Earth", earth::point_mass());
    debug_assert_eq!(earth_idx, EARTH_IDX);
    // Moon and Sun positions are overwritten each step by the
    // ephemeris stage; the seed values here are immaterial.
    let moon_idx = sb.add_source(
        "Moon",
        moon::third_body(
            DVec3::new(3.85e8, 0.0, 0.0).m_at::<astrodyn_quantities::frame::RootInertial>(),
        ),
    );
    debug_assert_eq!(moon_idx, MOON_IDX);
    let sun_idx = sb.add_source(
        "Sun",
        sun::third_body(
            DVec3::new(1.496e11, 0.0, 0.0).m_at::<astrodyn_quantities::frame::RootInertial>(),
        ),
    );
    debug_assert_eq!(sun_idx, SUN_IDX);
    sb = sb.sun(sun_idx).moon(moon_idx);

    // CSM in a 200 km parking orbit, equatorial.
    let r = 6_578_137.0; // Earth radius + 200 km
    let v = (constants::mu_ggm05c().value / r).sqrt();
    use astrodyn_dynamics::state::TranslationalStateTyped;
    use astrodyn_quantities::frame::RootInertial;
    let csm_state = TranslationalStateTyped::<RootInertial> {
        position: DVec3::new(r, 0.0, 0.0).m_at::<RootInertial>(),
        velocity: DVec3::new(0.0, v, 0.0).m_per_s_at::<RootInertial>(),
    };

    let csm = VehicleBuilder::new()
        .with_translational(csm_state)
        .three_dof_point_mass(vehicle::apollo_csm_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(
            earth_idx,
            GravityRole::Central,
        ))
        .gravity(GravityControl::new_third_body(moon_idx))
        .gravity(GravityControl::new_third_body(sun_idx))
        .build();
    sb.add_body(csm);
    sb
}
