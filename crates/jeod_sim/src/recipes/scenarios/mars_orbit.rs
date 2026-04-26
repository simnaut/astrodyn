//! Mars-orbit scenario.
//!
//! ```
//! use jeod_sim::recipes::scenarios;
//! let sb = scenarios::mars_orbit();
//! assert_eq!(sb.sources.len(), 2);
//! ```

use glam::DVec3;
use jeod_gravity::GravityControl;

use crate::recipes::{constants, epoch, mars, sun, vehicle};
use crate::vehicle_builder::VehicleBuilder;
use crate::SimulationBuilder;

/// Dawn-class spacecraft in a Mars orbit. Mars central body
/// (point-mass + IAU rotation), Sun as a third-body source. Step size
/// 10 s, Dawn-arrival epoch.
///
/// The point-mass scenario produces a substantially less accurate
/// trajectory than the JEOD reference (which uses the MRO110B2
/// 110×110 SH model). Mission code wanting verification-grade
/// accuracy substitutes the central-body source with
/// `verification::reference_data::mars_mro110b2()` and adds a
/// non-spherical [`GravityControl`].
pub fn mars_orbit() -> SimulationBuilder {
    let mut sb = SimulationBuilder::new(epoch::dawn_mars_2009(), 10.0);
    let mars_idx = sb.add_source("Mars", mars::point_mass());
    let sun_idx = sb.add_source("Sun", sun::third_body(DVec3::new(2.27e11, 0.0, 0.0)));
    sb = sb.sun(sun_idx);

    // Dawn initial state at Mars (Mars-centered inertial frame).
    use jeod_dynamics::TranslationalState;
    let trans = TranslationalState {
        position: DVec3::new(11_563_355.680_2, -14_356_668.897_7, 6_293_704.616_9),
        velocity: DVec3::new(-2_273.107_8, 2_380.132_4, -22.911),
    };

    let vehicle = VehicleBuilder::new()
        .with_state(trans)
        .three_dof_point_mass(vehicle::dawn_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(mars_idx, false))
        .gravity(GravityControl::new_third_body(sun_idx))
        .build();
    sb.add_body(vehicle);

    let _ = constants::mu_mars(); // touch the constant so doctest sees it referenced
    sb
}
