//! Geostationary orbit scenario.
//!
//! ```
//! use astrodyn::recipes::Mission;
//! let sb = Mission::geo().into_builder();
//! assert_eq!(sb.bodies.len(), 1);
//! ```

use astrodyn_gravity::{GravityControl, GravityGradient};

use crate::recipes::{constants, earth, epoch, orbital_elements, vehicle};
use crate::vehicle_builder::VehicleBuilder;
use crate::SimulationBuilder;

/// Geostationary point-mass satellite at 42164 km, 0° inclination.
/// Earth point-mass gravity, RK4, J2000 epoch, 300 s step.
///
/// ```
/// use astrodyn::recipes::scenarios::geostationary;
/// let sb = geostationary::geo();
/// assert_eq!(sb.sources.len(), 1);
/// assert_eq!(sb.bodies.len(), 1);
/// assert_eq!(sb.dt, 300.0);
/// ```
pub fn geo() -> SimulationBuilder {
    let mut sb = SimulationBuilder::new(epoch::j2000(), 300.0);
    let earth_idx = sb.add_source("Earth", earth::point_mass());
    let vehicle = VehicleBuilder::new()
        .vehicle_named("geo-sat")
        .from_orbital_elements(orbital_elements::geostationary(), constants::mu_ggm05c())
        .three_dof_point_mass(vehicle::unit_sphere_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(
            earth_idx,
            GravityGradient::Skip,
        ))
        .build();
    sb.add_body(vehicle);
    sb
}
