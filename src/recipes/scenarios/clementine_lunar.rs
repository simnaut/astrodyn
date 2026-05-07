//! Clementine 1994 lunar-orbit scenario.
//!
//! ```
//! use astrodyn::recipes::Mission;
//! let sb = Mission::clementine_lunar().into_builder();
//! // Three sources: Moon (central), Earth (third-body), Sun (third-body / SRP).
//! assert_eq!(sb.sources.len(), 3);
//! ```

use astrodyn_gravity::GravityControl;
use astrodyn_quantities::ext::Vec3Ext;
use glam::DVec3;

use crate::recipes::{constants, earth, epoch, moon, sun, vehicle};
use crate::vehicle_builder::VehicleBuilder;
use crate::SimulationBuilder;

/// Clementine probe in low lunar orbit at the 1994 mission epoch.
///
/// - Moon as central point-mass body (rotation via JEOD's IAU model).
/// - Earth as third-body perturbation source (point-mass).
/// - Sun as third-body source for both gravity and SRP shadow geometry.
/// - Cannonball SRP (cx area 5 m², albedo 0.4, diffuse 0.4).
///
/// Mission code that wants high-fidelity Moon gravity (LP150Q) replaces
/// the Moon entry with
/// `astrodyn::recipes::verification::reference_data::moon_lp150q()` and
/// adds a non-spherical control. That's appropriate only for Tier 3
/// cross-validation (it requires `$JEOD_HOME`).
///
/// ```
/// use astrodyn::recipes::scenarios::clementine_lunar;
/// let sb = clementine_lunar::clementine_lunar();
/// // Moon (central) + Earth (third-body) + Sun (third-body / SRP).
/// assert_eq!(sb.sources.len(), 3);
/// assert_eq!(sb.bodies.len(), 1);
/// ```
pub fn clementine_lunar() -> SimulationBuilder {
    let mut sb = SimulationBuilder::new(epoch::clementine_1994(), 1.0);

    // Sources: Moon (central), Earth/Sun (third-body, positions
    // overwritten each step by the ephemeris stage if mission code
    // sets one up).
    let moon_idx = sb.add_source("Moon", moon::point_mass());
    let earth_idx = sb.add_source(
        "Earth",
        earth::third_body(
            DVec3::new(-3.85e8, 0.0, 0.0).m_at::<astrodyn_quantities::frame::RootInertial>(),
        ), // approx Earth–Moon distance
    );
    let sun_idx = sb.add_source(
        "Sun",
        sun::third_body(
            DVec3::new(1.496e11, 0.0, 0.0).m_at::<astrodyn_quantities::frame::RootInertial>(),
        ), // approx 1 AU
    );
    sb = sb.sun(sun_idx);

    // Initial state: 250 km altitude polar lunar orbit (representative —
    // the JEOD reference uses a more elaborate state from BPC kernels).
    let r_moon = 1_738_140.0_f64;
    let r = r_moon + 250_000.0;
    let mu_moon = constants::mu_moon().value;
    let v = (mu_moon / r).sqrt();

    use astrodyn_dynamics::TranslationalState;
    let trans = TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        velocity: DVec3::new(0.0, 0.0, v), // polar orbit
    };

    let vehicle = VehicleBuilder::new()
        .with_state(trans)
        .three_dof_point_mass(vehicle::clementine_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(moon_idx, false))
        .gravity(GravityControl::new_third_body(earth_idx))
        .gravity(GravityControl::new_third_body(sun_idx))
        .cannonball_srp(5.0, 0.4, 0.4)
        .shadow(moon_idx, &crate::MOON)
        .build();
    sb.add_body(vehicle);
    sb
}
