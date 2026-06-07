//! Mercury / GR-perihelion-advance scenario.
//!
//! ```
//! use astrodyn::recipes::Mission;
//! let sb = Mission::mercury_relativistic().into_builder();
//! assert_eq!(sb.bodies.len(), 1);
//! ```

use astrodyn_gravity::{GravityControl, GravityGradient};
use glam::DVec3;

use crate::recipes::{epoch, sun};
use crate::vehicle_builder::VehicleBuilder;
use crate::SimulationBuilder;
use astrodyn_quantities::frame::{PlanetInertial, Sun};
use astrodyn_quantities::frame_descriptor::FrameUid;

/// Mercury orbit around the Sun with GR (Schwarzschild) corrections
/// enabled — used to measure perihelion advance over many orbits.
///
/// 100 s step. Initial state matches Mercury at perihelion.
///
/// ```
/// use astrodyn::recipes::scenarios::mercury;
/// let sb = mercury::mercury_relativistic();
/// assert_eq!(sb.sources.len(), 1);
/// assert_eq!(sb.bodies.len(), 1);
/// assert_eq!(sb.dt, 100.0);
/// ```
pub fn mercury_relativistic() -> SimulationBuilder {
    let mut sb = SimulationBuilder::new(epoch::j2000(), 100.0);
    let _ = sb.add_source("Sun", sun::point_mass());

    // Mercury at perihelion (~46 Gm from Sun, ~58.98 km/s).
    use astrodyn_dynamics::state::TranslationalStateTyped;
    use astrodyn_quantities::ext::Vec3Ext;
    use astrodyn_quantities::frame::RootInertial;
    let trans = TranslationalStateTyped::<RootInertial> {
        position: DVec3::new(46.0e9, 0.0, 0.0).m_at::<RootInertial>(),
        velocity: DVec3::new(0.0, 58_980.0, 0.0).m_per_s_at::<RootInertial>(),
    };

    let mut ctrl =
        GravityControl::new_spherical(FrameUid::of::<PlanetInertial<Sun>>(), GravityGradient::Skip);
    ctrl.relativistic = true;

    let vehicle = VehicleBuilder::new()
        .vehicle_named("mercury-probe")
        .with_translational(trans)
        .three_dof_point_mass(uom::si::f64::Mass::new::<uom::si::mass::kilogram>(3.301e23))
        .rk4()
        .gravity(ctrl)
        .build();
    sb.add_body(vehicle);
    sb
}
