//! Mars-orbit scenario.
//!
//! ```
//! use astrodyn::recipes::Mission;
//! let sb = Mission::mars_orbit().into_builder();
//! assert_eq!(sb.sources.len(), 2);
//! ```

use astrodyn_gravity::{GravityControl, GravityGradient};
use astrodyn_quantities::ext::Vec3Ext;
use glam::DVec3;

use crate::recipes::{epoch, mars, sun, vehicle};
use crate::vehicle_builder::VehicleBuilder;
use crate::SimulationBuilder;
use astrodyn_quantities::frame::{Mars, PlanetInertial, Sun};
use astrodyn_quantities::frame_descriptor::FrameUid;

/// Dawn-class spacecraft in a Mars orbit. Mars central body
/// (point-mass + IAU rotation), Sun as a third-body source. Step size
/// 10 s, Dawn-arrival epoch.
///
/// The point-mass scenario produces a substantially less accurate
/// trajectory than the JEOD reference (which uses the MRO110B2
/// 110×110 SH model). Mission code wanting verification-grade
/// accuracy substitutes the central-body source with
/// [`crate::recipes::mars::mro110b2`] and adds a non-spherical
/// [`GravityControl`].
///
/// ```
/// use astrodyn::recipes::scenarios::mars_orbit;
/// let sb = mars_orbit::mars_orbit();
/// // Mars central + Sun third-body.
/// assert_eq!(sb.sources.len(), 2);
/// assert_eq!(sb.dt, 10.0);
/// ```
pub fn mars_orbit() -> SimulationBuilder {
    let mut sb = SimulationBuilder::new(epoch::dawn_mars_2009(), 10.0);
    let _ = sb.add_source("Mars", mars::point_mass());
    let sun_idx = sb.add_source(
        "Sun",
        sun::third_body(
            DVec3::new(2.27e11, 0.0, 0.0).m_at::<astrodyn_quantities::frame::RootInertial>(),
        ),
    );
    sb = sb.sun(sun_idx);

    // Dawn initial state at Mars (Mars-centered inertial frame).
    use astrodyn_dynamics::state::TranslationalStateTyped;
    use astrodyn_quantities::frame::RootInertial;
    let trans = TranslationalStateTyped::<RootInertial> {
        position: DVec3::new(11_563_355.680_2, -14_356_668.897_7, 6_293_704.616_9)
            .m_at::<RootInertial>(),
        velocity: DVec3::new(-2_273.107_8, 2_380.132_4, -22.911).m_per_s_at::<RootInertial>(),
    };

    let vehicle = VehicleBuilder::new()
        .vehicle_named("mars-orbiter")
        .with_translational(trans)
        .three_dof_point_mass(vehicle::dawn_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(
            FrameUid::of::<PlanetInertial<Mars>>(),
            GravityGradient::Skip,
        ))
        .gravity(GravityControl::new_third_body(FrameUid::of::<
            PlanetInertial<Sun>,
        >()))
        .build();
    sb.add_body(vehicle);
    sb
}
