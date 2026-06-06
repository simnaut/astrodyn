//! ISS-class LEO scenarios.
//!
//! ```
//! use astrodyn::recipes::Mission;
//! let sb = Mission::iss_leo().into_builder();
//! assert_eq!(sb.bodies.len(), 1);
//! assert_eq!(sb.sources.len(), 1);
//! ```

use astrodyn_gravity::{GravityControl, GravityGradient};

use crate::recipes::{constants, earth, epoch, orbital_elements, vehicle};
use crate::vehicle_builder::VehicleBuilder;
use crate::SimulationBuilder;

/// 3-DOF point-mass ISS-like LEO with Earth point-mass gravity.
///
/// Uses the ISS reference orbital elements
/// ([`crate::recipes::orbital_elements::iss`])
/// initialized via the typestate vehicle builder. Step size is 60 s,
/// epoch is J2000.
///
/// ```
/// use astrodyn::recipes::scenarios::iss_leo;
/// let sb = iss_leo::iss_leo();
/// assert_eq!(sb.bodies.len(), 1);
/// assert_eq!(sb.sources.len(), 1);
/// ```
pub fn iss_leo() -> SimulationBuilder {
    let mut sb = SimulationBuilder::new(epoch::j2000(), 60.0);
    let earth_idx = sb.add_source("Earth", earth::point_mass());
    let vehicle = VehicleBuilder::new()
        .vehicle_named("iss")
        .from_orbital_elements(orbital_elements::iss(), constants::mu_ggm05c())
        .three_dof_point_mass(vehicle::iss_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(
            earth_idx,
            GravityGradient::Skip,
        ))
        .build();
    sb.add_body(vehicle);
    sb
}

/// ISS-like LEO with atmospheric drag (MET solar-mean atmosphere).
///
/// Same orbit as [`iss_leo`] plus a 1900 m² Cd=2.2 drag config and
/// identity attitude (drag requires a rotational state for the
/// inertial → structural frame transform — see JEOD_INV: IN.15).
/// Demonstrates 24-hour altitude decay under solar-mean conditions.
///
/// ```
/// use astrodyn::recipes::scenarios::iss_leo;
/// let sb = iss_leo::iss_leo_drag();
/// assert_eq!(sb.bodies.len(), 1);
/// assert!(sb.atmosphere.is_some());
/// ```
pub fn iss_leo_drag() -> SimulationBuilder {
    use crate::recipes::atmosphere;
    use astrodyn_dynamics::{MassProperties, RotationalState};
    use astrodyn_interactions::DragConfig;
    use astrodyn_math::JeodQuat;

    let mut sb = SimulationBuilder::new(epoch::j2000(), 60.0);
    let earth_idx = sb.add_source("Earth", earth::point_mass());
    sb = sb.atmosphere(atmosphere::met_solar_mean(), earth_idx);

    // Drag needs an attitude — use identity quaternion + zero ω. The
    // typestate's `.sixdof()` requires `MassProperties` with an
    // inertia tensor; supply a minimal one here.
    let rot = RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: glam::DVec3::ZERO,
    };
    let mass =
        MassProperties::with_inertia(420_000.0, glam::DMat3::IDENTITY * 1.0e6, glam::DVec3::ZERO);

    let vehicle = VehicleBuilder::new()
        .vehicle_named("iss")
        .from_orbital_elements(
            orbital_elements::leo_400km_circular_iss_inclination(),
            constants::mu_ggm05c(),
        )
        .sixdof(rot, mass)
        .rk4()
        .gravity(GravityControl::new_spherical(
            earth_idx,
            GravityGradient::Skip,
        ))
        .drag(DragConfig {
            cd: 2.2,
            area: 1900.0,
            constant_density: None,
            ..Default::default()
        })
        .build();
    sb.add_body(vehicle);
    sb
}
