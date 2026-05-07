//! Named vehicle archetypes.
//!
//! These are starting-point [`MassProperties`] / mass values for common
//! reference missions. Mission code typically combines them with an
//! [`orbital_elements`](super::orbital_elements) preset and an
//! integrator choice via the typestate
//! [`VehicleBuilder`](crate::VehicleBuilder).
//!
//! ```
//! use astrodyn::recipes::vehicle;
//! use astrodyn_quantities::ext::F64Ext;
//! assert_eq!(vehicle::iss_mass().get::<uom::si::mass::kilogram>(), 420_000.0);
//! ```

use astrodyn_dynamics::MassProperties;
use astrodyn_quantities::ext::F64Ext;
use uom::si::f64::{Length, Mass};
use uom::si::length::meter;
use uom::si::mass::kilogram;

/// ISS-class mass (~420 t) as a typed [`Mass`].
///
/// ```
/// use astrodyn::recipes::vehicle;
/// use uom::si::mass::kilogram;
/// assert_eq!(vehicle::iss_mass().get::<kilogram>(), 420_000.0);
/// ```
pub fn iss_mass() -> Mass {
    420_000.0.kg()
}

/// 1 kg unit-sphere test particle. Used by drag / SRP verification
/// scenarios that match JEOD's "1 kg sphere in elliptical orbit"
/// pattern.
///
/// ```
/// use astrodyn::recipes::vehicle;
/// use uom::si::mass::kilogram;
/// assert_eq!(vehicle::unit_sphere_mass().get::<kilogram>(), 1.0);
/// ```
pub fn unit_sphere_mass() -> Mass {
    1.0.kg()
}

/// STS-114 (Discovery) mass at launch (~109 t).
///
/// ```
/// use astrodyn::recipes::vehicle;
/// use uom::si::mass::kilogram;
/// assert_eq!(vehicle::sts114_mass().get::<kilogram>(), 109_000.0);
/// ```
pub fn sts114_mass() -> Mass {
    109_000.0.kg()
}

/// Clementine probe wet mass at lunar arrival (424 kg).
///
/// Matches the JEOD `SIM_Earth_Moon` reference simulation mass; the
/// `tier3_sim_earth_moon` Tier 3 case asserts against this value. SRP
/// acceleration scales with mass so missions cross-validating against
/// JEOD must use 424 kg, not the 227 kg dry mass.
///
/// ```
/// use astrodyn::recipes::vehicle;
/// use uom::si::mass::kilogram;
/// assert_eq!(vehicle::clementine_mass().get::<kilogram>(), 424.0);
/// ```
pub fn clementine_mass() -> Mass {
    424.0.kg()
}

/// Dawn spacecraft mass at Mars arrival (~1217 kg total).
///
/// ```
/// use astrodyn::recipes::vehicle;
/// use uom::si::mass::kilogram;
/// assert_eq!(vehicle::dawn_mass().get::<kilogram>(), 1_217.0);
/// ```
pub fn dawn_mass() -> Mass {
    1_217.0.kg()
}

/// Apollo CSM mass (~30 t).
///
/// ```
/// use astrodyn::recipes::vehicle;
/// use uom::si::mass::kilogram;
/// assert_eq!(vehicle::apollo_csm_mass().get::<kilogram>(), 30_000.0);
/// ```
pub fn apollo_csm_mass() -> Mass {
    30_000.0.kg()
}

/// Apollo Lunar Module mass (~14.7 t fueled / lunar-descent
/// configuration).
///
/// ```
/// use astrodyn::recipes::vehicle;
/// use uom::si::mass::kilogram;
/// assert_eq!(vehicle::apollo_lm_mass().get::<kilogram>(), 14_700.0);
/// ```
pub fn apollo_lm_mass() -> Mass {
    14_700.0.kg()
}

/// 6-DoF rigid sphere mass properties: total mass `mass`, uniform
/// inertia `I = (2/5) m r²` along all body axes, CoM at structural
/// origin. Inputs are typed so call sites stay unit-safe ergonomically:
/// `vehicle::rigid_sphere(420_000.0.kg(), 5.0.m())`.
///
/// ```
/// use astrodyn_quantities::ext::F64Ext;
/// use astrodyn::recipes::vehicle;
/// let mp = vehicle::rigid_sphere(10.0.kg(), 1.0.m());
/// // I = (2/5) m r² = 4.0
/// assert!((mp.inertia.x_axis.x - 4.0).abs() < 1e-12);
/// assert_eq!(mp.mass, 10.0);
/// ```
pub fn rigid_sphere(mass: Mass, radius: Length) -> MassProperties {
    let m = mass.get::<kilogram>();
    let r = radius.get::<meter>();
    let i = 0.4 * m * r * r;
    let inertia = glam::DMat3::from_diagonal(glam::DVec3::new(i, i, i));
    MassProperties::with_inertia(m, inertia, glam::DVec3::ZERO)
}
