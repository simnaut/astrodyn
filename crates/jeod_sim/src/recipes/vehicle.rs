//! Named vehicle archetypes.
//!
//! These are starting-point [`MassProperties`] / mass values for common
//! reference missions. Mission code typically combines them with an
//! [`orbital_elements`](super::orbital_elements) preset and an
//! integrator choice via the typestate
//! [`VehicleBuilder`](crate::VehicleBuilder).
//!
//! ```
//! use jeod_sim::recipes::vehicle;
//! use jeod_quantities::ext::F64Ext;
//! assert_eq!(vehicle::iss_mass().get::<uom::si::mass::kilogram>(), 420_000.0);
//! ```

use jeod_dynamics::MassProperties;
use jeod_quantities::ext::F64Ext;
use uom::si::f64::Mass;

/// ISS-class mass (~420 t) as a typed [`Mass`].
pub fn iss_mass() -> Mass {
    420_000.0.kg()
}

/// 1 kg unit-sphere test particle. Used by drag / SRP verification
/// scenarios that match JEOD's "1 kg sphere in elliptical orbit"
/// pattern.
pub fn unit_sphere_mass() -> Mass {
    1.0.kg()
}

/// STS-114 (Discovery) mass at launch (~109 t).
pub fn sts114_mass() -> Mass {
    109_000.0.kg()
}

/// Clementine probe wet mass at lunar arrival (424 kg).
///
/// Matches the JEOD `SIM_Earth_Moon` reference simulation mass; the
/// `tier3_sim_earth_moon` Tier 3 case asserts against this value. SRP
/// acceleration scales with mass so missions cross-validating against
/// JEOD must use 424 kg, not the 227 kg dry mass.
pub fn clementine_mass() -> Mass {
    424.0.kg()
}

/// Dawn spacecraft mass at Mars arrival (~1217 kg total).
pub fn dawn_mass() -> Mass {
    1_217.0.kg()
}

/// Apollo CSM mass (~30 t).
pub fn apollo_csm_mass() -> Mass {
    30_000.0.kg()
}

/// 6-DoF rigid sphere mass properties: total mass `m`, uniform inertia
/// `I = (2/5) m r²` along all body axes, CoM at structural origin.
pub fn rigid_sphere(mass_kg: f64, radius_m: f64) -> MassProperties {
    let i = 0.4 * mass_kg * radius_m * radius_m;
    let inertia = glam::DMat3::from_diagonal(glam::DVec3::new(i, i, i));
    MassProperties::with_inertia(mass_kg, inertia, glam::DVec3::ZERO)
}
