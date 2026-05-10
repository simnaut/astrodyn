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

/// NESC Apollo body model: 6-DoF [`MassProperties`] for the
/// "Apollo Model" published in the NASA NESC GN&C Lunar Check Cases.
///
/// Source:
/// <https://nescacademy.nasa.gov/flightsim/2023/bodies#apollo-model>.
///
/// - Mass: 16 642.0 kg (distinct from [`apollo_lm_mass`], which is the
///   14.7 t LM lunar-descent configuration).
/// - Inertia tensor (kg·m², body frame about the CoM):
///   - Diagonal: Ixx = 36 502.7, Iyy = 38 372.4, Izz = 36 514.9.
///   - Off-diagonal: Ixy = −27.1, Iyz = 1 152.4, Izx = 233.2.
/// - Center-of-mass offset (m, body frame): (4.7, 0.01, −0.0075).
///
/// The NESC document notes that the body-fixed coordinate frame
/// "coincides with the orbit-defined (vo) system at the start of the
/// scenario." That is **descriptive**: the published initial quaternion
/// already encodes the body-from-inertial rotation at t=0 — the recipe
/// returns identity `t_struct_body` (the structural-to-body transform),
/// so mission code does not need to apply an additional rotation.
///
/// Used by `astrodyn_verif_nesc::cc8` and any future NESC GN&C check
/// case that names this body model.
///
/// ```
/// use astrodyn::recipes::vehicle;
/// use uom::si::mass::kilogram;
/// let mp = vehicle::nesc_apollo_lm();
/// assert_eq!(mp.mass, 16_642.0);
/// assert!((mp.inertia.x_axis.x - 36_502.7).abs() < 1e-6);
/// assert!((mp.position.x - 4.7).abs() < 1e-12);
/// ```
pub fn nesc_apollo_lm() -> MassProperties {
    let mass_kg = 16_642.0_f64;
    let ixx = 36_502.7;
    let iyy = 38_372.4;
    let izz = 36_514.9;
    let ixy = -27.1;
    let iyz = 1_152.4;
    let izx = 233.2;
    // Symmetric 3×3 inertia tensor. glam::DMat3::from_cols takes columns;
    // the (i,j) entry of the inertia tensor (and its symmetric pair) is
    // the same regardless of column-vs-row layout, so this is unambiguous.
    let inertia = glam::DMat3::from_cols(
        glam::DVec3::new(ixx, ixy, izx),
        glam::DVec3::new(ixy, iyy, iyz),
        glam::DVec3::new(izx, iyz, izz),
    );
    let com = glam::DVec3::new(4.7, 0.01, -0.0075);
    MassProperties::with_inertia(mass_kg, inertia, com)
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
