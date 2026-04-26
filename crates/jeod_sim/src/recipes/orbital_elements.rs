//! Orbital element presets for common reference orbits.
//!
//! These presets construct an [`OrbitalElements`] via
//! [`OrbitalElements::from_cartesian_typed`] from named state vectors,
//! so the resulting elements include all derived fields (mean anomaly,
//! orbital energy, angular momentum, …) consistently.
//!
//! ```
//! use jeod_sim::recipes::orbital_elements;
//! let oe = orbital_elements::iss();
//! assert!(oe.semi_major_axis > 6_700_000.0 && oe.semi_major_axis < 6_900_000.0);
//! assert!(oe.e_mag < 0.01);
//! ```

use jeod_math::OrbitalElements;
use jeod_quantities::aliases::{Position, Velocity};
use jeod_quantities::frame::Inertial;

use super::constants::mu_ggm05c;

fn from_pos_vel(pos: glam::DVec3, vel: glam::DVec3) -> OrbitalElements {
    let p = Position::<Inertial>::from_raw_si(pos);
    let v = Velocity::<Inertial>::from_raw_si(vel);
    OrbitalElements::from_cartesian_typed(mu_ggm05c(), p, v)
        .expect("preset state vector must produce well-defined orbital elements")
}

/// ISS-class circular LEO at 400 km altitude, inclination 51.6°.
///
/// Constructed analytically from the canonical altitude / inclination —
/// this is the same orbit as
/// [`leo_400km_circular_iss_inclination`], exposed under the
/// mission-friendly name `iss`. Mission code that needs the JEOD
/// reference state vector (with all of Earth's perturbations baked in)
/// constructs it directly from a CSV / Python preset; that's a Tier 3
/// concern, not a recipe.
pub fn iss() -> OrbitalElements {
    leo_400km_circular_iss_inclination()
}

/// Geostationary circular orbit at 42164 km, inclination 0°.
pub fn geostationary() -> OrbitalElements {
    let r = 42_164_172.0_f64;
    let mu = 3.986_004_415e14_f64;
    let v = (mu / r).sqrt();
    from_pos_vel(glam::DVec3::new(r, 0.0, 0.0), glam::DVec3::new(0.0, v, 0.0))
}

/// 400 km circular LEO at 51.6° inclination — the simplest ISS-like
/// orbit (analytic closed-form, vs. [`iss`] which is from the JEOD
/// reference state).
pub fn leo_400km_circular_iss_inclination() -> OrbitalElements {
    let r_eq = 6_378_137.0_f64;
    let r = r_eq + 400_000.0;
    let mu = 3.986_004_415e14_f64;
    let v = (mu / r).sqrt();
    let inc = 51.6_f64.to_radians();
    from_pos_vel(
        glam::DVec3::new(r, 0.0, 0.0),
        glam::DVec3::new(0.0, v * inc.cos(), v * inc.sin()),
    )
}

/// Polar circular LEO at 600 km altitude.
pub fn leo_polar_600km() -> OrbitalElements {
    let r_eq = 6_378_137.0_f64;
    let r = r_eq + 600_000.0;
    let mu = 3.986_004_415e14_f64;
    let v = (mu / r).sqrt();
    let inc = 90.0_f64.to_radians();
    from_pos_vel(
        glam::DVec3::new(r, 0.0, 0.0),
        glam::DVec3::new(0.0, v * inc.cos(), v * inc.sin()),
    )
}
