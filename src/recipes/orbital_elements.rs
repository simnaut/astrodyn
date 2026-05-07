//! Orbital element presets for common reference orbits.
//!
//! These presets construct an [`OrbitalElements<P>`] via
//! [`OrbitalElements::from_cartesian_typed`] from named state vectors,
//! so the resulting elements include all derived fields (mean anomaly,
//! orbital energy, angular momentum, …) consistently. Each preset
//! returns a planet-tagged [`OrbitalElements<P>`] so the central body
//! identity flows through to the consumer's type at compile time.
//!
//! ```
//! use astrodyn::recipes::orbital_elements;
//! let oe = orbital_elements::iss();
//! assert!(oe.semi_major_axis > 6_700_000.0 && oe.semi_major_axis < 6_900_000.0);
//! assert!(oe.e_mag < 0.01);
//! ```

use astrodyn_math::OrbitalElements;
use astrodyn_quantities::aliases::{Position, Velocity};
use astrodyn_quantities::dims::GravParam;
use astrodyn_quantities::frame::{Earth, Mars, Planet, PlanetInertial, Sun};

use super::constants::{mu_ggm05c, mu_mars, mu_sun};

/// Compute classical orbital elements for the given μ in the
/// gravitating body `P`'s inertial frame. Every preset in this module
/// funnels through here so the velocity computation (when present) and
/// the conversion share a single μ value.
///
/// Generic over `P: Planet` so callers pin their actual gravitating
/// body — Earth-orbit presets use `<Earth>`, the Mercury heliocentric
/// preset uses `<Sun>`, the Mars-arrival preset uses `<Mars>`.
///
/// What the phantom enforces:
///
/// - **Position-velocity frame agreement**: a `Position<PlanetInertial<Earth>>`
///   next to a `Velocity<PlanetInertial<Sun>>` is a compile error.
/// - **μ-vs-frame agreement**: `from_pos_vel_with_mu::<Earth>(…, mu_sun())`
///   is a compile error — `mu_sun()` returns `GravParam<Sun>`, but the
///   `<Earth>` instantiation requires `GravParam<Earth>`.
/// - **Output planet identity**: the returned `OrbitalElements<P>`
///   propagates the central-body phantom to downstream consumers.
fn from_pos_vel_with_mu<P: Planet>(
    pos: glam::DVec3,
    vel: glam::DVec3,
    mu: GravParam<P>,
) -> OrbitalElements<P> {
    let p = Position::<PlanetInertial<P>>::from_raw_si(pos);
    let v = Velocity::<PlanetInertial<P>>::from_raw_si(vel);
    OrbitalElements::<P>::from_cartesian_typed(mu, p, v)
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
///
/// ```
/// use astrodyn::recipes::orbital_elements;
/// let oe = orbital_elements::iss();
/// assert!(oe.semi_major_axis > 6_700_000.0);
/// assert!(oe.e_mag < 1e-6);
/// ```
pub fn iss() -> OrbitalElements<Earth> {
    leo_400km_circular_iss_inclination()
}

/// Build a circular orbit of radius `r` and inclination `inc` (radians)
/// around planet `P` whose gravitational parameter is `mu`. Used by the
/// circular-orbit presets so that the velocity computation and the
/// `from_pos_vel_with_mu` call share a single source of truth for μ —
/// mismatching them silently breaks the circular-orbit invariant.
fn circular_orbit_with_mu<P: Planet>(r: f64, inc: f64, mu: GravParam<P>) -> OrbitalElements<P> {
    let v = (mu.value / r).sqrt();
    from_pos_vel_with_mu::<P>(
        glam::DVec3::new(r, 0.0, 0.0),
        glam::DVec3::new(0.0, v * inc.cos(), v * inc.sin()),
        mu,
    )
}

/// Geostationary circular orbit at 42164 km, inclination 0°.
///
/// ```
/// use astrodyn::recipes::orbital_elements;
/// let oe = orbital_elements::geostationary();
/// assert!((oe.semi_major_axis - 42_164_172.0).abs() < 1.0);
/// assert!(oe.inclination.abs() < 1e-12);
/// ```
pub fn geostationary() -> OrbitalElements<Earth> {
    circular_orbit_with_mu::<Earth>(42_164_172.0_f64, 0.0, mu_ggm05c())
}

/// 400 km circular LEO at 51.6° inclination — the simplest ISS-like
/// orbit (analytic closed-form, vs. [`iss`] which is from the JEOD
/// reference state).
///
/// ```
/// use astrodyn::recipes::orbital_elements;
/// let oe = orbital_elements::leo_400km_circular_iss_inclination();
/// assert!((oe.inclination - 51.6_f64.to_radians()).abs() < 1e-12);
/// assert!(oe.e_mag < 1e-6);
/// ```
pub fn leo_400km_circular_iss_inclination() -> OrbitalElements<Earth> {
    let r_eq = 6_378_137.0_f64;
    circular_orbit_with_mu::<Earth>(r_eq + 400_000.0, 51.6_f64.to_radians(), mu_ggm05c())
}

/// Polar circular LEO at 600 km altitude.
///
/// ```
/// use astrodyn::recipes::orbital_elements;
/// let oe = orbital_elements::leo_polar_600km();
/// assert!((oe.inclination - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
/// assert!(oe.semi_major_axis > 6_900_000.0 && oe.semi_major_axis < 7_000_000.0);
/// ```
pub fn leo_polar_600km() -> OrbitalElements<Earth> {
    let r_eq = 6_378_137.0_f64;
    circular_orbit_with_mu::<Earth>(r_eq + 600_000.0, 90.0_f64.to_radians(), mu_ggm05c())
}

// ── Heliocentric / planetary presets ─────────────────────────────────

/// Mercury at perihelion in a heliocentric inertial frame.
///
/// Position ~46 Gm Sun-ward (perihelion distance, scalar), velocity
/// ~58.98 km/s perpendicular. Matches the inlined initial state used
/// by the GR perihelion-advance test (`tier3_sim_mercury`) and the
/// existing `scenarios::mercury_relativistic` setup.
///
/// ```
/// use astrodyn::recipes::orbital_elements;
/// let oe = orbital_elements::mercury_perihelion();
/// // Mercury's eccentricity is ~0.2 — definitely non-circular.
/// assert!(oe.e_mag > 0.1 && oe.e_mag < 0.3);
/// ```
pub fn mercury_perihelion() -> OrbitalElements<Sun> {
    from_pos_vel_with_mu::<Sun>(
        glam::DVec3::new(46.0e9, 0.0, 0.0),
        glam::DVec3::new(0.0, 58_980.0, 0.0),
        mu_sun(),
    )
}

/// Dawn-class spacecraft initial state at Mars (Mars-centered
/// inertial). Constants match `scenarios::mars_orbit` exactly so a
/// custom-loop test can pull the same starting state without
/// duplicating the literal vectors.
///
/// ```
/// use astrodyn::recipes::orbital_elements;
/// let oe = orbital_elements::mars_dawn_orbit();
/// // Dawn arrives at Mars on a hyperbolic flyby trajectory.
/// assert!(oe.e_mag > 1.0);
/// ```
pub fn mars_dawn_orbit() -> OrbitalElements<Mars> {
    from_pos_vel_with_mu::<Mars>(
        glam::DVec3::new(11_563_355.680_2, -14_356_668.897_7, 6_293_704.616_9),
        glam::DVec3::new(-2_273.107_8, 2_380.132_4, -22.911),
        mu_mars(),
    )
}

/// Apollo CSM parking orbit: 185 km circular at 32.5° inclination.
///
/// Used by Earth-orbit Apollo tests as the starting state before the
/// trans-lunar burn.
///
/// ```
/// use astrodyn::recipes::orbital_elements;
/// let oe = orbital_elements::apollo_parking();
/// assert!((oe.inclination - 32.5_f64.to_radians()).abs() < 1e-12);
/// assert!(oe.e_mag < 1e-6);
/// ```
pub fn apollo_parking() -> OrbitalElements<Earth> {
    let r_eq = 6_378_137.0_f64;
    circular_orbit_with_mu::<Earth>(r_eq + 185_000.0, 32.5_f64.to_radians(), mu_ggm05c())
}

/// Geostationary radius (42164 km) at non-zero inclination.
///
/// Specifically `geo_inclined(0.rad())` matches
/// [`geostationary`]. Pass the desired inclination
/// (e.g. `0.1.rad()` for an inclined-GEO study) and the function
/// returns the orbital elements of a circular orbit at that
/// inclination, with the line of nodes along the +x axis.
///
/// ```
/// use astrodyn_quantities::ext::F64Ext;
/// use astrodyn::recipes::orbital_elements;
/// let oe = orbital_elements::geo_inclined(0.1_f64.rad());
/// assert!((oe.inclination - 0.1).abs() < 1e-12);
/// assert!((oe.semi_major_axis - 42_164_172.0).abs() < 1.0);
/// ```
pub fn geo_inclined(inclination: uom::si::f64::Angle) -> OrbitalElements<Earth> {
    use uom::si::angle::radian;
    circular_orbit_with_mu::<Earth>(42_164_172.0_f64, inclination.get::<radian>(), mu_ggm05c())
}

#[cfg(test)]
mod tests {
    use super::*;
    use astrodyn_quantities::ext::F64Ext;

    #[test]
    fn mercury_perihelion_is_heliocentric_and_eccentric() {
        let oe = mercury_perihelion();
        // Mercury's semi-major axis is ~5.79e10 m.
        assert!(oe.semi_major_axis > 5.5e10 && oe.semi_major_axis < 6.1e10);
        // Eccentricity around 0.2 — definitely non-circular.
        assert!(oe.e_mag > 0.1 && oe.e_mag < 0.3);
    }

    #[test]
    fn mars_dawn_orbit_hyperbolic_at_arrival() {
        let oe = mars_dawn_orbit();
        // Dawn arrives at Mars on a hyperbolic flyby trajectory:
        // semi-major axis < 0, eccentricity > 1.
        assert!(oe.semi_major_axis < 0.0);
        assert!(oe.e_mag > 1.0);
    }

    #[test]
    fn apollo_parking_low_earth_orbit() {
        let oe = apollo_parking();
        // ~6563 km radius, near-circular.
        assert!((oe.semi_major_axis - 6_563_137.0).abs() < 1.0);
        assert!(oe.e_mag.abs() < 1e-10);
        // Inclination 32.5° = 0.5672 rad.
        assert!((oe.inclination - 32.5_f64.to_radians()).abs() < 1e-12);
    }

    #[test]
    fn geo_inclined_zero_matches_geostationary() {
        let g0 = geo_inclined(0.0_f64.rad());
        let g = geostationary();
        assert!((g0.semi_major_axis - g.semi_major_axis).abs() < 1.0);
        assert!(g0.inclination.abs() < 1e-12);
    }

    #[test]
    fn geo_inclined_nonzero_carries_inclination() {
        use uom::si::angle::radian;
        let g = geo_inclined(uom::si::f64::Angle::new::<radian>(0.1));
        assert!((g.inclination - 0.1).abs() < 1e-12);
        assert!(g.semi_major_axis > 4.2e7 && g.semi_major_axis < 4.3e7);
    }
}
