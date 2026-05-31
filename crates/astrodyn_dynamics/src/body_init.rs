// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Body initialization functions for translational state.
//!
//! Port of JEOD `DynBodyInitOrbit`, `DynBodyInitLvlh`, and NED initialization
//! from `models/dynamics/body_action/src/`.
//!
//! These functions initialize a vehicle's translational state from various
//! parameterizations: Keplerian orbital elements, LVLH-relative state, or
//! NED (North-East-Down) relative state.

use crate::rotational::RotationalState;
use crate::state::{TranslationalState, TranslationalStateTyped};
use astrodyn_math::{mat3_from_rows, GeodeticState, JeodQuat, OrbitalElements};
use astrodyn_quantities::aliases::{Position, Velocity};
use astrodyn_quantities::dims::GravParam;
use astrodyn_quantities::ext::Vec3Ext;
use astrodyn_quantities::frame::RootInertial;
use glam::{DMat3, DVec3};
use uom::si::angle::radian;
use uom::si::f64::{Angle, Length};
use uom::si::length::meter;

/// Initialize translational state from Keplerian orbital elements (true anomaly).
///
/// Port of JEOD `DynBodyInitOrbit::apply()` from `dyn_body_init_orbit.cc`,
/// for the `SmaEccIncAscnodeArgperTanom` element set.
///
/// # Arguments
/// * `semi_major_axis` - Semi-major axis (m)
/// * `eccentricity` - Orbital eccentricity
/// * `inclination` - Inclination (rad)
/// * `raan` - Right ascension of ascending node (rad)
/// * `arg_periapsis` - Argument of periapsis (rad)
/// * `true_anomaly` - True anomaly (rad)
/// * `mu` - Gravitational parameter of central body (m^3/s^2)
pub fn init_from_orbital_elements(
    semi_major_axis: f64,
    eccentricity: f64,
    inclination: f64,
    raan: f64,
    arg_periapsis: f64,
    true_anomaly: f64,
    mu: f64,
) -> TranslationalState {
    // JEOD_INV: BA.05 — orbit initializer requires a valid gravity source (mu > 0)
    // JEOD dyn_body_init_orbit.cc:101-111: validate mu before use.
    assert!(
        mu > 0.0,
        "init_from_orbital_elements: mu must be positive, got {mu}"
    );
    assert!(
        semi_major_axis.is_finite(),
        "init_from_orbital_elements: semi_major_axis must be finite, got {semi_major_axis}"
    );
    assert!(
        (0.0..1.0).contains(&eccentricity),
        "init_from_orbital_elements: eccentricity must be in [0, 1), got {eccentricity}"
    );

    // Build OrbitalElements with the provided Keplerian elements.
    // Following JEOD dyn_body_init_orbit.cc: populate semiparam, angles, true_anom,
    // then call nu_to_anomalies() and to_cartesian().
    use astrodyn_quantities::frame::SelfPlanet;
    let mut oe = OrbitalElements::<SelfPlanet>::default();
    oe.semi_major_axis = semi_major_axis;
    oe.e_mag = eccentricity;
    oe.inclination = inclination;
    oe.long_asc_node = raan;
    oe.arg_periapsis = arg_periapsis;
    oe.semiparam = semi_major_axis * (1.0 - eccentricity * eccentricity);
    oe.true_anom = true_anomaly;
    oe.nu_to_anomalies();

    let (position, velocity) = oe
        .to_cartesian(mu)
        .expect("init_from_orbital_elements: to_cartesian failed");

    TranslationalState { position, velocity }
}

/// Initialize translational state from Keplerian orbital elements using the
/// semi-latus rectum (rather than semi-major axis) plus true anomaly.
///
/// Port of the `SlrEccIncAscnodeArgperTanom` branch of JEOD
/// `DynBodyInitOrbit::apply()` from
/// `models/dynamics/body_action/src/dyn_body_init_orbit.cc:196-200, 285-321`.
///
/// JEOD selects `shape = ShapeSemiLatusRectum`, which **skips** the
/// `semi_latus_rectum = semi_major_axis * (1 - e²)` derivation (that block
/// runs only for `ShapeSemiMajorAxis`). The deck-supplied semi-latus rectum
/// is therefore used verbatim as `elem.semiparam`. To match JEOD bit-for-bit
/// we set `semiparam = semi_latus_rectum` directly here — routing through
/// `init_from_orbital_elements` (which recomputes `semiparam = a·(1-e²)` from
/// `a = p/(1-e²)`) would introduce a round-trip that JEOD never performs.
///
/// # Arguments
/// * `semi_latus_rectum` - Semi-latus rectum p (m)
/// * `eccentricity` - Orbital eccentricity
/// * `inclination` - Inclination (rad)
/// * `raan` - Right ascension of ascending node (rad)
/// * `arg_periapsis` - Argument of periapsis (rad)
/// * `true_anomaly` - True anomaly (rad)
/// * `mu` - Gravitational parameter of central body (m^3/s^2)
pub fn init_from_semi_latus_rectum_true_anomaly(
    semi_latus_rectum: f64,
    eccentricity: f64,
    inclination: f64,
    raan: f64,
    arg_periapsis: f64,
    true_anomaly: f64,
    mu: f64,
) -> TranslationalState {
    // JEOD_INV: BA.05 — orbit initializer requires a valid gravity source (mu > 0)
    // JEOD dyn_body_init_orbit.cc:98-111: validate mu before use.
    assert!(
        mu > 0.0,
        "init_from_semi_latus_rectum_true_anomaly: mu must be positive, got {mu}"
    );
    assert!(
        semi_latus_rectum > 0.0 && semi_latus_rectum.is_finite(),
        "init_from_semi_latus_rectum_true_anomaly: semi_latus_rectum must be positive and finite, \
         got {semi_latus_rectum}"
    );
    assert!(
        (0.0..1.0).contains(&eccentricity),
        "init_from_semi_latus_rectum_true_anomaly: eccentricity must be in [0, 1), \
         got {eccentricity}"
    );

    // JEOD dyn_body_init_orbit.cc: ShapeSemiLatusRectum leaves semiparam as
    // the deck value, then sets the angles, true_anom, and calls
    // nu_to_anomalies() followed by to_cartesian().
    use astrodyn_quantities::frame::SelfPlanet;
    let mut oe = OrbitalElements::<SelfPlanet>::default();
    oe.semiparam = semi_latus_rectum;
    oe.e_mag = eccentricity;
    oe.inclination = inclination;
    oe.long_asc_node = raan;
    oe.arg_periapsis = arg_periapsis;
    oe.true_anom = true_anomaly;
    oe.nu_to_anomalies();

    let (position, velocity) = oe
        .to_cartesian(mu)
        .expect("init_from_semi_latus_rectum_true_anomaly: to_cartesian failed");

    TranslationalState { position, velocity }
}

/// Typed sibling of [`init_from_orbital_elements`].
///
/// Returns a [`TranslationalStateTyped<RootInertial>`] — Phase 3 callers
/// can pipe the result directly into typed propagation paths without
/// hand-wrapping with `from_untyped_unchecked`. Numerically
/// bit-identical to the untyped variant: the typed entry unwraps
/// inputs to f64 base SI, calls the existing implementation, and
/// re-wraps the output.
///
/// Generic over `P: Planet` so `mu` carries its source-body identity;
/// the planet phantom is consumed at this boundary and the f64 kernel
/// runs unchanged.
pub fn init_from_orbital_elements_typed<P: astrodyn_quantities::frame::Planet>(
    semi_major_axis: Length,
    eccentricity: f64,
    inclination: Angle,
    raan: Angle,
    arg_periapsis: Angle,
    true_anomaly: Angle,
    mu: GravParam<P>,
) -> TranslationalStateTyped<RootInertial> {
    let untyped = init_from_orbital_elements(
        semi_major_axis.get::<meter>(),
        eccentricity,
        inclination.get::<radian>(),
        raan.get::<radian>(),
        arg_periapsis.get::<radian>(),
        true_anomaly.get::<radian>(),
        mu.value, // base SI: m³/s²
    );
    // allowed: typed↔raw kernel boundary
    TranslationalStateTyped::<RootInertial> {
        position: Position::<RootInertial>::from_raw_si(untyped.position),
        velocity: Velocity::<RootInertial>::from_raw_si(untyped.velocity),
    }
}

/// Initialize translational state from Keplerian orbital elements (mean anomaly).
///
/// Port of JEOD `DynBodyInitOrbit::apply()` from `dyn_body_init_orbit.cc`,
/// for the `SmaEccIncAscnodeArgperManom` element set.
///
/// Solves Kepler's equation internally to convert mean anomaly to true anomaly.
///
/// # Arguments
/// * `semi_major_axis` - Semi-major axis (m)
/// * `eccentricity` - Orbital eccentricity
/// * `inclination` - Inclination (rad)
/// * `raan` - Right ascension of ascending node (rad)
/// * `arg_periapsis` - Argument of periapsis (rad)
/// * `mean_anomaly` - Mean anomaly (rad)
/// * `mu` - Gravitational parameter of central body (m^3/s^2)
pub fn init_from_mean_anomaly(
    semi_major_axis: f64,
    eccentricity: f64,
    inclination: f64,
    raan: f64,
    arg_periapsis: f64,
    mean_anomaly: f64,
    mu: f64,
) -> TranslationalState {
    // JEOD_INV: BA.05 — orbit initializer requires a valid gravity source (mu > 0)
    // JEOD dyn_body_init_orbit.cc:101-111: validate mu before use.
    assert!(
        mu > 0.0,
        "init_from_mean_anomaly: mu must be positive, got {mu}"
    );
    assert!(
        semi_major_axis.is_finite(),
        "init_from_mean_anomaly: semi_major_axis must be finite, got {semi_major_axis}"
    );
    assert!(
        (0.0..1.0).contains(&eccentricity),
        "init_from_mean_anomaly: eccentricity must be in [0, 1), got {eccentricity}"
    );

    // Following JEOD dyn_body_init_orbit.cc lines 302-318:
    // Populate elem with semiparam, e_mag, inclination, arg_periapsis, long_asc_node,
    // set mean_anom, then call mean_anom_to_nu() to solve Kepler's equation.
    use astrodyn_quantities::frame::SelfPlanet;
    let mut oe = OrbitalElements::<SelfPlanet>::default();
    oe.semi_major_axis = semi_major_axis;
    oe.e_mag = eccentricity;
    oe.inclination = inclination;
    oe.long_asc_node = raan;
    oe.arg_periapsis = arg_periapsis;
    oe.semiparam = semi_major_axis * (1.0 - eccentricity * eccentricity);
    oe.mean_anom = mean_anomaly;
    oe.mean_anom_to_nu()
        .expect("init_from_mean_anomaly: Kepler solver failed");

    let (position, velocity) = oe
        .to_cartesian(mu)
        .expect("init_from_mean_anomaly: to_cartesian failed");

    TranslationalState { position, velocity }
}

/// Initialize translational state from Keplerian orbital elements with the
/// `SmaEccIncAscnodeArgperTimeperi` element set (time since periapsis).
///
/// Port of the `LocationTimePeri` branch of JEOD `DynBodyInitOrbit::apply()`
/// at `models/dynamics/body_action/src/dyn_body_init_orbit.cc:293-295`:
///
/// ```cpp
/// if (location == LocationTimePeri) {
///     mean_anomaly = time_periapsis * std::sqrt(planet->grav_source->mu / semi_major_axis) / semi_major_axis;
/// }
/// ```
///
/// Converts `time_periapsis` (seconds elapsed since periapsis passage) to
/// mean anomaly via `M = n · t_peri` where `n = sqrt(mu / a^3)`, then defers
/// to [`init_from_mean_anomaly`].
///
/// # Arguments
/// * `semi_major_axis` - Semi-major axis (m)
/// * `eccentricity` - Orbital eccentricity
/// * `inclination` - Inclination (rad)
/// * `raan` - Right ascension of ascending node (rad)
/// * `arg_periapsis` - Argument of periapsis (rad)
/// * `time_periapsis` - Time elapsed **since** periapsis passage (s). JEOD
///   convention: positive when t > t_peri (i.e., after periapsis).
/// * `mu` - Gravitational parameter of central body (m^3/s^2)
pub fn init_from_time_periapsis(
    semi_major_axis: f64,
    eccentricity: f64,
    inclination: f64,
    raan: f64,
    arg_periapsis: f64,
    time_periapsis: f64,
    mu: f64,
) -> TranslationalState {
    assert!(
        mu > 0.0,
        "init_from_time_periapsis: mu must be positive, got {mu}"
    );
    assert!(
        semi_major_axis > 0.0 && semi_major_axis.is_finite(),
        "init_from_time_periapsis: semi_major_axis must be positive and finite, got {semi_major_axis}"
    );

    // JEOD dyn_body_init_orbit.cc:295 factorization:
    //   mean_anomaly = t_peri * sqrt(mu / a) / a
    // Algebraically equivalent to M = n*t with n = sqrt(mu/a^3), but matches
    // JEOD's arithmetic order to minimize rounding differences in parity tests.
    let mean_anomaly = time_periapsis * (mu / semi_major_axis).sqrt() / semi_major_axis;

    init_from_mean_anomaly(
        semi_major_axis,
        eccentricity,
        inclination,
        raan,
        arg_periapsis,
        mean_anomaly,
        mu,
    )
}

/// Derive semi-major axis and eccentricity from apoapsis/periapsis altitudes,
/// following JEOD's `ShapeAltitudes` branch of `DynBodyInitOrbit::apply()`
/// (`models/dynamics/body_action/src/dyn_body_init_orbit.cc:277-283`):
///
/// ```cpp
/// if (shape == ShapeAltitudes) {
///     semi_major_axis = planet->r_eq + 0.5 * (alt_apoapsis + alt_periapsis);
///     eccentricity = (alt_apoapsis - alt_periapsis) / (2.0 * semi_major_axis);
/// }
/// ```
///
/// `r_eq` is the planet's **equatorial** radius (`Planet::r_eq`), to which both
/// altitudes are referenced. The arithmetic order is preserved verbatim so the
/// f64 bit-pattern matches JEOD's for parity tests.
///
/// # Arguments
/// * `r_eq` - Planet equatorial radius (m). For Earth this is JEOD's
///   `1000 * 6378.137 = 6_378_137.0` m (`environment/planet/data/src/earth.cc`).
/// * `alt_apoapsis` - Apoapsis altitude above `r_eq` (m).
/// * `alt_periapsis` - Periapsis altitude above `r_eq` (m).
fn sma_ecc_from_altitudes(r_eq: f64, alt_apoapsis: f64, alt_periapsis: f64) -> (f64, f64) {
    // JEOD_INV: BA.13 — altitude shape: a = r_eq + ½(alt_apo + alt_peri),
    // e = (alt_apo − alt_peri) / (2a), with `r_eq` the planet equatorial radius.
    let semi_major_axis = r_eq + 0.5 * (alt_apoapsis + alt_periapsis);
    // Guard the altitude-derived semi-major axis locally so a mis-specified deck
    // (e.g. altitudes that drive `a` non-positive) fails with an altitude-aware
    // diagnostic here, rather than reaching `to_cartesian` via a delegating
    // converter that only checks `is_finite`. The eccentricity range is validated
    // downstream by the converter we delegate to.
    assert!(
        semi_major_axis > 0.0 && semi_major_axis.is_finite(),
        "sma_ecc_from_altitudes: derived semi_major_axis must be positive and finite, \
         got {semi_major_axis} from r_eq={r_eq}, alt_apoapsis={alt_apoapsis}, \
         alt_periapsis={alt_periapsis}; check the apo/peri altitudes in the deck"
    );
    let eccentricity = (alt_apoapsis - alt_periapsis) / (2.0 * semi_major_axis);
    (semi_major_axis, eccentricity)
}

/// Initialize translational state from the JEOD
/// `IncAscnodeAltperAltapoArgperTanom` element set (set #04): inclination,
/// ascending node, peri/apo **altitudes**, argument of periapsis, and **true
/// anomaly**.
///
/// Port of the `ShapeAltitudes` + `LocationTrueAnom` path of JEOD
/// `DynBodyInitOrbit::apply()`
/// (`models/dynamics/body_action/src/dyn_body_init_orbit.cc:205-208, 277-318`):
/// the altitudes are converted to semi-major axis + eccentricity via
/// `sma_ecc_from_altitudes`, then JEOD sets `shape = ShapeSemiMajorAxis`
/// (so `semiparam = a·(1-e²)`) and resolves the true anomaly directly. This is
/// therefore exactly [`init_from_orbital_elements`] with the derived a/e.
///
/// # Arguments
/// * `r_eq` - Planet equatorial radius (m).
/// * `alt_apoapsis` - Apoapsis altitude above `r_eq` (m).
/// * `alt_periapsis` - Periapsis altitude above `r_eq` (m).
/// * `inclination` - Inclination (rad)
/// * `raan` - Right ascension of ascending node (rad)
/// * `arg_periapsis` - Argument of periapsis (rad)
/// * `true_anomaly` - True anomaly (rad)
/// * `mu` - Gravitational parameter of central body (m^3/s^2)
#[expect(
    clippy::too_many_arguments,
    reason = "JEOD orbital-element set is six elements plus r_eq and mu"
)]
pub fn init_from_altitudes_true_anomaly(
    r_eq: f64,
    alt_apoapsis: f64,
    alt_periapsis: f64,
    inclination: f64,
    raan: f64,
    arg_periapsis: f64,
    true_anomaly: f64,
    mu: f64,
) -> TranslationalState {
    let (semi_major_axis, eccentricity) = sma_ecc_from_altitudes(r_eq, alt_apoapsis, alt_periapsis);
    init_from_orbital_elements(
        semi_major_axis,
        eccentricity,
        inclination,
        raan,
        arg_periapsis,
        true_anomaly,
        mu,
    )
}

/// Initialize translational state from the JEOD
/// `IncAscnodeAltperAltapoArgperTimeperi` element set (set #05): inclination,
/// ascending node, peri/apo **altitudes**, argument of periapsis, and **time
/// since periapsis passage**.
///
/// Port of the `ShapeAltitudes` + `LocationTimePeri` path of JEOD
/// `DynBodyInitOrbit::apply()`
/// (`models/dynamics/body_action/src/dyn_body_init_orbit.cc:213-216, 277-318`):
/// the altitudes are converted to semi-major axis + eccentricity via
/// `sma_ecc_from_altitudes`, then the time-since-periapsis is mapped to mean
/// anomaly (`M = t_peri·√(μ/a)/a`) exactly as in [`init_from_time_periapsis`],
/// which this function delegates to with the derived a/e.
///
/// # Arguments
/// * `r_eq` - Planet equatorial radius (m).
/// * `alt_apoapsis` - Apoapsis altitude above `r_eq` (m).
/// * `alt_periapsis` - Periapsis altitude above `r_eq` (m).
/// * `inclination` - Inclination (rad)
/// * `raan` - Right ascension of ascending node (rad)
/// * `arg_periapsis` - Argument of periapsis (rad)
/// * `time_periapsis` - Time elapsed since periapsis passage (s)
/// * `mu` - Gravitational parameter of central body (m^3/s^2)
#[expect(
    clippy::too_many_arguments,
    reason = "JEOD orbital-element set is six elements plus r_eq and mu"
)]
pub fn init_from_altitudes_time_periapsis(
    r_eq: f64,
    alt_apoapsis: f64,
    alt_periapsis: f64,
    inclination: f64,
    raan: f64,
    arg_periapsis: f64,
    time_periapsis: f64,
    mu: f64,
) -> TranslationalState {
    let (semi_major_axis, eccentricity) = sma_ecc_from_altitudes(r_eq, alt_apoapsis, alt_periapsis);
    init_from_time_periapsis(
        semi_major_axis,
        eccentricity,
        inclination,
        raan,
        arg_periapsis,
        time_periapsis,
        mu,
    )
}

/// Initialize translational state from the JEOD `SmaIncAscnodeArglatRadRadvel`
/// element set (set #06): semi-major axis, inclination, ascending node,
/// **argument of latitude**, **orbital radius**, and **radial velocity**.
///
/// Port of the `SmaIncAscnodeArglatRadRadvel` branch of JEOD
/// `DynBodyInitOrbit::apply()`
/// (`models/dynamics/body_action/src/dyn_body_init_orbit.cc:221-261, 286-321`).
/// JEOD recovers the eccentricity, true anomaly, and argument of periapsis from
/// the radius / radial-velocity pair via the eccentric-anomaly identities:
///
/// ```cpp
/// ecosE = (semi_major_axis - orb_radius) / semi_major_axis;
/// esinE = (radial_vel * orb_radius) / sqrt(mu * semi_major_axis);
/// ecc_sq = ecosE*ecosE + esinE*esinE;
/// eccentricity = sqrt(ecc_sq);
/// if (eccentricity >= 1.0e-14) {
///     kcost = ecosE - ecc_sq;
///     ksint = sqrt(1.0 - ecc_sq) * esinE;
///     true_anomaly = atan2(ksint, kcost);
/// } else {
///     true_anomaly = 0.0;
/// }
/// arg_periapsis = arg_latitude - true_anomaly;
/// ```
///
/// JEOD then selects `shape = ShapeSemiMajorAxis` and `location =
/// LocationTrueAnom`, so the orbit resolves exactly as
/// [`init_from_orbital_elements`] with the derived `(e, ω, ν)`. The arithmetic
/// order above is preserved verbatim so the f64 bit-pattern matches JEOD's for
/// parity tests.
///
/// # Arguments
/// * `semi_major_axis` - Semi-major axis (m)
/// * `inclination` - Inclination (rad)
/// * `raan` - Right ascension of ascending node (rad)
/// * `arg_latitude` - Argument of latitude `ω + ν` (rad)
/// * `orb_radius` - Orbital radius, distance from planet centre (m)
/// * `radial_vel` - Radial component of velocity `dr/dt` (m/s)
/// * `mu` - Gravitational parameter of central body (m^3/s^2)
pub fn init_from_arg_latitude_radial_vel(
    semi_major_axis: f64,
    inclination: f64,
    raan: f64,
    arg_latitude: f64,
    orb_radius: f64,
    radial_vel: f64,
    mu: f64,
) -> TranslationalState {
    // JEOD_INV: BA.05 — orbit initializer requires a valid gravity source (mu > 0)
    // JEOD dyn_body_init_orbit.cc:98-111: validate mu before use.
    assert!(
        mu > 0.0,
        "init_from_arg_latitude_radial_vel: mu must be positive, got {mu}"
    );
    assert!(
        semi_major_axis > 0.0 && semi_major_axis.is_finite(),
        "init_from_arg_latitude_radial_vel: semi_major_axis must be positive and finite, \
         got {semi_major_axis}"
    );
    assert!(
        orb_radius > 0.0 && orb_radius.is_finite(),
        "init_from_arg_latitude_radial_vel: orb_radius must be positive and finite, \
         got {orb_radius}"
    );
    assert!(
        radial_vel.is_finite(),
        "init_from_arg_latitude_radial_vel: radial_vel must be finite, got {radial_vel}"
    );

    // JEOD_INV: BA.14 — set06 (SmaIncAscnodeArglatRadRadvel) derives (e, ν, ω)
    // from the radius / radial-velocity pair via the eccentric-anomaly
    // identities, then resolves the orbit as the sma + true-anomaly shape.
    // Arithmetic order matches dyn_body_init_orbit.cc:227-249 verbatim.
    let ecos_e = (semi_major_axis - orb_radius) / semi_major_axis;
    let esin_e = (radial_vel * orb_radius) / (mu * semi_major_axis).sqrt();
    let ecc_sq = ecos_e * ecos_e + esin_e * esin_e;
    let eccentricity = ecc_sq.sqrt();

    let true_anomaly = if eccentricity >= 1.0e-14 {
        let kcost = ecos_e - ecc_sq;
        let ksint = (1.0 - ecc_sq).sqrt() * esin_e;
        ksint.atan2(kcost)
    } else {
        // Circular orbit: JEOD sets the true anomaly to zero.
        0.0
    };

    let arg_periapsis = arg_latitude - true_anomaly;

    init_from_orbital_elements(
        semi_major_axis,
        eccentricity,
        inclination,
        raan,
        arg_periapsis,
        true_anomaly,
        mu,
    )
}

/// Initialize translational state from LVLH-relative position and velocity.
///
/// Computes the LVLH frame from a reference orbit state, then transforms the
/// given LVLH-relative offsets into the inertial frame.
///
/// # Arguments
/// * `lvlh_pos` - Position relative to reference in LVLH frame (m)
/// * `lvlh_vel` - Velocity relative to reference in LVLH frame (m/s)
/// * `ref_position` - Reference orbit position in inertial frame (m)
/// * `ref_velocity` - Reference orbit velocity in inertial frame (m/s)
pub fn init_from_lvlh(
    lvlh_pos: DVec3,
    lvlh_vel: DVec3,
    ref_position: DVec3,
    ref_velocity: DVec3,
) -> TranslationalState {
    // Typed entry: lift inertial inputs and use `LvlhFrame::compute`,
    // which returns the full struct (orientation + angular velocity +
    // origin state). LVLH is computed in the central body's
    // planet-inertial frame. Earth here is the documented assumption;
    // non-Earth init paths use their own constructors.
    use astrodyn_quantities::frame::{Earth, PlanetInertial};
    let lvlh = astrodyn_math::LvlhFrame::compute(
        ref_position.m_at::<PlanetInertial<Earth>>(),
        ref_velocity.m_per_s_at::<PlanetInertial<Earth>>(),
    );

    // The LVLH frame B sits as a child of the inertial frame A, co-located
    // and co-moving with the reference orbit, oriented by `t_parent_this`
    // and rotating at `ang_vel_this` (the orbital rate). Compose the user
    // offset S_B:C (chaser relative to LVLH) up to inertial via JEOD
    // `RefFrameState::incr_left` (A = inertial, B = LVLH, C = chaser).
    // The earlier translation-only port dropped the ω×r velocity term,
    // which is negligible only for a non-rotating reference frame; for an
    // orbiting target's LVLH it is required and is what the
    // vehicle-relative RUNs exercise.
    let frame = lvlh_reference_frame_state(
        lvlh.t_parent_this,
        lvlh.ang_vel_this,
        ref_position,
        ref_velocity,
    );
    init_trans_relative_to_frame(&frame, lvlh_pos, lvlh_vel)
}

/// Build an [`astrodyn_frames::RefFrameState`] for a reference frame `B`
/// that is a child of the inertial frame, from its inertial origin
/// state, parent→B orientation, and B-frame angular velocity wrt the
/// inertial frame. The attitude quaternion cache is derived from
/// `t_parent_this` (JEOD `Q_parent_this` is canonical).
fn lvlh_reference_frame_state(
    t_parent_this: DMat3,
    ang_vel_this: DVec3,
    origin_position: DVec3,
    origin_velocity: DVec3,
) -> astrodyn_frames::RefFrameState {
    astrodyn_frames::RefFrameState {
        trans: astrodyn_frames::RefFrameTrans {
            position: origin_position,
            velocity: origin_velocity,
        },
        rot: astrodyn_frames::RefFrameRot {
            q_parent_this: JeodQuat::left_quat_from_transformation(&t_parent_this),
            t_parent_this,
            ang_vel_this,
        },
    }
}

/// Initialize a subject vehicle's translational state from an offset
/// expressed in a reference frame `B` that is a child of the inertial
/// integration frame, composing through JEOD `RefFrameState::incr_left`
/// (the canonical port lives in
/// [`astrodyn_frames::RefFrameState::incr_left`]). Used for the
/// body-relative and vehicle-LVLH / vehicle-NED translation-only inits,
/// where `frame` carries the reference frame's inertial state and
/// `offset_*` the user offset. The composition includes the reference
/// frame's `ω × r` velocity term.
///
/// # Arguments
/// * `frame` - Reference frame B's state wrt inertial (origin
///   position/velocity in inertial coordinates, `T_inertial_B`,
///   `ω_inertial_B` in B).
/// * `offset_position` - Subject position offset in frame B (m).
/// * `offset_velocity` - Subject velocity offset in frame B (m/s).
// JEOD_INV: BA.15 — relative-state init composes the reference-frame
// state with the user offset via incr_left, including the ω_A:B × x_B:C
// frame-rate term in the velocity.
pub fn init_trans_relative_to_frame(
    frame: &astrodyn_frames::RefFrameState,
    offset_position: DVec3,
    offset_velocity: DVec3,
) -> TranslationalState {
    // S_B:C — the subject's offset wrt B, with identity relative
    // orientation / zero relative rate (translation-only init).
    let mut composed = astrodyn_frames::RefFrameState {
        trans: astrodyn_frames::RefFrameTrans {
            position: offset_position,
            velocity: offset_velocity,
        },
        rot: astrodyn_frames::RefFrameRot::default(),
    };
    composed.incr_left(frame);
    TranslationalState {
        position: composed.trans.position,
        velocity: composed.trans.velocity,
    }
}

/// Initialize a subject vehicle's full rotational state from an attitude
/// and angular velocity expressed relative to a reference frame `B` that
/// is a child of the inertial integration frame, via JEOD
/// `RefFrameState::incr_left`.
///
/// `q_frame_subject` is the user-supplied B→subject attitude (scalar-first
/// left-transformation, JEOD convention) and `ang_vel_frame_to_subject`
/// the angular velocity of the subject wrt B, expressed in the subject
/// body frame (`ang_vel_this`, JEOD `rate_in_parent = false`). The
/// reference frame's own inertial→B attitude and angular velocity come
/// from `frame`. The returned [`RotationalState`] is the subject's
/// inertial→body attitude and body-frame angular velocity wrt inertial:
/// `Q_A:C = Q_B:C · Q_A:B`, `w_A:C = T_B:C · w_A:B + w_B:C`.
// JEOD_INV: BA.15 — full relative-state init composes attitude and rate
// via incr_left.
pub fn init_rot_relative_to_frame(
    frame: &astrodyn_frames::RefFrameState,
    q_frame_subject: JeodQuat,
    ang_vel_frame_to_subject: DVec3,
) -> RotationalState {
    let mut q_frame_subject = q_frame_subject;
    q_frame_subject.normalize();

    // S_B:C — subject wrt B: user attitude + body-frame relative rate.
    let mut composed = astrodyn_frames::RefFrameState {
        trans: astrodyn_frames::RefFrameTrans::default(),
        rot: astrodyn_frames::RefFrameRot {
            q_parent_this: q_frame_subject,
            t_parent_this: q_frame_subject.left_quat_to_transformation(),
            ang_vel_this: ang_vel_frame_to_subject,
        },
    };
    composed.incr_left(frame);

    RotationalState {
        quaternion: composed.rot.q_parent_this,
        ang_vel_body: composed.rot.ang_vel_this,
    }
}

/// Frame in which the user-supplied LVLH-relative angular velocity is expressed.
///
/// JEOD `DynBodyInit::ang_velocity` is interpreted via two flags
/// (`reverse_sense`, `rate_in_parent`); for `DynBodyInitLvlhRotState` the
/// only sane combinations reduce to "ang vel of body wrt LVLH, expressed
/// in body frame" (`rate_in_parent = false`) or "...expressed in LVLH"
/// (`rate_in_parent = true`). The `reverse_sense` flag is irrelevant to
/// the LVLH-rot init in JEOD's own verif tests; we expose only the
/// `rate_in_parent` choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LvlhAngularVelocityFrame {
    /// User input is the angular velocity of the body wrt the LVLH frame,
    /// expressed in the body frame (rad/s). Default — matches JEOD
    /// `rate_in_parent = false`.
    #[default]
    Body,
    /// User input is the angular velocity of the body wrt the LVLH frame,
    /// expressed in the LVLH frame (rad/s). JEOD `rate_in_parent = true`.
    Lvlh,
}

/// Initialize rotational state from an LVLH-relative attitude + angular velocity.
///
/// Port of JEOD `DynBodyInitLvlhRotState::initialize` /
/// `DynBodyInitLvlhState::apply` /
/// `DynBodyInit::apply_user_inputs` for the rotational sub-state
/// (`set_items = Att | Rate`). See
/// `models/dynamics/body_action/src/dyn_body_init_lvlh_rot_state.cc`,
/// `dyn_body_init_lvlh_state.cc`, and the rotational branches of
/// `dyn_body_init.cc:273-328`.
///
/// The LVLH frame is constructed from the reference orbit
/// (`ref_position`, `ref_velocity`) in the central body's planet-inertial
/// frame, then composed with the user-supplied LVLH→body attitude /
/// LVLH-relative angular velocity to produce the body's
/// inertial→body attitude and the body-frame angular velocity of the body
/// wrt the inertial frame.
///
/// The composition follows JEOD's `RefFrameState::incr_left` (with `A` =
/// inertial, `B` = LVLH, `C` = body):
///
/// ```text
/// Q_inertial_body         = Q_lvlh_body * Q_inertial_lvlh
/// w_inertial_body_in_body = T_lvlh_body * w_inertial_lvlh_in_lvlh
///                         + w_lvlh_body_in_body
/// ```
///
/// where `Q_lvlh_body` is the user-supplied attitude (LVLH-frame attitude
/// of the body) — composed with `Q_inertial_lvlh` via quaternion
/// multiplication — and `T_lvlh_body` is the equivalent rotation matrix
/// derived from `Q_lvlh_body`, used to project the LVLH-frame angular
/// velocity into the body frame. `w_lvlh_body_in_body` is the
/// user-supplied LVLH-relative angular velocity already expressed in the
/// body frame (when the user supplies it in the LVLH frame, the same
/// `T_lvlh_body` matrix lifts it into the body frame first — see
/// `ang_vel_frame` below).
///
/// # Arguments
/// * `q_lvlh_body` - LVLH→body attitude quaternion (scalar-first,
///   left-transformation, JEOD convention). Renormalized once at function
///   entry; both the returned attitude and the angular-velocity
///   composition use the renormalized form so a slightly-off-unit input
///   cannot produce a returned attitude / ang-vel pair that disagree on
///   the body axes.
/// * `ang_vel_lvlh_to_body` - Angular velocity of the body wrt the LVLH
///   frame (rad/s), expressed per `ang_vel_frame`.
/// * `ang_vel_frame` - Coordinate frame of `ang_vel_lvlh_to_body`.
/// * `ref_position` - Reference orbit position in the planet-inertial
///   frame (m).
/// * `ref_velocity` - Reference orbit velocity in the planet-inertial
///   frame (m/s).
// JEOD_INV: BA.11 — LVLH-rot init composes inertial→LVLH (from reference
// orbit) with LVLH→body (user input) per RefFrameState::incr_left, then
// projects the LVLH-frame ang vel into the body frame and adds the
// LVLH-body ang vel.
pub fn init_rot_from_lvlh(
    q_lvlh_body: JeodQuat,
    ang_vel_lvlh_to_body: DVec3,
    ang_vel_frame: LvlhAngularVelocityFrame,
    ref_position: DVec3,
    ref_velocity: DVec3,
) -> RotationalState {
    // Renormalize the user-supplied LVLH→body attitude once at the entry
    // and use the renormalized value as the canonical input for both the
    // returned attitude and the `T_lvlh_body` matrix that drives the
    // angular-velocity composition. If the caller supplies a
    // slightly-off-unit quaternion (which this function explicitly
    // tolerates), `left_quat_to_transformation()` would otherwise produce
    // a scaled / skewed matrix and the returned `ang_vel_body` would no
    // longer match the rotation defined by the returned attitude — the
    // attitude and ang-vel outputs must describe the same body axes.
    let mut q_lvlh_body = q_lvlh_body;
    q_lvlh_body.normalize();

    // Reference-orbit LVLH frame (typed input, raw f64 inputs at the
    // boundary). Earth here is the documented assumption — the LVLH
    // construction is planet-agnostic so this matches the existing
    // `init_from_lvlh` translational sibling.
    use astrodyn_quantities::frame::{Earth, PlanetInertial};
    let lvlh = astrodyn_math::LvlhFrame::compute(
        ref_position.m_at::<PlanetInertial<Earth>>(),
        ref_velocity.m_per_s_at::<PlanetInertial<Earth>>(),
    );

    // Inertial→LVLH attitude as a JEOD scalar-first left quaternion.
    let q_inertial_lvlh = JeodQuat::left_quat_from_transformation(&lvlh.t_parent_this);

    // Inertial→body attitude: composition order matches
    // RefFrameState::incr_left line 270 (`Q_A:C = Q_B:C * Q_A:B`),
    // i.e. post-multiply the (already renormalized) LVLH→body by the
    // LVLH frame's inertial→LVLH.
    let q_inertial_body = q_lvlh_body.multiply(&q_inertial_lvlh);

    // T_lvlh_body matrix derived from the renormalized quaternion. Used
    // both to lift a parent-frame-expressed `ang_vel_lvlh_to_body` into
    // the body frame *and* to project the LVLH frame's inertial-relative
    // ang vel into the body frame — both must use the same matrix that
    // matches the returned attitude.
    let t_lvlh_body = q_lvlh_body.left_quat_to_transformation();

    // Angular velocity of the body wrt LVLH, expressed in the body frame.
    // JEOD `apply_user_inputs` lines 304-315: when `rate_in_parent` is
    // set, transform the parent-frame-expressed ang vel through
    // `T_parent_this` (LVLH→body); otherwise the user already provided
    // it in the body frame.
    let ang_vel_lvlh_to_body_in_body = match ang_vel_frame {
        LvlhAngularVelocityFrame::Body => ang_vel_lvlh_to_body,
        LvlhAngularVelocityFrame::Lvlh => t_lvlh_body * ang_vel_lvlh_to_body,
    };

    // Angular velocity of the LVLH frame wrt inertial, expressed in the
    // body frame: T_lvlh_body * w_inertial_lvlh_in_lvlh (the
    // `T_B:C * w_A:B` term from the `incr_left` formula).
    let ang_vel_inertial_lvlh_in_body = t_lvlh_body * lvlh.ang_vel_this;

    // Final body-frame angular velocity:
    //   w_inertial_body_in_body = T_lvlh_body * w_inertial_lvlh_in_lvlh
    //                           + w_lvlh_body_in_body
    let ang_vel_body = ang_vel_inertial_lvlh_in_body + ang_vel_lvlh_to_body_in_body;

    RotationalState {
        quaternion: q_inertial_body,
        ang_vel_body,
    }
}

/// Initialize translational state from NED (North-East-Down) position and velocity.
///
/// Converts geodetic coordinates to PCPF Cartesian, applies NED-to-PCPF rotation
/// for velocity, rotates from PCPF to ECI, and adds the ω×r frame-rotation term
/// to account for the planet's rotation.
///
/// The `ned_velocity` is a **planet-fixed** velocity (the natural NED meaning):
/// the velocity as measured by an observer rotating with the planet. The returned
/// ECI velocity includes the contribution from the planet's rotation via
/// `v_eci = T_pcpf→eci * v_pcpf + ω_planet × r_eci`.
///
/// This matches JEOD's `DynBodyInitNedState`, which applies the frame-rotation
/// term through `RefFrameState::incr_left()` when composing the rotating PCPF
/// frame with the inertial integration frame.
///
/// # Arguments
/// * `geodetic` - Geodetic position (latitude rad, longitude rad, altitude m)
/// * `ned_velocity` - Planet-fixed velocity in NED frame (m/s)
/// * `r_eq` - Equatorial radius (m)
/// * `r_pol` - Polar radius (m)
/// * `t_eci_pcpf` - Rotation matrix from ECI to PCPF (planet-fixed) frame
/// * `omega_planet` - Planet angular velocity in ECI frame (rad/s)
pub fn init_from_ned(
    geodetic: &GeodeticState,
    ned_velocity: DVec3,
    r_eq: f64,
    r_pol: f64,
    t_eci_pcpf: &DMat3,
    omega_planet: DVec3,
) -> TranslationalState {
    // Convert geodetic to PCPF cartesian via the planet-agnostic
    // `GeodeticState::to_planet_fixed`; bit-identical to the deprecated
    // `geodetic_to_cartesian` removed in Phase 10.
    let pcpf_pos = geodetic.to_planet_fixed(r_eq, r_pol);

    // Compute NED-to-PCPF rotation at this geodetic location.
    // t_pcpf_ned transforms vectors from PCPF to NED, so its transpose
    // transforms from NED to PCPF.
    let t_pcpf_ned = compute_ned_rotation(geodetic.latitude, geodetic.longitude);
    let pcpf_vel = t_pcpf_ned.transpose() * ned_velocity;

    // Convert PCPF to ECI.
    // t_eci_pcpf transforms from ECI to PCPF, so its transpose goes PCPF to ECI.
    let t_pcpf_to_eci = t_eci_pcpf.transpose();
    let position = t_pcpf_to_eci * pcpf_pos;

    // ECI velocity = rotated PCPF velocity + ω_planet × r_eci
    // The cross product accounts for the rotating frame contribution:
    // a point fixed in PCPF still has inertial velocity due to planet rotation.
    let velocity = t_pcpf_to_eci * pcpf_vel + omega_planet.cross(position);

    TranslationalState { position, velocity }
}

/// Build the inertial→NED reference-frame state for a single ground point
/// (no reference body), reproducing JEOD `DynBodyInitNedState::apply` for
/// the `ref_body == nullptr` branch
/// (`models/dynamics/body_action/src/dyn_body_init_ned_state.cc:114-146`,
/// `models/utils/planet_fixed/north_east_down/src/north_east_down.cc`).
///
/// JEOD constructs the NED frame as a child of the planet-centered,
/// planet-fixed (pfix) frame:
/// - its origin is the geodetic/spherical reference point converted to
///   planet-fixed Cartesian (`update_from_ellip` / `update_from_spher`),
/// - it is stationary wrt the rotating planet
///   (`Vector3::initialize(ned_frame.state.trans.velocity)` —
///   `dyn_body_init_ned_state.cc:117`),
/// - its orientation is the NED-axes matrix `T_pfix_ned`
///   (`build_ned_orientation`, identical to [`compute_ned_rotation`]),
/// - it has *zero* angular velocity wrt pfix
///   (`build_ned_orientation` zeroes `ang_vel_this`).
///
/// Because pfix itself rotates at `ω_planet` wrt the inertial integration
/// frame, the NED frame's inertial state is obtained by composing
/// pfix→inertial via [`astrodyn_frames::RefFrameState::incr_left`]
/// (A = inertial, B = pfix, C = NED), which carries the `ω_planet × r`
/// velocity term into the NED origin's inertial velocity and `ω_planet`
/// (rotated into NED coordinates) into the NED frame's inertial rate.
///
/// `geodetic` must carry **geodetic** latitude/longitude when the deck
/// specifies `altlatlong_type = elliptical` and **geocentric/spherical**
/// latitude/longitude when it specifies `spherical`; the caller is
/// responsible for supplying the matching `GeodeticState` and the matching
/// position conversion (this function uses [`GeodeticState::to_planet_fixed`],
/// the ellipsoid conversion, for the origin). The NED-axes matrix uses the
/// same `(lat, lon)` so the orientation is consistent with the origin.
///
/// # Arguments
/// * `geodetic` - Reference point (latitude rad, longitude rad, altitude m).
/// * `r_eq` - Planet equatorial radius (m).
/// * `r_pol` - Planet polar radius (m).
/// * `t_eci_pcpf` - Rotation matrix from ECI (inertial) to PCPF (pfix).
/// * `omega_planet` - Planet angular velocity expressed in **pfix
///   coordinates** (rad/s). This is the pfix frame's `ang_vel_this` in
///   JEOD's convention: `[0, 0, planet_omega]` about the pfix z-axis
///   (`planet_rnp.cc:199-201, 245-247` sets the pfix angular velocity in
///   the pfix frame, *not* the ECI frame). For Earth this is
///   `[0, 0, EARTH.omega]`. Because precession+nutation tilt the pfix
///   z-axis off the ECI z-axis, this is **not** the same as the ECI-frame
///   `[0, 0, ω]` rotated into pfix — JEOD anchors the rotation rate to the
///   pfix axes, so the caller must pass the pfix-frame value directly.
// JEOD_INV: BA.16 — single-point NED frame is a child of pfix with zero
// rate wrt pfix; its inertial state is incr_left(pfix-wrt-inertial,
// NED-wrt-pfix), so the planet-rotation ω×r velocity and ω_planet rate
// (anchored to the pfix axes) enter from pfix.
pub fn ned_reference_frame_state(
    geodetic: &GeodeticState,
    r_eq: f64,
    r_pol: f64,
    t_eci_pcpf: &DMat3,
    omega_planet: DVec3,
) -> astrodyn_frames::RefFrameState {
    // NED origin in pfix coordinates (ellipsoid conversion), stationary
    // wrt the rotating planet (zero pfix-frame velocity).
    let pcpf_pos = geodetic.to_planet_fixed(r_eq, r_pol);

    // NED-axes matrix T_pfix_ned (JEOD `build_ned_orientation`); rows are
    // North, East, Down.
    let t_pfix_ned = compute_ned_rotation(geodetic.latitude, geodetic.longitude);

    // S_inertial:pfix — pfix frame wrt inertial (origin coincident with the
    // planet center, rotating at `omega_planet` already in pfix
    // coordinates — JEOD `planet_rnp.cc` stores the pfix `ang_vel_this`
    // about the pfix z-axis).
    let s_inertial_pfix = astrodyn_frames::RefFrameState {
        trans: astrodyn_frames::RefFrameTrans {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        },
        rot: astrodyn_frames::RefFrameRot {
            q_parent_this: JeodQuat::left_quat_from_transformation(t_eci_pcpf),
            t_parent_this: *t_eci_pcpf,
            ang_vel_this: omega_planet,
        },
    };

    // S_pfix:ned — NED frame wrt pfix (origin at the ground point, NED
    // axes, zero rate and zero velocity wrt pfix). Composed up to inertial
    // in place via incr_left.
    let mut s_inertial_ned = astrodyn_frames::RefFrameState {
        trans: astrodyn_frames::RefFrameTrans {
            position: pcpf_pos,
            velocity: DVec3::ZERO,
        },
        rot: astrodyn_frames::RefFrameRot {
            q_parent_this: JeodQuat::left_quat_from_transformation(&t_pfix_ned),
            t_parent_this: t_pfix_ned,
            ang_vel_this: DVec3::ZERO,
        },
    };
    s_inertial_ned.incr_left(&s_inertial_pfix);
    s_inertial_ned
}

/// Initialize rotational state from an attitude + angular velocity
/// expressed relative to the local North-East-Down (NED) frame at a single
/// ground point.
///
/// Port of JEOD `DynBodyInitNedRotState` / `DynBodyInitNedState::apply`
/// (`ref_body == nullptr`) / `DynBodyInit::apply_user_inputs` for the
/// rotational sub-state (`set_items = Att | Rate`). See
/// `models/dynamics/body_action/src/dyn_body_init_ned_rot_state.cc`,
/// `dyn_body_init_ned_state.cc`, and the rotational branches of
/// `dyn_body_init.cc:298-315`.
///
/// The inertial→NED frame is built by [`ned_reference_frame_state`] (NED
/// axes from the geodetic location, NED frame stationary wrt pfix but
/// carrying pfix's `ω_planet` rate wrt inertial), then composed with the
/// user-supplied NED→body attitude / NED-relative angular velocity via
/// [`init_rot_relative_to_frame`] (JEOD `RefFrameState::incr_left`,
/// A = inertial, B = NED, C = body):
///
/// ```text
/// Q_inertial_body         = Q_ned_body * Q_inertial_ned
/// w_inertial_body_in_body = T_ned_body * w_inertial_ned_in_ned
///                         + w_ned_body_in_body
/// ```
///
/// Because the NED frame rotates with the planet, `w_inertial_ned_in_ned`
/// is non-zero (it is `ω_planet` rotated into NED coordinates) even when
/// the user-supplied NED-relative rate is zero — so a body "aligned with
/// and at rest in NED" still has the planet's inertial angular velocity.
/// This is the rotational analog of the `ω_planet × r` velocity term in
/// [`init_from_ned`].
///
/// # Arguments
/// * `q_ned_body` - NED→body attitude quaternion (scalar-first,
///   left-transformation; JEOD convention). The angular velocity is
///   interpreted in the body frame (JEOD `rate_in_parent = false`, the
///   default and the only sense the NED-rot verif decks use).
/// * `ang_vel_ned_to_body` - Angular velocity of the body wrt the NED
///   frame, expressed in the body frame (rad/s).
/// * `geodetic` - Reference point (latitude rad, longitude rad, altitude m).
/// * `r_eq` - Planet equatorial radius (m).
/// * `r_pol` - Planet polar radius (m).
/// * `t_eci_pcpf` - Rotation matrix from ECI (inertial) to PCPF (pfix).
/// * `omega_planet` - Planet angular velocity expressed in **pfix
///   coordinates** (rad/s), e.g. `[0, 0, EARTH.omega]` (see
///   [`ned_reference_frame_state`] for why this is the pfix-frame value,
///   not the ECI-frame value).
// JEOD_INV: BA.16 — NED-rot init composes inertial→NED (carrying pfix's
// ω_planet rate) with NED→body (user input) per RefFrameState::incr_left.
pub fn init_rot_from_ned(
    q_ned_body: JeodQuat,
    ang_vel_ned_to_body: DVec3,
    geodetic: &GeodeticState,
    r_eq: f64,
    r_pol: f64,
    t_eci_pcpf: &DMat3,
    omega_planet: DVec3,
) -> RotationalState {
    let frame = ned_reference_frame_state(geodetic, r_eq, r_pol, t_eci_pcpf, omega_planet);
    init_rot_relative_to_frame(&frame, q_ned_body, ang_vel_ned_to_body)
}

/// Compute the PCPF-to-NED transformation matrix at a given geodetic location.
///
/// The NED frame axes expressed in the PCPF frame are:
/// - North = [-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)]
/// - East  = [-sin(lon), cos(lon), 0]
/// - Down  = [-cos(lat)*cos(lon), -cos(lat)*sin(lon), -sin(lat)]
///
/// These vectors form the rows of the PCPF-to-NED transformation matrix.
///
/// # Arguments
/// * `lat` - Geodetic latitude (rad)
/// * `lon` - Geodetic longitude (rad)
pub fn compute_ned_rotation(lat: f64, lon: f64) -> DMat3 {
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let sin_lon = lon.sin();
    let cos_lon = lon.cos();

    // Rows of the PCPF-to-NED transformation matrix
    let north = DVec3::new(-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat);
    let east = DVec3::new(-sin_lon, cos_lon, 0.0);
    let down = DVec3::new(-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat);

    mat3_from_rows(north, east, down)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EARTH_MU: f64 = 3.986_004_415e14; // m^3/s^2
    const EARTH_R_EQ: f64 = 6_378_137.0; // WGS84 equatorial radius (m)
    const EARTH_R_POL: f64 = EARTH_R_EQ * (1.0 - 1.0 / 298.257_223_563); // JEOD: r_eq * (1 - flat_coeff)

    // =======================================================================
    // Test 1: Circular orbit from elements
    // =======================================================================

    #[test]
    fn circular_orbit_from_elements() {
        let alt = 400_000.0; // 400 km altitude
        let r = EARTH_R_EQ + alt;
        let a = r; // circular orbit: a = r
        let e = 0.0;
        let inc = 0.0; // equatorial
        let raan = 0.0;
        let argp = 0.0;
        let nu = 0.0; // at periapsis (== anywhere for circular)

        let state = init_from_orbital_elements(a, e, inc, raan, argp, nu, EARTH_MU);

        // Position magnitude should be r
        let r_mag = state.position.length();
        assert!(
            (r_mag - r).abs() < 1e-6,
            "Position magnitude: expected {}, got {}, error = {} m",
            r,
            r_mag,
            (r_mag - r).abs()
        );

        // Velocity magnitude should be sqrt(mu/r) for circular orbit
        let v_circ = (EARTH_MU / r).sqrt();
        let v_mag = state.velocity.length();
        assert!(
            (v_mag - v_circ).abs() < 1e-6,
            "Velocity magnitude: expected {}, got {}, error = {} m/s",
            v_circ,
            v_mag,
            (v_mag - v_circ).abs()
        );
    }

    // =======================================================================
    // Test 2: ISS reference state (Tier 2)
    // =======================================================================

    #[test]
    fn iss_reference_state_from_elements() {
        // Inputs come from the committed `test_data/body_init/iss.json`
        // fixture (regenerated via the `extract_body_init` binary), not
        // `$JEOD_HOME` at runtime.
        let init = astrodyn_verif_jeod_fixtures::orbital_init::load_orbital_init(
            "ISS",
            "trans_Orbit_inertial_body_set01",
        );
        let expected =
            astrodyn_verif_jeod_fixtures::reference_state::load_reference_state("ISS", "inertial");

        // ISS set01 uses SmaEccIncAscnodeArgperTimeperi.
        let t_peri = init
            .time_periapsis
            .expect("ISS set01 should have time_periapsis");
        let state = init_from_time_periapsis(
            init.semi_major_axis
                .expect("ISS set01 should have semi_major_axis"),
            init.eccentricity
                .expect("ISS set01 should have eccentricity"),
            init.inclination,
            init.ascending_node,
            init.arg_periapsis
                .expect("ISS set01 should have arg_periapsis"),
            t_peri,
            EARTH_MU,
        );

        let pos_err = (state.position - expected.position).length();
        let vel_err = (state.velocity - expected.velocity).length();

        println!("ISS position error: {:.2} m", pos_err);
        println!("ISS velocity error: {:.6} m/s", vel_err);
        println!(
            "Computed pos: [{:.2}, {:.2}, {:.2}]",
            state.position.x, state.position.y, state.position.z
        );
        println!(
            "Expected pos: [{:.2}, {:.2}, {:.2}]",
            expected.position.x, expected.position.y, expected.position.z
        );

        // Position tolerance: 1 km (conservative for time_periapsis interpretation)
        assert!(
            pos_err < 1000.0,
            "ISS position error {:.2} m exceeds 1 km tolerance",
            pos_err
        );

        // Velocity tolerance: 1 m/s
        assert!(
            vel_err < 1.0,
            "ISS velocity error {:.6} m/s exceeds 1 m/s tolerance",
            vel_err
        );
    }

    // =======================================================================
    // Test 3: LVLH zero offset returns reference state
    // =======================================================================

    #[test]
    fn lvlh_zero_offset_returns_reference() {
        let r = EARTH_R_EQ + 400_000.0;
        let v = (EARTH_MU / r).sqrt();

        let ref_pos = DVec3::new(r, 0.0, 0.0);
        let ref_vel = DVec3::new(0.0, v, 0.0);

        let state = init_from_lvlh(DVec3::ZERO, DVec3::ZERO, ref_pos, ref_vel);

        let pos_err = (state.position - ref_pos).length();
        let vel_err = (state.velocity - ref_vel).length();

        assert!(
            pos_err < 1e-10,
            "LVLH zero offset position error: {} m",
            pos_err
        );
        assert!(
            vel_err < 1e-10,
            "LVLH zero offset velocity error: {} m/s",
            vel_err
        );
    }

    // =======================================================================
    // Test 4: LVLH round-trip
    // =======================================================================

    #[test]
    fn lvlh_round_trip() {
        // Reference orbit: ISS-like inclined circular orbit
        let r = EARTH_R_EQ + 400_000.0;
        let v = (EARTH_MU / r).sqrt();
        let inc = 51.6_f64.to_radians();

        let ref_pos = DVec3::new(r, 0.0, 0.0);
        let ref_vel = DVec3::new(0.0, v * inc.cos(), v * inc.sin());

        // Set a known LVLH offset: 100m ahead, 50m below, 20m left
        let lvlh_offset_pos = DVec3::new(100.0, 20.0, 50.0); // x=along-track, y=cross-track, z=nadir
        let lvlh_offset_vel = DVec3::new(0.1, 0.05, -0.02);

        // Initialize from LVLH
        let state = init_from_lvlh(lvlh_offset_pos, lvlh_offset_vel, ref_pos, ref_vel);

        // Now compute the LVLH frame at the reference orbit and transform back
        use astrodyn_quantities::frame::{Earth, PlanetInertial};
        let lvlh = astrodyn_math::LvlhFrame::compute(
            ref_pos.m_at::<PlanetInertial<Earth>>(),
            ref_vel.m_per_s_at::<PlanetInertial<Earth>>(),
        );
        let t = lvlh.t_parent_this;

        // Recover LVLH-relative position and velocity. This is the
        // forward `compute_relative_state` direction (JEOD `decr_left`):
        //   x_B:C = T_A:B · (x_A:C - x_A:B)
        //   v_B:C = T_A:B · (v_A:C - v_A:B) - ω_A:B × x_B:C
        // The ω×r Coriolis term must be subtracted here to invert the
        // `incr_left` composition `init_from_lvlh` now performs — an
        // inversion that dropped it would only round-trip a non-rotating
        // reference frame.
        let delta_pos = state.position - ref_pos;
        let delta_vel = state.velocity - ref_vel;
        let recovered_lvlh_pos = t * delta_pos;
        let recovered_lvlh_vel = t * delta_vel - lvlh.ang_vel_this.cross(recovered_lvlh_pos);

        let pos_err = (recovered_lvlh_pos - lvlh_offset_pos).length();
        let vel_err = (recovered_lvlh_vel - lvlh_offset_vel).length();

        assert!(
            pos_err < 1e-10,
            "LVLH round-trip position error: {} m",
            pos_err
        );
        assert!(
            vel_err < 1e-10,
            "LVLH round-trip velocity error: {} m/s",
            vel_err
        );
    }

    // =======================================================================
    // incr_left composition kernels
    // =======================================================================

    fn frame_state(
        position: DVec3,
        velocity: DVec3,
        t_parent_this: DMat3,
        ang_vel_this: DVec3,
    ) -> astrodyn_frames::RefFrameState {
        astrodyn_frames::RefFrameState {
            trans: astrodyn_frames::RefFrameTrans { position, velocity },
            rot: astrodyn_frames::RefFrameRot {
                q_parent_this: JeodQuat::left_quat_from_transformation(&t_parent_this),
                t_parent_this,
                ang_vel_this,
            },
        }
    }

    #[test]
    fn init_trans_relative_to_frame_identity_is_plain_add() {
        // With an identity, non-rotating reference frame, the relative
        // translation init reduces to a vector add onto the frame origin.
        let frame = frame_state(
            DVec3::new(1.0, 2.0, 3.0),
            DVec3::new(0.1, 0.2, 0.3),
            DMat3::IDENTITY,
            DVec3::ZERO,
        );
        let out = init_trans_relative_to_frame(
            &frame,
            DVec3::new(10.0, 20.0, 30.0),
            DVec3::new(1.0, 1.0, 1.0),
        );
        assert!((out.position - DVec3::new(11.0, 22.0, 33.0)).length() < 1e-12);
        assert!((out.velocity - DVec3::new(1.1, 1.2, 1.3)).length() < 1e-12);
    }

    #[test]
    fn init_trans_relative_to_frame_includes_omega_cross_r() {
        // A reference frame rotating about +z at ω; an offset purely
        // along +x with zero relative velocity acquires inertial velocity
        // ω × r = (0,0,ω) × (r,0,0) = (0, ω·r, 0).
        let omega = 0.001;
        let r = 100.0;
        let frame = frame_state(
            DVec3::ZERO,
            DVec3::ZERO,
            DMat3::IDENTITY,
            DVec3::new(0.0, 0.0, omega),
        );
        let out = init_trans_relative_to_frame(&frame, DVec3::new(r, 0.0, 0.0), DVec3::ZERO);
        let expected = DVec3::new(0.0, omega * r, 0.0);
        assert!(
            (out.velocity - expected).length() < 1e-12,
            "ω×r term missing: got {:?}, expected {:?}",
            out.velocity,
            expected
        );
    }

    #[test]
    fn init_rot_relative_to_frame_composes_attitude_and_rate() {
        // Frame B rotates wrt inertial; identity B→subject attitude with
        // zero relative rate must forward the frame's orientation and
        // rate to the subject (w_A:C = T_B:C·w_A:B = w_A:B for identity).
        let t_parent_this = DMat3::from_axis_angle(DVec3::Z, 0.3);
        let w_frame = DVec3::new(0.0, 0.0, 0.002);
        let frame = frame_state(DVec3::ZERO, DVec3::ZERO, t_parent_this, w_frame);
        let out = init_rot_relative_to_frame(&frame, JeodQuat::identity(), DVec3::ZERO);
        // Inertial→subject attitude equals inertial→frame attitude.
        let q_frame = JeodQuat::left_quat_from_transformation(&t_parent_this);
        let dot = out.quaternion.scalar() * q_frame.scalar()
            + out.quaternion.vector().dot(q_frame.vector());
        assert!(dot.abs() > 1.0 - 1e-12, "attitude not forwarded");
        // Subject inertial rate equals the frame rate (identity B→C).
        assert!((out.ang_vel_body - w_frame).length() < 1e-12);
    }

    // =======================================================================
    // Test 5: NED at equator prime meridian
    // =======================================================================

    #[test]
    fn ned_equator_prime_meridian() {
        let geodetic = GeodeticState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
        };

        // Identity ECI-to-PCPF rotation (no Earth rotation offset)
        let t_eci_pcpf = DMat3::IDENTITY;

        let state = init_from_ned(
            &geodetic,
            DVec3::ZERO, // no velocity
            EARTH_R_EQ,
            EARTH_R_POL,
            &t_eci_pcpf,
            DVec3::ZERO, // no planet rotation
        );

        // At lat=0, lon=0, alt=0, the PCPF position should be [r_eq, 0, 0]
        // With identity ECI-to-PCPF, ECI position is the same.
        assert!(
            (state.position.x - EARTH_R_EQ).abs() < 1e-6,
            "Position X: expected {}, got {}",
            EARTH_R_EQ,
            state.position.x
        );
        assert!(
            state.position.y.abs() < 1e-6,
            "Position Y: expected 0, got {}",
            state.position.y
        );
        assert!(
            state.position.z.abs() < 1e-6,
            "Position Z: expected 0, got {}",
            state.position.z
        );
    }

    // =======================================================================
    // NED rotational init
    // =======================================================================

    #[test]
    fn ned_rot_frame_origin_velocity_agrees_with_init_from_ned() {
        // The NED frame origin's inertial position/velocity (zero
        // NED-relative offset) must match `init_from_ned` with a zero NED
        // velocity, for a non-trivial location and a non-trivial planet
        // rotation. This pins the translational consistency between the
        // frame builder and the existing NED translation kernel.
        let geodetic = GeodeticState {
            latitude: 28.6082_f64.to_radians(),
            longitude: (-80.6040_f64).to_radians(),
            altitude: 3.0,
        };
        // Non-trivial ECI→PCPF rotation about +z (a GMST-like angle).
        let t_eci_pcpf = DMat3::from_axis_angle(DVec3::Z, 1.234);
        let omega_planet = DVec3::new(0.0, 0.0, 7.292_115e-5);

        let frame = ned_reference_frame_state(
            &geodetic,
            EARTH_R_EQ,
            EARTH_R_POL,
            &t_eci_pcpf,
            omega_planet,
        );

        let via_init = init_from_ned(
            &geodetic,
            DVec3::ZERO,
            EARTH_R_EQ,
            EARTH_R_POL,
            &t_eci_pcpf,
            omega_planet,
        );

        assert!(
            (frame.trans.position - via_init.position).length() < 1e-6,
            "NED frame origin position disagrees with init_from_ned: {:?} vs {:?}",
            frame.trans.position,
            via_init.position
        );
        assert!(
            (frame.trans.velocity - via_init.velocity).length() < 1e-9,
            "NED frame origin velocity disagrees with init_from_ned: {:?} vs {:?}",
            frame.trans.velocity,
            via_init.velocity
        );
    }

    #[test]
    fn ned_rot_aligned_body_recovers_planet_rate() {
        // A body aligned with and at rest in the NED frame (identity
        // NED→body, zero NED-relative rate) at a non-trivial location must
        // still carry the planet's inertial angular velocity: the body
        // rotates with the planet. The magnitude of the inertial body rate
        // must equal |ω_planet|, and its representation in the body frame
        // must equal the inertial ω rotated through the inertial→body
        // attitude.
        let geodetic = GeodeticState {
            latitude: 28.6082_f64.to_radians(),
            longitude: (-80.6040_f64).to_radians(),
            altitude: 3.0,
        };
        let t_eci_pcpf = DMat3::from_axis_angle(DVec3::Z, 1.234);
        let omega_planet = DVec3::new(0.0, 0.0, 7.292_115e-5);

        let rot = init_rot_from_ned(
            JeodQuat::identity(),
            DVec3::ZERO,
            &geodetic,
            EARTH_R_EQ,
            EARTH_R_POL,
            &t_eci_pcpf,
            omega_planet,
        );

        // |ω_body| == |ω_planet| (a pure rotation of the same vector).
        assert!(
            (rot.ang_vel_body.length() - omega_planet.length()).abs() < 1e-15,
            "inertial body rate magnitude {} != |omega_planet| {}",
            rot.ang_vel_body.length(),
            omega_planet.length()
        );

        // The body-frame ω equals T_inertial_body · ω_planet (ω expressed
        // in the inertial frame, rotated into the body frame).
        let t_inertial_body = rot.quaternion.left_quat_to_transformation();
        let expected_body = t_inertial_body * omega_planet;
        assert!(
            (rot.ang_vel_body - expected_body).length() < 1e-18,
            "body-frame ω {:?} != T_inertial_body·ω_planet {:?}",
            rot.ang_vel_body,
            expected_body
        );

        // Non-trivial inertial attitude: identity NED→body at this lat/lon
        // composes with the non-identity inertial→NED rotation, so the
        // body's inertial attitude must NOT be identity.
        assert!(
            rot.quaternion.vector().length() > 1e-3,
            "expected a non-trivial inertial→body attitude at 28.6N/-80.6E"
        );
    }

    // =======================================================================
    // Test 6: Elements round-trip
    // =======================================================================

    #[test]
    fn elements_round_trip() {
        // Non-trivial orbit with distinct elements
        let a = 7_000_000.0; // m
        let e = 0.01;
        let inc = 51.6_f64.to_radians();
        let raan = 30.0_f64.to_radians();
        let argp = 45.0_f64.to_radians();
        let nu = 60.0_f64.to_radians();

        // Initialize from elements
        let state = init_from_orbital_elements(a, e, inc, raan, argp, nu, EARTH_MU);

        // Convert back to orbital elements via the typed sibling.
        use astrodyn_quantities::frame::{Earth, PlanetInertial};
        let oe = OrbitalElements::<Earth>::from_cartesian_typed(
            astrodyn_quantities::ext::F64Ext::m3_per_s2_for::<Earth>(EARTH_MU),
            state.position.m_at::<PlanetInertial<Earth>>(),
            state.velocity.m_per_s_at::<PlanetInertial<Earth>>(),
        )
        .expect("from_cartesian_typed failed");

        // Compare recovered elements against originals
        assert!(
            (oe.semi_major_axis - a).abs() / a < 1e-10,
            "semi_major_axis: expected {}, got {}, rel_err = {}",
            a,
            oe.semi_major_axis,
            (oe.semi_major_axis - a).abs() / a
        );
        assert!(
            (oe.e_mag - e).abs() < 1e-10,
            "eccentricity: expected {}, got {}, error = {}",
            e,
            oe.e_mag,
            (oe.e_mag - e).abs()
        );
        assert!(
            (oe.inclination - inc).abs() < 1e-10,
            "inclination: expected {}, got {}, error = {}",
            inc,
            oe.inclination,
            (oe.inclination - inc).abs()
        );
        assert!(
            (oe.long_asc_node - raan).abs() < 1e-10,
            "RAAN: expected {}, got {}, error = {}",
            raan,
            oe.long_asc_node,
            (oe.long_asc_node - raan).abs()
        );
        assert!(
            (oe.arg_periapsis - argp).abs() < 1e-10,
            "arg_periapsis: expected {}, got {}, error = {}",
            argp,
            oe.arg_periapsis,
            (oe.arg_periapsis - argp).abs()
        );
        assert!(
            (oe.true_anom - nu).abs() < 1e-10,
            "true_anomaly: expected {}, got {}, error = {}",
            nu,
            oe.true_anom,
            (oe.true_anom - nu).abs()
        );
    }

    // =======================================================================
    // Additional tests
    // =======================================================================

    #[test]
    fn mean_anomaly_agrees_with_true_anomaly_for_circular() {
        // For a circular orbit, mean anomaly == true anomaly
        let a = EARTH_R_EQ + 400_000.0;
        let e = 0.0;
        let inc = 0.0;
        let raan = 0.0;
        let argp = 0.0;
        let nu = 1.0; // radians

        let state_true = init_from_orbital_elements(a, e, inc, raan, argp, nu, EARTH_MU);
        let state_mean = init_from_mean_anomaly(a, e, inc, raan, argp, nu, EARTH_MU);

        let pos_err = (state_true.position - state_mean.position).length();
        let vel_err = (state_true.velocity - state_mean.velocity).length();

        assert!(
            pos_err < 1e-6,
            "Circular orbit: true vs mean anomaly position error = {} m",
            pos_err
        );
        assert!(
            vel_err < 1e-6,
            "Circular orbit: true vs mean anomaly velocity error = {} m/s",
            vel_err
        );
    }

    #[test]
    fn ned_rotation_orthonormal() {
        // Verify NED rotation matrix is orthonormal at several locations
        let test_cases = [
            (0.0, 0.0),             // equator, prime meridian
            (PI / 4.0, PI / 3.0),   // 45N, 60E
            (-PI / 6.0, -PI / 2.0), // 30S, 90W
            (PI / 2.0 - 0.01, 1.0), // near north pole
        ];

        for (lat, lon) in test_cases {
            let t = compute_ned_rotation(lat, lon);

            // T * T^T should be identity
            let product = t * t.transpose();
            let diff = product - DMat3::IDENTITY;
            assert!(
                diff.x_axis.length() < 1e-14,
                "NED rotation not orthonormal at lat={}, lon={}",
                lat,
                lon
            );
            assert!(diff.y_axis.length() < 1e-14);
            assert!(diff.z_axis.length() < 1e-14);

            // Determinant should be +1
            assert!(
                (t.determinant() - 1.0).abs() < 1e-14,
                "NED rotation determinant != 1 at lat={}, lon={}",
                lat,
                lon
            );
        }
    }

    #[test]
    fn ned_north_velocity_at_equator() {
        // At the equator (lat=0, lon=0), a pure North velocity in NED
        // should map to the +Z direction in PCPF (since North points toward
        // the pole at the equator).
        let geodetic = GeodeticState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
        };
        let t_eci_pcpf = DMat3::IDENTITY;

        let ned_vel = DVec3::new(1000.0, 0.0, 0.0); // 1 km/s North
        let state = init_from_ned(
            &geodetic,
            ned_vel,
            EARTH_R_EQ,
            EARTH_R_POL,
            &t_eci_pcpf,
            DVec3::ZERO,
        );

        // North at (lat=0, lon=0) in PCPF is [-sin(0)*cos(0), -sin(0)*sin(0), cos(0)] = [0, 0, 1]
        // NED-to-PCPF = T_pcpf_ned^T, where row0 of T_pcpf_ned is North = [0,0,1].
        // So column 0 of T^T = [0,0,1]. Thus NED [1000,0,0] -> PCPF [0,0,1000].
        assert!(
            state.velocity.x.abs() < 1e-6,
            "Vel X: expected 0, got {}",
            state.velocity.x
        );
        assert!(
            state.velocity.y.abs() < 1e-6,
            "Vel Y: expected 0, got {}",
            state.velocity.y
        );
        assert!(
            (state.velocity.z - 1000.0).abs() < 1e-6,
            "Vel Z: expected 1000, got {}",
            state.velocity.z
        );
    }

    #[test]
    fn ned_omega_cross_r_contribution() {
        // Verify that planet rotation adds ω×r to ECI velocity.
        // At equator (lat=0, lon=0), position is [r_eq, 0, 0] in PCPF.
        // With identity T_eci_pcpf, ECI position is the same.
        // ω = [0, 0, ω_earth], so ω × r = [0, 0, ω] × [r, 0, 0] = [0, ω*r, 0].
        let geodetic = GeodeticState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
        };
        let t_eci_pcpf = DMat3::IDENTITY;
        let omega_earth = 7.292_115_0e-5; // rad/s
        let omega = DVec3::new(0.0, 0.0, omega_earth);

        // Zero NED velocity: the only ECI velocity comes from planet rotation.
        let state = init_from_ned(
            &geodetic,
            DVec3::ZERO,
            EARTH_R_EQ,
            EARTH_R_POL,
            &t_eci_pcpf,
            omega,
        );

        // Expected: ω × r = [0, ω*r_eq, 0] ≈ [0, 465.1, 0] m/s
        let expected_vy = omega_earth * EARTH_R_EQ;
        assert!(
            state.velocity.x.abs() < 1e-6,
            "Vel X: expected 0, got {}",
            state.velocity.x
        );
        assert!(
            (state.velocity.y - expected_vy).abs() < 1e-3,
            "Vel Y: expected {:.1}, got {:.1}",
            expected_vy,
            state.velocity.y
        );
        assert!(
            state.velocity.z.abs() < 1e-6,
            "Vel Z: expected 0, got {}",
            state.velocity.z
        );
    }

    #[test]
    fn lvlh_with_inclined_orbit() {
        // Test LVLH with a non-trivial inclined orbit and non-zero offset
        let r = EARTH_R_EQ + 400_000.0;
        let v = (EARTH_MU / r).sqrt();
        let inc = 51.6_f64.to_radians();

        // Position along X-axis, velocity in the Y-Z plane (inclined orbit)
        let ref_pos = DVec3::new(r, 0.0, 0.0);
        let ref_vel = DVec3::new(0.0, v * inc.cos(), v * inc.sin());

        // Zero offset should still give reference state
        let state = init_from_lvlh(DVec3::ZERO, DVec3::ZERO, ref_pos, ref_vel);
        assert!(
            (state.position - ref_pos).length() < 1e-10,
            "Inclined LVLH zero offset position error"
        );
        assert!(
            (state.velocity - ref_vel).length() < 1e-10,
            "Inclined LVLH zero offset velocity error"
        );

        // 1 km nadir offset (Z in LVLH = toward planet center)
        let lvlh_pos = DVec3::new(0.0, 0.0, 1000.0);
        let state_nadir = init_from_lvlh(lvlh_pos, DVec3::ZERO, ref_pos, ref_vel);

        // The offset in inertial should reduce position magnitude (closer to Earth)
        let r_offset = state_nadir.position.length();
        assert!(
            r_offset < r,
            "1 km nadir offset should reduce position magnitude: {} vs {}",
            r_offset,
            r
        );
        // And the offset magnitude should be approximately 1 km
        let delta = (state_nadir.position - ref_pos).length();
        assert!(
            (delta - 1000.0).abs() < 1e-6,
            "Offset magnitude: expected 1000 m, got {} m",
            delta
        );
    }

    #[test]
    fn typed_orbital_init_matches_untyped_bit_for_bit() {
        use astrodyn_quantities::ext::F64Ext;
        use uom::si::angle::radian;
        use uom::si::f64::{Angle, Length};
        use uom::si::length::meter;

        let alt = 400_000.0;
        let r = EARTH_R_EQ + alt;
        let a = r;
        let e = 0.0;
        let inc = 0.0;
        let raan = 0.0;
        let argp = 0.0;
        let nu = 0.0;

        let untyped = init_from_orbital_elements(a, e, inc, raan, argp, nu, EARTH_MU);
        let typed = init_from_orbital_elements_typed(
            Length::new::<meter>(a),
            e,
            Angle::new::<radian>(inc),
            Angle::new::<radian>(raan),
            Angle::new::<radian>(argp),
            Angle::new::<radian>(nu),
            EARTH_MU.m3_per_s2_for::<astrodyn_quantities::frame::Earth>(),
        );

        assert_eq!(typed.position.raw_si(), untyped.position);
        assert_eq!(typed.velocity.raw_si(), untyped.velocity);
    }

    // =======================================================================
    // LVLH-relative rotational init
    // =======================================================================

    /// Build the same LVLH frame the kernel sees, for use in test
    /// expectations. Matches `init_from_lvlh`'s typed entry.
    fn lvlh_frame_at(ref_pos: DVec3, ref_vel: DVec3) -> astrodyn_math::LvlhFrame {
        use astrodyn_quantities::frame::{Earth, PlanetInertial};
        astrodyn_math::LvlhFrame::compute(
            ref_pos.m_at::<PlanetInertial<Earth>>(),
            ref_vel.m_per_s_at::<PlanetInertial<Earth>>(),
        )
    }

    #[test]
    fn lvlh_rot_identity_attitude_zero_rate_recovers_lvlh_frame() {
        // Identity LVLH→body attitude with zero LVLH-relative angular
        // velocity must yield the LVLH frame's own attitude / angular
        // velocity wrt inertial: Q_inertial_body = Q_inertial_lvlh, and
        // w_body = w_inertial_lvlh_in_lvlh = [0, -wmag, 0].
        let r = EARTH_R_EQ + 400_000.0;
        let v = (EARTH_MU / r).sqrt();
        let inc = 51.6_f64.to_radians();
        let ref_pos = DVec3::new(r, 0.0, 0.0);
        let ref_vel = DVec3::new(0.0, v * inc.cos(), v * inc.sin());

        let lvlh = lvlh_frame_at(ref_pos, ref_vel);
        let expected_q = JeodQuat::left_quat_from_transformation(&lvlh.t_parent_this);

        let state = init_rot_from_lvlh(
            JeodQuat::identity(),
            DVec3::ZERO,
            LvlhAngularVelocityFrame::Body,
            ref_pos,
            ref_vel,
        );

        // Compare quaternion components (canonical hemisphere is enforced
        // by `normalize`, so the sign convention matches).
        let dq: f64 = (0..4)
            .map(|i| {
                let a = state.quaternion.data[i];
                let b = expected_q.data[i];
                (a - b).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        assert!(
            dq < 1e-14,
            "identity LVLH→body should match LVLH frame quaternion exactly: dq = {dq}"
        );

        // Body-frame angular velocity must be the LVLH frame's own
        // angular velocity (in LVLH coords, since identity LVLH→body
        // means body axes == LVLH axes).
        let ang_err = (state.ang_vel_body - lvlh.ang_vel_this).length();
        assert!(
            ang_err < 1e-14,
            "identity LVLH→body should recover LVLH ang vel: err = {ang_err}"
        );
    }

    #[test]
    fn lvlh_rot_inverse_rate_zeros_inertial_ang_vel() {
        // If the user supplies w_lvlh_body_in_lvlh = -w_inertial_lvlh_in_lvlh,
        // the body is non-rotating wrt inertial: w_inertial_body_in_body = 0.
        // Use identity LVLH→body so the LVLH-coord ang vel maps directly
        // into the body frame.
        let r = EARTH_R_EQ + 400_000.0;
        let v = (EARTH_MU / r).sqrt();
        let ref_pos = DVec3::new(r, 0.0, 0.0);
        let ref_vel = DVec3::new(0.0, v, 0.0);

        let lvlh = lvlh_frame_at(ref_pos, ref_vel);
        // w_lvlh_body = -w_inertial_lvlh: cancels exactly.
        let cancel = -lvlh.ang_vel_this;

        let state = init_rot_from_lvlh(
            JeodQuat::identity(),
            cancel,
            LvlhAngularVelocityFrame::Lvlh,
            ref_pos,
            ref_vel,
        );

        let mag = state.ang_vel_body.length();
        assert!(
            mag < 1e-14,
            "cancelling LVLH-relative rate should null inertial ang vel: |w| = {mag}"
        );
    }

    #[test]
    fn lvlh_rot_nontrivial_attitude_round_trips() {
        // With a non-trivial LVLH→body rotation (60° about an arbitrary
        // axis), recover the user input by composing
        // Q_inertial_body * Q_inertial_lvlh^conj and comparing.
        let r = EARTH_R_EQ + 400_000.0;
        let v = (EARTH_MU / r).sqrt();
        let inc = 28.5_f64.to_radians();
        let ref_pos = DVec3::new(r * 0.6, r * 0.8, 0.0);
        let ref_vel = DVec3::new(-v * 0.8 * inc.cos(), v * 0.6 * inc.cos(), v * inc.sin());

        // Non-trivial axis-angle attitude: 1.2 rad about a non-axis-
        // aligned direction.
        let axis = DVec3::new(1.0, 2.0, 3.0).normalize();
        let q_lvlh_body = JeodQuat::left_quat_from_eigen_rotation(1.2, axis);

        // Non-trivial body-frame LVLH-relative angular velocity (rad/s).
        let w_lvlh_body_in_body = DVec3::new(0.01, -0.02, 0.03);

        let state = init_rot_from_lvlh(
            q_lvlh_body,
            w_lvlh_body_in_body,
            LvlhAngularVelocityFrame::Body,
            ref_pos,
            ref_vel,
        );

        // Recover Q_lvlh_body = Q_inertial_body * conj(Q_inertial_lvlh).
        let lvlh = lvlh_frame_at(ref_pos, ref_vel);
        let q_inertial_lvlh = JeodQuat::left_quat_from_transformation(&lvlh.t_parent_this);
        let mut recovered = state.quaternion.multiply(&q_inertial_lvlh.conjugate());
        recovered.normalize();
        // Compare with the user input (also normalized to canonical
        // hemisphere by `left_quat_from_eigen_rotation`).
        let dq: f64 = (0..4)
            .map(|i| {
                let a = recovered.data[i];
                let b = q_lvlh_body.data[i];
                (a - b).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        assert!(
            dq < 1e-12,
            "recovered Q_lvlh_body should match input: dq = {dq}, recovered = {:?}, input = {:?}",
            recovered.data,
            q_lvlh_body.data
        );

        // Recover w_lvlh_body_in_body =
        //   w_inertial_body_in_body - T_lvlh_body * w_inertial_lvlh_in_lvlh
        let t_lvlh_body = q_lvlh_body.left_quat_to_transformation();
        let recovered_rate = state.ang_vel_body - t_lvlh_body * lvlh.ang_vel_this;
        let rate_err = (recovered_rate - w_lvlh_body_in_body).length();
        assert!(
            rate_err < 1e-14,
            "recovered LVLH-relative rate should match input: err = {rate_err}"
        );
    }

    #[test]
    fn lvlh_rot_body_vs_lvlh_rate_frame_agree_when_rotated_back() {
        // The two `LvlhAngularVelocityFrame` choices must agree when the
        // user rotates the LVLH-frame input by T_lvlh_body before calling
        // with `Body`.
        let r = EARTH_R_EQ + 400_000.0;
        let v = (EARTH_MU / r).sqrt();
        let ref_pos = DVec3::new(r, 0.0, 0.0);
        let ref_vel = DVec3::new(0.0, v, 0.0);

        let axis = DVec3::new(0.0, 0.0, 1.0);
        let q_lvlh_body = JeodQuat::left_quat_from_eigen_rotation(0.5, axis);

        let w_in_lvlh = DVec3::new(0.001, -0.002, 0.003);
        let t_lvlh_body = q_lvlh_body.left_quat_to_transformation();
        let w_in_body = t_lvlh_body * w_in_lvlh;

        let s_lvlh = init_rot_from_lvlh(
            q_lvlh_body,
            w_in_lvlh,
            LvlhAngularVelocityFrame::Lvlh,
            ref_pos,
            ref_vel,
        );
        let s_body = init_rot_from_lvlh(
            q_lvlh_body,
            w_in_body,
            LvlhAngularVelocityFrame::Body,
            ref_pos,
            ref_vel,
        );

        let dq: f64 = (0..4)
            .map(|i| {
                let a = s_lvlh.quaternion.data[i];
                let b = s_body.quaternion.data[i];
                (a - b).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        assert!(dq < 1e-14, "quaternion mismatch across rate-frame: {dq}");
        let dw = (s_lvlh.ang_vel_body - s_body.ang_vel_body).length();
        assert!(dw < 1e-14, "ang vel mismatch across rate-frame: {dw}");
    }

    #[test]
    fn lvlh_rot_renormalizes_off_unit_input_consistently() {
        // A slightly-off-unit input quaternion must be renormalized once
        // at the entry of `init_rot_from_lvlh`, with the renormalized
        // value used for *both* the returned attitude and the
        // `T_lvlh_body` matrix that lifts the LVLH-frame ang vel into
        // the body frame. The test feeds an off-unit input and the
        // pre-normalized equivalent and asserts both forms produce the
        // same `RotationalState` — pinning the consistency property
        // (without renormalizing first, the off-unit input would drive
        // a scaled `T_lvlh_body` matrix and the returned ang vel would
        // not match the renormalized attitude).
        let r = EARTH_R_EQ + 400_000.0;
        let v = (EARTH_MU / r).sqrt();
        let inc = 51.6_f64.to_radians();
        let ref_pos = DVec3::new(r, 0.0, 0.0);
        let ref_vel = DVec3::new(0.0, v * inc.cos(), v * inc.sin());

        // Non-trivial LVLH→body: 30° around an oblique axis.
        let axis = DVec3::new(1.0, 2.0, 3.0).normalize();
        let q_unit = JeodQuat::left_quat_from_eigen_rotation(0.5, axis);

        // Off-unit copy: scale every component by 1.001 so |q|² ≈ 1.002,
        // i.e. ~0.1% off unit length — well outside the
        // `left_quat_to_transformation` tolerance for an exact match
        // but inside what `JeodQuat::normalize` accepts.
        let mut q_off = q_unit;
        for d in q_off.data.iter_mut() {
            *d *= 1.001;
        }

        // LVLH-frame angular velocity, exercising the Lvlh branch (the
        // branch where the unrenormalized quaternion previously drove
        // an inconsistent matrix).
        let w_in_lvlh = DVec3::new(0.001, -0.002, 0.003);

        let s_unit = init_rot_from_lvlh(
            q_unit,
            w_in_lvlh,
            LvlhAngularVelocityFrame::Lvlh,
            ref_pos,
            ref_vel,
        );
        let s_off = init_rot_from_lvlh(
            q_off,
            w_in_lvlh,
            LvlhAngularVelocityFrame::Lvlh,
            ref_pos,
            ref_vel,
        );

        let dq: f64 = (0..4)
            .map(|i| (s_unit.quaternion.data[i] - s_off.quaternion.data[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            dq < 1e-14,
            "off-unit input must produce the same attitude as the pre-normalized input: dq = {dq}"
        );
        let dw = (s_unit.ang_vel_body - s_off.ang_vel_body).length();
        assert!(
            dw < 1e-14,
            "off-unit input must produce the same ang vel as the pre-normalized input: dw = {dw}"
        );

        // Independent cross-check: build the expected ang vel from the
        // renormalized quaternion's matrix directly, and confirm the
        // function's result agrees. This pins the ang-vel formula to
        // the renormalized matrix specifically (an implementation that
        // used the raw input matrix would fail this even though the two
        // calls above produce equal output by sheer coincidence).
        let lvlh = lvlh_frame_at(ref_pos, ref_vel);
        let t_lvlh_body_unit = q_unit.left_quat_to_transformation();
        let expected_w = t_lvlh_body_unit * lvlh.ang_vel_this + t_lvlh_body_unit * w_in_lvlh;
        let err_unit = (s_unit.ang_vel_body - expected_w).length();
        let err_off = (s_off.ang_vel_body - expected_w).length();
        assert!(
            err_unit < 1e-14,
            "ang vel must match the renormalized-matrix formula (unit input): err = {err_unit}"
        );
        assert!(
            err_off < 1e-14,
            "ang vel must match the renormalized-matrix formula (off-unit input): err = {err_off}"
        );
    }

    // =======================================================================
    // BA.05 negative tests — orbit initializer requires `mu > 0`.
    //
    // JEOD `dyn_body_init_orbit.cc:98-111` validates the central body's
    // gravity source before computing the position/velocity. Our port
    // asserts `mu > 0.0` at the entry of every orbital-init kernel; the
    // three tests below drive the same misconfiguration through each
    // public entry point so a future refactor cannot neuter a single
    // assert and leave the others intact.
    // =======================================================================

    #[test]
    fn slr_true_anomaly_matches_orbital_elements_within_roundoff() {
        // The slr+true-anomaly converter sets semiparam = p directly (JEOD's
        // SlrEccIncAscnodeArgperTanom path), whereas init_from_orbital_elements
        // takes sma and recomputes semiparam = a·(1-e²). Feeding the
        // algebraically-equivalent sma = p/(1-e²) into the sma path must agree
        // to within the round-trip roundoff (a few ULP × radius).
        let p = 6_732_889.984_55;
        let e = 0.00129073350;
        let inc = 51.670450765_f64.to_radians();
        let raan = 49.708417385_f64.to_radians();
        let argp = 100.582445989_f64.to_radians();
        let nu = 299.884499026_f64.to_radians();

        let slr = init_from_semi_latus_rectum_true_anomaly(p, e, inc, raan, argp, nu, EARTH_MU);
        let a = p / (1.0 - e * e);
        let sma = init_from_orbital_elements(a, e, inc, raan, argp, nu, EARTH_MU);

        let pos_err = (slr.position - sma.position).length();
        let vel_err = (slr.velocity - sma.velocity).length();
        // ~7e6 m radius × ~1e-15 relative ULP ≈ 1e-8 m; allow generous margin.
        assert!(
            pos_err < 1e-6,
            "slr vs sma position roundoff too large: {pos_err} m"
        );
        assert!(
            vel_err < 1e-9,
            "slr vs sma velocity roundoff too large: {vel_err} m/s"
        );
    }

    #[test]
    fn slr_true_anomaly_position_magnitude_matches_conic() {
        // r = p / (1 + e·cos ν) — verify the converter reproduces the conic
        // radius for a non-trivial true anomaly.
        let p = 6_700_000.0;
        let e = 0.01;
        let nu = 1.3_f64; // rad
        let state = init_from_semi_latus_rectum_true_anomaly(p, e, 0.5, 0.3, 0.7, nu, EARTH_MU);
        let r_expected = p / (1.0 + e * nu.cos());
        let r_actual = state.position.length();
        assert!(
            (r_actual - r_expected).abs() < 1e-6,
            "conic radius: expected {r_expected}, got {r_actual}"
        );
    }

    #[test]
    #[should_panic(expected = "mu must be positive")]
    fn ba_05_panics_on_zero_mu_in_slr_true_anomaly_init() {
        // JEOD_INV: BA.05 — `init_from_semi_latus_rectum_true_anomaly` shares
        // the mu>0 guard so the set03 path can't slip a misconfigured gravity
        // source past the others.
        let _ =
            init_from_semi_latus_rectum_true_anomaly(6_700_000.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "mu must be positive")]
    fn ba_05_panics_on_zero_mu_in_orbital_elements_init() {
        // JEOD_INV: BA.05 — `init_from_orbital_elements` rejects mu = 0
        // before the Keplerian-to-Cartesian conversion would otherwise
        // produce NaN propagation downstream.
        let _ = init_from_orbital_elements(
            EARTH_R_EQ + 400_000.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0, // misconfigured: no gravity source
        );
    }

    #[test]
    #[should_panic(expected = "mu must be positive")]
    fn ba_05_panics_on_negative_mu_in_mean_anomaly_init() {
        // JEOD_INV: BA.05 — `init_from_mean_anomaly` shares the same
        // guard; a negative mu is non-physical and would silently flip
        // the orbit sense if allowed through.
        let _ = init_from_mean_anomaly(EARTH_R_EQ + 400_000.0, 0.01, 0.0, 0.0, 0.0, 0.0, -EARTH_MU);
    }

    #[test]
    #[should_panic(expected = "mu must be positive")]
    fn ba_05_panics_on_zero_mu_in_time_periapsis_init() {
        // JEOD_INV: BA.05 — `init_from_time_periapsis` defers to the
        // mean-anomaly path but checks `mu > 0` itself first so the
        // diagnostic names the entry point the caller invoked.
        let _ = init_from_time_periapsis(EARTH_R_EQ + 400_000.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    #[test]
    fn ba_13_altitudes_true_anomaly_matches_sma_ecc_derivation() {
        // JEOD_INV: BA.13 — the altitude shape derives a/e from the apo/peri
        // altitudes (a = r_eq + ½(alt_apo+alt_peri), e = (alt_apo-alt_peri)/2a)
        // and then resolves the true anomaly. Feeding the same a/e directly to
        // `init_from_orbital_elements` must reproduce the state bit-for-bit.
        let alt_apo = 363_454.582_64;
        let alt_peri = 346_073.820_40;
        let a = EARTH_R_EQ + 0.5 * (alt_apo + alt_peri);
        let e = (alt_apo - alt_peri) / (2.0 * a);
        let via_alt = init_from_altitudes_true_anomaly(
            EARTH_R_EQ, alt_apo, alt_peri, 0.9, 0.86, 1.75, 5.23, EARTH_MU,
        );
        let via_sma = init_from_orbital_elements(a, e, 0.9, 0.86, 1.75, 5.23, EARTH_MU);
        assert_eq!(via_alt.position, via_sma.position);
        assert_eq!(via_alt.velocity, via_sma.velocity);
    }

    #[test]
    #[should_panic(expected = "eccentricity must be in [0, 1)")]
    fn ba_13_panics_on_swapped_altitudes() {
        // JEOD_INV: BA.13 — apo below peri yields an invalid negative derived
        // eccentricity, which `init_from_orbital_elements` rejects rather than
        // silently accepting a non-physical orbit. (A swapped-altitude deck is
        // the obvious way to mis-specify the altitude shape.)
        let _ = init_from_altitudes_true_anomaly(
            EARTH_R_EQ, 346_073.0, 363_454.0, 0.9, 0.86, 1.75, 5.23, EARTH_MU,
        );
    }

    #[test]
    #[should_panic(expected = "mu must be positive")]
    fn ba_13_panics_on_zero_mu_in_altitudes_time_periapsis_init() {
        // JEOD_INV: BA.13 — the set05 altitude path defers to the
        // time-periapsis converter, which keeps the mu>0 guard so a
        // misconfigured gravity source can't slip through the altitude shape.
        let _ = init_from_altitudes_time_periapsis(
            EARTH_R_EQ, 363_454.0, 346_073.0, 0.9, 0.86, 1.75, 4581.96, 0.0,
        );
    }

    // =======================================================================
    // BA.14 — set06 (SmaIncAscnodeArglatRadRadvel) derives (e, ν, ω) from the
    // orbital radius / radial-velocity pair, then resolves the sma + true-
    // anomaly shape.
    // =======================================================================

    #[test]
    fn ba_14_arg_latitude_radial_vel_matches_iss_set06_deck() {
        // JEOD_INV: BA.14 — recover (e, ν, ω) from the radius/radial-velocity
        // pair and confirm the derived state matches `init_from_orbital_elements`
        // fed the JEOD-derived elements. The inputs are the ISS set06 deck
        // (`Modified_data/ISS/trans_Orbit_inertial_body_set06.py`).
        let a = 6_732_901.201_52; // km → m
        let i = 51.670450765_f64.to_radians();
        let raan = 49.708417385_f64.to_radians();
        let arg_lat = 400.466945015_f64.to_radians();
        let r = 6_728_562.764_55; // km → m
        let rdot = -8.61072308;

        // Reproduce JEOD's derivation independently (same arithmetic order).
        let ecos_e = (a - r) / a;
        let esin_e = (rdot * r) / (EARTH_MU * a).sqrt();
        let ecc_sq = ecos_e * ecos_e + esin_e * esin_e;
        let e = ecc_sq.sqrt();
        let kcost = ecos_e - ecc_sq;
        let ksint = (1.0 - ecc_sq).sqrt() * esin_e;
        let nu = ksint.atan2(kcost);
        let argp = arg_lat - nu;

        let via_set06 = init_from_arg_latitude_radial_vel(a, i, raan, arg_lat, r, rdot, EARTH_MU);
        let via_elem = init_from_orbital_elements(a, e, i, raan, argp, nu, EARTH_MU);

        assert_eq!(via_set06.position, via_elem.position);
        assert_eq!(via_set06.velocity, via_elem.velocity);

        // The recovered orbital radius must reproduce the deck's orb_radius
        // (the radius is what set06 is parameterized on); a stray sign or a
        // swapped ecosE/esinE would shift it kilometres.
        let r_actual = via_set06.position.length();
        assert!(
            (r_actual - r).abs() < 1e-6,
            "set06 recovered radius: expected {r} m, got {r_actual} m"
        );
    }

    #[test]
    fn ba_14_arg_latitude_radial_vel_circular_sets_nu_zero() {
        // JEOD_INV: BA.14 — for a circular orbit (r = a, rdot = 0) the derived
        // eccentricity is below the 1e-14 cutoff, so JEOD pins ν = 0 and
        // ω = arg_latitude. The state must then equal the circular orbit at
        // true anomaly 0 with arg_periapsis = arg_latitude.
        let a = EARTH_R_EQ + 400_000.0;
        let i = 0.5;
        let raan = 0.3;
        let arg_lat = 1.1;

        let via_set06 = init_from_arg_latitude_radial_vel(a, i, raan, arg_lat, a, 0.0, EARTH_MU);
        let via_elem = init_from_orbital_elements(a, 0.0, i, raan, arg_lat, 0.0, EARTH_MU);

        assert_eq!(via_set06.position, via_elem.position);
        assert_eq!(via_set06.velocity, via_elem.velocity);
    }

    #[test]
    #[should_panic(expected = "mu must be positive")]
    fn ba_14_panics_on_zero_mu_in_arg_latitude_radial_vel_init() {
        // JEOD_INV: BA.14 — set06 shares the mu>0 guard so a misconfigured
        // gravity source can't slip through the arg-latitude/radial-vel shape.
        let _ = init_from_arg_latitude_radial_vel(
            EARTH_R_EQ + 400_000.0,
            0.5,
            0.3,
            1.1,
            EARTH_R_EQ + 400_000.0,
            0.0,
            0.0,
        );
    }
}
