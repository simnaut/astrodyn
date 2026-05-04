//! Atmosphere stage: configuration and the per-body
//! [`evaluate_atmosphere`] / [`evaluate_atmosphere_typed`] orchestration
//! that converts inertial position to geodetic coordinates and queries
//! the configured density / temperature / wind model.

use glam::{DMat3, DVec3};
use jeod_atmosphere::exponential::ExponentialAtmosphere;
use jeod_atmosphere::met::MetAtmosphere;
use jeod_atmosphere::AtmosphereState;
use jeod_math::GeodeticState;
use jeod_quantities::aliases::Position;
use jeod_quantities::frame::{Planet, PlanetInertial, SelfPlanet};

use crate::planet_config::PlanetConfig;

/// Selectable atmosphere model.
#[derive(Debug, Clone)]
pub enum AtmosphereModel {
    /// Simple exponential: `rho = rho_0 * exp(-(h - h_0) / H)`.
    /// No time, latitude, or longitude dependence.
    Exponential(ExponentialAtmosphere),
    /// Marshall Engineering Thermosphere (Jacchia 1970/1971).
    /// Full altitude/latitude/longitude/time/solar-activity dependence.
    Met(MetAtmosphere),
}

/// Planet-level atmosphere configuration (ECS-agnostic).
///
/// Bevy adapter wraps this in a `Resource` and adds an `Entity` for planet lookup.
/// `Simulation` stores the planet source index separately.
///
/// Use [`from_planet`](AtmosphereConfig::from_planet) to construct from a
/// [`PlanetConfig`] preset, avoiding scattered planet constants.
#[derive(Debug, Clone)]
pub struct AtmosphereConfig {
    /// The atmosphere model to evaluate.
    pub model: AtmosphereModel,
    /// Equatorial radius of the planet (m). Used for geodetic conversion.
    pub r_eq: f64,
    /// Polar radius of the planet (m). Used for geodetic conversion.
    pub r_pol: f64,
    /// Planet angular velocity in rad/s for atmospheric co-rotation wind.
    /// Set to 0.0 to disable wind computation.
    /// Earth: 7.292115146706388e-5 rad/s (from JEOD RNPJ2000 data).
    pub planet_omega: f64,
}

impl AtmosphereConfig {
    /// Create an atmosphere configuration from a [`PlanetConfig`] preset.
    ///
    /// Extracts r_eq, r_pol, and omega from the planet config so that
    /// constants aren't scattered across multiple configuration sites.
    ///
    /// # Example
    /// ```
    /// use jeod_sim::{AtmosphereConfig, AtmosphereModel, EARTH};
    /// use jeod_atmosphere::exponential::ExponentialAtmosphere;
    ///
    /// let config = AtmosphereConfig::from_planet(
    ///     AtmosphereModel::Exponential(ExponentialAtmosphere::default()),
    ///     &EARTH,
    /// );
    /// assert_eq!(config.r_eq, EARTH.shape.r_eq);
    /// ```
    pub fn from_planet(model: AtmosphereModel, planet: &PlanetConfig) -> Self {
        Self {
            model,
            r_eq: planet.shape.r_eq,
            r_pol: planet.shape.r_pol,
            planet_omega: planet.omega,
        }
    }
}

/// Evaluate atmosphere at a body's position.
///
/// Pipeline: rotate position to planet-fixed, convert to geodetic,
/// evaluate atmosphere model, compute optional co-rotation wind.
///
/// # Arguments
/// - `config`: atmosphere model and planet parameters
/// - `position`: body position in the inertial frame (m)
/// - `t_inertial_pfix`: optional inertial-to-planet-fixed rotation matrix.
///   If `None`, position is assumed to already be in planet-fixed coordinates.
/// - `tai_tjt`: truncated Julian time (required for MET model seasonal variation)
///
/// # Panics
/// Panics if `AtmosphereModel::Met` is used without providing `tai_tjt`.
// JEOD_INV: AT.01 — active flag gates computation (caller checks presence)
// JEOD_INV: AT.02 — atmosphere model pointer non-null (caller provides config)
// JEOD_INV: AT.03 — planet-fixed position required for geodetic altitude
// JEOD_INV: AT.04 — wind velocity computed as omega x position (co-rotation)
pub fn evaluate_atmosphere(
    config: &AtmosphereConfig,
    position: DVec3,
    t_inertial_pfix: Option<&DMat3>,
    tai_tjt: Option<f64>,
) -> AtmosphereState<SelfPlanet> {
    // Rotate inertial position to planet-fixed frame
    let pos_pfix = if let Some(rot) = t_inertial_pfix {
        *rot * position
    } else {
        position
    };

    // Convert to geodetic coordinates via the planet-agnostic
    // `GeodeticState::from_planet_fixed`; bit-identical to the deprecated
    // `jeod_math::cartesian_to_geodetic` removed in Phase 10.
    let geodetic = GeodeticState::from_planet_fixed(pos_pfix, config.r_eq, config.r_pol);

    // Evaluate atmosphere model
    let result = match &config.model {
        AtmosphereModel::Exponential(exp) => exp.density(geodetic.altitude),
        AtmosphereModel::Met(met) => {
            let tjt = tai_tjt.expect(
                "MET atmosphere requires tai_tjt (truncated Julian time). \
                 Provide SimulationTime or pass tai_tjt explicitly.",
            );
            met.density_si(
                geodetic.altitude,
                geodetic.latitude,
                geodetic.longitude,
                tjt,
            )
        }
    };

    // Co-rotation wind override. The runtime planet identity is
    // determined by `config` (a runtime-typed `AtmosphereConfig`); the
    // returned state is therefore tagged `SelfPlanet`. Mission code
    // that knows the planet at compile time should call
    // [`evaluate_atmosphere_typed`] instead, which produces a
    // planet-pinned `AtmosphereState<P>`.
    // JEOD_INV: AT.04 — wind velocity computed as omega x position (co-rotation)
    let wind_raw = if config.planet_omega != 0.0 {
        jeod_atmosphere::compute_corotation_wind(config.planet_omega, position)
    } else {
        result.wind.raw_si()
    };

    AtmosphereState::<SelfPlanet>::from_raw(
        result.density,
        result.temperature,
        result.pressure,
        wind_raw,
    )
}

/// Typed sibling of [`evaluate_atmosphere`].
///
/// Generic over the atmosphere planet `P`: accepts the vehicle position
/// in the planet's own inertial frame (`Position<PlanetInertial<P>>`),
/// not in the simulation's root inertial frame. This is the structural
/// distinction enforced by RF.10 — atmosphere geodetic altitude is
/// computed against the planet's center, so the input must be
/// planet-centered. Callers with a body in the planet's integration
/// frame should relabel via `from_raw_si` (bit-identical).
///
/// Bit-identical kernel — wraps the raw f64 implementation via
/// `.raw_si()` at the boundary. Returns `AtmosphereState<P>` so the
/// wind vector and any downstream consumer (`compute_drag_typed`,
/// `compute_ballistic_drag_typed`) can structurally enforce that the
/// vehicle's planet-inertial velocity matches the wind's planet at
/// the type level.
pub fn evaluate_atmosphere_typed<P: Planet>(
    config: &AtmosphereConfig,
    position: Position<PlanetInertial<P>>,
    t_inertial_pfix: Option<&DMat3>,
    tai_tjt: Option<f64>,
) -> AtmosphereState<P> {
    // Internal evaluator works against the runtime-typed config, so the
    // result is `<SelfPlanet>` — relabel to the caller-chosen `<P>` at
    // the typed boundary. Bit-identical (no arithmetic).
    evaluate_atmosphere(config, position.raw_si(), t_inertial_pfix, tai_tjt).relabel::<P>()
}
