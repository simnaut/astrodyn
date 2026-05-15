//! Atmosphere stage: configuration and the per-body
//! [`evaluate_atmosphere`] / [`evaluate_atmosphere_typed`] orchestration
//! that converts inertial position to geodetic coordinates and queries
//! the configured density / temperature / wind model.
//!
//! JEOD_INV: TS.01 — the `<SelfPlanet>`-tagged returns from the untyped
//! [`evaluate_atmosphere`] entry point exist for adapter storage whose
//! planet identity is keyed by a dynamic source index (the runner's
//! `atmosphere_planet_source: Option<usize>`). Mission code that knows
//! the planet at compile time uses the [`evaluate_atmosphere_typed`]
//! sibling, which carries `<P: Planet>` end-to-end. The lint at
//! `tests/self_ref_self_planet_discipline.rs` enforces this discipline
//! globally.

use astrodyn_atmosphere::exponential::ExponentialAtmosphere;
use astrodyn_atmosphere::met::MetAtmosphere;
use astrodyn_atmosphere::AtmosphereState;
use astrodyn_math::GeodeticState;
use astrodyn_quantities::aliases::Position;
use astrodyn_quantities::frame::{Planet, PlanetInertial, SelfPlanet};
use glam::{DMat3, DVec3};

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
    /// use astrodyn::{AtmosphereConfig, AtmosphereModel, EARTH};
    /// use astrodyn_atmosphere::exponential::ExponentialAtmosphere;
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
    // `astrodyn_math::cartesian_to_geodetic` removed in Phase 10.
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
        astrodyn_atmosphere::compute_corotation_wind(config.planet_omega, position)
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
/// wind vector and any downstream consumer (`compute_ballistic_drag_typed`)
/// can structurally enforce that the vehicle's planet-inertial
/// velocity matches the wind's planet at the type level.
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

/// Per-body atmosphere kernel — the atmospheric counterpart of
/// [`crate::gravity::evaluate_body_gravity_typed`] for the stage-runner
/// abstraction (audit §2.2).
///
/// Atmosphere is an **RF.10 NON-shift site**: geodetic altitude,
/// planet-fixed rotation, and the co-rotation wind are all computed
/// against the *atmosphere planet's* center, not against the
/// simulation's root inertial frame. The kernel therefore takes
/// `Position<PlanetInertial<P>>` and passes it through unchanged — no
/// `IntegOrigin` shift is applied here. Callers whose storage frame is
/// the body's `IntegrationFrame` (the runner) relabel at the boundary
/// via `from_raw_si`; callers whose storage frame is already
/// `PlanetInertial<P>` (the Bevy `TranslationalStateC<P>`) move the
/// value through as-is.
///
/// In every realistic config the body integrates in the atmosphere
/// planet's inertial frame (e.g., a body integrating in
/// `Earth.inertial` with the atmosphere keyed to Earth), so the relabel
/// is bit-identical. A body integrated in some *other* planet's
/// inertial frame would need an inertial-to-inertial transform before
/// reaching this kernel; that case isn't exercised today and would be
/// rejected by the type system at the call site if attempted via
/// relabel alone (the adapter would need a real
/// `FrameTransform<…, PlanetInertial<P>>` first).
///
/// # Arguments
/// - `config`: planet-level atmosphere model and shape parameters
/// - `position`: body position in the atmosphere planet's inertial frame (m)
/// - `t_inertial_pfix`: optional inertial-to-planet-fixed rotation;
///   passing `None` treats `position` as already in planet-fixed coords
/// - `tai_tjt`: truncated Julian time (required for [`AtmosphereModel::Met`])
// JEOD_INV: AT.01 — active flag gates computation (caller checks presence)
// JEOD_INV: AT.02 — atmosphere model pointer non-null (caller provides config)
// JEOD_INV: AT.03 — planet-fixed position required for geodetic altitude
// JEOD_INV: AT.04 — wind velocity computed as omega x position (co-rotation)
pub fn evaluate_body_atmosphere_typed<P: Planet>(
    config: &AtmosphereConfig,
    position: Position<PlanetInertial<P>>,
    t_inertial_pfix: Option<&DMat3>,
    tai_tjt: Option<f64>,
) -> AtmosphereState<P> {
    evaluate_atmosphere_typed::<P>(config, position, t_inertial_pfix, tai_tjt)
}

/// Per-body inputs the atmosphere stage driver consumes.
///
/// Mirrors [`crate::gravity::GravityBodyInputs`] in shape: each adapter
/// `.map(...)`s its native row iterator (Bevy `Query::iter_mut()`,
/// runner `self.bodies.iter_mut()`) into one of these per body. The
/// `position` field is `Position<PlanetInertial<P>>` because atmosphere
/// is a non-shift site — the body's storage frame is relabeled to the
/// atmosphere planet's inertial frame at the adapter boundary, never
/// shifted via an `IntegOrigin`.
pub struct AtmosphereBodyInputs<P: Planet> {
    /// Body position in the atmosphere planet's inertial frame.
    pub position: Position<PlanetInertial<P>>,
}

/// Drive the atmosphere stage across an adapter-supplied iterator of
/// bodies.
///
/// `bodies` yields one `(key, inputs, store)` triple per body. The
/// driver:
///
/// 1. Calls [`evaluate_body_atmosphere_typed`] for each body. The
///    kernel performs no `IntegOrigin` shift; `inputs.position` is the
///    body position in the atmosphere planet's inertial frame and is
///    passed through unchanged. This preserves the **RF.10 non-shift**
///    discipline that distinguishes atmosphere from gravity — the
///    geodetic altitude must be measured against the atmosphere
///    planet's center, not against the simulation root.
/// 2. Hands the result to the per-body `store` closure, which writes
///    it back into adapter-owned storage. Both adapters now store the
///    typed [`AtmosphereState<P>`] (the Bevy `AtmosphericStateC<P>`
///    newtype with `<P>` pinned at the type level, the runner with
///    `<P> = SelfPlanet` because its planet identity is keyed by a
///    runtime source index), so each closure is a direct move on the
///    adapter's mutable handle.
///
/// `config`, `t_inertial_pfix`, and `tai_tjt` are stage-level constants
/// (planet-keyed atmosphere model, planet-fixed rotation, time tag) and
/// are passed alongside the body iterator rather than per-row. The
/// atmosphere stage has no per-source resolver — atmosphere is keyed
/// to a single planet, not to a per-body source list — so the driver
/// takes only the body iterator plus the stage constants. If a future
/// config grows per-body source resolution, add a resolver argument
/// here mirroring [`crate::gravity::run_gravity_stage`].
///
/// The closure-based writer mirrors the gravity stage: each adapter
/// uses whatever mutable handle its iterator yields
/// (`Mut<'_, AtmosphericStateC<P>>` for Bevy, `&mut AtmosphereState<SelfPlanet>`
/// for the runner) without a trait impl on a foreign type.
pub fn run_atmosphere_stage<P, K, Store, BodyIter>(
    bodies: BodyIter,
    config: &AtmosphereConfig,
    t_inertial_pfix: Option<&DMat3>,
    tai_tjt: Option<f64>,
) where
    P: Planet,
    K: Copy,
    Store: FnOnce(AtmosphereState<P>),
    BodyIter: IntoIterator<Item = (K, AtmosphereBodyInputs<P>, Store)>,
{
    for (_key, inputs, store) in bodies {
        let result =
            evaluate_body_atmosphere_typed::<P>(config, inputs.position, t_inertial_pfix, tai_tjt);
        store(result);
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "atmosphere recipe tests assert bit-exact recovery of literal-built state values"
)]
mod tests {
    use super::*;
    use astrodyn_atmosphere::exponential::ExponentialAtmosphere;
    use astrodyn_quantities::frame::Earth;

    /// Build a plain exponential-atmosphere config with no co-rotation
    /// wind so the typed-kernel-vs-`evaluate_atmosphere` comparison is
    /// bit-identical.
    fn earth_exponential_config() -> AtmosphereConfig {
        AtmosphereConfig {
            model: AtmosphereModel::Exponential(ExponentialAtmosphere::default()),
            r_eq: 6_378_136.3,
            r_pol: 6_356_751.0,
            planet_omega: 0.0,
        }
    }

    /// `evaluate_body_atmosphere_typed` reproduces the open-coded
    /// `evaluate_atmosphere` sequence the runner has carried since
    /// before the kernel existed: pass body position in the atmosphere
    /// planet's inertial frame straight to `evaluate_atmosphere` —
    /// *no* `IntegOrigin` shift. The typed kernel is a relabel-only
    /// wrapper, so `Position<PlanetInertial<P>> -> AtmosphereState<P>`
    /// returns numerics bit-identical to the planet-erased
    /// `evaluate_atmosphere` reference.
    #[test]
    fn evaluate_body_atmosphere_typed_matches_evaluate_atmosphere() {
        let config = earth_exponential_config();
        let raw_position = DVec3::new(7e6, 1e5, -2e5);
        let typed_position = Position::<PlanetInertial<Earth>>::from_raw_si(raw_position);

        let manual = evaluate_atmosphere(&config, raw_position, None, None);
        let typed = evaluate_body_atmosphere_typed::<Earth>(&config, typed_position, None, None);

        assert_eq!(typed.density, manual.density);
        assert_eq!(typed.temperature, manual.temperature);
        assert_eq!(typed.pressure, manual.pressure);
        assert_eq!(typed.wind.raw_si(), manual.wind.raw_si());
    }

    /// Type alias for the row-tuple shape `run_atmosphere_stage`
    /// expects in tests, factored out so the clippy `type_complexity`
    /// lint doesn't fire on the array-literal type ascriptions below.
    /// The `'a` lifetime threads through the boxed closure so the
    /// per-body writer can borrow a stack-local output slot rather
    /// than being forced to `'static`.
    type AtmosphereStageRow<'a, P, K> = (
        K,
        AtmosphereBodyInputs<P>,
        Box<dyn FnOnce(AtmosphereState<P>) + 'a>,
    );

    /// The driver does *not* shift body position by any integ-origin —
    /// atmosphere is an RF.10 non-shift site. Pass a body's
    /// planet-inertial position straight through and confirm the
    /// driver-routed result equals the per-body kernel called directly:
    /// the driver is a thin pass-through and any extra shift would
    /// surface here as a discrepancy in geodetic altitude (and
    /// therefore density).
    #[test]
    fn run_atmosphere_stage_is_a_non_shift_pass_through() {
        let config = earth_exponential_config();
        let body_position =
            Position::<PlanetInertial<Earth>>::from_raw_si(DVec3::new(7e6, 0.0, 0.0));

        let mut out = AtmosphereState::<Earth>::default();
        {
            let entries: [AtmosphereStageRow<Earth, usize>; 1] = [(
                0,
                AtmosphereBodyInputs {
                    position: body_position,
                },
                Box::new(|r| out = r),
            )];
            run_atmosphere_stage::<Earth, _, _, _>(entries, &config, None, None);
        }

        let direct = evaluate_body_atmosphere_typed::<Earth>(&config, body_position, None, None);
        assert_eq!(out.density, direct.density);
        assert_eq!(out.temperature, direct.temperature);
        assert_eq!(out.pressure, direct.pressure);
        assert_eq!(out.wind.raw_si(), direct.wind.raw_si());
    }

    /// `run_atmosphere_stage` over two bodies produces the same
    /// per-body results as calling [`evaluate_body_atmosphere_typed`]
    /// directly — the driver is a thin loop over the kernel.
    #[test]
    fn run_atmosphere_stage_drives_kernel_per_body() {
        let config = earth_exponential_config();
        let body_a = Position::<PlanetInertial<Earth>>::from_raw_si(DVec3::new(7e6, 0.0, 0.0));
        let body_b = Position::<PlanetInertial<Earth>>::from_raw_si(DVec3::new(0.0, 7.5e6, 1e5));

        let baseline_a = evaluate_body_atmosphere_typed::<Earth>(&config, body_a, None, None);
        let baseline_b = evaluate_body_atmosphere_typed::<Earth>(&config, body_b, None, None);

        let mut out_a = AtmosphereState::<Earth>::default();
        let mut out_b = AtmosphereState::<Earth>::default();
        {
            let entries: [AtmosphereStageRow<Earth, usize>; 2] = [
                (
                    0,
                    AtmosphereBodyInputs { position: body_a },
                    Box::new(|r| out_a = r),
                ),
                (
                    1,
                    AtmosphereBodyInputs { position: body_b },
                    Box::new(|r| out_b = r),
                ),
            ];
            run_atmosphere_stage::<Earth, _, _, _>(entries, &config, None, None);
        }

        assert_eq!(out_a.density, baseline_a.density);
        assert_eq!(out_a.wind.raw_si(), baseline_a.wind.raw_si());
        assert_eq!(out_b.density, baseline_b.density);
        assert_eq!(out_b.wind.raw_si(), baseline_b.wind.raw_si());
    }
}
