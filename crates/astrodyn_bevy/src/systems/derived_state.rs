//! Bevy systems for [`AstrodynSet::DerivedState`](crate::AstrodynSet::DerivedState).
//!
//! Per-step derived states: orbital elements, Euler angles, LVLH frame,
//! geodetic state, and solar beta angle.

use astrodyn::{OrbitalError, Planet, RootInertial};
use bevy::prelude::*;

use crate::components::*;
use crate::frame_param::FrameOrigin;

use super::util::body_integ_origin_in_root;

/// Compute orbital elements for entities with `OrbitalElementsConfigC`.
///
/// Generic over `P: Planet` so the result is correctly typed. The
/// `mu` value read from the configured `gravity_source` entity must
/// physically correspond to planet `P` (RF.11): for an Earth-orbit
/// instantiation `<Earth>` the `gravity_source` should point at the
/// Earth entity, not at Sun/Moon. The system instantiation's `<P>`
/// determines which bodies it processes (only those carrying
/// `OrbitalElementsC<P>`).
///
/// Placed in `AstrodynSet::DerivedState`.
///
/// # Panics
///
/// Panics in two cases:
///
/// 1. `OrbitalElementsConfigC.gravity_source` does not resolve to an
///    entity carrying `GravitySourceC`. The diagnostic names the body,
///    the failing source entity, and the bundle remediation.
/// 2. The typed kernel rejects the instantaneous state (non-positive
///    μ, degenerate orbit with `|h| ≈ 0`, or Kepler iteration
///    non-convergence). Each variant emits a per-cause diagnostic
///    naming the entity and the caller fix.
///
/// Silent zero-element fallback would let geometrically-impossible
/// `(a, e, i) = (0, 0, 0)` values reach downstream consumers as if
/// they were correct. CLAUDE.md "Fail Loudly".
pub fn orbital_elements_system<P: Planet>(
    mut query: Query<(
        Entity,
        &TranslationalStateC<P>,
        &OrbitalElementsConfigC,
        &mut OrbitalElementsC<P>,
    )>,
    sources: Query<&GravitySourceC>,
) {
    for (entity, state, config, mut elements) in &mut query {
        // JEOD_INV: OE.01 — surface a `gravity_source` wiring miss as
        // a per-cause panic naming the offending entity and the broken
        // wiring rather than silently writing zero orbital elements.
        // The previous `Default::default()` fallback (the
        // `OrbitalElementsConfigC.gravity_source` points at an entity
        // that doesn't carry `GravitySourceC` case) is exactly the
        // silent-numerically-wrong shape "Fail Loudly" forbids: a
        // downstream consumer reading `(a, e, i) = (0, 0, 0)` cannot
        // distinguish "no orbital elements computed" from
        // "geometrically-impossible orbit". A standalone wiring guard
        // here is in defense-in-depth with the startup `validation`
        // pass — that pass only fires on `Added<GravityControlsC>`
        // bodies, so a `OrbitalElementsConfigC` inserted after the
        // body's spawn tick (or on an entity without
        // `GravityControlsC`) reaches the system unchecked.
        let source = sources.get(config.gravity_source).unwrap_or_else(|_| {
            panic!(
                "{entity:?} orbital elements: \
                 OrbitalElementsConfigC.gravity_source = {gravity_source:?} \
                 does not resolve to a GravitySourceC component. Insert \
                 GravitySourceC on that entity (via SourceBundle / PlanetBundle \
                 / SunBundle / MoonBundle at spawn time), or point \
                 gravity_source at a gravity-source entity that already has \
                 the component.",
                gravity_source = config.gravity_source
            )
        });
        // `OrbitalElementsC<P>` and the typed kernel result both pin
        // the planet to `P`. Mint a `GravParam<P>` from the source's
        // f64 mu at the call boundary; the caller is responsible for
        // wiring `gravity_source` to a source whose `mu` matches `P`
        // (RF.11). Misconfigurations (e.g. an Earth-orbit body whose
        // `OrbitalElementsConfigC.gravity_source` points at Sun)
        // produce numerically-wrong elements at *runtime*, not at
        // compile time — Bevy's runtime ECS link cannot enforce the
        // mu↔planet match structurally.
        let mu_p = astrodyn::GravParam::<P>::from_si(source.mu);
        // JEOD_INV: OE.01 / OE.06 / OE.07 — surface the kernel's
        // structural failure modes (non-positive μ, Kepler
        // non-convergence, degenerate orbit) as a per-cause panic
        // rather than silently writing zero orbital elements.
        let result =
            astrodyn::compute_orbital_elements_typed::<P>(mu_p, state.position, state.velocity);
        elements.0 = result.unwrap_or_else(|err| panic_for_orbital_error(entity, err));
    }
}

/// Format the per-cause panic message for a failed
/// `compute_orbital_elements_typed` call.
///
/// Diverges by `OrbitalError` variant so each panic names the broken
/// invariant, the carried diagnostic (μ value, iteration count), and a
/// concrete caller fix — required by the "Fail Loudly" convention.
///
/// Factored out of the loop body so each arm has a dedicated unit
/// test. The `KeplerConvergence` variant is structurally unreachable
/// from `compute_orbital_elements_typed` today (its kernel only walks
/// `nu → M` analytically and never invokes `kep_eqtn_e` / `kep_eqtn_h`),
/// but the match is exhaustive against the `OrbitalError` enum so a
/// future kernel change that adds an iterative path would surface
/// here. The test pins the diagnostic shape so the message stays
/// useful if that happens.
fn panic_for_orbital_error(entity: Entity, err: OrbitalError) -> ! {
    match err {
        OrbitalError::InvalidMu(mu) => panic!(
            "{entity:?} orbital elements: source gravity has μ <= 0 (got {mu}). \
             Configure a positive μ on the source body before adding orbital-elements \
             derived state."
        ),
        OrbitalError::DegenerateOrbit => panic!(
            "{entity:?} orbital elements: orbit degenerate (|h| ≈ 0, where h = r × v is \
             the specific angular momentum). Common causes: zero velocity, purely radial \
             trajectory, or position and velocity parallel. Either don't request orbital \
             elements at this instant or initialize a non-degenerate orbit."
        ),
        OrbitalError::KeplerConvergence(iters) => panic!(
            "{entity:?} orbital elements: Kepler iteration failed to converge after \
             {iters} iterations. Inspect the eccentric / hyperbolic orbit input."
        ),
    }
}

/// Compute Euler angles for entities with `EulerAnglesConfigC`.
///
/// Placed in `AstrodynSet::DerivedState`.
pub fn euler_angles_system(
    mut query: Query<(
        Option<&RotationalStateC>,
        &EulerAnglesConfigC,
        &mut EulerAnglesC,
    )>,
) {
    for (rot_opt, config, mut angles) in &mut query {
        if let Some(rot) = rot_opt {
            // The "_typed" function takes untyped input but returns
            // typed `[Angle; 3]` (the typed-output naming convention
            // documented in astrodyn::derived). Convert at the call.
            // allowed: typed↔raw kernel boundary
            let rot_untyped = astrodyn::typed_bridge::rot_typed_to_raw(&rot.0);
            angles.0 = astrodyn::compute_body_euler_angles_typed(&rot_untyped, config.sequence);
        } else {
            angles.0 = Default::default();
        }
    }
}

/// Compute LVLH frame for entities with `LvlhFrameC`.
///
/// Presence of `LvlhFrameC` alone enables computation (no separate config needed).
///
/// Placed in `AstrodynSet::DerivedState`.
pub fn lvlh_system<P: Planet>(mut query: Query<(&TranslationalStateC<P>, &mut LvlhFrameC)>) {
    for (state, mut lvlh) in &mut query {
        // `TranslationalStateC<P>` already carries `PlanetInertial<P>`,
        // matching the typed kernel's `P` parameter directly — no
        // relabel needed. LVLH stays in planet-inertial throughout
        // (no integ-origin shift).
        lvlh.0 = astrodyn::compute_body_lvlh_frame_typed::<P>(state.position, state.velocity);
    }
}

/// Compute geodetic state for entities with `GeodeticConfigC`.
///
/// Placed in `AstrodynSet::DerivedState`.
///
/// # Panics
///
/// Panics when `GeodeticConfigC.planet` does not resolve to an entity
/// carrying `PlanetFixedRotationC<P>` (the per-step inertial→pfix
/// rotation required by the geodetic kernel). The diagnostic names the
/// offending body, the planet entity that failed lookup, and the
/// remediation — silent `GeodeticState::default()` would write
/// `(lat, lon, alt) = (0, 0, 0)` (the Gulf of Guinea), geographically
/// indistinguishable from a real fix. The reference-ellipsoid radii
/// `r_eq` / `r_pol` live on the config directly (mirroring the runner's
/// `body.geodetic_planet: (idx, r_eq, r_pol)` shape) so they cannot be
/// missing at the system level — a misconfiguration there is a type
/// error, not a runtime miss. CLAUDE.md "Fail Loudly".
pub fn geodetic_system<P: Planet>(
    mut query: Query<(
        Entity,
        &TranslationalStateC<P>,
        &GeodeticConfigC,
        &mut GeodeticStateC,
    )>,
    planets: Query<&PlanetFixedRotationC<P>>,
) {
    for (entity, state, config, mut geodetic) in &mut query {
        // JEOD_INV: AT.03 — surface a `GeodeticConfigC.planet` wiring
        // miss as a per-cause panic naming the offending entity and
        // the missing component rather than silently writing zero
        // geodetic state. The previous `Default::default()` fallback
        // (lat = lon = alt = 0) is geographically indistinguishable
        // from "vehicle off the coast of Africa", so downstream
        // consumers reading the silent-default state cannot detect
        // the misconfiguration. The planet entity must carry
        // `PlanetFixedRotationC<P>` (the inertial→pfix rotation); the
        // typical misconfiguration is pointing `GeodeticConfigC.planet`
        // at a gravity-source entity spawned without a `rotation_model`
        // (so `spawn_source` skipped the rotation insert), or at a
        // non-planet entity entirely.
        let rot = planets.get(config.planet).unwrap_or_else(|_| {
            panic!(
                "{entity:?} geodetic state: \
                 GeodeticConfigC.planet = {planet_entity:?} does not resolve \
                 to PlanetFixedRotationC<{p}>. Spawn the planet source with a \
                 non-`None` `rotation_model` (or hand-insert \
                 `PlanetFixedRotationC<{p}>` on the existing planet entity) and \
                 point GeodeticConfigC.planet at it.",
                planet_entity = config.planet,
                p = std::any::type_name::<P>(),
            )
        });
        // Position is already typed `Position<PlanetInertial<P>>` —
        // matches the typed kernel's `P` directly, no relabel needed.
        // Geodetic stays in planet-inertial throughout (no integ-origin
        // shift). The ellipsoid-radii lift below is the typed-units
        // boundary on planet shape (a config-time conversion, not a
        // per-step bypass).
        use astrodyn::F64Ext;
        geodetic.0 = astrodyn::compute_body_geodetic_typed::<P>(
            state.position,
            rot.0.matrix_ref(),
            config.r_eq.m(),
            config.r_pol.m(),
        );
    }
}

/// Compute solar beta angle for entities with `SolarBetaC`.
///
/// Requires a `SunMarker` entity to exist in the world.
///
/// Generic over `P: Planet` so the body's planet-inertial state and
/// the Sun's `TranslationalStateC<P>` (which by convention stores the
/// Sun position in the body's planet-inertial frame for the
/// single-planet pipeline) match at the type level. Multi-planet
/// instantiation registers a separate Sun-state component per planet.
///
/// Placed in `AstrodynSet::DerivedState`.
///
/// # Panics
///
/// Panics when at least one body carries `SolarBetaC` but no
/// `SunMarker` entity exists in the world. Silent
/// `SolarBeta::default() = 0.0` fallback writes the
/// geometrically-plausible "perfectly noon" value into every requesting
/// body, which silently corrupts downstream consumers (thermal / power
/// / pointing budgets). The diagnostic names the affected body and the
/// two remediations (spawn a Sun source via `SunBundle`, or remove
/// `SolarBetaC` from the body). Also panics on multiple `SunMarker`
/// entities — JEOD assumes exactly one Sun. CLAUDE.md "Fail Loudly".
#[allow(clippy::type_complexity)]
pub fn solar_beta_system<P: Planet>(
    frame_origin: FrameOrigin,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    parents: Query<&ChildOf>,
    mut query: Query<
        (
            Entity,
            &TranslationalStateC<P>,
            Option<&FrameEntityC>,
            &mut SolarBetaC,
        ),
        Without<SunMarker>,
    >,
    sun_query: Query<&TranslationalStateC<P>, With<SunMarker>>,
) {
    let sun_state = match sun_query.single() {
        Ok(s) => s,
        Err(bevy::ecs::query::QuerySingleError::NoEntities(_)) => {
            // JEOD_INV: IN.09 — surface a missing `SunMarker` as a
            // per-cause panic naming the affected body and the wiring
            // fix rather than silently writing `SolarBeta = 0.0` into
            // every body's `SolarBetaC`. The previous fallback gave
            // every body the "perfectly-noon" solar-beta value, which
            // is geometrically plausible (β ∈ [-90°, 90°] includes 0)
            // and silently corrupts downstream consumers (thermal /
            // power / pointing budgets). When no body carries
            // `SolarBetaC` the query is empty and the system is a
            // no-op — the panic fires only when a body is configured
            // to read solar beta and the scenario is missing its Sun
            // source. The startup `validation` pass enforces this
            // invariant on `Added<GravityControlsC>` bodies; this
            // per-step guard catches bodies that bypass that path
            // (e.g. a body with `SolarBetaC` but no `GravityControlsC`,
            // or a `SolarBetaC` inserted after the body's spawn tick).
            if let Some((entity, _, _, _)) = query.iter().next() {
                panic!(
                    "{entity:?} solar beta: no SunMarker entity exists in the World, \
                     but {entity:?} carries SolarBetaC. Spawn a Sun source body via \
                     SunBundle (which inserts SunMarker + TranslationalStateC and \
                     registers the body in the source frame tree), or remove \
                     SolarBetaC from {entity:?}."
                );
            }
            return;
        }
        Err(bevy::ecs::query::QuerySingleError::MultipleEntities(_)) => {
            panic!(
                "Multiple entities with SunMarker found in solar_beta_system. \
                 JEOD assumes exactly one Sun body; ensure exactly one SunMarker entity exists."
            );
        }
    };
    for (_entity, state, body_frame, mut beta) in &mut query {
        // Solar beta is a root-inertial-shift consumer (RF.10): the
        // kernel mixes the body state with the Sun position in
        // absolute root-inertial coordinates. For non-root-integrated
        // bodies the body's `<PlanetInertial<P>>` storage is
        // integ-frame-relative, not absolute root-inertial — passing
        // it raw to the root-inertial kernel would compute solar beta
        // off by the inter-source separation distance. Lift to
        // absolute root-inertial via the integ-origin shift, then
        // call the typed kernel. `Angle.value` reads radians (the SI
        // base unit), so the f64 `SolarBetaC` storage is bit-identical
        // for root-integrated bodies (where the shift is zero).
        let (integ_origin, integ_origin_vel) =
            body_integ_origin_in_root(body_frame, &parents, root_frame_entity.0, &frame_origin);
        let body_pos_rel = state.position.relabel_to::<RootInertial>();
        let body_vel_rel = state.velocity.relabel_to::<RootInertial>();
        let body_pos = body_pos_rel + integ_origin;
        let body_vel = body_vel_rel + integ_origin_vel;
        // Sun is registered through `SunBundle` and integrates in the
        // root frame, so its `<PlanetInertial<P>>` storage is
        // numerically root-inertial; the relabel here is the boundary
        // step that pins the framing convention at the consumer call
        // site rather than asserting it once at registration.
        let sun_pos = sun_state.position.relabel_to::<RootInertial>();
        beta.0 = astrodyn::compute_body_solar_beta_typed(body_pos, body_vel, sun_pos).value;
    }
}

#[cfg(test)]
mod tests {
    //! Fail-loudly regressions for the derived-state systems. The
    //! orbital-elements site previously silently substituted
    //! `OrbitalElements::default()` for any kernel failure
    //! (`InvalidMu`, `DegenerateOrbit`, `KeplerConvergence`), which let
    //! geometrically-impossible `(a, e, i) = (0, 0, 0)` values reach
    //! downstream consumers. Each test now confirms the matching
    //! variant panics with a per-cause diagnostic.
    //!
    //! `InvalidMu` and `DegenerateOrbit` are driven end-to-end through
    //! `orbital_elements_system::<Earth>` because the kernel reaches
    //! both branches from plausible position/velocity inputs. The
    //! `KeplerConvergence` variant is structurally unreachable from
    //! `compute_orbital_elements_typed` (the kernel walks `nu → M`
    //! analytically and never iterates Kepler's equation), so the
    //! third test feeds an `OrbitalError::KeplerConvergence` straight
    //! into the shared `panic_for_orbital_error` formatter — the
    //! exhaustive `match` already proves the variant is wired into the
    //! same panic site, and the test pins the message shape so any
    //! future kernel change that exposes the variant surfaces a
    //! diagnostic the caller can act on.
    //!
    //! Three additional wiring-miss regressions cover the sibling
    //! silent-Default fallbacks retired in this module:
    //!
    //! - `orbital_gravity_source_lookup_miss_panics_with_caller_fix` —
    //!   `OrbitalElementsConfigC.gravity_source` resolves to an entity
    //!   that lacks `GravitySourceC`.
    //! - `geodetic_planet_lookup_miss_panics_with_caller_fix` —
    //!   `GeodeticConfigC.planet` resolves to an entity that lacks
    //!   `PlanetFixedRotationC<P>`.
    //! - `solar_beta_missing_sun_marker_panics_with_caller_fix` — a
    //!   body carries `SolarBetaC` but the World has no `SunMarker`
    //!   entity.

    use super::*;
    use crate::components::{
        GeodeticConfigC, GeodeticStateC, GravitySourceC, OrbitalElementsC, OrbitalElementsConfigC,
        SolarBetaC, TranslationalStateC,
    };
    use crate::test_utils::create_minimal_test_app;
    use astrodyn::{Earth, GravityModel, GravitySource, TranslationalState};
    use glam::DVec3;

    /// Spawn a gravity source entity carrying `mu`, then a vehicle
    /// entity with the given (position, velocity) wired to that
    /// source via `OrbitalElementsConfigC`. The two entities + an
    /// `Update` schedule that runs `orbital_elements_system::<Earth>`
    /// are enough to drive the system.
    fn spawn_vehicle_with_state(mu: f64, pos: DVec3, vel: DVec3) -> App {
        let mut app = create_minimal_test_app();
        let source = app
            .world_mut()
            .spawn(GravitySourceC(GravitySource {
                mu,
                model: GravityModel::PointMass,
            }))
            .id();
        app.world_mut().spawn((
            TranslationalStateC::<Earth>::from_untyped(TranslationalState {
                position: pos,
                velocity: vel,
            }),
            OrbitalElementsConfigC {
                gravity_source: source,
            },
            OrbitalElementsC::<Earth>::default(),
        ));
        app.add_systems(Update, orbital_elements_system::<Earth>);
        app
    }

    /// Non-positive μ on the configured `GravitySourceC` panics the
    /// orbital-elements system with the `InvalidMu` diagnostic instead
    /// of silently writing zero orbital elements.
    // JEOD_INV: OE.01 — non-positive μ in `from_cartesian_impl` returns
    // `OrbitalError::InvalidMu`, which the Bevy adapter escalates via
    // `panic_for_orbital_error` (per-cause diagnostic), so this test
    // drives the runtime failure end-to-end through
    // `orbital_elements_system::<Earth>`.
    #[test]
    #[should_panic(expected = "source gravity has \u{3bc} <= 0")]
    fn invalid_mu_panics_with_caller_fix() {
        let mut app = spawn_vehicle_with_state(
            -1.0,
            DVec3::new(7e6, 0.0, 0.0),
            DVec3::new(0.0, 7500.0, 0.0),
        );
        app.update();
    }

    /// A degenerate orbit (`|h| ≈ 0`: zero relative velocity at t=0,
    /// the classic circular-insertion fixture mentioned in the audit)
    /// panics with the `DegenerateOrbit` diagnostic instead of writing
    /// `(a, e, i) = (0, 0, 0)`. The expected substring names the
    /// geometric condition (`|h| ≈ 0`) so the diagnostic stays useful
    /// when triggered by the other r × v = 0 cause patterns (radial
    /// trajectory, parallel r and v) — the user-facing message lists
    /// all three.
    // JEOD_INV: OE.07 — zero velocity at non-zero position degenerates
    // `h = r × v` to ‖h‖ ≈ 0; `from_cartesian_impl` returns
    // `OrbitalError::DegenerateOrbit`, which the Bevy adapter escalates
    // via `panic_for_orbital_error`. Same kernel guard also fires for
    // the symmetric (zero position) case from OE.07's catalog row.
    #[test]
    #[should_panic(expected = "|h| \u{2248} 0")]
    fn degenerate_orbit_panics_with_caller_fix() {
        // mu = Earth standard; non-zero position; zero velocity → |h|≈0.
        let mut app =
            spawn_vehicle_with_state(3.986004418e14, DVec3::new(7e6, 0.0, 0.0), DVec3::ZERO);
        app.update();
    }

    /// `OrbitalError::KeplerConvergence` is structurally unreachable
    /// from `compute_orbital_elements_typed` today (see the module
    /// preamble), but the exhaustive `match` in the system wires it
    /// into `panic_for_orbital_error` so a future kernel change that
    /// adds an iterative path would surface here. The shared formatter
    /// is exercised directly with a synthetic error to pin the
    /// diagnostic shape — substring covers the variant name, the
    /// iteration count from the carrier, and the "inspect …
    /// orbit input" caller fix. The world built by `add_test_app`
    /// gives `Entity::PLACEHOLDER` a real ECS provenance for the
    /// formatter's `{entity:?}` debug print. Complements the
    /// kernel-layer panic test
    /// `astrodyn_math::orbital_elements::tests::oe_06_panics_on_kepler_non_convergence`,
    /// which drives the `Err` path via NaN-poisoned Newton iteration.
    #[test]
    #[should_panic(expected = "Kepler iteration failed to converge after 1234 iterations")]
    fn kepler_convergence_panics_with_caller_fix() {
        // JEOD_INV: OE.06 — `OrbitalError::KeplerConvergence` (raised by
        // `kep_eqtn_e` / `kep_eqtn_h` after 1000 non-converging
        // Newton-Raphson iterations) routes through
        // `panic_for_orbital_error` for the adapter-layer escalation.
        let mut app = create_minimal_test_app();
        let entity = app.world_mut().spawn_empty().id();
        panic_for_orbital_error(entity, OrbitalError::KeplerConvergence(1234));
    }

    /// `OrbitalElementsConfigC.gravity_source` pointed at an entity
    /// without `GravitySourceC` previously wrote
    /// `OrbitalElements::default()` (a = e = i = 0) into the body and
    /// continued, leaving downstream consumers unable to distinguish
    /// "wiring broken" from "geometrically-impossible orbit". The
    /// system now panics naming the failing entity and the missing
    /// `GravitySourceC` so the caller can fix the wiring at spawn
    /// time. The bait entity here is a bare `spawn_empty()` — the
    /// minimal misconfiguration shape, which is the most common
    /// real-world failure mode (the caller spawned a planet entity
    /// before the bundle that inserts `GravitySourceC` ran).
    #[test]
    #[should_panic(expected = "does not resolve to a GravitySourceC component")]
    fn orbital_gravity_source_lookup_miss_panics_with_caller_fix() {
        let mut app = create_minimal_test_app();
        let bad_source = app.world_mut().spawn_empty().id();
        app.world_mut().spawn((
            TranslationalStateC::<Earth>::from_untyped(TranslationalState {
                position: DVec3::new(7e6, 0.0, 0.0),
                velocity: DVec3::new(0.0, 7500.0, 0.0),
            }),
            OrbitalElementsConfigC {
                gravity_source: bad_source,
            },
            OrbitalElementsC::<Earth>::default(),
        ));
        app.add_systems(Update, orbital_elements_system::<Earth>);
        app.update();
    }

    /// `GeodeticConfigC.planet` pointed at an entity without
    /// `PlanetFixedRotationC<P>` previously wrote
    /// `GeodeticState::default()` (lat = lon = alt = 0 — the Gulf of
    /// Guinea) into the body and continued. The system now panics
    /// naming the failing entity and the missing component, so the
    /// caller can fix the wiring at spawn time. The bait entity is a
    /// bare `spawn_empty()` (no `PlanetFixedRotationC`), the minimal
    /// misconfiguration shape.
    // JEOD_INV: AT.03 — planet-fixed rotation required for geodetic altitude
    #[test]
    #[should_panic(expected = "does not resolve to PlanetFixedRotationC")]
    fn geodetic_planet_lookup_miss_panics_with_caller_fix() {
        let mut app = create_minimal_test_app();
        let bad_planet = app.world_mut().spawn_empty().id();
        app.world_mut().spawn((
            TranslationalStateC::<Earth>::from_untyped(TranslationalState {
                position: DVec3::new(7e6, 0.0, 0.0),
                velocity: DVec3::ZERO,
            }),
            GeodeticConfigC {
                planet: bad_planet,
                r_eq: astrodyn::EARTH.shape.r_eq(),
                r_pol: astrodyn::EARTH.shape.r_pol(),
            },
            GeodeticStateC::default(),
        ));
        app.add_systems(Update, geodetic_system::<Earth>);
        app.update();
    }

    /// A body carrying `SolarBetaC` in a world with no `SunMarker`
    /// entity previously had `SolarBeta::default() = 0.0` written to
    /// every step — the geometrically-plausible "perfectly noon"
    /// value that silently corrupts thermal / power / pointing
    /// budgets. The system now panics naming the affected body and
    /// the two remediations (spawn a Sun via `SunBundle`, or remove
    /// `SolarBetaC` from the body). The `RootFrameEntityR` resource
    /// is required by the system's signature but the panic fires
    /// before any frame-tree access, so a placeholder entity
    /// (no `FrameTransC` / `FrameRotC` / `FrameAngVelC`) suffices.
    #[test]
    #[should_panic(expected = "no SunMarker entity exists in the World")]
    fn solar_beta_missing_sun_marker_panics_with_caller_fix() {
        let mut app = create_minimal_test_app();
        let root_frame_e = app.world_mut().spawn_empty().id();
        app.insert_resource(crate::RootFrameEntityR(root_frame_e));
        app.world_mut().spawn((
            TranslationalStateC::<Earth>::from_untyped(TranslationalState {
                position: DVec3::new(7e6, 0.0, 0.0),
                velocity: DVec3::new(0.0, 7500.0, 0.0),
            }),
            SolarBetaC::default(),
        ));
        // Deliberately no SunMarker / SunBundle in the world; the
        // `Without<SunMarker>` body still appears in the system's
        // query, and the missing-Sun branch panics naming it.
        // JEOD_INV: IN.09 — body carrying SolarBetaC without a
        // SunMarker in the World; per-step safety net inside
        // solar_beta_system (the startup `validation` pass is the
        // first line, this is the fallback for bodies that bypass it).
        app.add_systems(Update, solar_beta_system::<Earth>);
        app.update();
    }

    /// Regression for the `populate_app`-shape geodetic scenario
    /// (e.g. the SIM_NED Tier 3 family) that previously panicked
    /// at the geodetic site because the gravity-source entity was
    /// spawned with `PlanetFixedRotationC<P>` (from the source's
    /// `rotation_model`) but **without** `PlanetC` (the source
    /// doesn't carry planet shape — `mu` only). The geodetic
    /// system now reads ellipsoid radii from `GeodeticConfigC`
    /// directly, mirroring the runner's
    /// `body.geodetic_planet: (idx, r_eq, r_pol)` shape, so this
    /// arrangement runs without a `PlanetC` on the planet entity.
    /// One tick is enough to drive the system; the assertion is
    /// just "does not panic and writes a non-default altitude".
    #[test]
    fn geodetic_system_runs_when_planet_entity_has_rotation_but_no_planet_shape() {
        use crate::components::{GeodeticConfigC, GeodeticStateC, PlanetFixedRotationC};
        use astrodyn::{FrameTransform, EARTH};

        let mut app = create_minimal_test_app();
        // Planet entity carries `PlanetFixedRotationC<P>` (what
        // `spawn_source` inserts when `rotation_model != None`) but
        // *not* `PlanetC` (what `PlanetBundle::from_config` inserts).
        let planet = app
            .world_mut()
            .spawn(PlanetFixedRotationC::<Earth>(FrameTransform::from_matrix(
                glam::DMat3::IDENTITY,
            )))
            .id();
        let vehicle = app
            .world_mut()
            .spawn((
                TranslationalStateC::<Earth>::from_untyped(TranslationalState {
                    position: DVec3::new(EARTH.shape.r_eq() + 400_000.0, 0.0, 0.0),
                    velocity: DVec3::new(0.0, 7500.0, 0.0),
                }),
                GeodeticConfigC {
                    planet,
                    r_eq: EARTH.shape.r_eq(),
                    r_pol: EARTH.shape.r_pol(),
                },
                GeodeticStateC::default(),
            ))
            .id();
        app.add_systems(Update, geodetic_system::<Earth>);
        app.update();

        let geo = app
            .world()
            .get::<GeodeticStateC>(vehicle)
            .expect("GeodeticStateC must remain after one tick")
            .0;
        assert!(
            geo.altitude > 1.0e5,
            "geodetic altitude = {} m — expected ~4e5 m, indicating the kernel ran",
            geo.altitude,
        );
    }
}
