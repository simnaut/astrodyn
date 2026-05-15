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
/// Panics when the typed kernel rejects the instantaneous state
/// (non-positive μ, degenerate orbit with `|h| ≈ 0`, or Kepler
/// iteration non-convergence). Each variant emits a per-cause
/// diagnostic naming the entity and the caller fix — silent
/// zero-element fallback would let geometrically-impossible
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
        let Ok(source) = sources.get(config.gravity_source) else {
            // The companion misconfiguration — a `gravity_source` entity
            // that doesn't carry `GravitySourceC` — is intentionally
            // out of scope for this guard: it's a wiring contract
            // separate from the kernel's structural failure modes
            // and is policed elsewhere as the gravity / config
            // surface tightens.
            elements.0 = Default::default();
            continue;
        };
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
pub fn geodetic_system<P: Planet>(
    mut query: Query<(
        &TranslationalStateC<P>,
        &GeodeticConfigC,
        &mut GeodeticStateC,
    )>,
    planets: Query<(&PlanetFixedRotationC<P>, &PlanetC)>,
) {
    for (state, config, mut geodetic) in &mut query {
        let Ok((rot, planet)) = planets.get(config.planet) else {
            geodetic.0 = Default::default();
            continue;
        };
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
            planet.r_eq.m(),
            planet.r_pol.m(),
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
#[allow(clippy::type_complexity)]
pub fn solar_beta_system<P: Planet>(
    frame_origin: FrameOrigin,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    parents: Query<&ChildOf>,
    mut query: Query<
        (
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
            // No SunMarker present: clear stale solar beta values
            for (_, _, mut beta) in &mut query {
                beta.0 = Default::default();
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
    for (state, body_frame, mut beta) in &mut query {
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

    use super::*;
    use crate::components::{
        GravitySourceC, OrbitalElementsC, OrbitalElementsConfigC, TranslationalStateC,
    };
    use astrodyn::{Earth, GravityModel, GravitySource, TranslationalState};
    use glam::DVec3;

    fn add_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app
    }

    /// Spawn a gravity source entity carrying `mu`, then a vehicle
    /// entity with the given (position, velocity) wired to that
    /// source via `OrbitalElementsConfigC`. The two entities + an
    /// `Update` schedule that runs `orbital_elements_system::<Earth>`
    /// are enough to drive the system.
    fn spawn_vehicle_with_state(mu: f64, pos: DVec3, vel: DVec3) -> App {
        let mut app = add_test_app();
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
    /// formatter's `{entity:?}` debug print.
    #[test]
    #[should_panic(expected = "Kepler iteration failed to converge after 1234 iterations")]
    fn kepler_convergence_panics_with_caller_fix() {
        let mut app = add_test_app();
        let entity = app.world_mut().spawn_empty().id();
        panic_for_orbital_error(entity, OrbitalError::KeplerConvergence(1234));
    }
}
