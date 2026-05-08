//! Bevy systems for [`AstrodynSet::DerivedState`](crate::AstrodynSet::DerivedState).
//!
//! Per-step derived states: orbital elements, Euler angles, LVLH frame,
//! geodetic state, and solar beta angle.

use astrodyn::{Planet, RootInertial};
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
pub fn orbital_elements_system<P: Planet>(
    mut query: Query<(
        &TranslationalStateC<P>,
        &OrbitalElementsConfigC,
        &mut OrbitalElementsC<P>,
    )>,
    sources: Query<&GravitySourceC>,
) {
    for (state, config, mut elements) in &mut query {
        let Ok(source) = sources.get(config.gravity_source) else {
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
        match astrodyn::compute_orbital_elements_typed::<P>(mu_p, state.position, state.velocity) {
            Ok(oe) => elements.0 = oe,
            Err(_) => elements.0 = Default::default(),
        }
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
            let rot_untyped = rot.0.to_untyped();
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
