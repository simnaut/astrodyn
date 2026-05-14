//! Runtime validation of JEOD invariants.
//!
//! Delegates to [`astrodyn::validate_body`] for the per-body invariant checks,
//! wrapping results with Bevy entity context and panicking on errors.
//!
//! This system fires whenever a body with [`GravityControlsC`] is added —
//! once per startup tick (covering the bodies spawned at app build) and
//! again when bodies are spawned mid-mission (staging, dynamic constellation
//! growth). Bodies added without `GravityControlsC` are not validated; this
//! matches JEOD's `DynManager` which only invariants bodies that participate
//! in integration.
//!
//! Non-root integration-frame checks walk the ECS frame hierarchy via
//! `Query<&ChildOf>` on the body's [`FrameEntityC`] — there is no
//! explicit integration-frame handle component to read.

use astrodyn::Planet;
use bevy::prelude::*;

use crate::components::{
    CannonballSrpC, DragConfigC, DynamicsConfigC, EarthLightingConfigC, EulerAnglesConfigC,
    FlatPlateConfigC, FrameAngVelC, FrameEntityC, FrameRotC, FrameSwitchesC, FrameTransC,
    GeodeticConfigC, GravityAccelerationC, GravityControlsC, GravitySourceC, LvlhFrameC,
    MassPropertiesC, MoonMarker, OrbitalElementsConfigC, PlanetFixedRotationC, RotationalStateC,
    SolarBetaC, SunMarker, TidalConfigC, TidalDeltaC20C, TranslationalStateC,
};
use crate::RootFrameEntityR;

/// In `astrodyn_runner` the central body's inertial frame *is* the root
/// frame (the runner renames the root to `<central>.inertial`), so
/// `body.integ_frame_id != root_frame_id` cleanly distinguishes
/// "central" from "third body". The Bevy adapter instead registers
/// every gravity source — including whatever the mission treats as
/// central — as a child of a generic root inertial frame, so a
/// perfectly normal Earth-centered body with `IntegSourceC(Some(earth))`
/// is one level below root and would trip the "non-root integration"
/// check despite being numerically root-equivalent.
///
/// This helper folds the Bevy topology back onto the runner's
/// semantics: a frame entity counts as root-equivalent when it is the
/// root entity itself, or when it is a direct child of the root entity
/// whose stored frame state is identity (zero position/velocity,
/// identity rotation, zero angular velocity). The check is evaluated
/// against the validation-tick snapshot of the frame components —
/// sources whose state is non-zero at startup (Moon at 384 Mm,
/// ephemeris-driven bodies at their epoch position) correctly remain
/// "non-root" and continue to surface the drag/SRP/etc. warning the
/// non-root check is meant to catch.
pub(crate) fn is_root_equivalent_entity(
    frame_entity: Entity,
    root_entity: Entity,
    parents: &Query<&ChildOf>,
    frame_states: &Query<(&FrameTransC, &FrameRotC, &FrameAngVelC)>,
) -> bool {
    if frame_entity == root_entity {
        return true;
    }
    let Ok(child_of) = parents.get(frame_entity) else {
        return false;
    };
    if child_of.parent() != root_entity {
        return false;
    }
    let Ok((trans, rot, ang_vel)) = frame_states.get(frame_entity) else {
        return false;
    };
    trans.position == glam::DVec3::ZERO
        && trans.velocity == glam::DVec3::ZERO
        && rot.t_parent_this == glam::DMat3::IDENTITY
        && ang_vel.0 == glam::DVec3::ZERO
}

/// Validates JEOD invariants on dynamic body entities.
///
/// Triggered by [`Added<GravityControlsC>`] on the body query: the system
/// short-circuits to a no-op on ticks where no body with
/// `GravityControlsC` has been newly attached. When the trigger fires,
/// the system runs the validation pass below; bodies attached to the app
/// at build time are validated on the first `FixedUpdate` tick, and
/// bodies spawned later (staging, hot-attach, runtime spawn events) are
/// validated on the tick following their insertion.
///
/// Generic over `P: Planet` so each planet-tagged pipeline gets its own
/// validator. `AstrodynPlugin::build` registers `<astrodyn::Earth>` (preserving
/// single-planet-Earth behavior); `register_planet_systems::<P>` registers
/// the additional instance for multi-planet missions. Each instantiation
/// queries `Option<&TranslationalStateC<P>>` and
/// `Option<&PlanetFixedRotationC<P>>` so per-planet trans-state and
/// tidal-rotation checks fire on the matching planet only — a body tagged
/// `<Mars>` is validated by the `<Mars>` registration, not silently passed
/// by an Earth-pinned validator.
///
/// Two scopes participate:
///
/// * **Global state checks** (Sun/Moon marker counts, tidal-config pairing
///   on gravity sources) iterate the *unfiltered* `derived_state_markers`
///   and `tidal_sources` queries, so they re-evaluate the entire world's
///   marker/source set on every trigger. Adding a stray second
///   `SunMarker` mid-mission is therefore caught the next tick a body
///   with `GravityControlsC` is added. In multi-planet configs these
///   global checks run once per registered planet — wasteful but
///   idempotent (each pass evaluates the same world state and reaches
///   the same conclusion).
/// * **Per-body invariant checks** (SRP mutual exclusion, the full
///   `astrodyn::validate_body` pass, gravity-control `check_validity`
///   auto-corrections) iterate the `Added`-filtered `bodies` query, so
///   they validate only newly-attached bodies. Existing bodies were
///   validated on the tick they first appeared, and the per-body
///   invariants do not depend on inter-body state, so re-running them
///   for unchanged bodies would be wasteful.
///
/// Delegates per-body checks to [`astrodyn::validate_body`] and applies
/// gravity control auto-corrections via `check_validity()`.
///
/// # Panics
/// Panics with a descriptive message for any violated invariant.
// JEOD_INV: DM.03 — `Added<GravityControlsC>` filter on the body query fires on every body addition; bodies added mid-simulation are validated on the following tick
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn validate_jeod_invariants<P: Planet>(
    mut bodies: Query<
        (
            Entity,
            &DynamicsConfigC,
            &mut GravityControlsC,
            Option<&GravityAccelerationC>,
            Option<&MassPropertiesC>,
            Option<&RotationalStateC>,
            Option<&TranslationalStateC<P>>,
            Option<&FlatPlateConfigC>,
        ),
        Added<GravityControlsC>,
    >,
    sources: Query<(Entity, &GravitySourceC)>,
    tidal_sources: Query<(
        Entity,
        &TidalConfigC,
        Option<&TidalDeltaC20C>,
        Option<&PlanetFixedRotationC<P>>,
    )>,
    srp_exclusion: Query<Entity, With<CannonballSrpC>>,
    derived_state_markers: Query<(
        Entity,
        Option<&SolarBetaC>,
        Option<&EarthLightingConfigC>,
        Option<&SunMarker>,
        Option<&MoonMarker>,
        Option<&TranslationalStateC<P>>,
    )>,
    // Frame-tree state for non-root validation. Mirrors
    // `Simulation::validate()`'s frame-switch + non-root checks
    // (`crates/astrodyn_runner/src/simulation/validate.rs:129-184`) so the
    // typed `VehicleBuilder` API (which plumbs `integ_source` and
    // `frame_switches` through `spawn_bevy`) doesn't silently land
    // bodies in misconfigured non-root setups. The body's integration
    // frame is its `FrameEntityC`'s `ChildOf` parent; gravity sources
    // expose their frame entity via `FrameEntityC` for the
    // frame-switch target check.
    body_frame_state: Query<(Option<&FrameEntityC>, Option<&FrameSwitchesC>)>,
    source_frames: Query<&FrameEntityC, With<GravitySourceC>>,
    parents: Query<&ChildOf>,
    frame_states: Query<(&FrameTransC, &FrameRotC, &FrameAngVelC)>,
    // Needed by `is_root_equivalent_entity` to fold Bevy's "every
    // source is a child of root" topology back onto the runner's
    // "central body's inertial == root" semantics.
    root_frame_entity: Option<Res<RootFrameEntityR>>,
    // Root-dependent features. Presence of any of these on a body that
    // integrates in (or switches into) a non-root frame produces a
    // warning — the underlying systems consume root-inertial state.
    #[allow(clippy::type_complexity)] root_dependent_features: Query<(
        Has<DragConfigC>,
        Has<FlatPlateConfigC>,
        Has<CannonballSrpC>,
        Has<OrbitalElementsConfigC>,
        Has<EulerAnglesConfigC>,
        Has<GeodeticConfigC>,
        Has<LvlhFrameC>,
        Has<SolarBetaC>,
        Has<EarthLightingConfigC>,
    )>,
) {
    if bodies.is_empty() {
        // No body with `GravityControlsC` has appeared this tick — nothing
        // new to validate. Existing bodies were validated on the tick they
        // were spawned and the work is otherwise idempotent.
        return;
    }

    // Validate derived-state marker prerequisites.
    // Matches Simulation::validate() which errors on missing sun_source/moon_source.
    // Count markers and validate they have TranslationalStateC (required by
    // solar_beta_system/earth_lighting_system queries).
    let mut sun_count = 0;
    let mut moon_count = 0;
    for (entity, _, _, sun, moon, trans) in &derived_state_markers {
        if sun.is_some() {
            sun_count += 1;
            assert!(
                trans.is_some(),
                "Entity {entity:?}: SunMarker present but TranslationalStateC is missing. \
                 Sun entity requires TranslationalStateC for position queries."
            );
        }
        if moon.is_some() {
            moon_count += 1;
            assert!(
                trans.is_some(),
                "Entity {entity:?}: MoonMarker present but TranslationalStateC is missing. \
                 Moon entity requires TranslationalStateC for position queries."
            );
        }
    }
    assert!(
        sun_count <= 1,
        "Multiple SunMarker entities found. JEOD assumes exactly one Sun body."
    );
    assert!(
        moon_count <= 1,
        "Multiple MoonMarker entities found. JEOD assumes exactly one Moon body."
    );
    for (entity, solar_beta, earth_lighting, _, _, _) in &derived_state_markers {
        if solar_beta.is_some() && sun_count == 0 {
            panic!(
                "Entity {entity:?}: SolarBetaC present but no SunMarker entity exists. \
                 Solar beta computation requires exactly one SunMarker entity."
            );
        }
        if earth_lighting.is_some() {
            if sun_count == 0 {
                panic!(
                    "Entity {entity:?}: EarthLightingConfigC present but no SunMarker entity. \
                     Earth lighting requires both SunMarker and MoonMarker entities."
                );
            }
            if moon_count == 0 {
                panic!(
                    "Entity {entity:?}: EarthLightingConfigC present but no MoonMarker entity. \
                     Earth lighting requires both SunMarker and MoonMarker entities."
                );
            }
        }
    }

    // Validate tidal component pairing on gravity sources.
    for (entity, _config, delta, rotation) in &tidal_sources {
        assert!(
            delta.is_some(),
            "Entity {entity:?}: TidalConfigC is present but TidalDeltaC20C is missing. \
             Add TidalDeltaC20C::default() to the entity so tidal_update_system can write ΔC20."
        );
        assert!(
            rotation.is_some(),
            "Entity {entity:?}: TidalConfigC is present but PlanetFixedRotationC is missing. \
             tidal_update_system requires PlanetFixedRotationC to transform tidal body \
             positions into the planet-fixed frame."
        );
    }

    // Validate SRP mutual exclusion: CannonballSrpC and FlatPlateConfigC
    // must not coexist on the same entity (both write RadiationForceC).
    for (entity, _, _, _, _, _, _, flat_plates) in &bodies {
        if flat_plates.is_some() && srp_exclusion.get(entity).is_ok() {
            panic!(
                "Entity {entity:?}: both FlatPlateConfigC and CannonballSrpC are present. \
                 These are mutually exclusive — use one SRP model per entity."
            );
        }
    }

    for (entity, config, mut controls, grav_accel, mass, rot_state, trans_state, flat_plates) in
        &mut bodies
    {
        // Compute plate counts for validation
        let plate_counts = flat_plates.map(|fp| {
            (
                fp.plates.len(),
                fp.temperatures.len(),
                fp.t_pow4_cached.len(),
            )
        });

        // Delegate structural validation to astrodyn. The kernel
        // signature still consumes the untyped forms; convert at the
        // boundary. (Per-step calls in this validation system are rare
        // — runs once at startup — so the per-call conversion cost is
        // negligible compared to the typed-storage win.)
        // allowed: typed↔raw kernel boundary
        let mass_untyped = mass.map(|m| astrodyn::typed_bridge::mass_typed_to_raw(&m.0));
        let trans_untyped = trans_state.map(|t| astrodyn::typed_bridge::trans_typed_to_raw(&t.0));
        let errors = astrodyn::validate_body(
            config,
            &controls.0,
            grav_accel.is_some(),
            mass_untyped.as_ref(),
            rot_state.is_some(),
            trans_untyped.as_ref(),
            |source_entity| sources.get(source_entity).ok().map(|(_, source)| &source.0),
            plate_counts,
        );

        for error in &errors {
            if error.is_warning() {
                // FAIL_LOUD_EXEMPT: operational report path for warning-class
                // ValidationErrors (see `ValidationError::is_warning`). The
                // warning category is reserved for suspicious-but-valid
                // states (uninitialized at origin, non-root integ with proper
                // IntegOrigin shifts per RF.10) where a panic would
                // false-positive on legitimate setups.
                bevy::log::warn!("Entity {entity:?}: {error}");
            } else {
                panic!("Entity {entity:?}: {error}");
            }
        }

        // ── Frame-switch + non-root frame validation ──
        // Mirrors `Simulation::validate()` checks at
        // `crates/astrodyn_runner/src/simulation/validate.rs:129-184`: every
        // active `FrameSwitchConfig.target_source`
        // must (a) be a registered gravity source and (b) appear in the
        // body's `gravity_controls` so the post-switch differential flip
        // leaves a non-differential central body. Bodies whose
        // integration frame is non-root, or which have an active switch
        // into a non-root frame, also warn when carrying root-dependent
        // features (drag, SRP, orbital elements, etc.).
        let (body_frame_handle, switches) = body_frame_state.get(entity).unwrap_or((None, None));
        let root_entity_value = root_frame_entity.as_ref().map(|r| r.0);
        // Resolve the body's current integration frame entity via
        // `Query<&ChildOf>` on its frame entity. Bodies registered
        // before the frames-as-entities components landed have no
        // `FrameEntityC` — treat those as root-integrated (the
        // pre-migration default) so they don't trip the warning.
        let body_integ_frame_entity = body_frame_handle
            .and_then(|fe| parents.get(fe.0).ok().map(|child_of| child_of.parent()));
        // Use `is_root_equivalent_entity` instead of raw entity equality so
        // Earth-centered bodies with `IntegSourceC(Some(earth))`
        // (Earth.inertial sits one level below the generic root with
        // identity state) don't trip the warning. See helper doc.
        let non_root_integ = match (body_integ_frame_entity, root_entity_value) {
            (Some(integ_e), Some(root_e)) => {
                !is_root_equivalent_entity(integ_e, root_e, &parents, &frame_states)
            }
            _ => false,
        };
        let mut non_root_switch = false;
        if let Some(switches) = switches {
            for sw in &switches.0 {
                if !sw.active {
                    continue;
                }
                // (a) target_source must be a registered gravity source
                // (Bevy analog of runner's `target >= num_sources` check).
                // The `source_frames` query is filtered by
                // `With<GravitySourceC>`, so a missing match means the
                // target is missing `GravitySourceC` and/or
                // `FrameEntityC`; the diagnostic enumerates both
                // because either misconfiguration produces the same
                // failure here.
                let target_frame_entity = match source_frames.get(sw.target_source) {
                    Ok(fe) => Some(fe.0),
                    Err(_) => {
                        panic!(
                            "Entity {entity:?}: FrameSwitchConfig.target_source = {target:?} \
                             is not a registered gravity source — it is missing \
                             GravitySourceC and/or FrameEntityC. Spawn it with PlanetBundle \
                             (which inserts both) before adding the body.",
                            target = sw.target_source,
                        );
                    }
                };
                // (b) target_source must appear in the body's
                // gravity_controls — without it, the post-switch
                // `differential = true` flip leaves no central body and
                // the body integrates under the wrong gravity model.
                if !controls
                    .0
                    .controls
                    .iter()
                    .any(|c| c.source_name == sw.target_source)
                {
                    panic!(
                        "Entity {entity:?}: FrameSwitchConfig.target_source = {target:?} \
                         is not in the body's GravityControlsC. The post-switch gravity \
                         reclassification needs the target source to have a GravityControl \
                         entry (it becomes the non-differential central body). Add \
                         GravityControl::new_spherical({target:?}, ...) to the body's \
                         controls before configuring the switch.",
                        target = sw.target_source,
                    );
                }
                // Same root-equivalence adjustment as `non_root_integ`
                // above — switching back to the central body in Bevy
                // still produces a target frame entity one level below
                // root, but with identity state, so it is numerically
                // root-equivalent and must not trip the warning.
                if let (Some(tfe), Some(root_e)) = (target_frame_entity, root_entity_value) {
                    if !is_root_equivalent_entity(tfe, root_e, &parents, &frame_states) {
                        non_root_switch = true;
                    }
                }
            }
        }
        if non_root_integ || non_root_switch {
            let (
                has_drag,
                has_flat,
                has_cannonball,
                has_orbital,
                has_euler,
                has_geodetic,
                has_lvlh,
                has_solar_beta,
                has_earth_lighting,
            ) = root_dependent_features.get(entity).unwrap_or_default();
            let has_root_dependent = has_drag
                || has_flat
                || has_cannonball
                || has_orbital
                || has_euler
                || has_geodetic
                || has_lvlh
                || has_solar_beta
                || has_earth_lighting;
            if has_root_dependent {
                // FAIL_LOUD_EXEMPT: Bevy-side mirror of the runner-side
                // `NonRootFrameWithRootDependentFeatures` warning-class
                // ValidationError. The configuration is supported (see
                // `tests/integ_frame_translation_invariance.rs` for a
                // correctly-handled non-root setup with the same feature
                // set) provided the caller applies the per-step
                // `IntegOrigin` shift per RF.10. Promoting to panic
                // would false-positive on legitimate setups.
                bevy::log::warn!(
                    "Entity {entity:?}: non-root integration frame (or active \
                     frame switch into a non-root frame) paired with features \
                     that consume root-inertial coordinates (drag={has_drag}, \
                     flat_plate_srp={has_flat}, cannonball_srp={has_cannonball}, \
                     orbital_elements={has_orbital}, euler={has_euler}, \
                     geodetic={has_geodetic}, lvlh={has_lvlh}, \
                     solar_beta={has_solar_beta}, earth_lighting={has_earth_lighting}). \
                     The consumer must apply the per-step `IntegOrigin` shift \
                     (RF.10) to convert the body's integration-frame state into \
                     root-inertial before the consumer runs; without that shift \
                     the consumer mixes coordinates from different frames.",
                );
            }
        }

        // ── Gravity control auto-corrections ──
        // JEOD_INV: GV.03 — check_validity() called at startup
        // This mutates controls (degree/order clamping), so it must be done
        // after validation, not inside validate_body().
        for ctrl in &mut controls.0.controls {
            if let Ok((_source_entity, source)) = sources.get(ctrl.source_name) {
                ctrl.check_validity(&source.0);
            }
        }
    }
}
