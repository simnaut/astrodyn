//! Runtime validation of JEOD invariants.
//!
//! Delegates to [`jeod_sim::validate_body`] for the per-body invariant checks,
//! wrapping results with Bevy entity context and panicking on errors.
//!
//! This system fires whenever a body with [`GravityControlsC`] is added —
//! once per startup tick (covering the bodies spawned at app build) and
//! again when bodies are spawned mid-mission (staging, dynamic constellation
//! growth). Bodies added without `GravityControlsC` are not validated; this
//! matches JEOD's `DynManager` which only invariants bodies that participate
//! in integration.

use bevy::prelude::*;

use crate::components::{
    CannonballSrpC, DragConfigC, DynamicsConfigC, EarthLightingConfigC, EulerAnglesConfigC,
    FlatPlateConfigC, FrameSwitchesC, GeodeticConfigC, GravityAccelerationC, GravityControlsC,
    GravitySourceC, IntegFrameIdC, LvlhFrameC, MassPropertiesC, MoonMarker, OrbitalElementsConfigC,
    RotationalStateC, SolarBetaC, SourceFrameIdC, SunMarker, TidalConfigC, TidalDeltaC20C,
    TranslationalStateC,
};
use crate::RootFrameIdR;

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
/// Two scopes participate:
///
/// * **Global state checks** (Sun/Moon marker counts, tidal-config pairing
///   on gravity sources) iterate the *unfiltered* `derived_state_markers`
///   and `tidal_sources` queries, so they re-evaluate the entire world's
///   marker/source set on every trigger. Adding a stray second
///   `SunMarker` mid-mission is therefore caught the next tick a body
///   with `GravityControlsC` is added.
/// * **Per-body invariant checks** (SRP mutual exclusion, the full
///   `jeod_sim::validate_body` pass, gravity-control `check_validity`
///   auto-corrections) iterate the `Added`-filtered `bodies` query, so
///   they validate only newly-attached bodies. Existing bodies were
///   validated on the tick they first appeared, and the per-body
///   invariants do not depend on inter-body state, so re-running them
///   for unchanged bodies would be wasteful.
///
/// Delegates per-body checks to [`jeod_sim::validate_body`] and applies
/// gravity control auto-corrections via `check_validity()`.
///
/// # Panics
/// Panics with a descriptive message for any violated invariant.
// JEOD_INV: DM.03 — `Added<GravityControlsC>` filter on the body query fires on every body addition; bodies added mid-simulation are validated on the following tick
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn validate_jeod_invariants(
    mut bodies: Query<
        (
            Entity,
            &DynamicsConfigC,
            &mut GravityControlsC,
            Option<&GravityAccelerationC>,
            Option<&MassPropertiesC>,
            Option<&RotationalStateC>,
            Option<&TranslationalStateC>,
            Option<&FlatPlateConfigC>,
        ),
        Added<GravityControlsC>,
    >,
    sources: Query<(Entity, &GravitySourceC)>,
    tidal_sources: Query<(
        Entity,
        &TidalConfigC,
        Option<&TidalDeltaC20C>,
        Option<&crate::components::PlanetFixedRotationC>,
    )>,
    srp_exclusion: Query<Entity, With<CannonballSrpC>>,
    derived_state_markers: Query<(
        Entity,
        Option<&SolarBetaC>,
        Option<&EarthLightingConfigC>,
        Option<&SunMarker>,
        Option<&MoonMarker>,
        Option<&TranslationalStateC>,
    )>,
    // Frame-tree state for non-root validation. PR #260 round-8 fixup:
    // mirrors `Simulation::validate()`'s frame-switch + non-root checks
    // (`crates/jeod_runner/src/simulation/validate.rs:129-184`) so the
    // typed `VehicleBuilder` API (which now plumbs `integ_source` and
    // `frame_switches` through `spawn_bevy`) doesn't silently land
    // bodies in misconfigured non-root setups.
    body_frame_state: Query<(Option<&IntegFrameIdC>, Option<&FrameSwitchesC>)>,
    source_frames: Query<&SourceFrameIdC>,
    root_fid: Option<Res<RootFrameIdR>>,
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

        // Delegate structural validation to jeod_sim. The kernel
        // signature still consumes the untyped forms; convert at the
        // boundary. (Per-step calls in this validation system are rare
        // — runs once at startup — so the per-call conversion cost is
        // negligible compared to the typed-storage win.)
        let mass_untyped = mass.map(|m| m.0.to_untyped());
        let trans_untyped = trans_state.map(|t| t.0.to_untyped());
        let errors = jeod_sim::validate_body(
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
                bevy::log::warn!("Entity {entity:?}: {error}");
            } else {
                panic!("Entity {entity:?}: {error}");
            }
        }

        // ── Frame-switch + non-root frame validation ──
        // Mirrors `Simulation::validate()` checks at
        // `crates/jeod_runner/src/simulation/validate.rs:129-184` (PR #260
        // round-8 fixup): every active `FrameSwitchConfig.target_source`
        // must (a) be a registered gravity source and (b) appear in the
        // body's `gravity_controls` so the post-switch differential flip
        // leaves a non-differential central body. Bodies whose
        // integration frame is non-root, or which have an active switch
        // into a non-root frame, also warn when carrying root-dependent
        // features (drag, SRP, orbital elements, etc.).
        let (integ_frame, switches) = body_frame_state.get(entity).unwrap_or((None, None));
        let root_fid_value = root_fid.as_ref().map(|r| r.0);
        let non_root_integ = match (integ_frame, root_fid_value) {
            (Some(ifc), Some(root)) => ifc.0 != root,
            _ => false,
        };
        let mut non_root_switch = false;
        if let Some(switches) = switches {
            for sw in &switches.0 {
                if !sw.active {
                    continue;
                }
                // (a) target_source must be a registered gravity source
                // (Bevy analog of runner's `target >= num_sources` check —
                // here a missing `SourceFrameIdC` is the failure mode).
                let target_fid = match source_frames.get(sw.target_source) {
                    Ok(s) => Some(s.0),
                    Err(_) => {
                        panic!(
                            "Entity {entity:?}: FrameSwitchConfig.target_source = {target:?} \
                             is not a registered gravity source (no SourceFrameIdC). \
                             Spawn the source with PlanetBundle (or attach GravitySourceC + \
                             SourceInertialPositionC) before adding the body.",
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
                if let (Some(tf), Some(root)) = (target_fid, root_fid_value) {
                    if tf != root {
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
                bevy::log::warn!(
                    "Entity {entity:?}: non-root integration frame (or active \
                     frame switch into a non-root frame) with features that \
                     assume root-inertial coordinates (drag={has_drag}, \
                     flat_plate_srp={has_flat}, cannonball_srp={has_cannonball}, \
                     orbital_elements={has_orbital}, euler={has_euler}, \
                     geodetic={has_geodetic}, lvlh={has_lvlh}, \
                     solar_beta={has_solar_beta}, earth_lighting={has_earth_lighting}). \
                     These derived states assume the simulation's central-body inertial \
                     frame and will produce incorrect results in other frames.",
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
