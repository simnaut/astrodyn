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
/// * **Per-`<P>` global state checks** (Sun/Moon marker counts, tidal-config
///   pairing on gravity sources) iterate the unfiltered `derived_state_markers`
///   and `tidal_sources` queries, then narrow each entity to those whose
///   `TranslationalStateC<P>` matches *this* validator's `<P>`. The
///   marker count therefore counts only Sun/Moon entities tagged for
///   this planet's integration frame. Adding a stray second
///   `<P>`-tagged `SunMarker` mid-mission is caught the next tick a
///   body with `GravityControlsC` is added under the same `<P>`. In
///   multi-planet configs each `<P>` validator counts only its own
///   `<P>`-tagged markers — the `<Mars>` validator catches a duplicate
///   Mars-tagged Sun, the `<Earth>` validator catches a duplicate
///   Earth-tagged Sun, neither steps on the other. Without this
///   `<P>`-scoping, the `<Earth>` validator that `AstrodynPlugin::build`
///   registers by default would panic on a Sun tagged for any other
///   planet's frame — the marker check would observe the marker
///   without an `<Earth>`-tagged trans even though the matching
///   `<Mars>` validator owned the assertion.
/// * **Per-body invariant checks** (SRP mutual exclusion, the full
///   `astrodyn::validate_body` pass, gravity-control `check_validity`)
///   iterate the `Added`-filtered `bodies` query, so they validate only
///   newly-attached bodies. Existing bodies were validated on the tick
///   they first appeared, and the per-body invariants do not depend on
///   inter-body state, so re-running them for unchanged bodies would be
///   wasteful.
///
/// Delegates per-body checks to [`astrodyn::validate_body`] and runs
/// `GravityControl::check_validity()` on every control. The latter
/// panics on any gravity misconfiguration (out-of-range degree/order,
/// gradient ordinals outside their valid ranges, non-spherical against
/// a point-mass source) rather than silently auto-correcting — see
/// CLAUDE.md "Fail Loudly".
///
/// Per-body `astrodyn::validate_body` errors are split by
/// `ValidationError::is_warning()`:
///
/// * Fatal-class errors (e.g. `RotationalWithoutMass`,
///   `MissingGravityAcceleration`, `GravitySourceMissing`,
///   `ThreeDofWithRotational`, `InertiaInconsistent`,
///   `PlateTemperatureLengthMismatch`) panic on detection — these
///   describe a body that cannot integrate at all and would otherwise
///   propagate silently-wrong physics. Construction-time error return
///   from `VehicleConfig::spawn_bevy` is the future-preferred surface;
///   until that refactor lands, the Bevy-side validator panics on
///   detection so the misconfiguration surfaces at the first
///   validation tick.
/// * Warning-class errors (`UninitializedState`,
///   `NonRootFrameWithRootDependentFeatures`) emit
///   `bevy::log::warn!()` and let the entity continue. These flag
///   suspicious-but-valid configurations: zero translational state may
///   be the intended initial condition, and non-root integration with
///   root-dependent features is supported when the caller applies the
///   per-step `IntegOrigin` shift at every shift site.
///
/// The separate RF.10 non-root + root-dependent features check below
/// stays as `warn!` (FAIL_LOUD_EXEMPT) — see the inline rationale at
/// that site.
///
/// # Panics
/// Panics with a descriptive `Entity {entity:?} fails component
/// validation: ...` message for any fatal-class `ValidationError`. The
/// message names the offending entity, the specific failure, and how
/// to fix the call site (which component to spawn, which frame to
/// integrate in, etc.). Warning-class errors do not panic.
// JEOD_INV: DM.03 — `Added<GravityControlsC>` filter on the body query fires on every body addition; bodies added mid-simulation are validated on the following tick
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn validate_jeod_invariants<P: Planet>(
    bodies: Query<
        (
            Entity,
            &DynamicsConfigC,
            &GravityControlsC,
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
    //
    // The marker-trans assertion is scoped to entities whose `TranslationalStateC<P>`
    // matches *this* validator's `<P>`. In a multi-planet world the Sun /
    // Moon entity is tagged with the scenario's chosen integration frame
    // (e.g. `TranslationalStateC<Mars>` in a Mars-central scenario); the
    // `<Earth>` validator instance — which `AstrodynPlugin::build` registers
    // by default — would otherwise observe the marker without an
    // `<Earth>`-tagged trans and panic, even though the `<Mars>` validator
    // that `register_planet_systems::<Mars>` installs is the one
    // responsible for that entity. Skipping when this `<P>` has no trans
    // hands ownership of the assertion to the validator whose `<P>` does
    // match. A truly-bare SunMarker entity with no `TranslationalStateC<P>`
    // for any registered planet is caught downstream when
    // `solar_beta_system::<P>` / `earth_lighting_system::<P>` (or any
    // `Query<&TranslationalStateC<P>, With<SunMarker>>`) finds an empty
    // query result.
    let mut sun_count = 0;
    let mut moon_count = 0;
    for (_entity, _, _, sun, moon, trans) in &derived_state_markers {
        if trans.is_none() {
            continue;
        }
        if sun.is_some() {
            sun_count += 1;
        }
        if moon.is_some() {
            moon_count += 1;
        }
    }
    assert!(
        sun_count <= 1,
        "Multiple SunMarker entities found with TranslationalStateC for this validator's planet. \
         JEOD assumes exactly one Sun body per integration frame."
    );
    assert!(
        moon_count <= 1,
        "Multiple MoonMarker entities found with TranslationalStateC for this validator's planet. \
         JEOD assumes exactly one Moon body per integration frame."
    );
    // SolarBetaC / EarthLightingConfigC consumers are themselves
    // generic over `<P>` and query `Query<&TranslationalStateC<P>, With<SunMarker>>`.
    // Only bodies tagged for *this* validator's `<P>` can reach those
    // queries, so the "must have a Sun / Moon" precondition is also
    // scoped to `<P>`-tagged bodies. A `<Mars>`-tagged body with
    // `SolarBetaC` is validated by the `<Mars>` validator (which finds
    // the `<Mars>`-tagged Sun); this `<P>` validator skips it.
    for (entity, solar_beta, earth_lighting, _, _, body_trans) in &derived_state_markers {
        if body_trans.is_none() {
            continue;
        }
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

    for (entity, config, controls, grav_accel, mass, rot_state, trans_state, flat_plates) in &bodies
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

        // Split `astrodyn::validate_body` errors by
        // `ValidationError::is_warning()`. Fatal-class errors panic on
        // detection because they describe a body that cannot integrate
        // at all (missing gravity acceleration storage, a 6-DOF body
        // without mass, a degenerate inertia tensor, plate-temperature
        // length mismatch, etc.) — propagating them would yield
        // silently-wrong physics. Warning-class errors
        // (`UninitializedState`, `NonRootFrameWithRootDependentFeatures`)
        // flag suspicious-but-valid configurations and emit a Bevy
        // log warning so the entity is allowed to continue.
        // Construction-time error return from
        // `VehicleConfig::spawn_bevy` is the future-preferred surface
        // for fatal-class failures — see the `# Panics` rustdoc on
        // this system.
        //
        // The bullet-list shape (leading newline + `  - `) keeps
        // multi-error diagnostics readable when surfaced by Bevy's
        // logger. The remediation tail is configuration-neutral —
        // `validate_body` can fail for missing components
        // (`MissingGravityAcceleration`, `RotationalWithoutMass`,
        // `RotationalWithoutRotState`) *or* for inconsistent
        // configurations on entities whose components are all present
        // (`ThreeDofWithRotational`, `InertiaInconsistent`,
        // `PlateTemperatureLengthMismatch`, `GravitySourceMissing`),
        // so the message points at the misconfiguration list above
        // rather than naming one specific remedy.
        //
        // JEOD_INV: DM.05 — partial; warning-class `UninitializedState` warns, fatal-class siblings panic
        // JEOD_INV: DB.11 — partial; same warning-class/fatal-class split (no per-component `initialized_states` bitfield in our port)
        if !errors.is_empty() {
            let mut warn_detail = String::new();
            let mut fatal_detail = String::new();
            for error in &errors {
                if error.is_warning() {
                    warn_detail.push_str("\n  - ");
                    warn_detail.push_str(&error.to_string());
                } else {
                    fatal_detail.push_str("\n  - ");
                    fatal_detail.push_str(&error.to_string());
                }
            }
            if !warn_detail.is_empty() {
                // Mirror of `Simulation::validate`'s warning-class
                // handling: `is_warning()` failures
                // (`UninitializedState`,
                // `NonRootFrameWithRootDependentFeatures`) may be
                // intentional and must not panic — see the per-class
                // rationale in `ValidationError::is_warning`'s docs.
                // FAIL_LOUD_EXEMPT: operational report path for
                // `is_warning()`-class `ValidationError` (no JEOD
                // invariant counterpart).
                bevy::log::warn!("Entity {entity:?} component validation warnings:{warn_detail}");
            }
            if !fatal_detail.is_empty() {
                let remediation = "Validation is a hard gate — fix the misconfiguration named above and respawn the entity before adding it to the simulation.";
                panic!(
                    "Entity {entity:?} fails component validation:{fatal_detail}\n{remediation}"
                );
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
                // `tests/integ_frame_translation_invariance.rs`) provided
                // the caller applies the per-step `IntegOrigin` shift at
                // every RF.10 shift site (SRP, solar beta, earth lighting)
                // and consumes non-shift sites directly. Promoting to
                // panic would false-positive on legitimate setups.
                bevy::log::warn!(
                    "Entity {entity:?}: non-root integration frame (or active \
                     frame switch into a non-root frame) paired with features. \
                     Per RF.10, these fall into two groups: SHIFT SITES (need \
                     root-inertial conversion before mixing with root-inertial \
                     source positions — flat_plate_srp={has_flat}, \
                     cannonball_srp={has_cannonball}, \
                     solar_beta={has_solar_beta}, \
                     earth_lighting={has_earth_lighting}) and NON-SHIFT SITES \
                     (consume the body's typed `Position<PlanetInertial<P>>` \
                     directly; shifting would break them — drag={has_drag}, \
                     orbital_elements={has_orbital}, euler={has_euler}, \
                     geodetic={has_geodetic}, lvlh={has_lvlh}). If the \
                     configuration applies the shift at every shift site (and \
                     leaves the non-shift sites alone) this warning is \
                     informational.",
                );
            }
        }

        // ── Gravity control startup validation ──
        // JEOD_INV: GV.03 — check_validity() called at startup
        // Panics on any out-of-range gravity control parameter (degree,
        // order, gradient ordinals) — see CLAUDE.md "Fail Loudly".
        for ctrl in &controls.0.controls {
            if let Ok((_source_entity, source)) = sources.get(ctrl.source_name) {
                ctrl.check_validity(&source.0);
            }
        }
    }
}
