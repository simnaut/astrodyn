//! Runtime validation of JEOD invariants.
//!
//! Delegates to [`jeod_sim::validate_body`] for the per-body invariant checks,
//! wrapping results with Bevy entity context and panicking on errors.
//!
//! This system runs once at the start of the first `FixedUpdate` tick.

use bevy::prelude::*;

use crate::components::{
    CannonballSrpC, DynamicsConfigC, FlatPlateConfigC, GravityAccelerationC, GravityControlsC,
    GravitySourceC, MassPropertiesC, RotationalStateC, TidalConfigC, TidalDeltaC20C,
    TranslationalStateC,
};

/// Validates JEOD invariants on all dynamic body entities.
///
/// Runs once at startup (first `FixedUpdate` tick), matching JEOD's
/// `DynManager::initialize_simulation()` which validates all bodies
/// before the first integration step.
///
/// Delegates per-body checks to [`jeod_sim::validate_body`] and applies
/// gravity control auto-corrections via `check_validity()`.
///
/// # Panics
/// Panics with a descriptive message for any violated invariant.
// JEOD_INV: DM.03 — one-shot validation gate (Local<bool>); does not block integration like JEOD's initialized flag
#[allow(clippy::type_complexity)]
pub fn validate_jeod_invariants(
    mut bodies: Query<(
        Entity,
        &DynamicsConfigC,
        &mut GravityControlsC,
        Option<&GravityAccelerationC>,
        Option<&MassPropertiesC>,
        Option<&RotationalStateC>,
        Option<&TranslationalStateC>,
        Option<&FlatPlateConfigC>,
    )>,
    sources: Query<(Entity, &GravitySourceC)>,
    tidal_sources: Query<(
        Entity,
        &TidalConfigC,
        Option<&TidalDeltaC20C>,
        Option<&crate::components::PlanetFixedRotationC>,
    )>,
    srp_exclusion: Query<Entity, With<CannonballSrpC>>,
    mut has_run: Local<bool>,
) {
    if *has_run {
        return;
    }
    *has_run = true;

    // Validate tidal component pairing on gravity sources.
    for (entity, _config, delta, rotation) in &tidal_sources {
        assert!(
            delta.is_some(),
            "Entity {entity:?}: TidalConfigC is present but TidalDeltaC20C is missing. \
             Add TidalDeltaC20C(0.0) to the entity so tidal_update_system can write ΔC20."
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

        // Delegate structural validation to jeod_sim
        let errors = jeod_sim::validate_body(
            config,
            &controls.0,
            grav_accel.is_some(),
            mass.map(|m| &m.0),
            rot_state.is_some(),
            trans_state.map(|t| &t.0),
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
