//! Runtime validation of JEOD invariants.
//!
//! Delegates to [`jeod_sim::validate_body`] for the per-body invariant checks,
//! wrapping results with Bevy entity context and panicking on errors.
//!
//! This system runs once at the start of the first `FixedUpdate` tick.

use bevy::prelude::*;

use crate::components::{
    DynamicsConfigC, GravityAccelerationC, GravityControlsC, GravitySourceC, MassPropertiesC,
    RotationalStateC, TranslationalStateC,
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
    )>,
    sources: Query<(Entity, &GravitySourceC)>,
    mut has_run: Local<bool>,
) {
    if *has_run {
        return;
    }
    *has_run = true;

    for (entity, config, mut controls, grav_accel, mass, rot_state, trans_state) in &mut bodies {
        // Delegate structural validation to jeod_sim
        let errors = jeod_sim::validate_body(
            config,
            &controls.0,
            grav_accel.is_some(),
            mass.map(|m| &m.0),
            rot_state.is_some(),
            trans_state.map(|t| &t.0),
            |source_entity| {
                sources
                    .get(source_entity)
                    .ok()
                    .map(|(_, source)| &source.0)
            },
        );

        for error in &errors {
            match error {
                // Uninitialized state is a warning, not fatal
                jeod_sim::ValidationError::UninitializedState => {
                    bevy::log::warn!(
                        "Entity {entity:?}: {error}. If this entity is intentionally \
                         at the origin with zero velocity, ignore this warning."
                    );
                }
                // All other errors are fatal
                _ => {
                    panic!("Entity {entity:?}: {error}");
                }
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
