//! Runtime validation of JEOD invariants.
//!
//! JEOD's OOP architecture enforces many invariants structurally (mandatory
//! class members, initialization order, fatal errors during init). In ECS,
//! components are optional and can be added/removed freely, so we enforce
//! these invariants at runtime with descriptive panics.
//!
//! This system runs once at the start of the first `FixedUpdate` tick.

use bevy::prelude::*;

use crate::components::{
    DynamicsConfigC, GravityAccelerationC, GravityControlsC, GravitySourceC,
    MassPropertiesC, RotationalStateC, TranslationalStateC,
};

/// Validates JEOD invariants on all dynamic body entities.
///
/// Runs once at startup (first `FixedUpdate` tick), matching JEOD's
/// `DynManager::initialize_simulation()` which validates all bodies
/// before the first integration step.
///
/// # Checked invariants
///
/// **B: Gravity control validation** (JEOD: `check_validity()` during
/// `initialize_gravity_controls()`). Degree/order must not exceed the
/// source model's degree/order. Fatal in JEOD.
///
/// **G: Gravity source existence** (JEOD: `GravityManager::find_grav_source()`
/// during init). Every gravity control must reference an existing entity
/// with `GravitySourceC`. Non-fatal in JEOD (`MessageHandler::error`), but
/// we escalate to a panic to prevent silently skipping gravity.
///
/// **H: three_dof consistency** (JEOD: `create_body_integrators()` skips
/// rotational integrator when `three_dof=true`). If `three_dof` is true,
/// `rotational_dynamics` must be false.
///
/// **E: Integration frame existence** (partial — validates that
/// `MassPropertiesC` exists on entities with `rotational_dynamics=true`,
/// matching JEOD's mandatory `MassBody` on all `DynBody`s).
///
/// # Panics
/// Panics with a descriptive message for any violated invariant, matching
/// JEOD's `MessageHandler::fail()` behavior.
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
        // ── Invariant: GravityAccelerationC required for integration ──
        // In JEOD, grav_interaction is a value member of DynBody — always present.
        // In ECS, the component could be missing, causing silent integration skip.
        if grav_accel.is_none() {
            panic!(
                "Entity {entity:?}: has GravityControlsC but no GravityAccelerationC. \
                 In JEOD, grav_interaction is a value member of DynBody. \
                 Add GravityAccelerationC::default() to any entity with gravity controls."
            );
        }

        // ── Invariant H: three_dof consistency ──
        // ECS-context equivalent of DynamicsConfig::validate() in jeod_dynamics (DB.05, DB.06),
        // preserving entity-specific panic context.
        if config.three_dof && config.rotational_dynamics {
            panic!(
                "Entity {entity:?}: DynamicsConfig has three_dof=true AND \
                 rotational_dynamics=true. In JEOD, three_dof=true prevents \
                 creation of the rotational integrator."
            );
        }

        // ── Invariant E (partial): rotational dynamics requires mass ──
        // JEOD_INV: MA.01 — MassBody always present on DynBody (partial: only checked for rotational path)
        // JEOD: DynBody always has MassBody. Without mass, Euler's equation
        // (I^-1 * (tau - omega x I*omega)) cannot be evaluated.
        if config.rotational_dynamics {
            if mass.is_none() {
                panic!(
                    "Entity {entity:?}: rotational_dynamics=true but no MassPropertiesC. \
                     In JEOD, DynBody always has MassBody (inertia tensor required for \
                     Euler's equation). Add MassPropertiesC with valid inertia."
                );
            }
            if rot_state.is_none() {
                panic!(
                    "Entity {entity:?}: rotational_dynamics=true but no RotationalStateC. \
                     Add RotationalStateC with initial quaternion and angular velocity."
                );
            }
        }

        // ── Invariant C: inertia/inverse_inertia consistency ──
        // Delegates to MassProperties::validate_consistency() in jeod_dynamics (DB.19, MA.04)
        if let Some(m) = mass {
            m.validate_consistency(jeod_dynamics::INERTIA_CONSISTENCY_TOL);
        }

        // ── Invariant B: gravity control validation ──
        // JEOD_INV: GV.03 — check_validity() called at startup (auto-corrections applied in-place)
        // JEOD: check_validity() is called during initialize_gravity_controls().
        // degree > source degree is fatal. order > source order is fatal.
        // Non-fatal auto-corrections (GV.07-11) are applied in-place to the actual control.
        for ctrl in &mut controls.0.controls {
            // ── Invariant G: gravity source must exist ──
            // JEOD_INV: DM.08 — gravitation requires gravity source (init-time check; "initialized" gate not enforced)
            // JEOD_INV: GV.12 — gravity source must exist for control
            // JEOD: initialize_control() calls MessageHandler::error() (non-fatal,
            // severity 0) and returns, leaving the control uninitialized. We escalate
            // to a panic because silently skipping a gravity source would produce
            // incorrect physics.
            let Ok((_source_entity, source)) = sources.get(ctrl.source_name) else {
                panic!(
                    "Entity {entity:?}: GravityControl references entity {:?} which \
                     does not exist or has no GravitySourceC. JEOD logs a non-fatal \
                     error and skips; we panic to prevent silently wrong physics.",
                    ctrl.source_name
                );
            };

            // Validate degree/order against the source model and apply auto-corrections.
            ctrl.check_validity(&source.0);
        }

        // ── Invariant F (informational): uninitialized state detection ──
        // Delegates to TranslationalState::is_likely_uninitialized() in jeod_dynamics (DM.05, DB.11)
        if config.translational_dynamics {
            if let Some(trans) = trans_state {
                if trans.is_likely_uninitialized() {
                    bevy::log::warn!(
                        "Entity {entity:?}: TranslationalStateC is all zeros (position and \
                         velocity). In JEOD, uninitialized state is a fatal error. If this \
                         entity is intentionally at the origin with zero velocity, ignore \
                         this warning."
                    );
                }
            }
        }
    }
}
