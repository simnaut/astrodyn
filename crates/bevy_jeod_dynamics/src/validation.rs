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
    DynamicsConfigC, GravityControlsC, GravitySourceC, MassPropertiesC,
    RotationalStateC, TranslationalStateC,
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
/// with `GravitySourceC`. Fatal in JEOD.
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
// JEOD_INV: DM.05 — partial: warns on zero translational state only (JEOD fatally checks all states)
#[allow(clippy::type_complexity)]
pub fn validate_jeod_invariants(
    mut bodies: Query<(
        Entity,
        &DynamicsConfigC,
        &mut GravityControlsC,
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

    for (entity, config, mut controls, mass, rot_state, trans_state) in &mut bodies {
        // ── Invariant H: three_dof consistency ──
        // JEOD_INV: DB.05 — three_dof=true prevents rotational integrator creation
        // JEOD_INV: DB.06 — three_dof=true AND rotational_dynamics=true is invalid
        // JEOD: three_dof=true prevents rotational integrator creation.
        // rotational_dynamics=true with three_dof=true would attempt to
        // integrate rotation without an integrator (undefined behavior).
        if config.three_dof && config.rotational_dynamics {
            panic!(
                "Entity {entity:?}: DynamicsConfig has three_dof=true AND \
                 rotational_dynamics=true. In JEOD, three_dof=true prevents \
                 creation of the rotational integrator. Set rotational_dynamics=false \
                 when three_dof=true."
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
        // JEOD_INV: DB.19 — inverse_inertia used for Euler equation (validated I*I^-1 ≈ identity)
        // JEOD_INV: MA.04 — inverse_inertia consistent with inertia
        // JEOD: inverse_inertia is always recomputed from inertia.
        // Verify they are consistent (I * I^-1 ≈ identity).
        if let Some(m) = mass {
            let product = m.inertia * m.inverse_inertia;
            let identity_err = (product - glam::DMat3::IDENTITY).abs_diff_eq(glam::DMat3::ZERO, 1e-6);
            if !identity_err {
                panic!(
                    "Entity {entity:?}: MassPropertiesC.inertia and .inverse_inertia are \
                     inconsistent (I * I^-1 != identity to 1e-6). In JEOD, inverse_inertia \
                     is always recomputed from inertia. Use MassProperties::with_inertia() \
                     which computes the inverse automatically."
                );
            }
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
            // JEOD: find_grav_source() fatally fails if source not found.
            let Ok((_source_entity, source)) = sources.get(ctrl.source_name) else {
                panic!(
                    "Entity {entity:?}: GravityControl references entity {:?} which \
                     does not exist or has no GravitySourceC. In JEOD, gravity source \
                     resolution is fatal during initialization.",
                    ctrl.source_name
                );
            };

            // Validate degree/order against the source model and apply auto-corrections.
            ctrl.check_validity(&source.0);
        }

        // ── Invariant F (informational): uninitialized state detection ──
        // JEOD_INV: DM.05 — partial: warns on zero translational state (JEOD fatally checks all states)
        // JEOD_INV: DB.11 — partial: zero-state heuristic only (no initialized_states bitfield)
        // JEOD: check_for_uninitialized_states() fatally fails if required
        // state is not set. We check for exact-zero state, which is almost
        // certainly unintentional for orbital mechanics.
        if config.translational_dynamics {
            if let Some(trans) = trans_state {
                if trans.position == glam::DVec3::ZERO && trans.velocity == glam::DVec3::ZERO {
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
