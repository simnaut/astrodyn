//! Glue: per-step recompute of derived mass quantities.
//!
//! Recomputes `inverse_mass` / `inverse_inertia` for every entity
//! whose [`crate::components::MassPropertiesC`] is marked `dirty`.
//! Runs between [`crate::AstrodynSet::TimeUpdate`] and
//! [`crate::AstrodynSet::EphemerisUpdate`] so gravity, force
//! collection, and integration see current mass properties.

use bevy::prelude::*;

use crate::components::*;

/// Recompute derived mass quantities (`inverse_mass`, `inverse_inertia`) each step.
///
/// Port of JEOD's `(DYNAMICS, "scheduled") dyn_body.mass.update_mass_properties()`.
/// JEOD runs this every timestep so that runtime mass changes (fuel burn,
/// staging, attach/detach) are reflected in the dynamics before the next
/// derivative computation.
///
/// Placed before `AstrodynSet::EphemerisUpdate` so gravity and force collection
/// see current mass properties.
///
/// **Change-detection contract**: the dirty-flag check below is read through
/// `Mut::deref` (immutable access), and `recompute_derived()` is only
/// invoked — triggering `DerefMut` and marking the component as
/// `Changed` — when the entity actually needs updating. Without this
/// gate, an unconditional `mass.recompute_derived()` (whose body is a
/// `dirty`-guarded no-op) still triggers `DerefMut` on every entity
/// every tick, and `composite_mass_system`'s downstream
/// `Changed<MassPropertiesC>` filter would match every parent every
/// tick — corrupting the `CoreMassPropertiesC` cache by reseeding it
/// from the previous-tick composite. The `dirty` field is only set
/// `true` by mission code that genuinely mutates `mass`/`inertia`, so
/// it is the correct signal here.
pub fn mass_update_system(mut query: Query<&mut MassPropertiesC>) {
    for mut mass in &mut query {
        // Read `dirty` via `Mut::deref` (no `DerefMut`), so entities
        // that don't need recomputation are not falsely marked
        // `Changed`. `recompute_derived` is itself a no-op when
        // `!dirty`, so the gate preserves behavior.
        if mass.0.dirty {
            mass.recompute_derived();
        }
    }
}
