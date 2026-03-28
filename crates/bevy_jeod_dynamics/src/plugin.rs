use bevy::prelude::*;

use crate::sets::JeodSet;
use crate::systems::{force_collection_system, integration_system};
use crate::validation::validate_jeod_invariants;

pub struct JeodDynamicsPlugin;

impl Plugin for JeodDynamicsPlugin {
    fn build(&self, app: &mut App) {
        // JEOD_INV: DM.04 — init order: time -> ephemeris -> environment -> interaction -> forces -> integration -> derived
        // JEOD_INV: DM.13 — ephemeris updated before gravity (EphemerisUpdate before Environment)
        app.configure_sets(
            FixedUpdate,
            (
                JeodSet::TimeUpdate,
                JeodSet::EphemerisUpdate.after(JeodSet::TimeUpdate),
                JeodSet::Environment.after(JeodSet::EphemerisUpdate),
                JeodSet::Interaction.after(JeodSet::Environment),
                JeodSet::ForceCollection.after(JeodSet::Interaction),
                JeodSet::Integration.after(JeodSet::ForceCollection),
                JeodSet::DerivedState.after(JeodSet::Integration),
            ),
        );

        app.add_systems(
            FixedUpdate,
            (
                // Validation runs first — matches JEOD's initialize_simulation()
                // which validates all bodies before the first integration step.
                // Uses Local<bool> to run only once.
                validate_jeod_invariants.before(JeodSet::TimeUpdate),
                force_collection_system.in_set(JeodSet::ForceCollection),
                integration_system.in_set(JeodSet::Integration),
            ),
        );
    }
}
