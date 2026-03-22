use bevy::prelude::*;

use crate::sets::JeodSet;
use crate::systems::{force_collection_system, integration_system};

pub struct JeodDynamicsPlugin;

impl Plugin for JeodDynamicsPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            FixedUpdate,
            (
                JeodSet::Environment,
                JeodSet::ForceCollection.after(JeodSet::Environment),
                JeodSet::Integration.after(JeodSet::ForceCollection),
                JeodSet::DerivedState.after(JeodSet::Integration),
            ),
        );

        app.add_systems(
            FixedUpdate,
            (
                force_collection_system.in_set(JeodSet::ForceCollection),
                integration_system.in_set(JeodSet::Integration),
            ),
        );
    }
}
