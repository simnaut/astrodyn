use bevy::prelude::*;
use bevy_jeod_dynamics::JeodSet;

use crate::systems::gravity_computation_system;

pub struct JeodGravityPlugin;

impl Plugin for JeodGravityPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            FixedUpdate,
            gravity_computation_system.in_set(JeodSet::Environment),
        );
    }
}
