use bevy::prelude::*;
use bevy_jeod_dynamics::JeodSet;

use crate::systems::atmosphere_update_system;

pub struct JeodAtmospherePlugin;

impl Plugin for JeodAtmospherePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            FixedUpdate,
            atmosphere_update_system.in_set(JeodSet::Environment),
        );
    }
}
