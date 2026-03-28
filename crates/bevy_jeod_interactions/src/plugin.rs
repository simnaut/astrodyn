use bevy::prelude::*;
use bevy_jeod_dynamics::JeodSet;

use crate::systems::{aero_drag_system, gravity_torque_system, radiation_pressure_system};

pub struct JeodInteractionsPlugin;

impl Plugin for JeodInteractionsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            FixedUpdate,
            (
                aero_drag_system,
                gravity_torque_system,
                radiation_pressure_system,
            ).in_set(JeodSet::Interaction),
        );
    }
}
