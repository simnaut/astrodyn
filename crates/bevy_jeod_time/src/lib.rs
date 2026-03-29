use bevy::prelude::*;
use bevy_jeod_dynamics::JeodSet;
use jeod_sim::{default_leap_second_table, SimulationTime};

// JEOD_INV: TM.07 — JEOD uses -1.0 sentinel; we call recompute_derived() at construction instead
/// Bevy resource wrapping `SimulationTime`.
#[derive(Resource, Debug, Deref, DerefMut)]
pub struct SimulationTimeR(pub SimulationTime);

impl Default for SimulationTimeR {
    fn default() -> Self {
        Self(SimulationTime::at_j2000(default_leap_second_table()))
    }
}

/// Plugin that advances simulation time each fixed update step.
pub struct JeodTimePlugin;

impl Plugin for JeodTimePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimulationTimeR>();
        app.add_systems(FixedUpdate, time_advance_system.in_set(JeodSet::TimeUpdate));
    }
}

// JEOD_INV: TM.03 — time types updated in dependency order (delegates to SimulationTime::advance)
fn time_advance_system(mut sim_time: ResMut<SimulationTimeR>, time: Res<Time<Fixed>>) {
    let dt = time.delta_secs_f64();
    sim_time.advance(dt);
}
