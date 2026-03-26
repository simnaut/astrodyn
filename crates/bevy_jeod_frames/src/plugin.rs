use bevy::prelude::*;
use bevy_jeod_dynamics::{JeodSet, PlanetFixedRotationC};
use bevy_jeod_time::SimulationTimeR;

/// Plugin that computes inertial-to-planet-fixed rotation (RNP) each timestep.
///
/// **Requires** `JeodTimePlugin` (or manual insertion of [`SimulationTimeR`]) —
/// will panic during app construction if the resource is missing.
pub struct JeodFramesPlugin;

impl Plugin for JeodFramesPlugin {
    fn build(&self, app: &mut App) {
        assert!(
            app.world().contains_resource::<SimulationTimeR>(),
            "JeodFramesPlugin requires SimulationTimeR. Add JeodTimePlugin first."
        );
        app.add_systems(
            FixedUpdate,
            planet_fixed_rotation_system.in_set(JeodSet::EphemerisUpdate),
        );
    }
}

/// Computes the inertial-to-planet-fixed rotation matrix (RNP) for each entity
/// that carries a `PlanetFixedRotationC` component.
///
/// This replaces the `DMat3::IDENTITY` placeholder so that spherical-harmonic
/// gravity evaluation uses the correct body-fixed coordinates.
fn planet_fixed_rotation_system(
    sim_time: Res<SimulationTimeR>,
    mut query: Query<&mut PlanetFixedRotationC>,
) {
    let tt_tjt = sim_time.tai_tjt
        + jeod_time::epoch::TAI_TT_OFFSET / jeod_time::epoch::SECONDS_PER_DAY;
    let rotation =
        jeod_frames::rotation_j2000::compute_t_parent_this_from_tjt(sim_time.gmst_seconds, tt_tjt);
    for mut rot in &mut query {
        rot.0 = rotation;
    }
}
