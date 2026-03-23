use bevy::prelude::*;
use jeod_planet::PlanetShape;

/// Bevy component wrapping `PlanetShape`.
#[derive(Component, Debug, Clone, Deref, DerefMut)]
pub struct PlanetC(pub PlanetShape);

/// Plugin for planet entities.
pub struct JeodPlanetPlugin;

impl Plugin for JeodPlanetPlugin {
    fn build(&self, _app: &mut App) {
        // Phase 2: component registration only.
        // Phase 3+: planet-fixed frame propagation systems.
    }
}
