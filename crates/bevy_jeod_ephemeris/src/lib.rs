use bevy::prelude::*;
use jeod_sim::Ephemeris;

/// Bevy resource wrapping the planetary ephemeris.
#[derive(Resource)]
pub struct EphemerisR(pub Ephemeris);

/// Plugin for ephemeris-driven planet position updates.
pub struct JeodEphemerisPlugin;

impl Plugin for JeodEphemerisPlugin {
    fn build(&self, _app: &mut App) {
        // EphemerisR is not inserted by default — the user must provide a .bsp
        // file path and insert it manually, or use a setup system.
        // Phase 3: ephemeris_update_system will go in JeodSet::EphemerisUpdate.
    }
}
