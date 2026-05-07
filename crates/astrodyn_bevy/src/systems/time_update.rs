//! Bevy systems for [`AstrodynSet::TimeUpdate`](crate::AstrodynSet::TimeUpdate).
//!
//! Time-scale advance for the JEOD time pipeline (TAI/UTC/UT1/TDB/TT/GMST).

use bevy::prelude::*;

use crate::SimulationTimeR;

/// Advance every JEOD-tracked time scale by the Bevy `Time<Fixed>` delta
/// each step (TAI/UTC/UT1/TDB/TT/GMST). Runs in
/// [`AstrodynSet::TimeUpdate`](crate::AstrodynSet::TimeUpdate).
// JEOD_INV: TM.03 — time types updated in dependency order (delegates to SimulationTime::advance)
pub fn time_advance_system(mut sim_time: ResMut<SimulationTimeR>, time: Res<Time<Fixed>>) {
    let dt = time.delta_secs_f64();
    sim_time.advance(dt);
}
