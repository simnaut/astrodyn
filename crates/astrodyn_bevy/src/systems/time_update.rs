//! Bevy systems for [`AstrodynSet::TimeUpdate`](crate::AstrodynSet::TimeUpdate).
//!
//! Time-scale advance for the JEOD time pipeline (TAI/UTC/UT1/TDB/TT/GMST).

use bevy::prelude::*;

use crate::{IntegrationDtR, SimulationTimeR};

/// Advance every JEOD-tracked time scale by the pipeline's `dt`
/// each step (TAI/UTC/UT1/TDB/TT/GMST). Runs in
/// [`AstrodynSet::TimeUpdate`](crate::AstrodynSet::TimeUpdate).
///
/// `dt` comes from the optional [`IntegrationDtR`] override when
/// installed (bit-exact f64 — required for `runner ↔ bevy` parity on
/// irrational-in-seconds timesteps); otherwise falls back to
/// `Time<Fixed>::delta_secs_f64()`, preserving the historical path for
/// callers driving the schedule via Bevy's `Time<Fixed>::advance_by`.
// JEOD_INV: TM.03 — time types updated in dependency order (delegates to SimulationTime::advance)
pub fn time_advance_system(
    mut sim_time: ResMut<SimulationTimeR>,
    dt_override: Option<Res<IntegrationDtR>>,
    time: Res<Time<Fixed>>,
) {
    let dt = dt_override
        .map(|r| r.0)
        .unwrap_or_else(|| time.delta_secs_f64());
    sim_time.advance(dt);
}
