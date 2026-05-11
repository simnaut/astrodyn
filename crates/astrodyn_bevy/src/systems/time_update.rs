//! Bevy systems for [`AstrodynSet::TimeUpdate`](crate::AstrodynSet::TimeUpdate).
//!
//! Time-scale advance for the JEOD time pipeline (TAI/UTC/UT1/TDB/TT/GMST).

use bevy::prelude::*;

use crate::{IntegrationDtR, SimulationTimeR};

/// Advance every JEOD-tracked time scale by the pipeline's `dt`
/// each step (TAI/UTC/UT1/TDB/TT/GMST). Runs in
/// [`AstrodynSet::TimeUpdate`](crate::AstrodynSet::TimeUpdate).
///
/// `dt` is read from the mandatory [`IntegrationDtR`] resource (bit-
/// exact f64; see the type doc). If the resource is missing, Bevy
/// panics on the first FixedUpdate run with a "resource does not
/// exist" diagnostic — install it via
/// [`crate::AstrodynAppExt::add_astrodyn`],
/// [`crate::AstrodynAppExt::step_fixed_dt`], or
/// [`crate::SimulationBuilderBevyExt::populate_app`].
// JEOD_INV: TM.03 — time types updated in dependency order (delegates to SimulationTime::advance)
pub fn time_advance_system(mut sim_time: ResMut<SimulationTimeR>, dt: Res<IntegrationDtR>) {
    sim_time.advance(dt.0);
}
