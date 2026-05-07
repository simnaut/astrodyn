//! Gravity stage re-exports.
//!
//! The kernels and typed siblings live in `astrodyn_gravity::accumulate`.
//! This module preserves the `astrodyn::gravity::*` import path that
//! mission code, the Bevy adapter, and `astrodyn_runner` use today.

pub use astrodyn_gravity::accumulate::{
    accumulate_gravity, accumulate_gravity_typed, accumulate_relativistic_corrections,
    accumulate_relativistic_corrections_typed, evaluate_body_gravity_typed, run_gravity_stage,
    GravityBodyInputs, ResolvedRelativisticSource, ResolvedSource,
};
