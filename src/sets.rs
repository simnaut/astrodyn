//! [`JeodSet`] — the Bevy `SystemSet` partition that mirrors JEOD's
//! per-step pipeline.
//!
//! # Schedule as integration group
//!
//! JEOD's `JeodIntegrationGroup` collects a set of bodies that advance
//! together under a single shared time step, with one integrator stage
//! that consumes the group's force/torque accumulators and produces the
//! next state for every member. In `bevy_jeod`, the Bevy `FixedUpdate`
//! schedule **is** that integration group:
//!
//! - One `FixedUpdate` tick = one group advance at the schedule's fixed
//!   `dt`. Every entity matched by the integrating system's query
//!   shares that `dt` by construction — no per-body bookkeeping.
//! - The seven [`JeodSet`] stages below run once per tick, in the order
//!   declared, mirroring JEOD's init/update pipeline (force collection
//!   feeds integration feeds derived state).
//! - **Multi-stage integrators (RK4, etc.) are an inner loop inside
//!   [`JeodSet::Integration`], not multiple schedule passes.** A four-
//!   stage RK4 evaluates four sub-steps within the integration system
//!   for the same outer `FixedUpdate` tick.
//! - Multi-body sims (Apollo, Earth–Moon) need no extra wiring: every
//!   body registered in the schedule is automatically a member of the
//!   "group" in JEOD's sense.
//!
//! # Multiple integration groups
//!
//! When a scenario genuinely needs *separate* groups — different bodies
//! integrating at different cadences — register a second Bevy schedule
//! and run it on its own tick rate. This is plain Bevy mechanics, not
//! a JEOD-specific construct, and no current scenario in this workspace
//! uses it; the single-`FixedUpdate` model covers every shipped sim.

use bevy::prelude::*;

/// Bevy system-set partition mirroring JEOD's per-step pipeline. Stages
/// run in declaration order; `JeodPlugin::build` configures the ordering
/// when the plugin is added to the app.
///
/// All seven stages execute together inside one `FixedUpdate` tick at
/// the schedule's shared `dt`, defining a single JEOD-style integration
/// group; see the module-level docs for the full mapping.
// JEOD_INV: DM.04 — system set ordering mirrors JEOD init/update pipeline
// JEOD_INV: DM.13 — EphemerisUpdate before Environment ensures ephemeris is current for gravity
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum JeodSet {
    /// Time scale update (TAI, UTC, TDB, GMST, etc.).
    TimeUpdate,
    /// Ephemeris update (planet positions from DE4xx data).
    EphemerisUpdate,
    /// Environment computation (gravity, atmosphere).
    Environment,
    /// Interaction computation (aero drag, SRP, gravity torque).
    Interaction,
    /// Force and torque collection.
    ForceCollection,
    /// State integration (RK4, etc.).
    Integration,
    /// Derived state computation (orbital elements, Euler angles, etc.).
    DerivedState,
}
