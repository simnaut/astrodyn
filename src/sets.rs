//! [`JeodSet`] — the Bevy `SystemSet` partition that mirrors JEOD's
//! per-step pipeline.

use bevy::prelude::*;

/// Bevy system-set partition mirroring JEOD's per-step pipeline. Stages
/// run in declaration order; `JeodPlugin::build` configures the ordering
/// when the plugin is added to the app.
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
