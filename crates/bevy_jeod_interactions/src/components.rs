use bevy::prelude::*;
use jeod_sim::{DragConfig, FlatPlate, FlatPlateParams, FlatPlateThermal};

/// Vehicle drag configuration (Cd, area).
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
pub struct DragConfigC(pub DragConfig);

/// Flat-plate SRP configuration with thermal state.
///
/// Contains per-plate geometry, optical/thermal properties, and temperature state.
/// Temperatures are integrated via forward Euler each step (matching the
/// Simulation runner's approach).
#[derive(Component, Debug, Clone)]
pub struct FlatPlateConfigC {
    /// Per-plate geometry, optical, and thermal properties.
    pub plates: Vec<(FlatPlate, FlatPlateParams, FlatPlateThermal)>,
    /// Per-plate temperatures (K). Same length as `plates`.
    pub temperatures: Vec<f64>,
    /// Cached T^4 per plate from previous step (for thermal emission).
    pub t_pow4_cached: Vec<f64>,
}

/// Marker for an entity that casts shadows (e.g., Earth).
///
/// The shadow detection system queries all entities with this component
/// and computes the illumination factor for SRP. Place on any planet
/// entity along with `TranslationalStateC`.
#[derive(Component, Debug, Clone, Copy)]
pub struct ShadowBodyC {
    /// Body radius (m) for conical shadow computation.
    pub radius: f64,
}
