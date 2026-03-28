// Atmosphere components are defined in bevy_jeod_dynamics (AtmosphericStateC)
// to avoid circular dependencies with the force collection system.
//
// This module re-exports them for convenience.
pub use bevy_jeod_dynamics::AtmosphericStateC;
