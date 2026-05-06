//! Bevy `Component` newtypes for interactions: aerodynamic / radiation
//! force-and-torque accumulators, atmosphere state, drag and SRP
//! configuration, and the shadow-body marker.

use bevy::prelude::*;
use glam::DVec3;
use jeod_sim::{DragConfig, DragConfigTyped, Earth, Planet};

/// Aerodynamic force and torque in the **structural** frame (N, N*m).
///
/// Written by `aero_drag_system`.
/// `force_collection_system` rotates force to inertial and torque to body
/// via `StructuralTransformC`.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct AerodynamicForceC {
    /// Force in the body structural frame (N).
    pub force: DVec3,
    /// Torque about the body structural origin (N·m).
    pub torque: DVec3,
}

/// Solar radiation pressure force and torque.
///
/// Force is always in the **inertial** frame (`flat_plate_srp_system` rotates
/// from structural to inertial before writing).
/// Torque is always in the **structural** frame.
/// Written by `flat_plate_srp_system`.
/// `force_collection_system` rotates torque to body via `StructuralTransformC`.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct RadiationForceC {
    /// Force in the body structural frame (N).
    pub force: DVec3,
    /// Torque about the body structural origin (N·m).
    pub torque: DVec3,
}

// JEOD_INV: AT.01 — active flag gates computation (presence of AtmosphericStateC = active)
/// Atmospheric state at the vehicle's position.
///
/// Wraps a typed `AtmosphereState<P>` whose `wind` field is
/// `Velocity<PlanetInertial<P>>`. Every call site must pin `P` explicitly
/// — there is no fallback. Mission code with multiple planets in one
/// `World` instantiates the type per planet. Written by the atmosphere
/// system; read by the aerodynamic drag system.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
pub struct AtmosphericStateC<P: Planet>(pub jeod_sim::AtmosphereState<P>);

impl<P: Planet> Default for AtmosphericStateC<P> {
    #[inline]
    fn default() -> Self {
        Self(jeod_sim::AtmosphereState::default())
    }
}

// ── Interactions ──

/// Vehicle drag configuration (Cd, area).
///
/// Wraps [`DragConfigTyped`] (typed sibling of [`DragConfig`]) so the
/// untyped → typed conversion happens **once at insertion**, not per tick
/// in `aero_drag_system`. Convenience constructors `from_untyped` /
/// `new` are provided for callers building from raw `f64` fields.
///
/// Auto-inserts [`AtmosphericStateC`] and [`AerodynamicForceC`] when added.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
#[require(AtmosphericStateC::<Earth>, AerodynamicForceC)]
pub struct DragConfigC(pub DragConfigTyped);

impl DragConfigC {
    /// Wrap an untyped [`DragConfig`] as a typed Bevy component.
    ///
    /// The dimensional lift (`f64` → `Ratio`/`Area`/`MassDensity`) happens
    /// here at insertion. After that, the wrapped value is already typed
    /// for the lifetime of the component, eliminating per-tick
    /// per-tick unchecked conversions in `aero_drag_system`. Per #172 H1,
    /// this is the documented insertion-time boundary: DragConfig has no
    /// spatial fields (Cd / area / optional density override only), so
    /// the lift here is the JEOD-CSV-style boundary the audit carves out.
    /// The component then stores DragConfigTyped for the rest of its
    /// lifetime; no per-step re-minting occurs.
    #[inline]
    pub fn from_untyped(config: &DragConfig) -> Self {
        Self(DragConfigTyped::from_untyped_unchecked(config)) // allowed: #172 H1 insertion-time boundary, see docstring
    }
}

impl From<DragConfig> for DragConfigC {
    fn from(config: DragConfig) -> Self {
        Self::from_untyped(&config)
    }
}

impl From<DragConfigTyped> for DragConfigC {
    fn from(config: DragConfigTyped) -> Self {
        Self(config)
    }
}

/// Flat-plate SRP configuration with thermal state.
///
/// Wraps [`jeod_sim::FlatPlateState`] so the same type (and its
/// `integrate_temperatures` method) is shared with the `Simulation` runner.
///
/// The wrapped state is `FlatPlateState<SelfRef>` — the canonical
/// runtime-resolved instantiation at the Bevy adapter boundary, where
/// per-entity storage decides the vehicle identity at runtime. The
/// underlying `jeod_interactions::FlatPlate<V>` is `<V: Vehicle>`-
/// parametric so mission code that pins a concrete vehicle (e.g.
/// `<Iss>`) can demonstrate cross-vehicle compile-time blocking
/// upstream of the adapter; the adapter Component always lands at
/// `<SelfRef>`.
///
/// Auto-inserts [`RadiationForceC`] when added.
#[derive(Component, Debug, Clone, Deref, DerefMut)]
#[require(RadiationForceC)]
pub struct FlatPlateConfigC(pub jeod_sim::FlatPlateState<jeod_sim::SelfRef>);

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

/// Cannonball SRP configuration using JEOD's `RadiationDefaultSurface` formula.
///
/// Force = (flux/c) * cx_area * [1 + albedo*diffuse*(4/9)] * flux_hat * illum_factor.
/// Mutually exclusive with `FlatPlateConfigC` (use one or the other).
///
/// Requires `SunMarker` entity in the world. Optional `ShadowBodyC` for eclipse.
/// Writes to `RadiationForceC`.
///
/// Auto-inserts [`RadiationForceC`] when added.
#[derive(Component, Debug, Clone, Copy)]
#[require(RadiationForceC)]
pub struct CannonballSrpC {
    /// Cross-section area * Cr (m²).
    pub cx_area: f64,
    /// Surface albedo (0–1).
    pub albedo: f64,
    /// Diffuse reflection fraction (0–1).
    pub diffuse: f64,
}
