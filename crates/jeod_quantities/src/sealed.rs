//! Per-domain sealed-trait pattern.
//!
//! Each `*Sealed` trait gates a single category of public marker trait so
//! that downstream crates cannot impl that category directly. Splitting
//! the seals (rather than using one shared `Sealed`) lets the
//! `define_vehicle!` / `define_planet!` macros open exactly the
//! `Vehicle` and `Planet` catalogs (via the `__macro_support`
//! re-export) while keeping `Frame`, `TimeScale`, `Layout`, and
//! `Transform` closed at the type-system level.
//!
//! - `FrameSealed`: gates `Frame` impls. Closed.
//! - `TimeScaleSealed`: gates `TimeScale` impls. Closed.
//! - `QuatSealed`: gates `Layout` and `Transform` impls. Closed.
//! - `VehicleSealed`: gates `Vehicle` impls. Re-exported via
//!   `__macro_support` so `define_vehicle!` can satisfy the bound from
//!   downstream call sites. Convention-sealed: only the macro should
//!   produce impls.
//! - `PlanetSealed`: gates `Planet` impls. Same treatment as
//!   `VehicleSealed`.

pub trait FrameSealed {}
pub trait VehicleSealed {}
pub trait PlanetSealed {}
pub trait TimeScaleSealed {}
pub trait QuatSealed {}
