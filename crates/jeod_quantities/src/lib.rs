//! # jeod_quantities
//!
//! Dimensional-analysis and phantom-tag foundation for `bevy_jeod`.
//!
//! This crate sits at the bottom of the workspace dependency graph. Every other
//! `jeod_*` crate, plus `jeod_sim`, `jeod_runner`, and the `bevy_jeod` Bevy glue
//! all depend on it for typed quantities and phantom frame / time-scale tags.
//!
//! ## Three-layer facade
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │ Facade  (bevy_jeod::prelude, jeod_sim::recipes)          │
//! │   F64Ext: 400.0.km(), 51.6.deg(), 420_000.0.kg()         │
//! │   Concrete Component wrappers (no visible generics)      │
//! │   Custom #[diagnostic::on_unimplemented] messages        │
//! ├──────────────────────────────────────────────────────────┤
//! │ Typed jeod_* siblings                                     │
//! │   Position<F: Frame>, SecondsSince<S: TimeScale>,        │
//! │   Quat<L, T>, NormalizedQuat, FrameTransform<From, To>   │
//! ├──────────────────────────────────────────────────────────┤
//! │ jeod_quantities  (you are here)                          │
//! │   uom re-exports, Qty3<D, F>, phantom frames/scales,     │
//! │   F64Ext / Vec3Ext / Array3Ext                           │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! Mission-crate code consumes the facade layer and never sees `PhantomData`
//! or `uom::si::*` paths. Internal physics kernels drop down to raw
//! `glam::DVec3`/`f64` for arithmetic density via `.raw_si()` and re-wrap on
//! exit. See `docs/type_system.md` and `STRATEGY.md` §8 for architecture.
//!
//! ## What this crate provides
//!
//! - Reference-frame and time-scale phantom markers (`Inertial`, `Ecef`,
//!   `PlanetFixed<P>`, `BodyFrame<V>`, `Lvlh<Chief>`, `TAI`, `TT`, …)
//! - `uom`-backed componentwise 3-vectors `Qty3<D, F>` with aliases
//!   `Position<F>`, `Velocity<F>`, `Acceleration<F>`, `Force<F>`, `Torque<F>`, …
//! - Quaternion convention tags (`ScalarFirst`/`ScalarLast`,
//!   `LeftTransform`/`RightTransform`) plus a `NormalizedQuat`
//!   constructor-gated witness
//! - Typed `FrameTransform<From, To>` composing only when inner frames match
//! - The `F64Ext` facade (`400.0.km()`, `51.6.deg()`, `420_000.0.kg()`)
//! - Compiler error messages in physics language via
//!   `#[diagnostic::on_unimplemented]`
//!
//! ## Quick start
//!
//! ```
//! use jeod_quantities::prelude::*;
//!
//! let altitude = 400.0.km();
//! let inclination = 51.6.deg();
//! let mass = 420_000.0.kg();
//! ```

#![forbid(unsafe_code)]

mod sealed;

pub mod aliases;
pub mod diagnostics;
pub mod dims;
pub mod ext;
pub mod frame;
pub mod frame_transform;
pub mod harmonic;
pub mod inertia;
pub mod ops;
pub mod prelude;
pub mod qty3;
pub mod quat;
pub mod time_scale;

pub use aliases::*;
pub use dims::*;
pub use frame::*;
pub use frame_transform::*;
pub use qty3::*;
pub use quat::*;
pub use time_scale::*;
// `inertia::InertiaTensor` is re-exported via `aliases::*`.

/// Internal re-exports used by the `define_vehicle!` / `define_planet!`
/// macros to satisfy the per-domain sealed-trait bound at the downstream
/// call site.
///
/// Only `VehicleSealed` and `PlanetSealed` are re-exported. The other
/// three seal traits (`FrameSealed`, `TimeScaleSealed`, `QuatSealed`)
/// stay private to the crate, so `Frame`, `TimeScale`, `Layout`, and
/// `Transform` remain sealed at the type-system level — downstream code
/// cannot impl them at all.
///
/// **Do not import items from this module directly.** It exists only so
/// that macro expansions can satisfy the seal bounds. A direct
/// `impl VehicleSealed for X` outside the `define_vehicle!` macro is
/// technically possible but unsupported and may break in any release.
#[doc(hidden)] // allowed: macro infrastructure for define_vehicle!/define_planet!
pub mod __macro_support {
    pub use crate::frame::{Planet, Vehicle};
    pub use crate::sealed::{PlanetSealed, VehicleSealed};
}
