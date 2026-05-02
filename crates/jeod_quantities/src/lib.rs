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
//! exit. See the Type-System and Strategy wiki pages
//! (<https://github.com/simnaut/bevy_jeod/wiki>) for architecture.
//!
//! ## What this crate provides
//!
//! - Reference-frame and time-scale phantom markers (`RootInertial`, `Ecef`,
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
//! ## Compile-time guards
//!
//! Beyond `uom`'s built-in dimensional analysis, this crate adds layered
//! type-system guards specific to orbital-mechanics conventions. The
//! actively-wired set:
//!
//! - **Dimensional mismatch** (uom-native): `Position + Mass` is rejected
//!   because `Qty3`'s `Add` impl requires identical dimension `D`.
//! - **Frame mismatch on `+`/`-`/`+=`/`-=`** — `Position<Ecef> +
//!   Position<RootInertial>` fails with a tailored
//!   `#[diagnostic::on_unimplemented]` message pointing at
//!   `FrameTransform`. Wired via the
//!   [`diagnostics::CompatibleFrames`]`<Fl, Fr>` bound in [`ops`].
//! - **Time-scale separation** — `SecondsSince<S>` has no `Add`/`Sub` impl
//!   across distinct scales; the only way to combine scales is via
//!   `TimeConverter::apply`. Cross-scale arithmetic is structurally
//!   impossible.
//! - **Quaternion convention separation** — layout (`ScalarFirst` vs
//!   `ScalarLast`), transform convention (`LeftTransform` vs
//!   `RightTransform`), and normalization status are distinct phantom
//!   tags. `NormalizedQuat<L, T>` is a separate type from `Quat<L, T>`,
//!   so a raw `Quat` cannot stand in where a unit quaternion is required.
//! - **`FrameTransform<From, To>` composition** only typechecks when the
//!   two inner frames align (`A→B ∘ B→C`); the identity is only defined
//!   for `From = To`.
//! - **Cross-dimension multiply / divide on `Qty3`** uses `typenum`
//!   exponent arithmetic, so `Velocity × Time → Position` and
//!   `Acceleration × Time → Velocity` are type-safe by construction; a
//!   bad combination produces a dimension that won't unify with the
//!   target type.
//! - **Inertial-flavor distinctions** (issue #255 / `RF.10`):
//!   `RootInertial`, `PlanetInertial<P>`, and `IntegrationFrame` are
//!   kind-distinct phantoms. Body integration-frame state cannot silently
//!   flow into root-inertial-only consumers (gravity, SRP, relativistic);
//!   the only safe transition is via [`IntegOrigin`].
//!
//! ### Scaffolded but not currently wired
//!
//! [`diagnostics::IntoLength`], [`diagnostics::IntoAngle`],
//! [`diagnostics::IntoGravParam`], [`diagnostics::CompatibleTimeScales`],
//! [`diagnostics::CompatibleQuatLayouts`],
//! [`diagnostics::CompatibleQuatTransforms`],
//! [`diagnostics::RequiresNormalizedQuat`],
//! [`diagnostics::InertialOnly`], and [`diagnostics::NoVectorVectorMul`]
//! carry tailored diagnostic messages but are not currently used as
//! `where` bounds by any impl in the workspace. Today, for example,
//! passing `400_000.0` where a `Length` is expected produces uom's stock
//! "mismatched types" error rather than the `IntoLength` hint, and
//! `Qty3 * Qty3` produces a default "no `Mul` impl" rather than the
//! `NoVectorVectorMul` hint. `F64Ext` discoverability today comes from
//! the prelude and worked examples. These scaffolds let a future
//! contributor flip on an active guard without touching call sites — the
//! diagnostic message is already in place.
//!
//! ### Where the guards stop
//!
//! [`Qty3::raw_si`] and [`Qty3::from_raw_si`] are the documented escape
//! hatches into raw `glam::DVec3`. Inside the `_inner` / `_impl` kernels
//! of `jeod_*` crates that operate on `f64` / `DVec3`, the dimensional
//! and frame guards are absent by design — that's where arithmetic
//! density lives. Unit slips *inside* a kernel (m vs ft) are caught only
//! at the `F64Ext` ingestion boundary; once you've called `.raw_si()` the
//! caller is responsible for keeping things in SI base units.
//!
//! See the [Type-System wiki page] for the contributor primer (phantom-tag
//! pattern, adding a new frame/scale/quantity, reading compiler errors,
//! escape hatches) and `examples/typed_mission.rs` for the canonical
//! worked example.
//!
//! [Type-System wiki page]: https://github.com/simnaut/bevy_jeod/wiki/Type-System
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
#![deny(missing_docs)]

mod sealed;

pub mod aliases;
pub mod body_attitude;
pub mod diagnostics;
pub mod dims;
pub mod ext;
pub mod frame;
pub mod frame_transform;
pub mod harmonic;
pub mod inertia;
pub mod integ_origin;
pub mod ops;
pub mod prelude;
pub mod qty3;
pub mod quat;
pub mod time_scale;

pub use aliases::*;
pub use body_attitude::BodyAttitude;
pub use dims::*;
pub use frame::*;
pub use frame_transform::*;
pub use integ_origin::IntegOrigin;
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
