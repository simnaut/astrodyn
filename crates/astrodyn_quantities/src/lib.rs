//! # astrodyn_quantities
//!
//! Dimensional-analysis and phantom-tag foundation for `astrodyn_bevy`.
//!
//! This crate sits at the bottom of the workspace dependency graph. Every other
//! `astrodyn_*` crate, plus `astrodyn`, `astrodyn_runner`, and the `astrodyn_bevy` Bevy glue
//! all depend on it for typed quantities and phantom frame / time-scale tags.
//!
//! ## Three-layer facade
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │ Facade  (astrodyn_bevy::prelude, astrodyn::recipes)          │
//! │   F64Ext: 400.0.km(), 51.6.deg(), 420_000.0.kg()         │
//! │   Concrete Component wrappers (no visible generics)      │
//! │   Custom #[diagnostic::on_unimplemented] messages        │
//! ├──────────────────────────────────────────────────────────┤
//! │ Typed astrodyn_* siblings                                     │
//! │   Position<F: Frame>, SecondsSince<S: TimeScale>,        │
//! │   Quat<L, T>, NormalizedQuat, FrameTransform<From, To>   │
//! ├──────────────────────────────────────────────────────────┤
//! │ astrodyn_quantities  (you are here)                          │
//! │   uom re-exports, Qty3<D, F>, phantom frames/scales,     │
//! │   F64Ext / Vec3Ext / Array3Ext                           │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! Mission-crate code consumes the facade layer and never sees `PhantomData`
//! or `uom::si::*` paths. Internal physics kernels drop down to raw
//! `glam::DVec3`/`f64` for arithmetic density via `.raw_si()` and re-wrap on
//! exit. See the Type-System and Strategy wiki pages
//! (<https://github.com/simnaut/astrodyn/wiki>) for architecture.
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
//! - **Time-scale separation** — `SecondsSince<S>` has no `Add`/`Sub`
//!   impl across distinct scales, so direct arithmetic between (e.g.)
//!   `SecondsSince<TAI>` and `SecondsSince<TT>` is rejected. The intended
//!   way to combine scales is via `TimeConverter::apply`. The raw
//!   `SecondsSince::from_seconds` / `as_seconds` boundaries can still
//!   relabel an `f64` across scales without applying the offset — see
//!   "Where the guards stop" below.
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
//! - **Vehicle-phantom mismatch** on the Act-5 phantom-wrapped types
//!   (`FlatPlate<V>`, `AttachEvent<VParent, VChild>`,
//!   `RelativeState<Subject, Reference>`,
//!   `RelativeTranslation<Reference>`, `LvlhRelativeState<Chief>`) is
//!   surfaced via the [`diagnostics::CompatibleVehicles`] /
//!   [`diagnostics::CompatibleVehiclePair`] markers. Each type exposes
//!   a zero-cost `assert_*` witness method whose `where` bound resolves
//!   only when the vehicle phantoms agree; on mismatch the diagnostic
//!   names the expected and found vehicles in physics language instead
//!   of a `PhantomData<…>` wall. Mission code reaches for the witness
//!   at the boundary that hands the value to a slot of a specific
//!   identity (e.g. `plate.assert_vehicle::<Iss>()`).
//!
//! ### Scaffolded but not currently wired
//!
//! [`diagnostics::IntoLength`], [`diagnostics::IntoAngle`],
//! [`diagnostics::IntoGravParam`], [`diagnostics::CompatibleGravParam`],
//! [`diagnostics::CompatibleTimeScales`],
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
//! The crate has several public raw-value boundaries where the
//! type-system guards end and the caller takes responsibility for
//! correctness:
//!
//! - [`Qty3::raw_si`] / [`Qty3::from_raw_si`] — into and out of raw
//!   `glam::DVec3` (frame tag and dimension are erased on `raw_si`,
//!   reattached without check on `from_raw_si`).
//! - `SecondsSince::from_seconds` / `as_seconds` — into and out of raw
//!   `f64` seconds (time-scale tag is erased on `as_seconds`, reattached
//!   without check on `from_seconds`, which is how a `TAI` reading can
//!   be relabeled as `TT` without applying the 32.184 s offset).
//! - `JeodQuat::from_array` — accepts a raw `[f64; 4]` without
//!   normalization or convention checks. Use `NormalizedQuat::new(q)?`
//!   to recover the unit-norm witness.
//! - `FrameTransform::from_matrix` — accepts a raw `DMat3` without
//!   checking orthogonality. The validating sibling is
//!   `FrameTransform::from_matrix_validated`.
//!
//! These boundaries are deliberate: inside the `_inner` / `_impl`
//! kernels of `astrodyn_*` crates the math runs on `f64` / `DVec3` /
//! `DMat3` for arithmetic density. Unit, frame, time-scale, and
//! quaternion-convention slips inside a kernel are caught only at the
//! `F64Ext` and typed-API ingestion boundary — once you've crossed into
//! a raw representation the caller is responsible.
//!
//! See the [Type-System wiki page] for the contributor primer (phantom-tag
//! pattern, adding a new frame/scale/quantity, reading compiler errors,
//! escape hatches) and `examples/typed_mission.rs` for the canonical
//! worked example.
//!
//! [Type-System wiki page]: https://github.com/simnaut/astrodyn/wiki/Type-System
//!
//! ## Quick start
//!
//! ```
//! use astrodyn_quantities::prelude::*;
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
pub mod body_constants;
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
