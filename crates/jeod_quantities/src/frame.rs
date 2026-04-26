//! Reference-frame phantom markers.
//!
//! JEOD distinguishes reference frames at runtime via `RefFrameKind` + string
//! names. We lift that distinction to compile time: `Position<Inertial>` and
//! `Position<Ecef>` are distinct types and cannot be added.
//!
//! Downstream crates cannot define new frames — the `Frame` trait is sealed.
//! To add a frame, define it here.

use core::marker::PhantomData;

use crate::sealed::Sealed;

/// Compile-time reference frame tag.
///
/// Sealed: only `jeod_quantities` can implement this trait.
pub trait Frame: Sealed + 'static {
    /// Human-readable name for error messages and debug output.
    const NAME: &'static str;
}

/// Compile-time planet tag used to parameterize planet-fixed frames.
pub trait Planet: Sealed + 'static {
    const NAME: &'static str;
}

/// Compile-time vehicle tag used to parameterize vehicle-relative frames.
///
/// Sealed: only `jeod_quantities` can implement this trait.
///
/// Vehicle marker types are used with [`BodyFrame`], [`StructuralFrame`],
/// [`Lvlh`], and [`Ned`]. Because `Vehicle` is sealed, downstream crates
/// cannot define new vehicle tags — add them in `jeod_quantities`. (A
/// future phase may relax this via a re-exported `define_vehicle!` macro
/// that emits the sealed impl.)
pub trait Vehicle: Sealed + 'static {
    const NAME: &'static str;
}

// --- Planet markers -----------------------------------------------------------

macro_rules! planet_marker {
    ($name:ident, $human:literal) => {
        #[doc = concat!("Planet marker for ", $human, ".")]
        #[derive(Debug, Clone, Copy)]
        pub struct $name;
        impl Sealed for $name {}
        impl Planet for $name {
            const NAME: &'static str = $human;
        }
    };
}

planet_marker!(Earth, "Earth");
planet_marker!(Moon, "Moon");
planet_marker!(Sun, "Sun");
planet_marker!(Mars, "Mars");

// --- Frame markers ------------------------------------------------------------

/// Quasi-inertial (ICRF / J2000 Earth-centered inertial) frame.
#[derive(Debug, Clone, Copy)]
pub struct Inertial;
impl Sealed for Inertial {}
impl Frame for Inertial {
    const NAME: &'static str = "Inertial";
}

/// Earth-centered Earth-fixed frame (ITRF-like). Rotates with Earth.
#[derive(Debug, Clone, Copy)]
pub struct Ecef;
impl Sealed for Ecef {}
impl Frame for Ecef {
    const NAME: &'static str = "Ecef";
}

// NOTE on `NAME`: each frame's `NAME` const identifies the *frame kind*
// (e.g. "BodyFrame") rather than the embedded planet/vehicle tag. This is
// a `const &'static str`, so we can't splice `V::NAME` into it at compile
// time. Callers that need a fully-qualified string (including the vehicle
// or planet) should use `std::any::type_name::<F>()`, which `Qty3`'s
// `Debug` impl does.

/// Planet-fixed frame for any planet `P`. Rotates with that planet.
#[derive(Debug, Clone, Copy)]
pub struct PlanetFixed<P: Planet>(PhantomData<P>);
impl<P: Planet> Sealed for PlanetFixed<P> {}
impl<P: Planet> Frame for PlanetFixed<P> {
    const NAME: &'static str = "PlanetFixed";
}

/// Body (CoM-centered) frame of vehicle `V`. Rotates with the vehicle.
#[derive(Debug, Clone, Copy)]
pub struct BodyFrame<V: Vehicle>(PhantomData<V>);
impl<V: Vehicle> Sealed for BodyFrame<V> {}
impl<V: Vehicle> Frame for BodyFrame<V> {
    const NAME: &'static str = "BodyFrame";
}

/// Structural (geometric-origin) frame of vehicle `V`. Rotates with vehicle.
#[derive(Debug, Clone, Copy)]
pub struct StructuralFrame<V: Vehicle>(PhantomData<V>);
impl<V: Vehicle> Sealed for StructuralFrame<V> {}
impl<V: Vehicle> Frame for StructuralFrame<V> {
    const NAME: &'static str = "StructuralFrame";
}

/// Local Vertical / Local Horizontal frame relative to chief vehicle `Chief`.
/// Z axis points planet-ward; Y opposes orbital angular momentum; X completes
/// the right-handed triad (approximately along-track in near-circular orbits).
#[derive(Debug, Clone, Copy)]
pub struct Lvlh<Chief: Vehicle>(PhantomData<Chief>);
impl<C: Vehicle> Sealed for Lvlh<C> {}
impl<C: Vehicle> Frame for Lvlh<C> {
    const NAME: &'static str = "Lvlh";
}

/// North-East-Down topocentric frame relative to chief vehicle `Chief`.
#[derive(Debug, Clone, Copy)]
pub struct Ned<Chief: Vehicle>(PhantomData<Chief>);
impl<C: Vehicle> Sealed for Ned<C> {}
impl<C: Vehicle> Frame for Ned<C> {
    const NAME: &'static str = "Ned";
}

// --- Self-referential vehicle marker ----------------------------------------
//
// `Vehicle` is sealed, so downstream crates cannot mint their own phantom
// tags. The Bevy adapter (and any other ECS adapter) needs *some* tag to
// instantiate vehicle-parameterized frames (`BodyFrame`, `StructuralFrame`)
// on per-entity components. `SelfRef` is the canonical "this entity's own
// vehicle frame" tag — it stands in for the (compile-time-unknown) entity
// without leaking generics into user-facing `Query`s.
//
// Distinct from `TestVehicle`: `SelfRef` is part of the production API
// surface (always compiled in), while `TestVehicle` is feature-gated for
// test harnesses only.

/// Phantom marker for "this entity's own vehicle frame" — used by ECS
/// adapters whose per-entity components carry frame phantoms but whose
/// vehicle identity is determined at runtime by the entity itself.
///
/// Use with [`BodyFrame`], [`StructuralFrame`], [`Lvlh`], [`Ned`], or any
/// `Vehicle`-parameterized type when the runtime entity *is* the vehicle.
/// Never appears in user-facing `Query`s — components wrap concrete
/// monomorphizations like `Position<Inertial>` or
/// `Torque<BodyFrame<SelfRef>>`, so the user sees only the wrapper newtype.
#[derive(Debug, Clone, Copy)]
pub struct SelfRef;
impl Sealed for SelfRef {}
impl Vehicle for SelfRef {
    const NAME: &'static str = "SelfRef";
}

// --- Test-only vehicle marker ------------------------------------------------
//
// Tests that exercise vehicle-parameterized frames (`Lvlh`, `Ned`,
// `BodyFrame`, `StructuralFrame`) sometimes need *another* tag distinct
// from `SelfRef`, so we expose a single test-only vehicle behind the
// `test-utils` feature. It is never compiled into production builds.

/// A no-op vehicle phantom marker for use in downstream test harnesses.
///
/// Available only when the crate is built with `--features test-utils`.
#[cfg(feature = "test-utils")]
#[derive(Debug, Clone, Copy)]
pub struct TestVehicle;
#[cfg(feature = "test-utils")]
impl Sealed for TestVehicle {}
#[cfg(feature = "test-utils")]
impl Vehicle for TestVehicle {
    const NAME: &'static str = "TestVehicle";
}
