//! Reference-frame phantom markers.
//!
//! JEOD distinguishes reference frames at runtime via `RefFrameKind` + string
//! names. We lift that distinction to compile time: `Position<Inertial>` and
//! `Position<Ecef>` are distinct types and cannot be added.
//!
//! ## Extension model
//!
//! - **Frame *kinds*** (`Inertial`, `Ecef`, `BodyFrame<V>`, …) are sealed
//!   to this crate. Adding a new kind requires editing this file.
//! - **Vehicle and Planet *parameter* tags** (the `V` in `BodyFrame<V>`,
//!   the `P` in `PlanetFixed<P>`) are extensible from downstream crates
//!   via the [`define_vehicle!`](crate::define_vehicle) and
//!   [`define_planet!`](crate::define_planet) macros, defined at the
//!   bottom of this file. Mission crates that model multiple vehicles
//!   should use those macros so each vehicle gets a distinct
//!   compile-time identity.

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
///
/// Sealed: only `jeod_quantities` can implement this trait directly.
/// Downstream crates extend the catalog via the
/// [`define_planet!`](crate::define_planet) macro.
pub trait Planet: Sealed + 'static {
    const NAME: &'static str;
}

/// Compile-time vehicle tag used to parameterize vehicle-relative frames.
///
/// Sealed: only `jeod_quantities` can implement this trait directly.
/// Downstream crates extend the catalog via the
/// [`define_vehicle!`](crate::define_vehicle) macro, which generates
/// the sealed impl from inside this crate so the seal is preserved.
///
/// Vehicle marker types are used with [`BodyFrame`], [`StructuralFrame`],
/// [`Lvlh`], and [`Ned`].
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

/// Phantom marker for "this entity's own planet" — used by ECS adapters
/// whose per-entity components carry `PlanetFixed<P>` phantoms but whose
/// planet identity is determined at runtime by the entity itself.
///
/// Planet-side analog of [`SelfRef`] (which serves the same role for
/// `Vehicle`-parameterized frames). Use when wrapping a per-entity
/// rotation Component such as `PlanetFixedRotationC` whose Bevy entity
/// already carries a `PlanetC` discriminator — the typed `FrameTransform`
/// inside the Component encodes the *direction* (Inertial → PlanetFixed)
/// while the planet identity stays at the entity level.
#[derive(Debug, Clone, Copy)]
pub struct SelfPlanet;
impl Sealed for SelfPlanet {}
impl Planet for SelfPlanet {
    const NAME: &'static str = "SelfPlanet";
}

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

// --- Macros for downstream marker types --------------------------------------
//
// Mission crates that model multiple vehicles (e.g., a chief + deputy
// formation, a tug + payload, the ISS plus a visiting Soyuz) need
// distinct compile-time `Vehicle` markers so `BodyFrame<Iss>` and
// `BodyFrame<Soyuz>` are type-distinct. The same applies to multi-planet
// scenarios (e.g., a Mars sample-return mission carrying state in both
// Mars-fixed and Earth-fixed frames).
//
// These macros generate the marker struct + sealed impl + trait impl in
// one statement. They are the **only** way for a downstream crate to add
// a `Vehicle` or `Planet` marker; direct `impl Vehicle for X {}` cannot
// satisfy the `Sealed` bound because `Sealed` is private to this crate.
// The macro reaches `Sealed` via the `$crate::__macro_support` re-export
// so the seal is preserved even though the impl is generated downstream.
//
// Per-instance names (`"Iss"`, `"Soyuz"`, …) come from `stringify!($name)`.
// `Frame::NAME` cannot splice `V::NAME` (it is a `&'static str` const, not
// a `const fn`); callers that need a fully-qualified name should use
// `std::any::type_name::<F>()`, which is what `Qty3`'s `Debug` impl does.

/// Define a new compile-time `Vehicle` marker type.
///
/// Generates `pub struct $name;` plus the sealed `Vehicle` impl. The
/// resulting type is zero-sized and `Copy`. After `define_vehicle!(Iss);`
/// you can use `BodyFrame<Iss>`, `StructuralFrame<Iss>`, `Lvlh<Iss>`, and
/// `Ned<Iss>` as distinct frame phantoms.
///
/// # Example
///
/// ```
/// use jeod_quantities::define_vehicle;
/// use jeod_quantities::prelude::*;
///
/// define_vehicle!(Iss);
/// define_vehicle!(Soyuz);
///
/// // Position<BodyFrame<Iss>> and Position<BodyFrame<Soyuz>> are distinct
/// // types — adding one to the other is a compile error.
/// let _iss_pos: Position<BodyFrame<Iss>> = Qty3::zero();
/// let _soyuz_pos: Position<BodyFrame<Soyuz>> = Qty3::zero();
/// ```
///
/// # Frame mismatches are caught at compile time
///
/// ```compile_fail
/// use jeod_quantities::define_vehicle;
/// use jeod_quantities::prelude::*;
///
/// define_vehicle!(Iss);
/// define_vehicle!(Soyuz);
///
/// let iss: Position<BodyFrame<Iss>> = Qty3::zero();
/// let soyuz: Position<BodyFrame<Soyuz>> = Qty3::zero();
/// // Mixing frames is a compile error with a physics-language diagnostic:
/// let _bad = iss + soyuz;
/// ```
///
/// # Sealing
///
/// Downstream crates cannot write `impl Vehicle for X {}` directly; the
/// `Vehicle` trait has a `Sealed` super-bound and `Sealed` is private to
/// `jeod_quantities`. This macro is the only way to extend the `Vehicle`
/// catalog. The macro body uses the crate's `__macro_support` re-export
/// to satisfy the bound from a downstream call site.
#[macro_export]
macro_rules! define_vehicle {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name;
        impl $crate::__macro_support::Sealed for $name {}
        impl $crate::__macro_support::Vehicle for $name {
            const NAME: &'static str = stringify!($name);
        }
    };
}

/// Define a new compile-time `Planet` marker type.
///
/// Generates `pub struct $name;` plus the sealed `Planet` impl. The
/// resulting type is zero-sized and `Copy`. After `define_planet!(Pluto);`
/// you can use `PlanetFixed<Pluto>` as a distinct frame phantom.
///
/// # Example
///
/// ```
/// use jeod_quantities::define_planet;
/// use jeod_quantities::prelude::*;
///
/// define_planet!(Pluto);
///
/// // PlanetFixed<Pluto> is a distinct frame from PlanetFixed<Earth>.
/// let _pluto_fixed: Position<PlanetFixed<Pluto>> = Qty3::zero();
/// ```
///
/// # Sealing
///
/// Same sealing pattern as [`define_vehicle!`]: downstream crates cannot
/// impl `Planet` directly, only via this macro.
#[macro_export]
macro_rules! define_planet {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name;
        impl $crate::__macro_support::Sealed for $name {}
        impl $crate::__macro_support::Planet for $name {
            const NAME: &'static str = stringify!($name);
        }
    };
}

#[cfg(test)]
mod macro_tests {
    use crate::aliases::Position;
    use crate::frame::{BodyFrame, PlanetFixed};
    use crate::qty3::Qty3;

    // Use the macros from inside this crate; they expand to
    // `$crate::__macro_support::*` which resolves to
    // `crate::__macro_support::*` here.
    crate::define_vehicle!(Iss);
    crate::define_vehicle!(Soyuz);
    crate::define_planet!(Pluto);

    #[test]
    fn distinct_vehicle_phantoms_construct_and_default() {
        let iss: Position<BodyFrame<Iss>> = Qty3::zero();
        let soyuz: Position<BodyFrame<Soyuz>> = Qty3::zero();
        // Both are valid Position values in their respective frames.
        // Mixing them is a compile error — see the trybuild-style
        // compile_fail doctests on the macros themselves.
        assert_eq!(iss.raw_si(), glam::DVec3::ZERO);
        assert_eq!(soyuz.raw_si(), glam::DVec3::ZERO);
    }

    #[test]
    fn distinct_planet_phantom_constructs() {
        let pluto: Position<PlanetFixed<Pluto>> = Qty3::zero();
        assert_eq!(pluto.raw_si(), glam::DVec3::ZERO);
    }

    #[test]
    fn vehicle_name_carries_through_typename() {
        // `Frame::NAME` is the kind ("BodyFrame"); the per-vehicle
        // identifier is recoverable via `type_name`. This is the
        // documented escape hatch for diagnostics.
        let n = std::any::type_name::<Iss>();
        assert!(
            n.ends_with("Iss"),
            "expected type_name to end in Iss, got {n}"
        );
    }
}
