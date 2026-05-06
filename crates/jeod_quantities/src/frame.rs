//! Reference-frame phantom markers.
//!
//! JEOD distinguishes reference frames at runtime via `RefFrameKind` + string
//! names. We lift that distinction to compile time: `Position<RootInertial>` and
//! `Position<Ecef>` are distinct types and cannot be added.
//!
//! ## Extension model
//!
//! - **Frame *kinds*** (`RootInertial`, `Ecef`, `BodyFrame<V>`, …) are sealed
//!   to this crate. Adding a new kind requires editing this file.
//! - **Vehicle and Planet *parameter* tags** (the `V` in `BodyFrame<V>`,
//!   the `P` in `PlanetFixed<P>`) are extensible from downstream crates
//!   via the [`define_vehicle!`](crate::define_vehicle) and
//!   [`define_planet!`](crate::define_planet) macros, defined at the
//!   bottom of this file. Mission crates that model multiple vehicles
//!   should use those macros so each vehicle gets a distinct
//!   compile-time identity.

use core::marker::PhantomData;

use crate::sealed::{FrameSealed, PlanetSealed, VehicleSealed};

/// Compile-time reference frame tag.
///
/// Sealed at the type-system level: only `jeod_quantities` can implement
/// this trait. The seal trait `FrameSealed` is private to this crate, so
/// downstream code cannot satisfy the supertrait bound.
///
/// # The seal is type-system enforced
///
/// Unlike [`Vehicle`] and [`Planet`] (which expose their seal traits via
/// `__macro_support` so the `define_*!` macros work cross-crate),
/// `Frame`'s seal is fully closed. Downstream code cannot impl `Frame`
/// even by reaching into the macro infrastructure:
///
/// ```compile_fail
/// // `FrameSealed` is private to jeod_quantities and is NOT re-exported
/// // via __macro_support, so the supertrait bound cannot be satisfied
/// // from outside the crate.
/// struct EvilFrame;
/// impl jeod_quantities::Frame for EvilFrame {
///     const NAME: &'static str = "Evil";
/// }
/// ```
pub trait Frame: FrameSealed + 'static {
    /// Human-readable name for error messages and debug output.
    const NAME: &'static str;
}

/// Compile-time planet tag used to parameterize planet-fixed frames.
///
/// Convention-sealed: the seal trait `PlanetSealed` is re-exported via
/// the crate's `__macro_support` module so [`define_planet!`](crate::define_planet)
/// can satisfy the bound from downstream call sites. Direct
/// `impl Planet for X` outside the macro is technically possible but
/// unsupported. Use the macro.
///
/// `Send + Sync` are required so `PhantomData<P>`-tagged ECS components
/// (e.g. `bevy_jeod::TranslationalStateC<P>`) can be moved between
/// threads under Bevy's parallel scheduler. Every Planet marker is a
/// zero-sized empty struct that trivially satisfies both — the
/// supertrait bound just makes the contract explicit.
pub trait Planet: PlanetSealed + Send + Sync + 'static {
    /// Human-readable name for error messages and debug output.
    const NAME: &'static str;
}

/// Compile-time vehicle tag used to parameterize vehicle-relative frames.
///
/// Convention-sealed: the seal trait `VehicleSealed` is re-exported via
/// the crate's `__macro_support` module so [`define_vehicle!`](crate::define_vehicle)
/// can satisfy the bound from downstream call sites. Direct
/// `impl Vehicle for X` outside the macro is technically possible but
/// unsupported. Use the macro.
///
/// Vehicle marker types are used with [`BodyFrame`], [`StructuralFrame`],
/// [`Lvlh`], and [`Ned`].
///
/// `Send + Sync` are required so `PhantomData<V>`-tagged ECS components
/// can be moved between threads under Bevy's parallel scheduler. Every
/// Vehicle marker is a zero-sized empty struct that trivially satisfies
/// both — the supertrait bound just makes the contract explicit and
/// mirrors the parallel symmetry with [`Planet`].
pub trait Vehicle: VehicleSealed + Send + Sync + 'static {
    /// Human-readable name for error messages and debug output.
    const NAME: &'static str;
}

// --- Planet markers -----------------------------------------------------------

macro_rules! planet_marker {
    ($name:ident, $human:literal) => {
        #[doc = concat!("Planet marker for ", $human, ".")]
        #[derive(Debug, Clone, Copy)]
        pub struct $name;
        impl PlanetSealed for $name {}
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
/// inside the Component encodes the *direction* (RootInertial → PlanetFixed)
/// while the planet identity stays at the entity level.
#[derive(Debug, Clone, Copy)]
pub struct SelfPlanet;
impl PlanetSealed for SelfPlanet {}
impl Planet for SelfPlanet {
    const NAME: &'static str = "SelfPlanet";
}

// --- Frame markers ------------------------------------------------------------

/// The simulation's **root inertial frame** — the unique inertial node
/// at the top of the frame tree.
///
/// In all current sims this is Earth-centered (ICRF / J2000) because
/// Earth is the central body, but the semantic is "whatever the
/// simulation's root inertial frame is" — for an SSB-rooted setup it
/// would be heliocentric/barycentric.
///
/// Distinct from [`PlanetInertial<P>`] (a particular planet's inertial
/// frame, which may be a non-root child of `RootInertial` in the tree) and
/// from [`IntegrationFrame`] (a body's integration frame, which equals
/// `RootInertial` when the body integrates in root and equals some
/// `PlanetInertial<P>` when it integrates in a child).
///
/// Consumers that mix body state with root-inertial source positions
/// (gravity sources, Sun, Moon) — gravity, relativistic, SRP, solar
/// beta, earth lighting — require their position arguments in this
/// frame. See issue #255 / RF.10.
#[derive(Debug, Clone, Copy)]
pub struct RootInertial;
impl FrameSealed for RootInertial {}
impl Frame for RootInertial {
    const NAME: &'static str = "RootInertial";
}

/// A particular planet's inertial (non-rotating, J2000-aligned) frame,
/// centered at the planet's CoM.
///
/// Distinct from [`RootInertial`] (the simulation root) and from
/// [`PlanetFixed<P>`] (the planet's rotating, body-fixed frame).
/// `PlanetInertial<P>` is the parent frame of `PlanetFixed<P>` in JEOD's
/// frame tree convention, so the rotation matrix
/// `PlanetFixed<P>::t_parent_this` rotates `PlanetInertial<P> →
/// PlanetFixed<P>`.
///
/// Consumers that operate around a single planet (atmosphere, drag
/// velocity vs corotation wind, LVLH, geodetic, orbital elements) take
/// their position/velocity arguments in this frame. In realistic
/// configs the body's integration frame *is* `PlanetInertial<P>` for
/// the planet the body orbits, so `body.trans.position` (typed
/// `Position<IntegrationFrame>`) can be relabeled to
/// `Position<PlanetInertial<P>>` at the call site without arithmetic.
///
/// The runner currently invokes consumers with raw `DVec3` (without
/// static `P` dispatch) and asserts the runtime invariant
/// `body.integ_source == consumer_planet_source`. Mission-crate code
/// that knows the planet at compile time should prefer the typed
/// `_typed_for_planet` siblings.
#[derive(Debug, Clone, Copy)]
pub struct PlanetInertial<P: Planet>(PhantomData<P>);
impl<P: Planet> FrameSealed for PlanetInertial<P> {}
impl<P: Planet> Frame for PlanetInertial<P> {
    const NAME: &'static str = "PlanetInertial";
}

/// Earth-centered Earth-fixed frame (ITRF-like). Rotates with Earth.
#[derive(Debug, Clone, Copy)]
pub struct Ecef;
impl FrameSealed for Ecef {}
impl Frame for Ecef {
    const NAME: &'static str = "Ecef";
}

/// A body's integration frame — a non-rotating quasi-inertial frame whose
/// origin generally differs from the root inertial origin.
///
/// `IntegrationFrame` is *kind-distinct* from [`RootInertial`] so that APIs
/// requiring root-inertial coordinates refuse `Position<IntegrationFrame>`
/// at compile time. **Root-inertial consumers** (the "shift sites" — they
/// mix body state with root-inertial source positions for Sun, Moon, or
/// gravity sources): gravity, relativistic corrections, SRP, solar beta,
/// earth lighting. **Planet-inertial consumers** (the "non-shift sites"
/// — they operate within a single planet's inertial frame, which equals
/// the body's integration frame in realistic configs): atmosphere, drag
/// velocity, LVLH, geodetic, orbital elements; these take
/// [`PlanetInertial<P>`] inputs and must NOT receive a root-inertial
/// shift, since shifting would change the planet-relative coordinates.
///
/// The only safe transition from `Position<IntegrationFrame>` to
/// `Position<RootInertial>` is the integration-origin shift, exposed via
/// [`IntegOrigin::shift_position`](crate::integ_origin::IntegOrigin::shift_position)
/// and the `to_inertial` method on the typed translational-state wrapper
/// in `jeod_dynamics`. There is no implicit `From`/`Into` — forgetting
/// the shift at a root-inertial consumer is the bug class this phantom
/// catches (issue #255 / `RF.10` in `docs/JEOD_invariants.md`).
///
/// For all sims that integrate in the root frame, the shift is by
/// `IntegOrigin::zero()` and is a no-op (bit-identical numerics).
#[derive(Debug, Clone, Copy)]
pub struct IntegrationFrame;
impl FrameSealed for IntegrationFrame {}
impl Frame for IntegrationFrame {
    const NAME: &'static str = "IntegrationFrame";
}

// NOTE on `NAME`: each frame's `NAME` const identifies the *frame kind*
// (e.g. "BodyFrame") rather than the embedded planet/vehicle tag. This is
// a `const &'static str`, so we can't splice `V::NAME` into it at compile
// time. Callers that need a fully-qualified string (including the vehicle
// or planet) should use `std::any::type_name::<F>()`, which `Qty3`'s
// `Debug` impl does.

/// Planet-fixed frame for any planet `P`. Rotates with that planet.
#[derive(Debug)]
pub struct PlanetFixed<P: Planet>(PhantomData<P>);
// Manual `Clone` / `Copy` impls so `FrameTransform<_, PlanetFixed<P>>` can
// be cloned/copied even when `P` itself doesn't bound `Clone` / `Copy`.
// `PlanetFixed<P>` is just `PhantomData<P>` — zero-sized — so both impls
// are trivially correct.
impl<P: Planet> Clone for PlanetFixed<P> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<P: Planet> Copy for PlanetFixed<P> {}
impl<P: Planet> FrameSealed for PlanetFixed<P> {}
impl<P: Planet> Frame for PlanetFixed<P> {
    const NAME: &'static str = "PlanetFixed";
}

/// Body (CoM-centered) frame of vehicle `V`. Rotates with the vehicle.
#[derive(Debug, Clone, Copy)]
pub struct BodyFrame<V: Vehicle>(PhantomData<V>);
impl<V: Vehicle> FrameSealed for BodyFrame<V> {}
impl<V: Vehicle> Frame for BodyFrame<V> {
    const NAME: &'static str = "BodyFrame";
}

/// Structural (geometric-origin) frame of vehicle `V`. Rotates with vehicle.
#[derive(Debug, Clone, Copy)]
pub struct StructuralFrame<V: Vehicle>(PhantomData<V>);
impl<V: Vehicle> FrameSealed for StructuralFrame<V> {}
impl<V: Vehicle> Frame for StructuralFrame<V> {
    const NAME: &'static str = "StructuralFrame";
}

/// Local Vertical / Local Horizontal frame relative to chief vehicle `Chief`.
/// Z axis points planet-ward; Y opposes orbital angular momentum; X completes
/// the right-handed triad (approximately along-track in near-circular orbits).
#[derive(Debug, Clone, Copy)]
pub struct Lvlh<Chief: Vehicle>(PhantomData<Chief>);
impl<C: Vehicle> FrameSealed for Lvlh<C> {}
impl<C: Vehicle> Frame for Lvlh<C> {
    const NAME: &'static str = "Lvlh";
}

/// North-East-Down topocentric frame relative to chief vehicle `Chief`.
#[derive(Debug, Clone, Copy)]
pub struct Ned<Chief: Vehicle>(PhantomData<Chief>);
impl<C: Vehicle> FrameSealed for Ned<C> {}
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
/// monomorphizations like `Position<RootInertial>` or
/// `Torque<BodyFrame<SelfRef>>`, so the user sees only the wrapper newtype.
#[derive(Debug, Clone, Copy)]
pub struct SelfRef;
impl VehicleSealed for SelfRef {}
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
impl VehicleSealed for TestVehicle {}
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
// These macros generate the marker struct + per-domain sealed impl +
// trait impl in one statement. They are the canonical way for a
// downstream crate to add a `Vehicle` or `Planet` marker.
//
// The seal traits `VehicleSealed` and `PlanetSealed` are re-exported via
// `$crate::__macro_support` so the macros can satisfy the bounds from
// downstream call sites. The other three seal traits — `FrameSealed`,
// `TimeScaleSealed`, and `QuatSealed` (which gates both `Layout` and
// `Transform`) — are *not* re-exported, so the four public traits they
// guard (`Frame`, `TimeScale`, `Layout`, `Transform`) remain
// type-system-sealed and downstream code cannot impl them at all.
// Direct `impl Vehicle for X` outside the macro is technically possible
// but unsupported.
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
/// `Vehicle` has a `VehicleSealed` super-bound. The seal trait is
/// re-exported via `__macro_support` so this macro can satisfy it from
/// a downstream call site. The other sealed-trait domains
/// (`FrameSealed`, `TimeScaleSealed`, `QuatSealed`) are not re-exported,
/// so `Frame`, `TimeScale`, `Layout`, and `Transform` remain
/// type-system-sealed.
///
/// Direct `impl Vehicle for X {}` outside this macro is technically
/// possible (the seal trait is reachable) but is unsupported and may
/// break in any release. Use the macro.
#[macro_export]
macro_rules! define_vehicle {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name;
        impl $crate::__macro_support::VehicleSealed for $name {}
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
/// Same per-domain seal as [`define_vehicle!`]: `Planet`'s super-bound
/// `PlanetSealed` is re-exported via `__macro_support` so this macro
/// can satisfy it from a downstream call site. Direct
/// `impl Planet for X {}` outside this macro is technically possible
/// but unsupported. Use the macro.
#[macro_export]
macro_rules! define_planet {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name;
        impl $crate::__macro_support::PlanetSealed for $name {}
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
