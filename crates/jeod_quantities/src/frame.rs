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

/// Compile-time vehicle tag used to parameterize body/structural frames.
///
/// In practice mission crates wire up empty marker structs per vehicle:
/// `struct Iss;` then `impl Vehicle for Iss {}` — but the `impl Sealed for Iss`
/// must live in `jeod_quantities`, which we accomplish below with a blanket-off
/// pattern: a mission crate wanting a new vehicle tag calls the
/// `define_vehicle!` macro (see below) which emits a `Sealed` impl guarded by
/// a private marker trait we re-export.
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

/// Planet-fixed frame for any planet `P`. Rotates with that planet.
#[derive(Debug, Clone, Copy)]
pub struct PlanetFixed<P: Planet>(PhantomData<P>);
impl<P: Planet> Sealed for PlanetFixed<P> {}
impl<P: Planet> Frame for PlanetFixed<P> {
    const NAME: &'static str = P::NAME;
}

/// Body (CoM-centered) frame of vehicle `V`. Rotates with the vehicle.
#[derive(Debug, Clone, Copy)]
pub struct BodyFrame<V: Vehicle>(PhantomData<V>);
impl<V: Vehicle> Sealed for BodyFrame<V> {}
impl<V: Vehicle> Frame for BodyFrame<V> {
    const NAME: &'static str = V::NAME;
}

/// Structural (geometric-origin) frame of vehicle `V`. Rotates with vehicle.
#[derive(Debug, Clone, Copy)]
pub struct StructuralFrame<V: Vehicle>(PhantomData<V>);
impl<V: Vehicle> Sealed for StructuralFrame<V> {}
impl<V: Vehicle> Frame for StructuralFrame<V> {
    const NAME: &'static str = V::NAME;
}

/// Local Vertical / Local Horizontal frame relative to chief vehicle `Chief`.
/// Z axis points planet-ward; Y opposes orbital angular momentum; X completes
/// the right-handed triad (approximately along-track in near-circular orbits).
#[derive(Debug, Clone, Copy)]
pub struct Lvlh<Chief: Vehicle>(PhantomData<Chief>);
impl<C: Vehicle> Sealed for Lvlh<C> {}
impl<C: Vehicle> Frame for Lvlh<C> {
    const NAME: &'static str = C::NAME;
}

/// North-East-Down topocentric frame relative to chief vehicle `Chief`.
#[derive(Debug, Clone, Copy)]
pub struct Ned<Chief: Vehicle>(PhantomData<Chief>);
impl<C: Vehicle> Sealed for Ned<C> {}
impl<C: Vehicle> Frame for Ned<C> {
    const NAME: &'static str = C::NAME;
}
