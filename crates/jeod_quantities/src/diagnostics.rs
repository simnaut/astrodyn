//! Custom compiler diagnostics via `#[diagnostic::on_unimplemented]`.
//!
//! The attribute is stable in Rust ≥ 1.78. The patterns below are expected
//! to fire when mission-crate code tries to use an operation that is not
//! implementable between the supplied types (e.g. adding `Position<RootInertial>`
//! to `Position<Ecef>`) and instead of the default terse "trait `Add` is not
//! implemented for ..." we give a hint in physics language.
//!
//! Each trait here is a zero-cost marker whose sole purpose is to carry a
//! tailored message. The `#[diagnostic::on_unimplemented]` attribute rides
//! along on the `impl Trait for X where ...` bound so the compiler can
//! surface it when the bound fails.

use core::marker::PhantomData;

use uom::si::{f64::Angle, f64::Length};

use crate::dims::GravParam;
use crate::frame::{Frame, Planet, Vehicle};
use crate::quat::{Layout, NormalizedQuat, Transform};
use crate::time_scale::TimeScale;

/// Fires when a user tries to add/subtract two frame-tagged vectors whose
/// frames differ. Implemented only for matching frames.
///
/// The `Add`/`Sub` impls on [`Qty3`](crate::qty3::Qty3) carry a
/// `(): CompatibleFrames<Fl, Fr>` bound, so a frame mismatch on `+`/`-`
/// surfaces the custom message below rather than the generic
/// `trait Add is not implemented` diagnostic.
///
/// ```compile_fail
/// use jeod_quantities::prelude::*;
/// use glam::DVec3;
/// let a: Position<RootInertial> = Qty3::from_raw_si(DVec3::new(1.0, 0.0, 0.0));
/// let b: Position<Ecef> = Qty3::from_raw_si(DVec3::new(1.0, 0.0, 0.0));
/// let _ = a + b; // ← frame mismatch, custom diagnostic fires
/// ```
#[diagnostic::on_unimplemented(
    message = "cannot combine values in frame `{FL}` with values in frame `{FR}`",
    label = "mismatched frame: {FL} vs {FR}",
    note = "apply a `FrameTransform<{FR}, {FL}>` (or its inverse) to bring both operands into the same frame before combining"
)]
pub trait CompatibleFrames<FL: Frame, FR: Frame> {}

impl<F: Frame> CompatibleFrames<F, F> for () {}

/// Fires when a `uom` `Length` is expected but a bare `f64` is supplied.
#[diagnostic::on_unimplemented(
    message = "bare `f64` is not a `Length` — attach a unit with `F64Ext`",
    label = "expected a `Length`, found `{Self}`",
    note = "use `.m()`, `.km()`, `.cm()`, `.mm()`, `.ft()`, `.mi()`, or `.nmi()` to produce a `Length`"
)]
pub trait IntoLength {
    /// Lift `self` into a typed [`Length`].
    fn into_length(self) -> Length;
}

impl IntoLength for Length {
    #[inline]
    fn into_length(self) -> Length {
        self
    }
}

/// Fires when a `uom` `Angle` is expected but a bare `f64` is supplied.
#[diagnostic::on_unimplemented(
    message = "bare `f64` is not an `Angle` — attach a unit with `F64Ext`",
    label = "expected an `Angle`, found `{Self}`",
    note = "use `.rad()`, `.deg()`, `.arcmin()`, or `.arcsec()` to produce an `Angle`"
)]
pub trait IntoAngle {
    /// Lift `self` into a typed [`Angle`].
    fn into_angle(self) -> Angle;
}

impl IntoAngle for Angle {
    #[inline]
    fn into_angle(self) -> Angle {
        self
    }
}

/// Fires when a gravitational parameter is expected but a bare `f64` is supplied.
///
/// Generic over the planet phantom `P` so the diagnostic carries the
/// expected source-body identity in the error message — a caller writing
/// `from_cartesian_typed::<Earth>(0.0, ...)` sees `expected
/// GravParam<Earth>`, not the planet-erased
/// `GravParam<SelfPlanet>` form.
#[diagnostic::on_unimplemented(
    message = "bare `f64` is not a `GravParam<{P}>` — attach a unit with `F64Ext`",
    label = "expected `GravParam<{P}>` (m³/s²), found `{Self}`",
    note = "use `.m3_per_s2_for::<{P}>()` or `.km3_per_s2_for::<{P}>()` to produce a planet-pinned `GravParam<{P}>`"
)]
pub trait IntoGravParam<P: Planet> {
    /// Lift `self` into a typed [`GravParam<P>`] (m³/s²).
    fn into_grav_param(self) -> GravParam<P>;
}

impl<P: Planet> IntoGravParam<P> for GravParam<P> {
    #[inline]
    fn into_grav_param(self) -> GravParam<P> {
        self
    }
}

/// Fires when a [`GravParam`] tagged with planet `PFound` is supplied
/// where the callee expected `GravParam<PExpected>`. Implemented only
/// when the planet phantoms match.
///
/// This guards the load-bearing mu-vs-frame agreement: a function that
/// takes `(mu: GravParam<P>, pos: Position<PlanetInertial<P>>, vel:
/// Velocity<PlanetInertial<P>>)` is structurally guaranteed to receive
/// matching planet phantoms when this bound is wired into its `where`
/// clause.
#[diagnostic::on_unimplemented(
    message = "gravitational-parameter source mismatch: expected `GravParam<{PExpected}>`, found `GravParam<{PFound}>`",
    label = "wrong source body: μ for {PFound} cannot stand in for μ of {PExpected}",
    note = "use the matching `mu_*()` constant or `f64::m3_per_s2_for::<{PExpected}>()` so the planet phantom on μ matches the position/velocity frames"
)]
pub trait CompatibleGravParam<PExpected: Planet, PFound: Planet> {}

impl<P: Planet> CompatibleGravParam<P, P> for () {}

/// Fires when a time in scale `TL` is combined with a time in scale `TR`.
/// Implemented only when the scales match.
#[diagnostic::on_unimplemented(
    message = "time-scale mismatch: cannot combine `{TL}` and `{TR}` directly",
    label = "scales differ: {TL} vs {TR}",
    note = "convert via `TimeConverter::<From, To>` (e.g. `TAI_TO_TT`, `GPS_TO_TAI`) before arithmetic"
)]
pub trait CompatibleTimeScales<TL: TimeScale, TR: TimeScale> {}

impl<S: TimeScale> CompatibleTimeScales<S, S> for () {}

/// Fires when a quaternion layout mismatch occurs (e.g. a `ScalarFirst`
/// quaternion is passed where a `ScalarLast` is expected).
#[diagnostic::on_unimplemented(
    message = "quaternion layout mismatch: expected `{LExpected}`, found `{LFound}`",
    label = "layout differs: {LFound}",
    note = "use `.to_scalar_first()` or `.to_scalar_last()` to convert between layouts"
)]
pub trait CompatibleQuatLayouts<LExpected: Layout, LFound: Layout> {}

impl<L: Layout> CompatibleQuatLayouts<L, L> for () {}

/// Fires on a quaternion transformation-convention mismatch (`LeftTransform`
/// vs `RightTransform`).
#[diagnostic::on_unimplemented(
    message = "quaternion transformation convention mismatch: expected `{TExpected}`, found `{TFound}`",
    label = "transform differs: {TFound}",
    note = "JEOD uses LeftTransform (`r' = q r q⁻¹`); RightTransform quaternions must be conjugated before use"
)]
pub trait CompatibleQuatTransforms<TExpected: Transform, TFound: Transform> {}

impl<T: Transform> CompatibleQuatTransforms<T, T> for () {}

/// Fires if a raw `Quat` is used where a `NormalizedQuat` is expected.
#[diagnostic::on_unimplemented(
    message = "this API requires a unit quaternion; got an unverified `Quat`",
    label = "found a raw `Quat`, expected `NormalizedQuat`",
    note = "wrap with `NormalizedQuat::new(q)?` (fails if |q| ≠ 1 within 1e-12) or `NormalizedQuat::renormalize(q)` (forces |q| = 1)"
)]
pub trait RequiresNormalizedQuat {}

impl<L: Layout, T: Transform> RequiresNormalizedQuat for NormalizedQuat<L, T> {}

/// Fires when a caller tries to use an ECEF velocity where the callee
/// expected an inertial velocity (or vice versa). Phrased from the common
/// mistake direction; the opposite flavor uses `CompatibleFrames`.
#[diagnostic::on_unimplemented(
    message = "this operation needs an inertial-frame quantity; got one in the rotating frame `{F}`",
    label = "non-inertial frame `{F}`",
    note = "rotate into RootInertial via `FrameTransform<{F}, RootInertial>::apply(...)`; remember to add the frame rotation velocity for derivatives"
)]
pub trait InertialOnly<F: Frame> {}

impl InertialOnly<crate::frame::RootInertial> for () {}

/// Fires when user code tries `Position * Position` or `Velocity * Velocity`
/// (an accidental componentwise multiply). There's no meaningful vector
/// product implementation, so the hint nudges toward `.dot()` or `.cross()`.
#[diagnostic::on_unimplemented(
    message = "two `Qty3`s cannot be multiplied componentwise",
    label = "ambiguous product of `Qty3<{D1}, {F}>` and `Qty3<{D2}, {F}>`",
    note = "use `.dot(other)` for scalar product (returns a scalar) or `.cross(other)` for vector product (returns a `Qty3`)"
)]
pub trait NoVectorVectorMul<D1, D2, F: Frame> {
    /// Marker method — never called; exists only so the trait carries
    /// its three type parameters into the compiler diagnostic.
    fn _do_not_call(&self) -> PhantomData<(D1, D2, F)>;
}

/// Fires when two distinct vehicle phantoms appear in a slot that demands
/// a single shared identity. Vehicle-side analog of [`CompatibleFrames`]
/// — implemented only when `VL` and `VR` resolve to the same vehicle
/// marker, so a mismatch surfaces a tailored physics-language message
/// instead of a `PhantomData<…>` type-mismatch wall.
///
/// Used as the `where` bound that gates "carries a single vehicle"
/// methods on the Act-5 phantom-wrapped types
/// (`FlatPlate<V>`, `RelativeTranslation<Reference>`,
/// `LvlhRelativeState<Chief>`, …): a method that requires the caller's
/// turbofish vehicle to match the receiver's vehicle takes
/// `(): CompatibleVehicles<VHere, VThere>` and the bound only resolves
/// when the two phantoms agree.
///
/// ```compile_fail
/// use jeod_quantities::define_vehicle;
/// use jeod_quantities::diagnostics::CompatibleVehicles;
///
/// define_vehicle!(Iss);
/// define_vehicle!(Soyuz);
///
/// fn require_match<VL, VR>()
/// where
///     VL: jeod_quantities::frame::Vehicle,
///     VR: jeod_quantities::frame::Vehicle,
///     (): CompatibleVehicles<VL, VR>,
/// {
/// }
///
/// // Same vehicle on both sides — compiles.
/// require_match::<Iss, Iss>();
/// // Mismatched vehicles — bound fails with the tailored diagnostic.
/// require_match::<Iss, Soyuz>();
/// ```
#[diagnostic::on_unimplemented(
    message = "vehicle mismatch: cannot combine values tagged `{VL}` with values tagged `{VR}`",
    label = "mismatched vehicle: `{VL}` vs `{VR}`",
    note = "the two vehicle phantoms must agree — pin the same `Vehicle` marker on both sides (e.g. via `define_vehicle!`), or rebuild the value for the right vehicle if it was constructed for a different one"
)]
pub trait CompatibleVehicles<VL: Vehicle, VR: Vehicle> {}

impl<V: Vehicle> CompatibleVehicles<V, V> for () {}

/// Paired-vehicle compatibility: fires when a `(Subject, Reference)` (or
/// `(Parent, Child)`) pair on one value does not match the pair on the
/// slot it's being passed into.
///
/// Implemented only when both the subject phantoms and the reference
/// phantoms agree. Used by types that carry two independent vehicle
/// identities — `RelativeState<Subject, Reference>` and
/// `AttachEvent<VParent, VChild>` — so a `<Iss, Soyuz>` value cannot
/// silently flow into a `<Iss, Cygnus>` slot.
///
/// ```compile_fail
/// use jeod_quantities::define_vehicle;
/// use jeod_quantities::diagnostics::CompatibleVehiclePair;
///
/// define_vehicle!(Iss);
/// define_vehicle!(Soyuz);
/// define_vehicle!(Cygnus);
///
/// fn require_match<S1, R1, S2, R2>()
/// where
///     S1: jeod_quantities::frame::Vehicle,
///     R1: jeod_quantities::frame::Vehicle,
///     S2: jeod_quantities::frame::Vehicle,
///     R2: jeod_quantities::frame::Vehicle,
///     (): CompatibleVehiclePair<S1, R1, S2, R2>,
/// {
/// }
///
/// // Both pairs match — compiles.
/// require_match::<Iss, Soyuz, Iss, Soyuz>();
/// // Reference half mismatches — bound fails with the tailored diagnostic.
/// require_match::<Iss, Soyuz, Iss, Cygnus>();
/// ```
#[diagnostic::on_unimplemented(
    message = "paired-vehicle mismatch: cannot combine `<{S1}, {R1}>` with `<{S2}, {R2}>`",
    label = "mismatched pair: `<{S1}, {R1}>` vs `<{S2}, {R2}>`",
    note = "both phantoms in the pair must match — the subject (or parent) and the reference (or child) are independent identities, and a mismatch on either side means the value was built for a different physical relationship"
)]
pub trait CompatibleVehiclePair<S1: Vehicle, R1: Vehicle, S2: Vehicle, R2: Vehicle> {}

impl<S: Vehicle, R: Vehicle> CompatibleVehiclePair<S, R, S, R> for () {}
