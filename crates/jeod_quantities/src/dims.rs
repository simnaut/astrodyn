//! Custom quantity dimensions not already present in `uom::si`.
//!
//! Dimensions are compile-time 7-tuples of `typenum` integers indexing
//! `ISQ<L, M, T, I, Th, N, J>`. Each alias below names a physical quantity
//! used by orbital mechanics that `uom::si` does not predefine.
//!
//! The `*Dim` aliases can be used as the dimension parameter of `Qty3<D, F>`;
//! the companion scalar aliases (`SpecificAngMom`, …) model the
//! corresponding `Quantity<_, SI<f64>, f64>` values used in scalar math.
//!
//! # Planet-tagged gravitational parameter
//!
//! [`GravParam<P>`] is the planet-phantom-tagged gravitational parameter:
//! `GravParam<Earth>` and `GravParam<Sun>` are distinct types. Mixing them
//! across a planet boundary is a compile error. The witness-gated
//! constructor pattern mirrors [`crate::BodyAttitude<V>`]: callers reach
//! the type only through factories that fix the planet (e.g.
//! `f64.m3_per_s2_for::<Earth>()`, `mu_ggm05c()` returning
//! `GravParam<Earth>`).

use core::marker::PhantomData;

use typenum::{N1, N2, P1, P2, P3, Z0};
use uom::si::{Quantity, ISQ, SI};

use crate::frame::{Planet, SelfPlanet};

/// Gravitational parameter μ = GM (L³T⁻²). Base SI unit: m³/s².
pub type GravParamDim = ISQ<P3, Z0, N2, Z0, Z0, Z0, Z0>;

/// Specific angular momentum h = |r × v| / m (L²T⁻¹). Base SI unit: m²/s.
pub type SpecificAngMomDim = ISQ<P2, Z0, N1, Z0, Z0, Z0, Z0>;

/// Specific energy ε (L²T⁻²). Base SI unit: J/kg = m²/s².
pub type SpecificEnergyDim = ISQ<P2, Z0, N2, Z0, Z0, Z0, Z0>;

/// Mass flow rate ṁ (MT⁻¹). Base SI unit: kg/s.
pub type MassFlowRateDim = ISQ<Z0, P1, N1, Z0, Z0, Z0, Z0>;

// --- Scalar `Quantity` aliases -----------------------------------------------

/// Scalar specific angular momentum.
pub type SpecificAngMom = Quantity<SpecificAngMomDim, SI<f64>, f64>;

/// Scalar specific energy.
pub type SpecificEnergy = Quantity<SpecificEnergyDim, SI<f64>, f64>;

/// Scalar mass-flow rate.
pub type MassFlowRate = Quantity<MassFlowRateDim, SI<f64>, f64>;

// --- Planet-tagged gravitational parameter -----------------------------------

/// Gravitational parameter μ = GM tagged with the source planet `P`.
///
/// `GravParam<Earth>` and `GravParam<Sun>` are distinct types so the
/// compiler refuses to silently feed `μ_Sun` into a function expecting
/// `μ_Earth`. The numeric value is stored in SI base units (m³/s²) and
/// reachable via [`GravParam::value`] (also a public field for the
/// kernel-level call sites that already do `mu.value`).
///
/// ## Construction
///
/// Mission code constructs a `GravParam<P>` through the inferred-planet
/// factory `f64::m3_per_s2()` (planet `<P>` inferred from the
/// expected-type context at the call site — see
/// [`crate::ext::F64Ext::m3_per_s2`] for the full story; a bare
/// `let mu = 1.0.m3_per_s2();` with no expected-type context fails to
/// compile), the explicit-planet factory `f64::m3_per_s2_for::<P>()`, or
/// one of the curated `mu_*()` constants in
/// `jeod_sim::recipes::constants`. Calling a typed consumer with a
/// [`SelfPlanet`]-tagged μ in a planet-pinned slot is rejected at compile
/// time, so a planet-erased μ only flows through `SelfPlanet`-typed
/// surfaces (the registry-side boundary code, `GravitySource`, etc.).
///
/// ```
/// use jeod_quantities::prelude::*;
///
/// // Planet-pinned construction (Earth):
/// let mu_earth: GravParam<Earth> = 3.986_004_415e14_f64.m3_per_s2_for::<Earth>();
/// // Type ascription supplies the inference context for `<P>`:
/// let mu_sun: GravParam<Sun> = 1.327_124_400_18e20_f64.m3_per_s2();
/// // The numeric SI value in m³/s² is reachable via `.value`:
/// assert!(mu_earth.value > 0.0);
/// assert!(mu_sun.value > 0.0);
/// ```
///
/// ## Compile-fail: cross-planet construction is rejected
///
/// Building a planet-pinned `GravParam<Earth>` directly from a
/// `GravParam<Sun>` is a type error — the compiler refuses the
/// assignment.
///
/// ```compile_fail
/// use jeod_quantities::prelude::*;
/// let mu_sun: GravParam<Sun> = 1.327e20.m3_per_s2_for::<Sun>();
/// let _bad: GravParam<Earth> = mu_sun;   // planet phantom mismatch
/// ```
///
/// A planet-erased `GravParam<SelfPlanet>` cannot be assigned to a
/// planet-pinned slot either — `SelfPlanet` is not `Earth`:
///
/// ```compile_fail
/// use jeod_quantities::prelude::*;
/// let mu_any: GravParam<SelfPlanet> = 3.986e14.m3_per_s2();
/// let _bad: GravParam<Earth> = mu_any;   // SelfPlanet vs Earth mismatch
/// ```
#[repr(C)]
pub struct GravParam<P: Planet = SelfPlanet> {
    /// Numeric value in SI base units (m³/s²).
    ///
    /// The field is public so internal physics kernels can read it via
    /// `mu.value` without going through an accessor — the typed planet
    /// phantom is the load-bearing part, not the wrapping ceremony.
    pub value: f64,
    _p: PhantomData<P>,
}

impl<P: Planet> GravParam<P> {
    /// Construct a `GravParam<P>` from its SI base value (m³/s²).
    ///
    /// This is the witness-gated constructor: the planet phantom is
    /// fixed by the turbofish or by the surrounding type context.
    /// Mission code typically reaches this through
    /// [`crate::ext::F64Ext::m3_per_s2_for`] or the curated `mu_*()`
    /// constants rather than calling `from_si` directly.
    // JEOD_INV: RF.11 — planet-pinned witness constructor; the phantom
    // `P` ties the resulting μ to its source body so downstream typed
    // consumers refuse a μ-vs-frame mismatch at compile time.
    #[inline]
    pub const fn from_si(value: f64) -> Self {
        Self {
            value,
            _p: PhantomData,
        }
    }

    /// Numeric value in SI base units (m³/s²).
    #[inline]
    pub const fn raw_si(&self) -> f64 {
        self.value
    }
}

impl GravParam<SelfPlanet> {
    /// Relabel a planet-erased ([`SelfPlanet`]) μ as belonging to a
    /// specific planet `Q`.
    ///
    /// Restricted to `impl GravParam<SelfPlanet>` so it can only retag a
    /// μ that is already planet-erased — a planet-pinned `GravParam<Sun>`
    /// cannot accidentally be relabeled as `GravParam<Earth>` via this
    /// method. Provided **only** for the registry-side boundary code
    /// that stores μ values keyed by a runtime source ID — the gravity
    /// source registry, `GravitySourceTyped`, the dynamic mu carried on
    /// `PlanetShape`, and similar surfaces where the planet identity is
    /// determined at runtime. Mission code that knows the planet at
    /// compile time should construct a planet-tagged `GravParam<Q>`
    /// directly via `m3_per_s2_for::<Q>()` or one of the `mu_*()`
    /// constants.
    ///
    /// A genuine `<P>` → `<Q>` retag for two distinct named planets is
    /// almost never the right operation (μ is per-body); if you need it
    /// for a different reason, add a separate, clearly-named escape
    /// hatch instead of widening this impl block.
    #[inline]
    pub fn relabel<Q: Planet>(self) -> GravParam<Q> {
        GravParam::<Q>::from_si(self.value)
    }
}

// Manual impls — the derive macros would demand `P: Default`/`P: Copy`/...
// which is wrong for a phantom-only type parameter.
impl<P: Planet> Copy for GravParam<P> {}

impl<P: Planet> Clone for GravParam<P> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: Planet> Default for GravParam<P> {
    #[inline]
    fn default() -> Self {
        Self::from_si(0.0)
    }
}

impl<P: Planet> PartialEq for GravParam<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<P: Planet> core::fmt::Debug for GravParam<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // `Planet::NAME` carries the per-planet identifier (e.g. "Earth").
        write!(f, "GravParam<{}>({} m^3/s^2)", P::NAME, self.value)
    }
}
