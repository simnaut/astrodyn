//! Declarative-macro helpers for phantom-only generic wrappers.
//!
//! Several typed-quantity wrappers in this crate parameterise on a phantom
//! generic (`Position<F: Frame>`, `GravParam<P: Planet>`, `Qty3<D, F>`,
//! `InertiaTensor<F>`, `PlanetFixed<P>`, …). The phantom is zero-sized
//! `PhantomData<G>`, so the storage is structurally `Copy + Clone + Debug +
//! PartialEq` whenever the *non-phantom* payload is. `#[derive(Copy)]` etc.
//! disagree: they unconditionally synthesize `where G: Copy + Clone + Debug +
//! PartialEq`, which is wrong for a phantom-only generic — the wrapper would
//! refuse to compile for any tag that does not opt into those derives, even
//! though the runtime storage never touches a `G` value.
//!
//! Hand-rolling the impls works, but at four sites the boilerplate becomes
//! a maintenance smell. The macros below consolidate it:
//!
//! - [`impl_copy_phantom!`] emits a trivial `impl Copy`.
//! - [`impl_clone_phantom!`] emits `impl Clone` whose `clone` is `*self`
//!   (free because the wrapper is `Copy`).
//! - [`impl_partial_eq_phantom!`] emits `impl PartialEq` whose `eq` is the
//!   caller-supplied boolean expression. The caller names the bound
//!   identifiers (typically `self, other`) so macro-hygiene keeps them
//!   reachable inside the body expression.
//! - [`impl_debug_phantom!`] emits `impl core::fmt::Debug` whose `fmt`
//!   evaluates the caller-supplied expression. The caller names the
//!   `&Self` and `&mut Formatter` identifiers (typically `self, f`) for
//!   the same hygiene reason.
//!
//! Both the generics-and-bounds list and the type-argument list are passed
//! in **square brackets** to keep `macro_rules!`'s `tt`-greedy matcher
//! unambiguous when the generics include `?Sized` or `+`-joined bounds. The
//! shape is `impl_*_phantom!([<generics>] Ty [<args>] $extras?)`.

/// Emit `impl Copy` for a phantom-only generic.
///
/// ```ignore
/// impl_copy_phantom!([P: Planet] GravParam[P]);
/// impl_copy_phantom!([D: ?Sized + Dimension, F: Frame] Qty3[D, F]);
/// ```
macro_rules! impl_copy_phantom {
    ([$($g:tt)*] $ty:ident [$($a:tt)*]) => {
        impl<$($g)*> Copy for $ty<$($a)*> {}
    };
}

/// Emit `impl Clone` whose `clone` returns `*self` (free because the
/// wrapper is `Copy`).
///
/// ```ignore
/// impl_clone_phantom!([P: Planet] GravParam[P]);
/// ```
macro_rules! impl_clone_phantom {
    ([$($g:tt)*] $ty:ident [$($a:tt)*]) => {
        impl<$($g)*> Clone for $ty<$($a)*> {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }
    };
}

/// Emit `impl PartialEq` whose `eq` is `$body`. The caller names the two
/// `&Self` parameters so they survive macro hygiene.
///
/// ```ignore
/// impl_partial_eq_phantom!(
///     [P: Planet] GravParam[P],
///     |this, other| this.value == other.value
/// );
/// impl_partial_eq_phantom!(
///     [D: ?Sized + Dimension, F: Frame] Qty3[D, F],
///     |this, other| this.x.value == other.x.value
///         && this.y.value == other.y.value
///         && this.z.value == other.z.value
/// );
/// ```
macro_rules! impl_partial_eq_phantom {
    (
        [$($g:tt)*] $ty:ident [$($a:tt)*],
        |$this:ident, $other:ident| $body:expr
    ) => {
        impl<$($g)*> PartialEq for $ty<$($a)*> {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                let $this: &Self = self;
                let $other: &Self = other;
                $body
            }
        }
    };
}

/// Emit `impl core::fmt::Debug` whose `fmt` evaluates `$body`. The caller
/// names the `&Self` receiver and the `&mut Formatter` so they survive
/// macro hygiene.
///
/// The caller-supplied body keeps the per-site format string (which often
/// names the phantom tag via `T::NAME` or `core::any::type_name::<T>()`)
/// visible at the call site rather than hidden in a macro.
///
/// ```ignore
/// impl_debug_phantom!(
///     [P: Planet] GravParam[P],
///     |this, f| write!(f, "GravParam<{}>({} m^3/s^2)", P::NAME, this.value)
/// );
/// ```
macro_rules! impl_debug_phantom {
    (
        [$($g:tt)*] $ty:ident [$($a:tt)*],
        |$this:ident, $f:ident| $body:expr
    ) => {
        impl<$($g)*> core::fmt::Debug for $ty<$($a)*> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let $this: &Self = self;
                let $f: &mut core::fmt::Formatter<'_> = f;
                $body
            }
        }
    };
}

pub(crate) use {
    impl_clone_phantom, impl_copy_phantom, impl_debug_phantom, impl_partial_eq_phantom,
};
