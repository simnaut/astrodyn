//! Declarative guard macros for finite/positive parameter validation.
//!
//! Two macros for the dominant *fail-loud* pattern at constructor and entry
//! boundaries:
//!
//! - [`crate::validate_finite`] panics if the value is `NaN` or `±∞`.
//! - [`crate::validate_finite_positive`] panics if the value is `NaN`, `±∞`,
//!   or `<= 0`.
//!
//! ## Diagnostic-message contract (partial Fail Loudly)
//!
//! CLAUDE.md *Fail Loudly* asks every panic message to name (a) the
//! parameter, (b) the failed condition, and (c) what the caller should
//! fix. These macros emit a uniform message that delivers (a), (b), and
//! the offending value — but **not** (c). They take both the parameter
//! name *and* the call-site context as literals so the panic message
//! names the entity being constructed *and* the field being validated:
//!
//! ```text
//! <context>: <name> must be finite, got {value}
//! <context>: <name> must be finite and > 0, got {value}
//! ```
//!
//! That covers the bulk of constructor-boundary `is_finite()` /
//! `is_finite() && > 0` checks, where the remediation is structural
//! ("don't pass NaN/∞/non-positive") and the `<context>: <name>` prefix
//! already points the caller at the offending argument. Sites whose
//! correct fix is *site-specific* — e.g. "Fix the upstream parent-time
//! advance ..." or "Use 0.0 for per-step refresh ..." — should *not*
//! migrate to these macros; they should keep their hand-written
//! `assert!` with the bespoke remediation tail (see the "When *not* to
//! use these" bullets below).
//!
//! The message wording is identical to the hand-written `assert!` macros
//! it replaces at the migrated sites — call sites that already have a
//! `#[should_panic(expected = "<name> must be finite ...")]` test stay
//! green because the substring is preserved verbatim.
//!
//! ## When *not* to use these
//!
//! - **`const fn` constructors**: formatted `panic!` is not allowed in
//!   `const fn` on stable Rust at our MSRV (compile-time evaluation
//!   cannot allocate the formatted message buffer). `PlanetShape::new`
//!   uses static-message `panic!` branches by design so the validation
//!   fires at compile time on `const PLANET: PlanetShape =
//!   PlanetShape::new(...)` declarations.
//! - **`Result::Err` boundaries**: validators that propagate an error rather
//!   than panicking (e.g. `OrbitalElements::from_cartesian_impl` returning
//!   `OrbitalError::InvalidMu`, `FrameTransform::from_matrix_validated`
//!   returning `FrameTransformError::NonFinite`, `AdaptiveConfig::check`
//!   returning `Result<(), &'static str>`) should keep the existing
//!   conditional and `return Err(...)`. The macros panic; they cannot model
//!   error-propagation.
//! - **Compound conditions other than `finite && > 0`**: a check like
//!   `finite && >= 0` (zero allowed) or `finite && in [0, 1]` doesn't fit
//!   either macro. Hand-write the `assert!`.
//! - **Sites whose existing message carries site-specific remediation
//!   text** ("Fix the upstream parent-time advance ...", "Use 0.0 for
//!   per-step refresh ...", etc.). The macros emit a uniform message
//!   that intentionally omits remediation guidance (see the contract
//!   above), so migrating such a site would strictly downgrade its
//!   diagnostic — a Fail-Loudly regression. Leave the hand-written
//!   `assert!` alone; that is the recommended pattern whenever
//!   remediation is site-specific.

/// Panic if `$value` is not finite (NaN or ±∞), naming the construction
/// context and the failing parameter.
///
/// # Panics
/// Panics with the message
///
/// ```text
/// <context>: <name> must be finite, got {value}
/// ```
///
/// where `<context>` and `<name>` are the string-literal arguments and
/// `{value}` is the offending `f64`.
///
/// # Examples
/// ```should_panic
/// use astrodyn_quantities::validate_finite;
/// validate_finite!("GroundFacet::new", "alt_offset", f64::NAN);
/// // panics: "GroundFacet::new: alt_offset must be finite, got NaN"
/// ```
#[macro_export]
macro_rules! validate_finite {
    ($context:literal, $name:literal, $value:expr $(,)?) => {{
        let __v: f64 = $value;
        ::core::assert!(
            __v.is_finite(),
            concat!($context, ": ", $name, " must be finite, got {}"),
            __v
        );
    }};
}

/// Panic if `$value` is not finite (NaN or ±∞) or not strictly positive,
/// naming the construction context and the failing parameter.
///
/// # Panics
/// Panics with the message
///
/// ```text
/// <context>: <name> must be finite and > 0, got {value}
/// ```
///
/// where `<context>` and `<name>` are the string-literal arguments and
/// `{value}` is the offending `f64`.
///
/// # Examples
/// ```should_panic
/// use astrodyn_quantities::validate_finite_positive;
/// validate_finite_positive!("SphericalTerrain::new", "radius", 0.0);
/// // panics: "SphericalTerrain::new: radius must be finite and > 0, got 0"
/// ```
#[macro_export]
macro_rules! validate_finite_positive {
    ($context:literal, $name:literal, $value:expr $(,)?) => {{
        let __v: f64 = $value;
        ::core::assert!(
            __v.is_finite() && __v > 0.0,
            concat!($context, ": ", $name, " must be finite and > 0, got {}"),
            __v
        );
    }};
}

#[cfg(test)]
mod tests {
    // The macros are imported from the crate root via `#[macro_export]`.

    #[test]
    fn validate_finite_passes_finite() {
        crate::validate_finite!("Test::new", "x", 1.0);
        crate::validate_finite!("Test::new", "x", -1.0);
        crate::validate_finite!("Test::new", "x", 0.0);
    }

    #[test]
    #[should_panic(expected = "Test::new: x must be finite, got NaN")]
    fn validate_finite_panics_on_nan() {
        crate::validate_finite!("Test::new", "x", f64::NAN);
    }

    #[test]
    #[should_panic(expected = "Test::new: x must be finite, got inf")]
    fn validate_finite_panics_on_positive_infinity() {
        crate::validate_finite!("Test::new", "x", f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "Test::new: x must be finite, got -inf")]
    fn validate_finite_panics_on_negative_infinity() {
        crate::validate_finite!("Test::new", "x", f64::NEG_INFINITY);
    }

    #[test]
    fn validate_finite_positive_passes_positive_finite() {
        crate::validate_finite_positive!("Test::new", "x", 1.0);
        crate::validate_finite_positive!("Test::new", "x", 1e-300);
    }

    #[test]
    #[should_panic(expected = "Test::new: x must be finite and > 0, got 0")]
    fn validate_finite_positive_panics_on_zero() {
        crate::validate_finite_positive!("Test::new", "x", 0.0);
    }

    #[test]
    #[should_panic(expected = "Test::new: x must be finite and > 0, got -1")]
    fn validate_finite_positive_panics_on_negative() {
        crate::validate_finite_positive!("Test::new", "x", -1.0);
    }

    #[test]
    #[should_panic(expected = "Test::new: x must be finite and > 0, got NaN")]
    fn validate_finite_positive_panics_on_nan() {
        crate::validate_finite_positive!("Test::new", "x", f64::NAN);
    }

    #[test]
    #[should_panic(expected = "Test::new: x must be finite and > 0, got inf")]
    fn validate_finite_positive_panics_on_positive_infinity() {
        crate::validate_finite_positive!("Test::new", "x", f64::INFINITY);
    }

    // Confirms the substring `"<name> must be finite, got"` is preserved
    // verbatim — call sites with `#[should_panic(expected = "<name> must
    // be finite")]` tests stay green after migration to `validate_finite!`.
    #[test]
    #[should_panic(expected = "alt_offset must be finite")]
    fn validate_finite_preserves_substring_for_should_panic() {
        crate::validate_finite!("GroundFacet::new", "alt_offset", f64::NAN);
    }

    // Confirms the substring `"<name> must be finite and > 0"` is preserved
    // verbatim for the positive variant.
    #[test]
    #[should_panic(expected = "radius must be finite and > 0")]
    fn validate_finite_positive_preserves_substring_for_should_panic() {
        crate::validate_finite_positive!("SphericalTerrain::new", "radius", 0.0);
    }
}
