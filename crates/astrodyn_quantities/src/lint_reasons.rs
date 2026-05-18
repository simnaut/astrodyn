//! Canonical lint-rationale catalog.
//!
//! `#[allow(clippy::float_cmp, reason = "...")]` is denied workspace-wide by
//! the numerics-half policy in the root `Cargo.toml` (`workspace.lints`); the
//! audit-log requirement in `CLAUDE.md` §"Lints & invariants" mandates a
//! `reason = "..."` justification at every bypass. This module catalogs the
//! recurring justifications so that the canonical phrasings live in one
//! audit-able place rather than being re-invented per site.
//!
//! Three recurring sub-themes have a single canonical phrasing repeated
//! verbatim across many files; this catalog pins each:
//!
//! - [`clippy_float_cmp::TYPED_RAW_PARITY`] — typed-vs-raw boundary parity
//!   tests, where the invariant is "the typed sibling returns the
//!   bit-identical `DVec3`/`f64`/`DQuat` that the raw kernel produces".
//! - [`clippy_float_cmp::TIER3_LITERAL_ANALYTIC`] — Tier 3 trajectory
//!   cross-validation tests, where literal-built initial conditions and
//!   analytic closed-form spot checks must recover bit-exactly.
//! - [`clippy_float_cmp::BEVY_PARITY_STATE_FIELDS`] /
//!   [`clippy_float_cmp::BEVY_PARITY_TIME_FIELDS`] — `bevy_parity_*`
//!   wrappers, where the lockstep invariant is bit-identity between the
//!   `astrodyn_runner` driver and the `astrodyn_bevy` adapter.
//!
//! Two further sub-themes recur as a *pattern* rather than a verbatim
//! string — each call site customizes the noun being tested while the
//! lint-and-pattern stays uniform. These are documented here so a future
//! contributor knows when their bespoke phrasing is on-pattern:
//!
//! - **`BITEXACT_RECOVERY` pattern** — integrator / closed-form
//!   round-trip tests where the invariant is "no rounding occurred
//!   between the literal input and the recovered output" (e.g. `RK4`
//!   `t=0` recovery, `ω·dt` analytic literals, mass / `μ` literal
//!   accessor round-trips, default-construction zero checks). The
//!   phrasing template is `"<subject> tests assert bit-exact recovery
//!   of <literal-or-analytic-source>"`.
//! - **`BITEXACT_SENTINEL` pattern** — bit-exact sentinels used as
//!   state-change detectors or as caller fast-path triggers, where a
//!   `< eps` window would introduce a discontinuity at the chosen
//!   tolerance. The phrasing template is `"bit-exact sentinel <for
//!   which fast-path / detector>"`.
//!
//! ## Why call sites still spell the string literally
//!
//! Rust's `reason = "..."` attribute requires a **string-literal** RHS at
//! parse time. A `const` path (`reason = TYPED_RAW_PARITY`) or a macro
//! invocation (`reason = my_macro!()`) is rejected by the compiler:
//!
//! ```text
//! error: expected unsuffixed literal, found `TYPED_RAW_PARITY`
//!   --> src/foo.rs:N:M
//!    |
//! N  | #[allow(clippy::float_cmp, reason = TYPED_RAW_PARITY)]
//!    |                                     ^^^^^^^^^^^^^^^^^
//! ```
//!
//! So each `#[allow]` site still spells its rationale verbatim as a string
//! literal. This module is the **canonical reference** for the recurring
//! phrasings: when you add a new bit-exact `#[allow]` whose rationale
//! matches one of the catalog sub-themes, copy the exact string from one
//! of the constants below rather than coining a new one. The
//! `astrodyn_quantities` test `tests/lint_reasons_catalog.rs` walks the
//! workspace and asserts every cataloged string still appears in at least
//! two `#[allow(...)]` `reason = "..."` attribute literals (the
//! minimum-cluster threshold that justifies centralization), so a stale
//! catalog entry — or a sub-theme that has shrunk back below the
//! centralization threshold — fails CI rather than drifting silently.
//!
//! ## Adding a new entry
//!
//! 1. Verify the new phrasing actually recurs verbatim across multiple
//!    files. Single-site rationales stay bespoke and out of the catalog —
//!    centralizing them adds indirection without saving anything.
//! 2. The constant name should describe the *invariant being tested*, not
//!    the lint being suppressed (which is always `clippy::float_cmp` in
//!    this submodule).
//! 3. Add the const here and extend the coverage check in
//!    `tests/lint_reasons_catalog.rs` so a future refactor that
//!    accidentally tweaks the wording at one site (but not the rest)
//!    fails CI loudly.

/// Canonical rationales for `#[allow(clippy::float_cmp, ...)]` sites.
///
/// Each `pub const` is the exact string used at one of the recurring
/// bypass sites in the physics crates. The string is what appears in the
/// `reason = "..."` attribute at the call site, not a paraphrase.
pub mod clippy_float_cmp {
    /// Typed-vs-raw boundary parity tests.
    ///
    /// Used by `#[cfg(test)] mod tests` blocks at typed-sibling
    /// boundaries (e.g. `astrodyn_gravity::compute`,
    /// `astrodyn_gravity::relativistic`,
    /// `astrodyn_dynamics::rotational`, `astrodyn_atmosphere::lib`,
    /// `astrodyn_math::solar_beta`). The invariant is that calling the
    /// typed function and dropping back to raw `DVec3`/`f64` returns
    /// bit-identical bytes to a direct raw-kernel call — any difference
    /// indicates an unwanted conversion path inside the typed wrapper.
    ///
    /// Topic-specific variants (e.g. `geodetic`, `orbital-element`,
    /// `SRP`, `tide-tensor`, `typed-vs-raw drag-config`) keep their
    /// bespoke wording so the prefix names the noun being tested.
    pub const TYPED_RAW_PARITY: &str =
        "typed-vs-raw parity tests assert bit-exact identity at the type boundary";

    /// Tier 3 trajectory cross-validation tests.
    ///
    /// Used by `tier3_*.rs` integration tests that propagate from
    /// JEOD-source initial conditions through `Simulation::step()` and
    /// compare against Trick reference CSVs. The invariant is that
    /// literal-built initial conditions and analytic closed-form spot
    /// checks recover bit-exactly under the simulation's own
    /// propagation — only the runtime trajectory comparisons use
    /// tolerances; the literal-vs-recovered checks are bit-exact.
    pub const TIER3_LITERAL_ANALYTIC: &str =
        "Tier 3 tests assert bit-exact recovery of literal-built / analytic state values";

    /// `bevy_parity_*` lockstep tests — state-field identity.
    ///
    /// Used by `crates/astrodyn_verif_parity/tests/bevy_parity_*.rs`
    /// wrappers. The transitivity argument behind `bevy ↔ JEOD`
    /// inheriting `runner ↔ JEOD` requires `runner ↔ bevy` to hold
    /// bit-for-bit — these tests pin that bit-identity on translational
    /// / rotational state fields.
    pub const BEVY_PARITY_STATE_FIELDS: &str =
        "bevy-parity tests assert bit-exact identity between runner and Bevy state fields";

    /// `bevy_parity_*` lockstep tests — time-field identity.
    ///
    /// Sibling of [`BEVY_PARITY_STATE_FIELDS`] for the time-scale
    /// scenarios that compare `SimulationTime` fields rather than
    /// translational state. Same lockstep transitivity argument.
    pub const BEVY_PARITY_TIME_FIELDS: &str =
        "bevy-parity tests assert bit-exact identity between runner and Bevy time fields";
}
