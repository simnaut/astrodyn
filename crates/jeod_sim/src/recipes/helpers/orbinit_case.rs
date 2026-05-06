// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Parametric orbital-init test-case bundle.
//!
//! Used by `tier3_sim_orbinit_*.rs` families that loop over many
//! `(orbital elements, tolerance)` pairs and assert a roundtrip
//! `state -> elements -> state` invariance.

use jeod_math::OrbitalElements;
use jeod_quantities::frame::SelfPlanet;

/// A single orbital-init test case: a label, its starting orbital
/// elements, and the position-roundtrip tolerance to assert. The
/// tolerance is in meters and is consumed verbatim by the test asserts
/// — see [the type-system refactor's Phase-0 baseline freeze
/// policy](../../../../../CLAUDE.md#cross-validation-tolerances) for
/// why test code (not this struct) owns tolerance values.
//
// Elements use `OrbitalElements<SelfPlanet>` because the orbinit-loop
// callers feed the planet-erased registry-side path; the planet
// identity is determined at runtime by the surrounding sim, not at
// the case-construction site.
#[derive(Debug, Clone)]
pub struct OrbInitCase {
    /// Test-case identifier used in assertion messages.
    pub label: &'static str,
    /// Starting Keplerian orbital elements.
    pub elements: OrbitalElements<SelfPlanet>,
    /// Position roundtrip tolerance in metres for the assertion.
    pub position_tol_m: f64,
}

impl OrbInitCase {
    /// Build a case from its label, elements, and position tolerance.
    pub fn new(
        label: &'static str,
        elements: OrbitalElements<SelfPlanet>,
        position_tol_m: f64,
    ) -> Self {
        Self {
            label,
            elements,
            position_tol_m,
        }
    }
}
