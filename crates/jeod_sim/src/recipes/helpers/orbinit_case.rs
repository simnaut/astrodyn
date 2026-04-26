//! Parametric orbital-init test-case bundle.
//!
//! Used by `tier3_sim_orbinit_*.rs` families that loop over many
//! `(orbital elements, tolerance)` pairs and assert a roundtrip
//! `state -> elements -> state` invariance.

use jeod_math::OrbitalElements;

/// A single orbital-init test case: a label, its starting orbital
/// elements, and the position-roundtrip tolerance to assert. The
/// tolerance is in meters and is consumed verbatim by the test asserts
/// — see [the type-system refactor's Phase-0 baseline freeze
/// policy](../../../../../CLAUDE.md#cross-validation-tolerances) for
/// why test code (not this struct) owns tolerance values.
#[derive(Debug, Clone)]
pub struct OrbInitCase {
    pub label: &'static str,
    pub elements: OrbitalElements,
    pub position_tol_m: f64,
}

impl OrbInitCase {
    pub fn new(label: &'static str, elements: OrbitalElements, position_tol_m: f64) -> Self {
        Self {
            label,
            elements,
            position_tol_m,
        }
    }
}
