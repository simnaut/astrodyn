//! Negative tests for quaternion normalization invariants (#531 seed
//! tests for the negative-test gate in `tests/invariant_coverage.rs`).
//!
//! These pin the `assert!`/`panic!` sites in
//! `crates/astrodyn_quantities/src/quat.rs::JeodQuat::normalize` and
//! prove that an attempt to normalize a zero-magnitude quaternion
//! produces the expected diagnostic, not a silent NaN. A future
//! refactor that downgrades the `assert!` to a `return` would trip
//! the `should_panic` here even though the existing
//! `tests/invariant_coverage.rs` tag-↔-catalog gate would still pass.

use astrodyn_quantities::prelude::*;

// JEOD_INV: QT.03 — cannot normalize a zero quaternion (panic site)
#[test]
#[should_panic(expected = "cannot normalize a zero quaternion")]
fn normalize_panics_on_zero_quaternion() {
    let mut q = JeodQuat::from_array([0.0, 0.0, 0.0, 0.0]);
    q.normalize();
}
