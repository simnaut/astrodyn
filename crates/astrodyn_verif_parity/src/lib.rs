//! Parity tests between `astrodyn_runner` (arena harness) and `astrodyn_bevy`
//! (ECS adapter).
//!
//! Each integration test in `tests/` runs an identical
//! scenario through both consumers of the `astrodyn` gateway crate and asserts
//! bit-identical state via the helpers in `tests/common/mod.rs`. The two
//! consumers depend on the same physics crates by design — these tests guard
//! against any divergence introduced by the ECS wiring.
