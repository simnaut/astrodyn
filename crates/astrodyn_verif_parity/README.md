# astrodyn_verif_parity

Parity tests asserting bit-identical state between `astrodyn_runner`
(arena harness) and `astrodyn_bevy` (ECS adapter).

This is a workspace-internal verification crate (`publish = false`). Each
test under `tests/bevy_parity_*.rs` runs an identical scenario through both
consumers of the `astrodyn` gateway and asserts every component is
bit-identical (`f64::to_bits()` equality), guarding against drift between
the two implementations of the pipeline.

Both subjects under test (`astrodyn_runner::Simulation` and
`astrodyn_bevy::App`) are direct dev-dependencies; that's the test contract,
not a layering violation. Shared helpers live in `tests/common/mod.rs`.

For the role this crate plays in the verification topology and how it
relates to `astrodyn_verif_jeod`'s Tier 3 cross-validation tests, see the
[Strategy](https://github.com/simnaut/astrodyn/wiki/Strategy) and
[Dependency-Graph](https://github.com/simnaut/astrodyn/wiki/Dependency-Graph)
wiki pages.
