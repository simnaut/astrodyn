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

## Test naming convention

Every `#[test] fn` in `tests/bevy_parity*.rs` shares a common prefix with
its containing file's stem. Concretely, a test in
`tests/bevy_parity_dyncomp_run3.rs` starts with `bevy_parity_dyncomp_run3`,
so a nextest filter expression `test(bevy_parity_dyncomp_run3)` selects
every test in that file. The invariant:

> *the `#[test] fn` name starts with the same stem as its containing
> file/binary*, so `test(<stem>)` always selects what a reader expects.

When adding a new parity wrapper, name the file
`bevy_parity_<topic>.rs` and prefix every test fn with
`bevy_parity_<topic>`. The `parity_coverage` meta-test
(`tests/parity_coverage.rs`) extracts topics from file stems, so it
will recognize the new wrapper as long as the file is named
correctly — but the function-name prefix is what makes
`test(bevy_parity_<topic>)` filter expressions work.
