# CI

`CLAUDE.md` lists the hot-path test commands. This page documents the full CI
matrix, the test-tier naming conventions, and the per-crate / per-test
invocations.

## Test execution

Use [`cargo-nextest`](https://nexte.st/) for test execution (matches CI,
parallel by default):

```bash
cargo build --workspace
cargo nextest run --workspace                                 # all tests
# `unit + tier 2 (fast)` excludes Tier 3 *and* every `bevy_parity_*`
# lockstep parity wrapper. The latter run each scenario through both
# `astrodyn_runner` and `astrodyn_bevy` for bit-identity, so their
# cost profile matches Tier 3 trajectory runs. Keep this filter in
# sync with the `test` and `test-parity-trajectory` jobs in CI.
cargo nextest run --workspace -E 'not test(tier3_) and not test(bevy_parity)'
cargo nextest run --workspace -E 'test(tier3_)'               # tier 3 only
cargo nextest run --workspace -E 'test(bevy_parity)'          # bevy_parity_* only
cargo nextest run -p astrodyn_math                                # single crate
cargo nextest run -p astrodyn_gravity -E 'test(verif)'            # gravity verification only
cargo nextest run -p astrodyn_verif_jeod --test tier3_sim_dyncomp_run2  # single Tier 3 test
```

Plain `cargo test` also works but runs tests serially per binary:

```bash
cargo test --workspace                          # all tests
cargo test --workspace -- --skip tier3_         # unit + tier 2
```

All three test tiers (`cargo nextest run --workspace`) run without
`$JEOD_HOME` set — `run_verification/sim_*.rs` reads everything from
the committed mirror under `crates/astrodyn_verif_jeod/test_data/jeod_inputs/`
plus the parsed gravity binaries under `crates/astrodyn_gravity/test_data/gravity/`.
The regen binaries (`extract_*`) and the Docker reference-CSV flow are the only
paths that still need `$JEOD_HOME`. `TRICK_HOME` follows the standard Trick
environment convention and is required only by the Docker reference-CSV regen
flow.

## Pre-commit

**Before every commit**, run the same checks CI runs:

```bash
cargo fmt --check && cargo clippy --workspace --tests -- -D warnings
```

Fix any issues before committing. This avoids lint-only CI failures.

## CI job matrix

CI lives in `.github/workflows/ci.yml` and uses the test-tier name filtering.

- **PRs**: `check` (fmt + clippy), `test` (unit + tier 2),
  `test-parity-trajectory` (the fast `bevy_parity_*` subset — the
  exclusion list lives inline in `.github/workflows/ci.yml`), and
  `test-tier3` (tier 3 excluding `earth_moon`) run in parallel for
  fast feedback.
- **Main push**: same jobs, plus `test-tier3-full` (includes the
  `earth_moon` test, ~17 min, generates the cross-validation report)
  and `test-parity-trajectory-full` (the heavy parity binaries excluded
  from PR CI; full bit-identity coverage on `main`).
- **Push to non-main branches**: no CI (only PRs and main trigger workflows).

## Naming conventions for CI routing

- New Tier 3 tests use the **`tier3_` function-name prefix** so cargo's
  name-based filtering picks them up automatically.
- New parity wrappers use the **`bevy_parity_` file-stem prefix**, enforced by
  `crates/astrodyn_verif_parity/tests/parity_coverage.rs`, so they route
  through `test-parity-trajectory` automatically.
- If a new wrapper joins the heavy bucket (e.g., a multi-hour SH+drag
  trajectory), extend the exclusion filter in `.github/workflows/ci.yml` so
  the PR lane stays under ~12 min — see the comment above the filter for the
  binaries currently deferred to `test-parity-trajectory-full`.

See `crates/astrodyn_bevy/tests/README.md` for tier conventions and the
tolerance / baseline workflow.

## Cross-validation tolerances

`CrossvalReport` (`crates/astrodyn_verif_jeod/src/crossval.rs`) computes
per-component max errors between our trajectory and JEOD's. It has no
tolerance fields — tolerances live exclusively in the test source code.

Tests assert tolerances via `report.assert_position(tol)`,
`report.assert_velocity(tol)`, `report.assert_quat_angle(tol)`,
`report.assert_ang_vel(tol)` (per-component checks), plus
`assert!(var < tol, "metric_name")` for extras added via
`report.add_extra(name, val, unit)`.

The report binary (`cargo run -p astrodyn_verif_jeod --bin tier3_report`)
extracts all tolerance values from test source files by regex-parsing the
`assert_*` call sites and `assert!(var < LITERAL, "name")` patterns. JSON
contains only errors — no tolerances.

**Tolerance policy:** each tolerance is set to 5% above the observed max error,
per component. Since JEOD reference CSVs are static and our code is
deterministic, errors are fixed numbers — no runtime-computed or conditional
tolerances.

When tightening tolerances after a code improvement: run the full test suite,
inspect the JSON reports in `target/tier3_crossval/`, compute `error * 1.05`
per component, and update the literal values in the test source.

See `crates/astrodyn_bevy/tests/README.md` "Baseline-freeze workflow" for the
`crates/astrodyn_verif_jeod/test_data/baselines.json` gating policy, the
`tier3_baseline_diff` check, and the refreeze workflow.
