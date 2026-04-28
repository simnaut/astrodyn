# Tests

This directory holds workspace-root integration tests. Per-crate unit tests
live under each crate's own `tests/` directory. This README documents the
naming conventions, the three-tier verification model, and the
`CrossvalReport` / baseline workflows used by Tier 3 tests.

For tolerance mechanics and CI lane definitions, see CLAUDE.md
"Cross-validation tolerances" and "Test tiers and CI" — this file does not
duplicate those sections.

## Naming conventions

Test discovery and CI filtering rely on filename and function-name prefixes.
Match exactly:

- `tier2_*.rs` and `tier2_*` — reference-vector tests that compare a function
  or a pipeline stage against a static JEOD-source-derived value
  (e.g. `crates/jeod_dynamics/tests/tier2_body_init.rs`).
- `tier3_*.rs` and `tier3_*` — full-pipeline trajectory cross-validation
  against a JEOD Trick CSV. The function-name prefix is the hook for CI's
  slow lane (`cargo nextest run -E 'test(tier3_)'`). E.g.
  `crates/jeod_runner/tests/tier3_apollo8_frame_switch.rs`.
- `bevy_parity_*.rs` (workspace root) — Bevy adapter must reproduce the
  pure-Rust `jeod_sim` numbers bit-for-bit.
- `integration_*.rs` and other workspace-root files
  (`mission_crate_sanity.rs`, `validation_added_trigger.rs`,
  `invariant_coverage.rs`) — cross-crate behavior.

## The three verification tiers

- **Tier 1 — unit tests.** Pure-function correctness, round-trips,
  convergence. Live alongside the function in `#[cfg(test)] mod tests`.
- **Tier 2 — JEOD reference data.** Static test vectors from JEOD source
  (`verif_out.txt`, Modified_data Python files, `.cc` coefficient arrays)
  validating one function or pipeline stage.
- **Tier 3 — trajectory cross-validation.** Propagate from JEOD initial
  conditions through `Simulation::step()` and compare against a JEOD Trick
  CSV across hours/days. Definition of done for any phase shipping new
  physics.

### Which tier do I need?

- Pure function (math, conversion, helper) → Tier 1.
- JEOD ships a static reference value for the exact input (a `verif_out.txt`
  row, a `Modified_data/*.py` constant) → Tier 2.
- The change touches a `Simulation::step()` stage (forces, torques,
  integration, derived state) and a JEOD verification SIM exercises it
  → Tier 3 against the matching CSV (regenerate with
  `cargo xtask regenerate-tier3` if missing; see CLAUDE.md
  "Generating Tier 3 Reference Data").
- New physics → usually all three.

## `CrossvalReport` API

`crates/jeod_test_data/src/crossval.rs` is the harness for every Tier 3 test
that compares a propagated trajectory against a JEOD reference CSV.

- `StateLog { time, position, velocity, acceleration, quaternion, ang_vel, ang_accel }`
  — one snapshot per timestep; each field is `Option`-typed so a test logs
  only what it cares about.
- `CrossvalReport::compute(test_name, ours, reference) -> Self` — per-component
  max absolute errors plus a rotation-invariant quaternion angle. Asserts
  equal length and time alignment (median-cadence based).
- `assert_position([f64; 3])`, `assert_velocity([f64; 3])`,
  `assert_ang_vel([f64; 3])` — per-component tolerance checks; tolerances
  are literals in the test source.
- `assert_quat_angle(f64)` — scalar rotation-angle tolerance in radians.
- `add_extra(var, val, unit)` — record a test-specific scalar; the test
  asserts via `assert!(var < tol, "name")`.
- `write()` — emit `target/tier3_crossval/<test_name>.json` for
  `tier3_baseline_diff`.
- Typed accessors `max_position_typed() -> Length`,
  `max_velocity_typed() -> Velocity`,
  `max_ang_vel_typed() -> AngularVelocity`,
  `max_quat_angle_typed() -> Angle` for mission-unit reporting.

Canonical usage from `crates/jeod_runner/tests/tier3_apollo8_frame_switch.rs`:

```rust
let report = CrossvalReport::compute("tier3_apollo8_eci_integ", &our_log, &ref_log);
report.write();

report.assert_position([4.8e-5, 3.9e-5, 3.6e-5]);
report.assert_velocity([9.6e-7, 7.7e-7, 7.2e-7]);
report.assert_quat_angle(1e-10);
report.assert_ang_vel([1e-15, 1e-15, 1e-15]);
```

## Tolerances live in test source

`CrossvalReport` carries no tolerance fields. Every tolerance is a literal
in the test's `assert_*` call (or in `assert!(var < LITERAL, "name")` for
extras). Policy: `error * 1.05` per component, rounded to two significant
figures. See CLAUDE.md "Cross-validation tolerances" for the exact policy
and the regex format the report binary uses to extract literals.

## Baseline-freeze workflow

`test_data/baselines.json` records frozen per-test, per-component max
errors. Refactor-only PRs must satisfy:

```text
max_error_new <= max(baseline * 1.0 + 1e-12 * |baseline|, 1e-12)
```

The check is automated by `crates/jeod_test_data/src/bin/tier3_baseline_diff.rs`:

```bash
# Run the Tier 3 suite to populate target/tier3_crossval/*.json
cargo nextest run --workspace -E 'test(tier3_)'

# Compare against the frozen baseline
cargo run -p jeod_test_data --bin tier3_baseline_diff
```

Use `--allow-missing NAME` (or `--allow-missing-from FILE`) to declare
intentionally-skipped tests (e.g. CI's fast lane omits the 17-minute
`tier3_earth_moon_clem`).

When a physics change legitimately moves errors, document the reason in
the PR body and refreeze:

```bash
cargo run -p jeod_test_data --bin tier3_report -- --freeze-baselines
```

Commit the updated `test_data/baselines.json` and `test_data/baselines.md`
together. See CLAUDE.md "Baseline freeze" for the full policy.
