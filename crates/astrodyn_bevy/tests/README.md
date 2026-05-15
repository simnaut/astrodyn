# Tests

This directory holds workspace-root integration tests. Per-crate integration
tests live under each crate's own `tests/` directory; per-crate unit tests
live alongside the code they exercise in `#[cfg(test)] mod tests`. This
README documents the naming conventions, the three-tier verification model,
and the `CrossvalReport` / baseline workflows used by Tier 3 tests.

For tolerance mechanics and CI lane definitions, see CLAUDE.md
"Cross-validation tolerances" and "Test tiers and CI" — this file does not
duplicate those sections.

## Naming conventions

Test discovery and CI filtering rely on filename and function-name prefixes.
Match exactly:

- `tier2_*.rs` and `tier2_*` — reference-vector tests that compare a function
  or a pipeline stage against a static JEOD-source-derived value
  (e.g. `crates/astrodyn_dynamics/tests/tier2_body_init.rs`).
- `tier3_*.rs` and `tier3_*` — full-pipeline trajectory cross-validation
  against a JEOD Trick CSV. The function-name prefix is the hook for CI's
  slow lane (`cargo nextest run -E 'test(tier3_)'`). E.g.
  `crates/astrodyn_verif_jeod/tests/tier3_apollo8_frame_switch.rs`.
- `bevy_parity*.rs` (in `crates/astrodyn_verif_parity/tests/`) — Bevy adapter must reproduce the
  pure-Rust `astrodyn` numbers bit-for-bit. Includes the umbrella
  `bevy_parity.rs` and per-feature files like `bevy_parity_drag.rs`.
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

## Tier 3 conventions

Every Tier 3 test must observe the following conventions in addition to
the tier-and-naming rules above. Failing any of these is a correctness
hazard, not a style preference.

1. **Sample cadence must match the JEOD CSV log cadence.** JEOD's logger
   frequently writes faster than the integrator runs (e.g. CSV rows at
   0.5 s while `IntegLoop ... DYNAMICS=1.0`). On off-cadence rows Trick
   re-emits the integrator's output from the previous integer second —
   so naive row-by-row comparison passes vacuously on the off-cadence
   rows and silently masks real residuals at the actual integrator-
   output instants. Pick one of:
   - **Cadence-aligned (preferred).** Choose an integrator step that
     evenly divides the CSV cadence (e.g. dt = 0.03125 s against a 60 s
     CSV — `60.0 / 0.03125 = 1920`). Then call
     `CrossvalReport::assert_cadence_matches(&reference_log,
     integrator_dt, 1e-6)` once before `compute` to fail loudly if the
     ratio is ever non-integer.
   - **Filter off-cadence rows.** When the integrator deliberately runs
     coarser than the CSV (e.g. SIM_ref_attach's dt = 1.0 s against a
     0.5 s CSV), skip rows that don't fall on an integrator-output
     instant before logging into the `StateLog` slice that
     `CrossvalReport::compute` sees. Use
     `CrossvalReport::is_on_integrator_cadence(row.time, dt)` in the
     row loop. The canonical template lives in
     [`crates/astrodyn_verif_jeod/tests/tier3_sim_ref_attach.rs`][cadence-template].
   - **Document the rationale.** Either choice must be justified in a
     pure-rationale comment near the top of the test (or at the row
     loop) that names the integrator step, the CSV cadence, the ratio,
     and what Trick does on the off-cadence rows. The
     [`tier3_sim_dyncomp_run_attach_to_ref_frame.rs`][cadence-rationale]
     header block is the worked example.

[cadence-template]: ../crates/astrodyn_verif_jeod/tests/tier3_sim_ref_attach.rs
[cadence-rationale]: ../crates/astrodyn_verif_jeod/tests/tier3_sim_dyncomp_run_attach_to_ref_frame.rs

## `CrossvalReport` API

`crates/astrodyn_verif_jeod/src/crossval.rs` is the harness for every Tier 3 test
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

Canonical usage from `crates/astrodyn_verif_jeod/tests/tier3_apollo8_frame_switch.rs`:

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

`crates/astrodyn_verif_jeod/test_data/baselines.json` records the per-test, per-component Tier 3 max
absolute errors. The snapshot was frozen at Phase 0 of the type-system
refactor (#101) and every refactor-only phase since (0, 2–6, 9–10) is
gated on it: a refactor-only PR must not regress past

```text
max_error_new <= max(baseline * 1.0 + 1e-12 * |magnitude|, 1e-12)
```

Baselines are **not silently widened**. Loosening a baseline requires a
PR comment citing the physical justification — a code change that
legitimately moves the error, not a tolerance papering over a regression.

This is enforced structurally by `scripts/check_baseline_widening.sh`,
which runs in the `check` CI lane and diffs the working
`baselines.json` against `origin/main`. For each (test, metric,
component) tuple it computes a widening ratio and emits a GitHub
Actions annotation in one of three buckets:

| widening ratio   | bucket  | CI effect                              |
| ---------------- | ------- | -------------------------------------- |
| ≤ 1.50x          | silent  | numerical drift, no signal             |
| 1.50x – 2.00x    | warning | `::warning::` annotation, lane passes  |
| > 2.00x          | error   | `::error::` annotation, lane fails     |

Tightenings, brand-new test entries, and removed test entries are
surfaced as `::notice::` lines but never fail the lane. The thresholds
are overridable via `BASELINE_WARN_RATIO` and `BASELINE_FAIL_RATIO`
environment variables; the base ref is overridable via
`BASELINE_BASE_REF` (default `origin/main`). When a refreeze
legitimately crosses the warning band, name the physical change in
the PR description; when it crosses the error band, the same applies
and the PR author should rerun the lane with adjusted thresholds set
locally to confirm the new ratios before pushing.

The check is automated by `crates/astrodyn_verif_jeod/src/bin/tier3_baseline_diff.rs`:

```bash
# Run the Tier 3 suite to populate target/tier3_crossval/*.json
cargo nextest run --workspace -E 'test(tier3_)'

# Compare against the frozen baseline
cargo run -p astrodyn_verif_jeod --bin tier3_baseline_diff
```

Use `--allow-missing NAME` (or `--allow-missing-from FILE`) to declare
intentionally-skipped tests (e.g. CI's fast lane omits the 17-minute
`tier3_earth_moon_clem`).

When a physics change legitimately moves errors, document the reason in
the PR body and refreeze:

```bash
# Run the full Tier 3 suite first (including earth_moon)
cargo nextest run --workspace -E 'test(tier3_)'

# Refreeze the snapshot
cargo run -p astrodyn_verif_jeod --bin tier3_report -- --freeze-baselines
```

Commit the updated `crates/astrodyn_verif_jeod/test_data/baselines.json` and `crates/astrodyn_verif_jeod/test_data/baselines.md`
together — `baselines.md` is the human-readable companion produced by the
same binary.
