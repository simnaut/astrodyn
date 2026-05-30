# CLAUDE.md

## Project

Pure-Rust port of [NASA JEOD](https://github.com/nasa/jeod) v5.4 — the JSC Engineering
Orbital Dynamics physics — as an **engine-agnostic** library. Ongoing work is tracked
as GitHub issues.

**Engine independence is a design goal, not an accident.** JEOD's physics carried a
dependence on NASA's Trick sim framework (Trick-generated `S_define` wiring, logging,
job scheduling threaded into the physics); that coupling is the mistake we are
deliberately avoiding. astrodyn's physics has *zero* dependence on any sim engine: it
is plain borrow-based Rust that a host drives. Bevy ECS (`astrodyn_bevy`) and the
standalone arena runner (`astrodyn_runner`) are two interchangeable drivers, and a
third-party host can drive the same pipeline with its own storage and scheduling.
Treat the engine as a removable consumer at the top of the stack — never let engine
concerns (ECS components, Bevy systems, schedules, async) leak into `astrodyn` or the
`astrodyn_*` physics crates.

**Read before refactoring**:
- [Strategy](https://github.com/simnaut/astrodyn/wiki/Strategy) — architecture and project history.
- [Audit-2026-05](https://github.com/simnaut/astrodyn/wiki/Audit-2026-05) — load-bearing guardrails
  (CI, parity-superset invariant, typed-quantity facade, JEOD invariant catalog).

Non-crate docs (architecture, contributor primers, audits, design discussions) live in
the [wiki](https://github.com/simnaut/astrodyn/wiki). `docs/` is reserved for files that
must travel with source — currently only `docs/JEOD_invariants.md`, consistency-checked
against `// JEOD_INV: XX.YY` source tags by `tests/invariant_coverage.rs`. Per-crate
`README.md` follows Rust convention. Default new prose to a new wiki page, not a new
`docs/` file.

## Non-negotiables

- **Three-layer architecture**. Physics lives in `astrodyn_*` crates (zero Bevy
  dependency). Orchestration lives in `astrodyn` (workspace root crate at `src/`, zero
  Bevy dependency) and composes `astrodyn_*` functions into pipeline stages. Bevy wiring
  lives in `astrodyn_bevy` (thin glue: component derives, systems that delegate to
  `astrodyn` functions, plugin registration). Every workspace consumer of physics —
  mission crates, `astrodyn_bevy`, `astrodyn_runner`, the verification crates — reads
  through `astrodyn` and only `astrodyn` (+ `bevy` for the adapter). `astrodyn_runner`
  is a parallel non-Bevy consumer for batch use and Tier 3 harness; mission crates
  never depend on it. Enforced by `scripts/check_no_bypass_deps.sh`. Never put physics
  algorithms directly in a Bevy system function — the system queries components, then
  calls a function in `astrodyn`. See [Architecture](https://github.com/simnaut/astrodyn/wiki/Architecture)
  for the full rationale.

- **Computational independence**. Every computation in the production path is our own
  ported Rust code. Never feed JEOD output (CSV files, logged matrices, precomputed
  values) into our pipeline. JEOD reference data is for *test comparison only*. When a
  test reveals missing physics, port the JEOD code — don't approximate it or read JEOD's
  output.

- **Tier 3 cross-validation is definition of done**. When new physics is delivered,
  a `tier3_*` test exercising it via the full `Simulation::step()` pipeline must
  accompany it. Initial conditions may come from JEOD source files (`Modified_data/*.py`,
  `S_define`, gravity coefficient files) or the t=0 row of a JEOD reference CSV — both
  are JEOD source data. The prohibition is against feeding JEOD output into our
  computation at intermediate time steps (e.g., setting position from CSV at t=100s).
  Bevy adapter coverage is inherited transitively via `bevy_parity_*` wrappers; the
  superset is policed by `crates/astrodyn_verif_parity/tests/parity_coverage.rs`.

- **No half-baked implementations**. Match JEOD's verification configuration exactly —
  same gravity model, rotation model, epoch, integration step, force terms. If matching
  reveals a bug or missing capability, fix the code; don't substitute simpler physics,
  use a convenient epoch, or widen tolerances. "Good enough" isn't acceptable when JEOD
  specifies the answer.

- **Fail Loudly**. Misconfigurations and invalid physics must panic immediately with a
  diagnostic message that names the broken invariant, the input that triggered it, and
  what the caller should change. Use `assert!`/`assert_eq!` (panics in release builds,
  where most runs actually execute), never `debug_assert!` (silently no-ops under
  `--release`). Patterns: `expect("<noun phrase>: <what to fix>")` or
  `unwrap_or_else(|err| panic!("<context with values>: {err:?}. <how to fix>"))`.
  Reserve `debug_assert!` for expensive perf-only checks with no correctness consequence.
  Deeply-internal kernels with structurally-proven preconditions may keep terse
  `expect()`. A test or implementation that compiles, runs, and silently produces wrong
  physics is the half-baked failure mode the previous rule forbids.

## Hot paths

```bash
cargo nextest run --workspace                                         # all tests
cargo nextest run --workspace -E 'not test(tier3_) and not test(bevy_parity)'  # unit + Tier 2
cargo nextest run --workspace -E 'test(tier3_)'                       # tier 3 only
cargo nextest run --workspace -E 'test(bevy_parity)'                  # parity wrappers

cargo fmt --check && cargo clippy --workspace --tests -- -D warnings  # before every commit
```

Fresh clone supports the full test suite (unit, Tier 2, **and Tier 3**) with no
`$JEOD_HOME` — every test reads committed fixtures under `test_data/`. CI fences enforce
that `JEOD_HOME` and `JEOD_PATH` are unset before the test jobs run. `$JEOD_HOME` is
only needed when **regenerating** fixtures via the `extract_*` binaries or the Docker
reference-CSV flow; see [Environment](https://github.com/simnaut/astrodyn/wiki/Environment)
for the regen workflow, NESC track, and ephemeris-kernel handling.

CI routing: new Tier 3 tests use the `tier3_` function-name prefix; new parity wrappers
use the `bevy_parity_` file-stem prefix. PR CI splits into fast (`tier3_` excluding
`earth_moon`) and main-only buckets; if a new parity wrapper joins the heavy bucket,
extend the exclusion filter in `.github/workflows/ci.yml`. See
[CI](https://github.com/simnaut/astrodyn/wiki/CI) for the full job matrix.

## Precision and conventions

- **`f64` everywhere**. Never use Bevy `Transform`/`GlobalTransform` (f32). Spherical-
  harmonics coefficients use `Vec<Vec<f64>>`. `nalgebra` is available transitively via
  `anise` but not used directly.
- **Typed quantities at API surfaces**, raw `glam` types inside kernels. Public/mission
  code uses `Position<F: Frame>`, `Velocity<F>`, `Acceleration<F>`,
  `SecondsSince<S: TimeScale>`, `Quat<L, T>`, `NormalizedQuat`,
  `FrameTransform<From, To>`, and the `F64Ext` facade (`400.0.km()`, `51.6.deg()`,
  `420_000.0.kg()`). Internal physics kernels drop to raw `glam::DVec3`/`DQuat`/`DMat3`
  via `.raw_si()` and re-wrap on exit. Three kind-distinct inertial-flavor phantoms:
  `RootInertial`, `PlanetInertial<P>`, `IntegrationFrame` (RF.10). See
  [Type-System](https://github.com/simnaut/astrodyn/wiki/Type-System) for the
  contributor primer (phantom-tag pattern, frame/scale additions, escape hatches);
  worked examples at `crates/astrodyn_bevy/examples/typed_mission.rs` and
  `crates/astrodyn_bevy/examples/multi_body_scenario.rs`.
- **Quaternion convention**: JEOD = scalar-first, left-transformation `[q0, q1, q2, q3]`
  with q0 scalar; `glam::DQuat` = scalar-last `[x, y, z, w]`. Convert at the boundary.
  Test with non-trivial rotations (not identity, not 90° axes).
- **JEOD Convention Rule**: for any field-name ambiguity (e.g., `time_periapsis` — to
  or since?), **read the JEOD C++ source**. Do not guess or reason by analogy. A wrong
  sign or direction guess produces code that compiles, passes trivial tests, and
  silently gives wrong answers at scale. (This rule was set after an agent guessed
  `M = 2π − n·t` instead of `M = n·t` for the periapsis time → mean-anomaly map,
  producing 11,668 km error against NASA flight data.)

## Lints

- Every workspace crate has `#![forbid(unsafe_code)]` at the crate root (`lib.rs`,
  every `main.rs`, every binary under `src/bin/`, every `example` target). FFI/SIMD
  exceptions opt out per-crate with `#![allow(unsafe_code)]` plus a documented
  justification in that crate's `lib.rs`.
- Workspace `[workspace.lints]` denies five Clippy lints that protect against silent
  FP bugs: `float_cmp`, `cast_precision_loss`, `lossy_float_literal`,
  `cast_possible_truncation`, `as_underscore`. Every `#[allow(clippy::*, reason = "…")]`
  carries a non-empty `reason` (the bypass is the audit log; a bypass without a
  justification is a TODO that someone forgot).
- Recurring "bit-exact" rationale strings are catalogued in
  [`astrodyn_quantities::lint_reasons::clippy_float_cmp`](crates/astrodyn_quantities/src/lint_reasons.rs).
  Clippy's `reason` requires a string literal (const paths don't typecheck), so copy
  the canonical phrasing verbatim. The
  `crates/astrodyn_quantities/tests/lint_reasons_catalog.rs` test enforces that every
  catalog string still appears at ≥2 sites.

## JEOD invariant tracking

Every JEOD invariant we encounter (or enforce) goes in `docs/JEOD_invariants.md` (one
row per invariant with a `Section.Tag` ID like `GV.04`, the JEOD enforcement mechanism,
and our status: `enforced`/`partial`/`deferred`/`n/a`/`structural`). Enforcement sites
carry `// JEOD_INV: XX.YY — <what the code does>` comments. Tags describe **what our
code does**, not JEOD's — note divergences. `tests/invariant_coverage.rs` enforces
bidirectional consistency: every non-`n/a`/`deferred` row needs ≥1 source tag, and
every source tag must reference a catalog entry.

New invariant discovered while reading JEOD source? Add a catalog row + the source
tag, then `cargo test --test invariant_coverage`. See
[JEOD-Invariant-Workflow](https://github.com/simnaut/astrodyn/wiki/JEOD-Invariant-Workflow)
for the negative-test convention (`enforced` rows should have a `#[should_panic]`
test) and the section-tag conventions.

## Cross-validation tolerances

`CrossvalReport` (`crates/astrodyn_verif_jeod/src/crossval.rs`) computes per-component
max errors. Tolerances live in test source code (literal values in
`assert_position(tol)` / `assert!(var < LITERAL, "name")`), set to **1.05× observed
max** per component. JEOD CSVs are static and our code is deterministic — tolerances
are fixed numbers, not runtime-computed. Tightening after a code improvement: run the
full suite, compute `error * 1.05` per component, update the literal. See
[`crates/astrodyn_bevy/tests/README.md`](crates/astrodyn_bevy/tests/README.md) for the
`baselines.json` freeze workflow.

## Mission crates / Bevy adapter

Mission crates depend on `astrodyn_bevy` (+ transitively `astrodyn`). Compose scenarios
via the typestate `VehicleBuilder` and either `SimulationBuilder::populate_app::<P>(&mut app)`
(full-scenario composition) or `VehicleConfig::spawn_bevy` (single-vehicle insertion
into a running App). The compiler rejects frame/unit mismatches at API boundaries with
`#[diagnostic::on_unimplemented]` messages in physics language. Worked examples:

- Full scenario: `crates/astrodyn_bevy/examples/multi_body_scenario.rs`
- Single vehicle: `crates/astrodyn_bevy/examples/typed_mission.rs`

Contributor primer and adding-a-frame walkthrough: [Type-System wiki](https://github.com/simnaut/astrodyn/wiki/Type-System).

## JEOD source and Tier 3 regeneration

For navigating JEOD source (DynBody, RNP, gravity, contact, etc.) and the catalog of
extractable verification data (the `extract_*` binaries and what they parse), see
[JEOD-Source-Data](https://github.com/simnaut/astrodyn/wiki/JEOD-Source-Data). For the
JEOD pipeline → `AstrodynSet` stage mapping and the `DynBody`/`DynManager` → ECS
class-by-class crosswalk, see [JEOD-ECS-Mapping](https://github.com/simnaut/astrodyn/wiki/JEOD-ECS-Mapping).

For Tier 3 reference-CSV regeneration (Docker, `cargo xtask regenerate-tier3`,
incremental vs `--force`, adding a new sim), see
[Tier3-Regeneration](https://github.com/simnaut/astrodyn/wiki/Tier3-Regeneration).

## Common pitfalls

- **JEOD `RefFrameState` stores state relative to parent frame, not global.** Don't
  confuse with absolute/inertial coordinates.
- **Gravity acceleration excludes the integration frame's own acceleration toward the
  source.** For Earth-centered inertial integration, the Sun's contribution is the
  *differential* acceleration (vehicle − Earth toward Sun), not absolute.
- **Trick DRAscii silently drops unregistered variables** in `generate_references.sh`:
  variable names must match the S_define's object names exactly, or the CSV column is
  silently missing with no error.
- **Trick sim working directory**: JEOD sims run from the SIM root
  (e.g., `verif/SIM_dyncomp/`), not from `SET_test/RUN_*/`. The `input.py` files use
  paths relative to the SIM root.

See [Common-Pitfalls](https://github.com/simnaut/astrodyn/wiki/Common-Pitfalls) for the
full catalog (geodetic longitude at the poles, DynBody three-frame structure,
`MassProperties` parallel-axis theorem, etc.).
