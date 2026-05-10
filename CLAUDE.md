# CLAUDE.md

## Project

Rust reimplementation of [NASA JEOD](https://github.com/nasa/jeod) (JSC Engineering
Orbital Dynamics, v5.4, 714 C++ source files) using Bevy ECS instead of NASA's Trick.
See the [Strategy wiki page](https://github.com/simnaut/astrodyn/wiki/Strategy)
for architecture and phase summaries. The original phased implementation plan
(Phases 1–7) closed in April 2026; ongoing work is tracked as GitHub issues.

### Environment Setup

`cargo build --workspace && cargo nextest run --workspace` works on a
fresh clone of this repo with no JEOD checkout — every test (unit,
Tier 2, **and Tier 3**) reads from committed fixtures under
`test_data/`. CI's "Test (unit + tier 2)" and Tier 3 jobs each include
a regression fence that asserts `JEOD_HOME` and `JEOD_PATH` are both
unset before running.

You only need `$JEOD_HOME` when **regenerating fixtures** after a JEOD
upgrade. The `extract_*` regen binaries are distributed by owner crate:
`extract_grav_coeffs` and `extract_mars_data` live in
`crates/astrodyn_gravity/src/bin/` (parsing JEOD `.cc` files into
`crates/astrodyn_gravity/test_data/gravity/*.bin`),
`extract_planet_pfixposn` lives in `crates/astrodyn_planet/src/bin/`,
and `extract_body_init` / `extract_jeod_validation` live in
`crates/astrodyn_verif_jeod/src/bin/`. The verbatim NASA JEOD source
mirror under `crates/astrodyn_verif_jeod/test_data/jeod_inputs/` is
refreshed via the `cp` recipe in that directory's `README.md`. Both
flows accept either `$JEOD_HOME` or `--jeod-home <PATH>`.

For the Docker reference-CSV regen (Tier 3 baselines):

```bash
cd /home/user/git   # or wherever your repos live
git clone https://github.com/nasa/jeod.git
git clone https://github.com/nasa/trick.git

export JEOD_HOME=$(pwd)/jeod
export TRICK_HOME=$(pwd)/trick

cargo xtask regenerate-tier3
```

`JEOD_HOME` is the standard NASA convention; the older `JEOD_PATH`
alias was retired in #239. `$TRICK_HOME` is required only by the
Docker reference-CSV regen flow.

## Three-Layer Architecture (non-negotiable)

All physics lives in **`astrodyn_*`** crates (pure Rust, zero Bevy dependency).
Orchestration lives in **`astrodyn`** — the workspace **root crate** (`src/` at
workspace root) — which composes `astrodyn_*` functions into pipeline stages
and re-exports all types; zero Bevy dependency. Bevy wiring lives in
**`astrodyn_bevy`** (`crates/astrodyn_bevy/` — thin glue: component derives,
systems that delegate to `astrodyn` functions, plugin registration).

`astrodyn_bevy` depends **only** on `astrodyn` + `bevy` — never on `astrodyn_*`
crates directly. **`astrodyn` is the single API surface for the production
path:** every Bevy system, every mission crate, every downstream consumer that
ships in a real simulation reads the workspace through `astrodyn` and only
through `astrodyn`. The narrower the production-path surface, the smaller the
contract that has to stay stable across phases.

This rule is scoped to the production path because the workspace also contains
a non-shipping test harness (`astrodyn_runner`) whose role is the inverse: it owns
its own state container and *needs* to construct concrete physics types
itself. See [`astrodyn_bevy` vs `astrodyn_runner`](#astrodyn_bevy-vs-astrodyn_runner-two-parallel-consumers-of-astrodyn)
below for why that asymmetry is intentional and what each consumer is allowed
to depend on.

Never put physics algorithms directly in a Bevy system function. The system queries
components, then calls a `astrodyn` function. This keeps physics portable to other
ECS frameworks, WASM, or standalone batch computation.

### `astrodyn_bevy` vs `astrodyn_runner`: two parallel consumers of `astrodyn`

The titular simulation environment is **`astrodyn_bevy`** (`crates/astrodyn_bevy/`).
It is the production target — Bevy ECS is the chosen runtime for mission code,
and the ECS world is the single source of truth for all state.

**`astrodyn_runner` is a parallel non-Bevy consumer of `astrodyn`**, not a
dependency of `astrodyn_bevy`. It exists because the `astrodyn_*` and `astrodyn`
crates have **zero Bevy dependency** by design (the layer rule above), so
they can be exercised directly from a plain Rust binary that owns its own
state. `astrodyn_runner` provides that owned-state harness for:

- **Tier 3 cross-validation tests** (`crates/astrodyn_verif_jeod/tests/tier3_*.rs`)
  — propagating from JEOD initial conditions and comparing against
  Trick reference CSVs without standing up a Bevy `App`. The verif crate
  drives `astrodyn_runner::Simulation` end-to-end.
- **Batch propagation, scripting, and offline studies** that don't need
  ECS scheduling, parallelism, or Bevy plugins.

`astrodyn_runner` and `astrodyn_bevy` sit *next to* each other in the dep graph —
both depend on `astrodyn` and *only* `astrodyn` for physics; neither
depends on the other (except that `astrodyn_bevy` carries
`astrodyn_runner` as a dev-dep for parity-style tests). Any improvement
that lands in `astrodyn_*` or `astrodyn` (typed quantities,
phantom-frame discipline, witness-gated constructors like
`BodyAttitude<V>`, …) benefits both consumers identically.

**Every workspace consumer of the `astrodyn` pipeline — mission crates,
`astrodyn_bevy`, `astrodyn_runner`, and the verification crates
(`astrodyn_verif_jeod`, `astrodyn_verif_parity`) — depends on
`astrodyn` and only `astrodyn` for physics** (+ `bevy` for the Bevy
adapter, + `astrodyn_runner` as a dev-dep on `astrodyn_bevy` for
parity tests, + `astrodyn_runner` and `astrodyn_verif_jeod` as
gateway-consumer deps for `astrodyn_verif_parity`). The "single API
surface" rule applies uniformly: every physics type, function, or
module that any consumer reaches must be reachable through `astrodyn`.
If something isn't, the fix is to widen `astrodyn`'s curated
re-export surface, not to add a direct `astrodyn_*` physics dep.

Owner-crate unit / Tier 2 / Tier 3 tests live inside their owning
crate's own `tests/` directory and reach the crate under test through
normal in-crate test access (no gateway, no verif crate). The verif
crates host the *cross-validation* against JEOD trajectories; they no
longer host kernel-level subsystem tests.

A CI lint (`scripts/check_no_bypass_deps.sh`) enforces the rule
structurally by failing the build if `astrodyn_runner`,
`astrodyn_bevy`, `astrodyn_verif_jeod`, or `astrodyn_verif_parity`
declare any direct `astrodyn_*` physics-crate dep.

Mission crates that target the production runtime depend on
`astrodyn_bevy` (and transitively `astrodyn`). They never depend on
`astrodyn_runner`, and they never reach around `astrodyn` to pull a
physics crate directly — if you find yourself writing
`astrodyn_dynamics = …` in a mission `Cargo.toml`, that is the bug.

## Computational Independence (non-negotiable)

Every computation in our pipeline must be our own ported Rust code. Never use JEOD's
output data (CSV files, logged matrices, precomputed values) as input to our
computation. JEOD reference data is used **only** for comparison in tests — never in
the production code path.

When a test reveals missing physics (e.g., Earth rotation needed for gravity
evaluation), the answer is to **port the JEOD code**, not to approximate it or read
JEOD's output. The purpose of this project is an independent reimplementation.

## Tier 3 Cross-Validation (non-negotiable)

Tier 3 tests (trajectory cross-validation against JEOD Trick simulations via Docker)
are part of the **definition of done** for every phase. They are not optional extras
or afterthoughts. When a phase delivers new physics, a Tier 3 test exercising that
physics must be included.

Every Tier 3 test must exercise the full `Simulation::step()` pipeline end-to-end.
Tests must use only initial conditions from JEOD source files — JEOD data must never
be injected into intermediate computation steps. The Simulation propagates entirely
under its own physics, and results are compared against JEOD's reference output at
checkpoints. Tests that call per-body functions directly or evaluate static data
points bypass the pipeline and must be upgraded to use `Simulation::step()`.

Initial conditions may come from JEOD source files (Modified_data/*.py, S_define,
gravity coefficient files) or from the t=0 row of a JEOD reference CSV — both are
"JEOD source data." The prohibition is against feeding JEOD output into our
computation at intermediate time steps (e.g., setting position from CSV at t=100s).

## No Half-Baked Implementations or Tests (non-negotiable)

Reject every urge to rationalize a simplified, approximate, or incomplete
implementation or test. When JEOD's verification sim exercises a specific gravity
model, rotation model, epoch, integration step size, or force configuration, our
test must match that configuration exactly — not substitute point-mass for spherical
harmonics, skip a force term, use a convenient epoch, or widen tolerances to paper
over a mismatch.

If matching JEOD's case definition reveals a bug or missing capability in our code,
the answer is to fix the code, not to weaken the test. "Good enough" and "close
enough" are not acceptable when the reference implementation specifies the answer.
Every implementation and test must include everything that the corresponding JEOD
implementation or test exercises.

The three verification tiers:
- **Tier 1**: Unit tests — pure function correctness, round-trips, convergence
- **Tier 2**: JEOD reference data — static test vectors from JEOD source files
- **Tier 3**: Trajectory cross-validation — propagate from same initial conditions
  through `Simulation::step()`, compare against JEOD Trick simulation output over
  hours/days

The Bevy adapter inherits Tier 3 coverage **transitively** for the
topics that have a `runner ↔ bevy` parity sibling: when both
`runner ↔ JEOD` (within tolerance) and `runner ↔ bevy` (bit-for-bit)
hold, `bevy ↔ JEOD` follows by transitivity within the same tolerance.
Issue #389 stands up the infrastructure
([`VerificationCaseParityExt::run_and_assert_parity`] +
[`SimulationBuilderBevyExt::populate_app`]) and seeds it with the
common topics; a long tail of tier3 topics is tracked individually in
[`KNOWN_PARITY_GAPS`](https://github.com/simnaut/astrodyn/blob/main/crates/astrodyn_verif_parity/tests/parity_coverage.rs)
for incremental closure (multi-planet scenarios, pre-recipe siblings,
analytical-only tests, scenarios with `pre_step` ephemeris updates that
need a Bevy-side `SimContext` impl — see issue #395). The
`crates/astrodyn_verif_parity/tests/parity_coverage.rs` meta-test
enforces the superset invariant: a new `tier3_*` topic that lands
without either a parity wrapper or a `KNOWN_PARITY_GAPS` exemption
fails CI, preventing silent regression of the transitivity argument
on the topics that *do* carry it.

## Fail Loudly (non-negotiable)

Physics simulations have no graceful-degradation mode. A trajectory that
silently propagates with a missing rotation matrix, an uninitialized mass,
or a misconfigured gravity source is not "approximately right" — it is
*wrong*, and downstream consumers (mission planners, GN&C developers,
analysts) will treat the wrong answer as correct because nothing failed.
The consequences range from wasted engineering time to mission-critical
errors. Therefore:

- **Misconfigurations and invalid physics must panic immediately.** A
  vehicle configured for `MoonDE421` rotation without `EphemerisR`
  loaded, an `AttachEvent` referencing an entity that isn't a mass body,
  a gravity source with a non-positive `mu`, or a quaternion that drifts
  past `NaN` — all of these must fail loudly with a diagnostic message
  that names the misconfiguration and tells the caller how to fix it. Do
  not `warn!` and continue; do not return a default; do not skip the
  step. Surface the failure at the point of detection.
- **Use `assert!` (and `assert_eq!`/`assert_ne!`), not `debug_assert!`.**
  Invariants that protect physics correctness must hold in release
  builds, where most simulation runs actually execute. `debug_assert!`
  is silently a no-op under `--release` and gives a false sense of
  safety. Reserve `debug_assert!` for invariants that are *purely
  expensive performance checks* with no correctness consequence — those
  are rare in this codebase.
- **Diagnostic messages name the broken assumption.** A panic message
  like `"unwrap on None"` is useless to a mission engineer. Write
  messages that state which invariant was violated, which input
  triggered it, and what the caller should change. The two patterns to
  use are `expect("<noun phrase>: <what to fix>")` and
  `unwrap_or_else(|err| panic!("<context with values>: {err:?}. <how to
  fix>"))`.
- **The exception**: deeply internal invariants that are unreachable by
  construction may keep terse `expect()` messages — e.g.
  `expect("stage 1 runs before stages 2-4")` inside a kernel that
  already proved the precondition holds. These are documentation for
  the next reader, not user-facing diagnostics.

This rule is not in tension with the *No Half-Baked Implementations*
rule above — it is its operational corollary. A test or implementation
that compiles and runs but silently produces wrong physics is the
half-baked failure mode the previous section forbids.

## Precision

Use `f64` everywhere. Do NOT use Bevy's `Transform`/`GlobalTransform` (f32).
Spherical harmonics coefficients use `Vec<Vec<f64>>`. `nalgebra` is available
transitively via `anise` but not used directly.

After the type-system refactor (#101), there are two layers to choose between:

- **Public/mission-crate code** uses typed quantities from `astrodyn_quantities`:
  `Position<F: Frame>`, `Velocity<F>`, `Acceleration<F>`, `SecondsSince<S: TimeScale>`,
  `Quat<L, T>`, `NormalizedQuat`, `FrameTransform<From, To>`, and the `F64Ext`
  facade (`400.0.km()`, `51.6.deg()`, `420_000.0.kg()`). Mission code never sees
  `DVec3`/`DQuat`/`DMat3` or `PhantomData`. The compiler rejects cross-frame
  mismatches, scalar-vs-vector quaternion confusion, and unit-dimensional errors
  at compile time. Custom `#[diagnostic::on_unimplemented]` messages render
  errors in physics language (e.g., *"expected `Position<RootInertial>`, found
  `Position<Ecef>` — apply a `FrameTransform<Ecef, RootInertial>` first"*).

- **Internal physics-crate kernels** (the inside of `astrodyn_*` `*_typed` functions
  and the underlying `_inner`/`_impl` math) use raw `glam::DVec3`/`DQuat`/`DMat3`
  for arithmetic density. The typed siblings call `.raw_si()` at the boundary
  to drop into the kernel and re-wrap on exit. This keeps numerics fast and the
  public surface typed.

See the [Type-System wiki page](https://github.com/simnaut/astrodyn/wiki/Type-System) for the contributor primer (phantom-tag pattern,
adding a new frame/scale/quantity, reading compiler errors, escape hatches)
and `examples/typed_mission.rs` for the canonical worked example.

### Inertial-frame phantoms (#255 / RF.10)

There are three kind-distinct inertial-flavor phantoms:

- `RootInertial` — the simulation's root inertial frame. Required by
  consumers that mix body state with root-inertial source positions
  (gravity, relativistic corrections, SRP, solar beta, earth lighting).
- `PlanetInertial<P: Planet>` — a particular planet's inertial frame.
  Required by planet-centered consumers (atmosphere, drag velocity,
  LVLH, geodetic, orbital elements). In realistic configs the body's
  integration frame *is* `PlanetInertial<P>` for the body's planet.
- `IntegrationFrame` — a body's integration frame. Stored on
  `SimBody.trans` so the compiler refuses to silently pass
  integration-frame state where root-inertial is required. The only
  safe transition is `body.trans.to_inertial(&integ_origin)` (the
  `IntegOrigin` shift) — applied at *shift sites* only.

`docs/JEOD_invariants.md` row `RF.10` enumerates which sites are
structurally guarded vs convention-only and why each consumer falls
into one or the other.

## Quaternion Convention

JEOD uses **scalar-first, left-transformation** quaternions: `[q0, q1, q2, q3]`
where q0 is scalar. `glam::DQuat` uses `[x, y, z, w]` where w is scalar.
Always convert at the boundary. Test with non-trivial rotations (never just identity
or 90-degree axes).

## JEOD Source Navigation

Key directories in `../jeod`:

```
models/dynamics/dyn_body/          DynBody — the central vehicle class (~1200 lines)
models/dynamics/dyn_manager/       DynManager — simulation orchestrator
models/dynamics/mass/              MassBody — rigid body mass properties and trees
models/dynamics/body_action/       BodyAction — initialization (orbit, LVLH, NED)
models/dynamics/derived_state/     EulerDerivedState, OrbElemDerivedState, etc.
models/environment/gravity/        Spherical harmonics gravity (Gottlieb algorithm)
models/environment/gravity/data/   Coefficient files (C++ headers with arrays)
models/environment/time/           Time scales (TAI/UTC/UT1/TDB/TT/GMST)
models/environment/time/data/      Leap_Second.dat
models/environment/ephemerides/    DE4xx binary ephemeris reader
models/environment/planet/         Planet shape, radius, flattening
models/environment/atmosphere/     MET atmosphere model
models/environment/RNP/            Earth rotation (precession, nutation, polar motion)
models/interactions/aerodynamics/  Drag force computation
models/interactions/radiation_pressure/  Solar radiation pressure
models/interactions/gravity_torque/      Gravity gradient torque
models/utils/ref_frames/           RefFrame tree (backbone of all coordinate systems)
models/utils/integration/          Gauss-Jackson, LSODE integrators
models/utils/orbital_elements/     Cartesian <-> Keplerian conversion
models/utils/quaternion/           Quaternion math
models/utils/planet_fixed/         Geodetic coordinates
models/utils/lvlh_frame/           LVLH frame
```

The spherical harmonics core algorithm is in:
`models/environment/gravity/src/spherical_harmonics_calc_nonspherical.cc`

Gravity coefficients are C++ arrays in:
`models/environment/gravity/data/include/earth_GGM05C.hh` (and similar)

## JEOD Verification Data

JEOD has 479 regression tests and 262 unit tests. Most unit tests are structural (empty
bodies, mock checks) — only two sources have extractable numerical test vectors:

- `models/environment/gravity/verif/unit_tests/grav_geospherical/data/verif_out.txt`
  40 test cases: position -> expected gravity acceleration/gradient/potential.
  Format: 18 space-separated numeric fields per line.

- `models/dynamics/derived_state/verif/unit_tests/euler_derived_state_ut.cc`
  6 test cases: rotation matrix -> expected Euler angles.

Reference state vectors (ISS, STS-114) are in Python files at:
`models/dynamics/body_action/verif/SIM_orbinit/Modified_data/ISS/`

These Python files come in three parsability tiers:
1. **Directly parseable**: `reference_*_trans_state.py`, `Leap_Second.dat`, `verif_out.txt`,
   simple `return [value]` files — plain regex extraction.
2. **Needs trick.attach_units() stripping**: orbital element files, mass files, attitude files.
   Pattern: `key = trick.attach_units("degree", 51.67)` — strip wrapper, apply unit conversion.
3. **Not parseable** (~30%): orchestration files with `exec()`, `eval()`. Don't contain
   unique data — they wire together the parseable files above. Ignore them.

## JEOD Integration Loop (maps to FixedUpdate)

JEOD's per-step pipeline collapses to **seven** `AstrodynSet` variants
(`src/sets.rs`), not nine. Two adjacent JEOD steps share a single set
where the bundling is natural — gravity + atmosphere both run in
`Environment`, and frame propagation rides inside the `Integration`
system as its post-step rather than as a separate schedule pass.

```
JEOD step                        →  AstrodynSet variant
1. Time update                   →  AstrodynSet::TimeUpdate
2. Ephemeris update              →  AstrodynSet::EphemerisUpdate
3. Gravity computation         ┐
                               ├─→  AstrodynSet::Environment
4. Atmosphere update           ┘
5. Aero / SRP / gravity torque   →  AstrodynSet::Interaction
6. Force collection              →  AstrodynSet::ForceCollection
7. State integration           ┐
                               ├─→  AstrodynSet::Integration
8. Frame propagation           ┘    (post-step inside the integration system)
9. Derived states                →  AstrodynSet::DerivedState
```

Multi-stage integrators (RK4 = 4 stages) run as an inner loop within the
integration system, not as multiple schedule passes.

## JEOD Key Classes -> ECS Mapping

```
DynBody (1200-line god-class)  ->  ~10 components on an entity
DynManager                     ->  system ordering + resources
GravityManager                 ->  gravity_computation_system
TimeManager                    ->  SimulationTime resource + time_advance_system
EphemerisManager               ->  EphemerisData resource + ephemeris_update_system
RefFrame tree                  ->  entities with Parent/Children hierarchy
BodyAction subclasses          ->  events or direct initialization functions
```

DynBody decomposes into: TranslationalState, RotationalState, MassProperties,
DynamicsConfig, GravityAcceleration, GravityControls, TotalForce, FrameDerivatives,
plus optional interaction components (AerodynamicForce, RadiationForce, GravityTorque).

## Build and Test

Use [`cargo-nextest`](https://nexte.st/) for test execution (matches CI, parallel
by default):

```bash
cargo build --workspace
cargo nextest run --workspace                                 # all tests
# `unit + tier 2 (fast)` excludes Tier 3 *and* the heavy bevy_parity_*
# trajectory tests (they run the same 8h–23-day propagations as
# their tier3_ siblings, twice each). Keep this filter in sync
# with the `test` and `test-parity-trajectory` jobs in CI.
cargo nextest run --workspace \
  -E 'not test(tier3_) and not test(/^bevy_parity_(dyncomp_run(2|4|5a|7)|solar_beta|srp_1st_order|tide_verif)/)'
cargo nextest run --workspace -E 'test(tier3_)'               # tier 3 only
# Heavy parity trajectory tests (the ones excluded above):
cargo nextest run --workspace \
  -E 'test(/^bevy_parity_(dyncomp_run(2|4|5a|7)|solar_beta|srp_1st_order|tide_verif)/)'
cargo nextest run -p astrodyn_math                                # single crate
cargo nextest run -p astrodyn_gravity -E 'test(verif)'            # gravity verification only
cargo nextest run -p astrodyn_runner --test tier3_sim_dyncomp_run2  # single Tier 3 test
```

Plain `cargo test` also works but runs tests serially per binary:

```bash
cargo test --workspace                          # all tests
cargo test --workspace -- --skip tier3_         # unit + tier 2
```

All three test tiers (`cargo nextest run --workspace`) run without
`$JEOD_HOME` set — `run_verification/sim_*.rs` reads everything from
the committed mirror under `crates/astrodyn_verif_jeod/test_data/jeod_inputs/` plus the parsed
gravity binaries under `crates/astrodyn_gravity/test_data/gravity/`. The regen binaries
(`extract_*`) and the Docker reference-CSV flow are the only paths
that still need `$JEOD_HOME`. `TRICK_HOME` follows the standard Trick
environment convention and is required only by the Docker
reference-CSV regen flow.

**Before every commit**, run the same checks CI runs:

```bash
cargo fmt --check && cargo clippy --workspace --tests -- -D warnings
```

Fix any issues before committing. This avoids lint-only CI failures.

### Cross-validation tolerances

`CrossvalReport` (`crates/astrodyn_verif_jeod/src/crossval.rs`) computes per-component
max errors between our trajectory and JEOD's. It has no tolerance fields — tolerances
live exclusively in the test source code.

Tests assert tolerances via `report.assert_position(tol)`, `report.assert_velocity(tol)`,
`report.assert_quat_angle(tol)`, `report.assert_ang_vel(tol)` (per-component checks),
plus `assert!(var < tol, "metric_name")` for extras added via
`report.add_extra(name, val, unit)`.

The report binary (`cargo run -p astrodyn_verif_jeod --bin tier3_report`) extracts all
tolerance values from test source files by regex-parsing the `assert_*` call sites
and `assert!(var < LITERAL, "name")` patterns. JSON contains only errors — no
tolerances.

**Tolerance policy:** each tolerance is set to 5% above the observed max error, per
component. Since JEOD reference CSVs are static and our code is deterministic, errors
are fixed numbers — no runtime-computed or conditional tolerances.

When tightening tolerances after a code improvement: run the full test suite, inspect
the JSON reports in `target/tier3_crossval/`, compute `error * 1.05` per component,
and update the literal values in the test source.

See `crates/astrodyn_bevy/tests/README.md` "Baseline-freeze workflow" for the `crates/astrodyn_verif_jeod/test_data/baselines.json`
gating policy, the `tier3_baseline_diff` check, and the refreeze workflow.

### Test tiers and CI

All Tier 3 test functions use the `tier3_` prefix, enabling cargo's name-based
filtering. CI (`.github/workflows/ci.yml`) uses this:

- **PRs**: `check` (fmt + clippy), `test` (unit + tier 2 + cheap parity),
  `test-parity-trajectory` (heavy `bevy_parity_*` trajectory tests),
  and `test-tier3` (tier 3 excluding `earth_moon`) run in parallel for
  fast feedback.
- **Main push**: same jobs, plus `test-tier3-full` which includes the
  `earth_moon` test (~17 min) and generates the cross-validation report.
- **Push to non-main branches**: no CI (only PRs and main trigger workflows).

When adding new Tier 3 tests, always prefix the function name with `tier3_` so
CI filtering picks it up automatically.

The `test` and `test-parity-trajectory` jobs split the `bevy_parity_*`
suite by cost: cheap mechanism-only parity tests (mass attach/detach,
kinematic propagation, body action lifecycle, …) stay in `test`, while
the trajectory tests that propagate 8 h–23 day simulations twice
(`bevy_parity_dyncomp_run{2,4,5a,7}*`, `bevy_parity_solar_beta_*`,
`bevy_parity_srp_1st_order_trajectory`, `bevy_parity_tide_verif_run01`)
run in their own job. When adding a new heavy parity trajectory test,
extend the regex in **both** the `test` (exclude) and
`test-parity-trajectory` (include) filters in
`.github/workflows/ci.yml`, or it will silently regress the
`test` job's wall time.

See `crates/astrodyn_bevy/tests/README.md` for tier conventions and the tolerance/baseline workflow.

## Generating Tier 3 Reference Data (Docker)

JEOD verification sims run inside a Rocky 9 container with Trick 25 + JEOD 5.4.
Trick is cloned at `../trick`, JEOD at `../jeod`. See
the [Tier3-Regeneration wiki page](https://github.com/simnaut/astrodyn/wiki/Tier3-Regeneration)
for the full workflow, troubleshooting, and "adding a new sim" recipe.

The canonical wrapper is the `xtask` binary (requires the
`.cargo/config.toml.example` alias copied into `.cargo/config.toml`):

```bash
cargo xtask regenerate-tier3            # incremental — skips existing CSVs
cargo xtask regenerate-tier3 --force    # regenerate everything
cargo xtask regenerate-tier3 --build    # force rebuild jeod-trick first
```

For environments without cargo (or for explicit reference), the equivalent
direct Docker invocation is:

```bash
# Build container (context is parent dir so trick/ and jeod/ are accessible)
docker build -f trick/Dockerfile -t jeod-trick ..

# Generate reference CSVs into test_data/ (incremental — skips existing outputs)
mkdir -p test_data
docker run --rm \
  -v $(pwd)/crates/astrodyn_verif_jeod/test_data:/output \
  -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro \
  jeod-trick

# Force regenerate all data (ignores existing outputs)
docker run --rm -e FORCE=1 \
  -v $(pwd)/crates/astrodyn_verif_jeod/test_data:/output \
  -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro \
  jeod-trick
```

The generation script is **incremental by default**: it checks for existing
`${label}_*.csv` files in the output directory and skips any sim whose data is
already present. This avoids expensive `trick-CP` builds and sim runs when adding
new sims to `generate_references.sh`. Set `FORCE=1` to regenerate everything.

The container runs sims from the SIM root directory (not from SET_test/RUN_*/) because
JEOD's `input.py` files use paths relative to the SIM root. Output CSVs land in
`test_data/` and are consumed by `crates/astrodyn_verif_jeod/tests/tier3_sim_*.rs`.

**Current results (Phase 1):** 0.4 m position error over 8 hours vs JEOD SIM_dyncomp
RUN_2 (ISS orbit, spherical gravity, 28800s, 481 data points at 60s intervals).

**Phase 2 Tier 3 tests** (require reference CSVs from Docker):
- RUN_3A: 4x4 spherical harmonics gravity, 8-hour ISS orbit
- RUN_3B: 8x8 spherical harmonics gravity, 8-hour ISS orbit
- Test: `crates/astrodyn_verif_jeod/tests/tier3_sim_dyncomp_run3.rs`

CSV column layout for `log_state_ASCII.csv`:
- Column 0: `sys.exec.out.time {s}`
- Columns 1,8,15: `composite_body.state.trans.position[0,1,2] {m}`
- Columns 2,9,16: `composite_body.state.trans.velocity[0,1,2] {m/s}`
- (interleaved with rotation matrix, quaternion, and angular velocity columns)

CSV and `.bsp` test data files are committed to the repository. Only binary `.trk` files
(Trick's native log format) are gitignored. Tests assert (panic) when required data is
absent — they never skip gracefully. The assert message includes the exact command to
obtain the data.

## JEOD Invariant Tracking (non-negotiable)

JEOD's C++ architecture enforces many invariants via `MessageHandler::fail()`,
`error()`, constructor logic, and structural guarantees (value members, deleted
copy ctors). We catalog every invariant we encounter in
`docs/JEOD_invariants.md` and trace enforcement sites in source with
`// JEOD_INV: XX.YY` comments.

### How it works

1. **Catalog** (`docs/JEOD_invariants.md`): one row per invariant with a
   `Section.Tag` ID (e.g., `GV.04`), description, JEOD enforcement mechanism,
   and our status (`enforced`, `partial`, `deferred`, `n/a`, `structural`).

2. **Source tags**: every enforcement site in our Rust code has a comment like
   `// JEOD_INV: GV.04 — degree <= source degree`. The tag text should
   accurately describe what the code does and note any divergence from JEOD.

3. **CI coverage** (`crates/astrodyn_bevy/tests/invariant_coverage.rs`): bidirectional test —
   every `enforced`/`partial`/`structural` invariant in the catalog must have
   at least one source tag, and every source tag must reference a catalog entry.

### When you encounter an unrecorded invariant

When reading JEOD source and you find a `MessageHandler::fail()`,
`MessageHandler::error()`, assert, or structural guarantee that is not
already in the catalog:

1. **Add a row** to `docs/JEOD_invariants.md` with the next available tag in
   the appropriate section (e.g., `DB.28`, `GV.19`).
2. **Add `// JEOD_INV: XX.YY`** at our enforcement site (or note `deferred`/
   `n/a` in the catalog if we don't enforce it yet).
3. **Run** `cargo test --test invariant_coverage` to verify consistency.

### When you encounter an untagged enforcement site

If our code enforces a JEOD invariant but the enforcement site lacks a
`// JEOD_INV` tag, add the tag. If the invariant isn't in the catalog yet,
add it there too.

### Tag accuracy

Tags must describe what the code **actually does**, not what JEOD does. When
our implementation diverges from JEOD (e.g., we divide by mass at runtime
instead of precomputing `inverse_mass`), the tag should note the divergence.
Never copy JEOD's description verbatim if our code works differently.

## JEOD Convention Rule

When JEOD uses a field name whose meaning could be ambiguous (e.g., `time_periapsis`
— is it time *to* periapsis or time *since* periapsis?), **always read the JEOD C++
source** to determine the convention. Do not guess or reason by analogy. A wrong guess
about a sign or direction convention produces code that compiles, passes trivial tests,
and silently gives wrong answers at scale.

This rule was established after an agent guessed the `time_periapsis` → mean anomaly
formula as `M = 2π - n·t` instead of the correct `M = n·t`, producing 11,668 km error
against NASA flight data. The bug was hidden for multiple commits because a broken
`jeod_path()` caused the validation test to silently skip. Reading
`models/dynamics/body_action/src/dyn_body_init_orbit.cc` would have given the correct
formula immediately.

## Building a Mission Crate

A "mission crate" is a downstream crate that depends on `astrodyn_bevy` to model a
specific scenario (an Earth-orbit constellation, a Mars approach, a station-
keeping study). After the type-system refactor (#101), mission code reads like
physics: typed building blocks compose via the typestate `VehicleBuilder`,
units flow through the `F64Ext` facade, and the compiler rejects frame/unit
mismatches before they become silent numerical bugs.

**Imports**: a mission crate needs only the prelude and the recipes module.

```rust
use bevy::prelude::*;
use astrodyn_bevy::prelude::*;        // AstrodynPlugin, typed Components, AstrodynSet
use astrodyn_bevy::recipes::*;        // earth, orbital_elements, vehicle, scenarios
```

**Compose a vehicle** with the typestate `VehicleBuilder` (re-exported by
`astrodyn_bevy::prelude`). The compiler refuses `.three_dof_point_mass(...)`
until a state is set, refuses `.rk4()` until mass is set, refuses `.build()`
until an integrator is chosen.

```rust
// `VehicleBuilder`, `GravityControl`, `GravityRole`, and `F64Ext` come from
// `astrodyn_bevy::prelude`. `earth`, `orbital_elements`, `vehicle` come from
// `astrodyn_bevy::recipes`.
let mu = earth::point_mass().source.mu.m3_per_s2();
let cfg = VehicleBuilder::new()
    .from_orbital_elements(orbital_elements::iss(), mu)
    .three_dof_point_mass(vehicle::iss_mass())
    .rk4()
    .gravity(GravityControl::new_spherical(0_usize, GravityRole::Central))
    .build();
let vehicle_entity = cfg.spawn_bevy::<astrodyn::Earth>(&mut commands, &[earth_entity]);
```

**Compiler errors as physics**: passing a `Position<Ecef>` where
`Position<RootInertial>` is required produces a custom diagnostic in physics
language pointing to the missing `FrameTransform<Ecef, RootInertial>` step, not a
PhantomData type-mismatch wall.

**Reference**:
- Canonical worked example: `examples/typed_mission.rs`.
- Contributor primer (phantom tags, adding new dimensions, escape hatches):
  [Type-System wiki page](https://github.com/simnaut/astrodyn/wiki/Type-System).
- Architecture and phase history:
  [Strategy wiki page](https://github.com/simnaut/astrodyn/wiki/Strategy)
  §8 "Phase 8: Type-System Refactor".

## Common Pitfalls

- **Trick sim working directory**: JEOD sims must be run from the SIM root directory
  (e.g., `verif/SIM_dyncomp/`), not from `SET_test/RUN_*/`. The `input.py` files use
  paths like `SET_test/common_input.py` and `Log_data/log_suite.py` relative to the SIM
  root. Running from the wrong directory produces no data output.
- JEOD's `RefFrameState` stores position/velocity **relative to parent frame**, not global.
  Don't confuse with absolute/inertial coordinates.
- JEOD uses **left-transformation** quaternions (`r' = q r q*`). Many references use
  right-transformation. Getting this wrong produces the transpose rotation.
- Gravity acceleration in `GravityInteraction` **excludes** the acceleration of the
  integration frame itself toward the planet. For Earth-centered inertial integration,
  the Sun's contribution is the differential acceleration (vehicle toward Sun minus
  Earth toward Sun), not the absolute acceleration toward the Sun.
- `MassProperties.inertia` is about the body frame axes through the center of mass.
  When composing masses, use the parallel axis theorem (Steiner's theorem) for the
  offset contribution.
- JEOD's `DynBody` has three reference frames: `structure` (geometric origin),
  `composite_body` (composite CoM), `core_body` (this body's CoM only).
  State is integrated in one of these, then propagated to the others.
- **Trick DRAscii silently drops unregistered variables**: When injecting ASCII logging
  snippets in `generate_references.sh`, variable names must match the S_define's object
  names exactly. If a variable doesn't exist in the sim, Trick silently omits it from
  the CSV — producing fewer columns than expected with no error message. Always verify
  the S_define (e.g., SIM_2A_SHADOW_CALC uses `radiation_simple`, not `radiation`).
- **Geodetic longitude at the poles**: At latitude ±90°, longitude is geometrically
  undefined (all meridians converge). `atan2(y, x)` becomes hypersensitive to position
  errors: at 89.8° latitude, ~3.7e-6 rad/m sensitivity. Polar orbit NED tests have
  larger longitude tolerances (~3.3e-5 rad) than inclined orbit tests (~6.5e-8 rad).
  This is not a code bug — both JEOD and our code produce valid but numerically
  unstable values.
