# CLAUDE.md

## Project

Rust reimplementation of [NASA JEOD](https://github.com/nasa/jeod) (JSC Engineering
Orbital Dynamics, v5.4, 714 C++ source files) using Bevy ECS instead of NASA's Trick.
See the [Strategy wiki page](https://github.com/simnaut/bevy_jeod/wiki/Strategy)
for architecture and phase summaries. The original phased implementation plan
(Phases 1–7) closed in April 2026; ongoing work is tracked as GitHub issues.

### Environment Setup

`cargo build --workspace && cargo nextest run --workspace -E 'not test(tier3_)'`
works on a fresh clone of this repo with no JEOD checkout — the unit and
Tier 2 tests read from committed fixtures under `test_data/`. CI's "Test
(unit + tier 2)" job verifies this with a regression fence that asserts
`JEOD_HOME` and `JEOD_PATH` are both unset before running.

**Tier 3 trajectory tests still need JEOD source.** The
`run_verification/sim_*.rs` rigs that build Tier 3 verification cases
have many `jeod_root()` callers (Moon GRAIL150 SH, S_define `#define
DYNAMICS` parsing, ISS / STS-114 mass.py loading, etc.). Migrating each
of those to a committed fixture is a sizeable follow-on; for now,
running Tier 3 tests requires `$JEOD_HOME` to point at a NASA JEOD
checkout. CI's Tier 3 jobs sparse-checkout the relevant subtrees and
set `JEOD_HOME` accordingly.

You also need `$JEOD_HOME` when:

1. Regenerating fixtures after a JEOD upgrade — invoked through the
   `extract_*` binaries under `crates/jeod_test_data/src/bin/` (e.g.
   `cargo run -p jeod_test_data --bin extract_grav_coeffs`).
2. Building / running the verification rigs in
   `crates/jeod_runner/src/run_verification/sim_*.rs`, which are gated
   behind the default-on `verification` cargo feature on `jeod_runner`.
   Production library consumers can opt out with `--no-default-features`
   to drop the JEOD-source dependency entirely.

When you do need it:

```bash
cd /home/user/git   # or wherever your repos live
git clone https://github.com/nasa/jeod.git
git clone https://github.com/nasa/trick.git

export JEOD_HOME=$(pwd)/jeod
export TRICK_HOME=$(pwd)/trick
```

`JEOD_HOME` is the standard NASA convention; the older `JEOD_PATH` alias
was retired in #239. The Trick container path (`$TRICK_HOME`) is unaffected
and still required for the Docker reference-CSV regen flow.

## Three-Layer Architecture (non-negotiable)

All physics lives in **`jeod_*`** crates (pure Rust, zero Bevy dependency).
Orchestration lives in **`jeod_sim`** (composes `jeod_*` functions into pipeline
stages, re-exports all types; zero Bevy dependency). Bevy wiring lives in the
**`bevy_jeod`** root package (`src/` — thin glue: component derives, systems
that delegate to `jeod_sim` functions, plugin registration).

The root package depends **only** on `jeod_sim` + `bevy` — never on `jeod_*`
crates directly. This makes `jeod_sim` the single API surface for any ECS adapter.

Never put physics algorithms directly in a Bevy system function. The system queries
components, then calls a `jeod_sim` function. This keeps physics portable to other
ECS frameworks, WASM, or standalone batch computation.

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

- **Public/mission-crate code** uses typed quantities from `jeod_quantities`:
  `Position<F: Frame>`, `Velocity<F>`, `Acceleration<F>`, `SecondsSince<S: TimeScale>`,
  `Quat<L, T>`, `NormalizedQuat`, `FrameTransform<From, To>`, and the `F64Ext`
  facade (`400.0.km()`, `51.6.deg()`, `420_000.0.kg()`). Mission code never sees
  `DVec3`/`DQuat`/`DMat3` or `PhantomData`. The compiler rejects cross-frame
  mismatches, scalar-vs-vector quaternion confusion, and unit-dimensional errors
  at compile time. Custom `#[diagnostic::on_unimplemented]` messages render
  errors in physics language (e.g., *"expected `Position<Inertial>`, found
  `Position<Ecef>` — apply a `FrameTransform<Ecef, Inertial>` first"*).

- **Internal physics-crate kernels** (the inside of `jeod_*` `*_typed` functions
  and the underlying `_inner`/`_impl` math) use raw `glam::DVec3`/`DQuat`/`DMat3`
  for arithmetic density. The typed siblings call `.raw_si()` at the boundary
  to drop into the kernel and re-wrap on exit. This keeps numerics fast and the
  public surface typed.

See the [Type-System wiki page](https://github.com/simnaut/bevy_jeod/wiki/Type-System) for the contributor primer (phantom-tag pattern,
adding a new frame/scale/quantity, reading compiler errors, escape hatches)
and `examples/typed_mission.rs` for the canonical worked example.

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

```
1. Time update         →  TimeUpdateSet
2. Ephemeris update    →  EphemerisUpdateSet
3. Gravity computation →  EnvironmentSet
4. Atmosphere update   →  EnvironmentSet
5. Aero/SRP/torque     →  InteractionSet
6. Force collection    →  ForceCollectionSet
7. State integration   →  IntegrationSet
8. Frame propagation   →  IntegrationSet
9. Derived states      →  DerivedStateSet
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
cargo nextest run --workspace -E 'not test(tier3_)'           # unit + tier 2 (fast)
cargo nextest run --workspace -E 'test(tier3_)'               # tier 3 only
cargo nextest run -p jeod_math                                # single crate
cargo nextest run -p jeod_gravity -E 'test(verif)'            # gravity verification only
cargo nextest run -p jeod_runner --test tier3_sim_dyncomp_run2  # single Tier 3 test
```

Plain `cargo test` also works but runs tests serially per binary:

```bash
cargo test --workspace                          # all tests
cargo test --workspace -- --skip tier3_         # unit + tier 2
```

The full test suite runs without `$JEOD_HOME` set; only the regen
binaries (`extract_*`) and the verification rigs need it. `TRICK_HOME`
follows the standard Trick environment convention and is required by
the Docker reference-CSV regen flow.

**Before every commit**, run the same checks CI runs:

```bash
cargo fmt --check && cargo clippy --workspace --tests -- -D warnings
```

Fix any issues before committing. This avoids lint-only CI failures.

### Cross-validation tolerances

`CrossvalReport` (`crates/jeod_test_data/src/crossval.rs`) computes per-component
max errors between our trajectory and JEOD's. It has no tolerance fields — tolerances
live exclusively in the test source code.

Tests assert tolerances via `report.assert_position(tol)`, `report.assert_velocity(tol)`,
`report.assert_quat_angle(tol)`, `report.assert_ang_vel(tol)` (per-component checks),
plus `assert!(var < tol, "metric_name")` for extras added via
`report.add_extra(name, val, unit)`.

The report binary (`cargo run -p jeod_test_data --bin tier3_report`) extracts all
tolerance values from test source files by regex-parsing the `assert_*` call sites
and `assert!(var < LITERAL, "name")` patterns. JSON contains only errors — no
tolerances.

**Tolerance policy:** each tolerance is set to 5% above the observed max error, per
component. Since JEOD reference CSVs are static and our code is deterministic, errors
are fixed numbers — no runtime-computed or conditional tolerances.

When tightening tolerances after a code improvement: run the full test suite, inspect
the JSON reports in `target/tier3_crossval/`, compute `error * 1.05` per component,
and update the literal values in the test source.

See `tests/README.md` "Baseline-freeze workflow" for the `test_data/baselines.json`
gating policy, the `tier3_baseline_diff` check, and the refreeze workflow.

### Test tiers and CI

All Tier 3 test functions use the `tier3_` prefix, enabling cargo's name-based
filtering. CI (`.github/workflows/ci.yml`) uses this:

- **PRs**: `check` (fmt + clippy), `test` (unit + tier 2), and `test-tier3`
  (tier 3 excluding `earth_moon`) run in parallel for fast feedback.
- **Main push**: same jobs, plus `test-tier3-full` which includes the
  `earth_moon` test (~17 min) and generates the cross-validation report.
- **Push to non-main branches**: no CI (only PRs and main trigger workflows).

When adding new Tier 3 tests, always prefix the function name with `tier3_` so
CI filtering picks it up automatically.

See `tests/README.md` for tier conventions and the tolerance/baseline workflow.

## Generating Tier 3 Reference Data (Docker)

JEOD verification sims run inside a Rocky 9 container with Trick 25 + JEOD 5.4.
Trick is cloned at `../trick`, JEOD at `../jeod`. See
the [Tier3-Regeneration wiki page](https://github.com/simnaut/bevy_jeod/wiki/Tier3-Regeneration)
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
  -v $(pwd)/test_data:/output \
  -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro \
  jeod-trick

# Force regenerate all data (ignores existing outputs)
docker run --rm -e FORCE=1 \
  -v $(pwd)/test_data:/output \
  -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro \
  jeod-trick
```

The generation script is **incremental by default**: it checks for existing
`${label}_*.csv` files in the output directory and skips any sim whose data is
already present. This avoids expensive `trick-CP` builds and sim runs when adding
new sims to `generate_references.sh`. Set `FORCE=1` to regenerate everything.

The container runs sims from the SIM root directory (not from SET_test/RUN_*/) because
JEOD's `input.py` files use paths relative to the SIM root. Output CSVs land in
`test_data/` and are consumed by `crates/jeod_runner/tests/tier3_sim_*.rs`.

**Current results (Phase 1):** 0.4 m position error over 8 hours vs JEOD SIM_dyncomp
RUN_2 (ISS orbit, spherical gravity, 28800s, 481 data points at 60s intervals).

**Phase 2 Tier 3 tests** (require reference CSVs from Docker):
- RUN_3A: 4x4 spherical harmonics gravity, 8-hour ISS orbit
- RUN_3B: 8x8 spherical harmonics gravity, 8-hour ISS orbit
- Test: `crates/jeod_runner/tests/tier3_sim_dyncomp_run3.rs`

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

3. **CI coverage** (`tests/invariant_coverage.rs`): bidirectional test —
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

A "mission crate" is a downstream crate that depends on `bevy_jeod` to model a
specific scenario (an Earth-orbit constellation, a Mars approach, a station-
keeping study). After the type-system refactor (#101), mission code reads like
physics: typed building blocks compose via the typestate `VehicleBuilder`,
units flow through the `F64Ext` facade, and the compiler rejects frame/unit
mismatches before they become silent numerical bugs.

**Imports**: a mission crate needs only the prelude and the recipes module.

```rust
use bevy::prelude::*;
use bevy_jeod::prelude::*;        // JeodPlugin, typed Components, JeodSet
use bevy_jeod::recipes::*;        // earth, orbital_elements, vehicle, scenarios
```

**Compose a vehicle** with the typestate `VehicleBuilder` (re-exported by
`bevy_jeod::prelude`). The compiler refuses `.three_dof_point_mass(...)`
until a state is set, refuses `.rk4()` until mass is set, refuses `.build()`
until an integrator is chosen.

```rust
// `VehicleBuilder`, `GravityControl`, and `F64Ext` come from `bevy_jeod::prelude`.
// `earth`, `orbital_elements`, `vehicle` come from `bevy_jeod::recipes`.
let mu = earth::point_mass().source.mu.m3_per_s2();
let cfg = VehicleBuilder::new()
    .from_orbital_elements(orbital_elements::iss(), mu)
    .three_dof_point_mass(vehicle::iss_mass())
    .rk4()
    .gravity(GravityControl::new_spherical(0_usize, false))
    .build();
let vehicle_entity = cfg.spawn_bevy(&mut commands, &[earth_entity]);
```

**Compiler errors as physics**: passing a `Position<Ecef>` where
`Position<Inertial>` is required produces a custom diagnostic in physics
language pointing to the missing `FrameTransform<Ecef, Inertial>` step, not a
PhantomData type-mismatch wall.

**Reference**:
- Canonical worked example: `examples/typed_mission.rs`.
- Contributor primer (phantom tags, adding new dimensions, escape hatches):
  [Type-System wiki page](https://github.com/simnaut/bevy_jeod/wiki/Type-System).
- Architecture and phase history:
  [Strategy wiki page](https://github.com/simnaut/bevy_jeod/wiki/Strategy)
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
