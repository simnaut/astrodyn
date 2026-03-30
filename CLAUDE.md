# CLAUDE.md

## Project

Rust reimplementation of [NASA JEOD](https://github.com/nasa/jeod) (JSC Engineering
Orbital Dynamics, v5.4, 714 C++ source files) using Bevy ECS instead of NASA's Trick.
See [STRATEGY.md](STRATEGY.md) for architecture and [PLAN.md](PLAN.md) for tasking.

Copy `.cargo/config.toml.example` to `.cargo/config.toml` and set `JEOD_HOME`
and `TRICK_HOME` to your local checkouts. Cargo resolves `relative = true`
paths from the workspace root.

## Three-Layer Architecture (non-negotiable)

All physics lives in **`jeod_*`** crates (pure Rust, zero Bevy dependency).
Orchestration lives in **`jeod_sim`** (composes `jeod_*` functions into pipeline
stages, re-exports all types; zero Bevy dependency). Bevy wiring lives in
**`bevy_jeod_*`** crates (thin glue: component derives, systems that delegate to
`jeod_sim` functions, plugin registration).

`bevy_jeod_*` crates depend **only** on `jeod_sim` + `bevy` — never on `jeod_*`
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

The three verification tiers:
- **Tier 1**: Unit tests — pure function correctness, round-trips, convergence
- **Tier 2**: JEOD reference data — static test vectors from JEOD source files
- **Tier 3**: Trajectory cross-validation — propagate from same initial conditions,
  compare against JEOD Trick simulation output over hours/days

## Precision

Use `f64` everywhere. Do NOT use Bevy's `Transform`/`GlobalTransform` (f32).
Use `glam::DVec3`, `glam::DQuat`, `glam::DMat3` for 3D types.
Use `nalgebra` only for variable-size matrices (spherical harmonics coefficients).

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

```bash
cargo build --workspace
cargo test --workspace                          # all tests (needs JEOD_HOME or JEOD_PATH)
cargo test --workspace -- --skip tier3_         # fast subset: unit + tier 2 (skip trajectory)
cargo test --workspace -- tier3_               # tier 3 only: trajectory cross-validation
JEOD_HOME=../jeod cargo test                    # explicit path
cargo test -p jeod_math                         # single crate
cargo test -p jeod_gravity -- verif             # gravity verification tests only
cargo test -p jeod_dynamics --test tier3_jeod_trajectory  # single Tier 3 test
```

Set `JEOD_HOME` (or `JEOD_PATH`) to the JEOD source checkout.
`JEOD_HOME` and `TRICK_HOME` follow the standard JEOD/Trick environment
variable conventions.

### Test tiers and CI

All Tier 3 test functions use the `tier3_` prefix, enabling cargo's name-based
filtering. CI (`.github/workflows/ci.yml`) uses this:

- **PRs**: `check` (fmt + clippy) and `test` (unit + tier 2, `--skip tier3_`)
  run in parallel for fast feedback.
- **Main push**: same jobs, plus `test-tier3` which runs only `tier3_` tests.
- **Push to non-main branches**: no CI (only PRs and main trigger workflows).

When adding new Tier 3 tests, always prefix the function name with `tier3_` so
CI filtering picks it up automatically.

## Generating Tier 3 Reference Data (Docker)

JEOD verification sims run inside a Rocky 9 container with Trick 25 + JEOD 5.4.
Trick is cloned at `../trick`, JEOD at `../jeod`.

```bash
# Build container (context is parent dir so trick/ and jeod/ are accessible)
docker build -f trick/Dockerfile -t jeod-trick ..

# Generate reference CSVs into test_data/ (incremental — skips existing outputs)
mkdir -p test_data
docker run --rm -v $(pwd)/test_data:/output jeod-trick

# Force regenerate all data (ignores existing outputs)
docker run --rm -e FORCE=1 -v $(pwd)/test_data:/output jeod-trick
```

The generation script is **incremental by default**: it checks for existing
`${label}_*.csv` files in the output directory and skips any sim whose data is
already present. This avoids expensive `trick-CP` builds and sim runs when adding
new sims to `generate_references.sh`. Set `FORCE=1` to regenerate everything.

The container runs sims from the SIM root directory (not from SET_test/RUN_*/) because
JEOD's `input.py` files use paths relative to the SIM root. Output CSVs land in
`test_data/` and are consumed by `tier3_jeod_trajectory.rs`.

**Current results (Phase 1):** 0.4 m position error over 8 hours vs JEOD SIM_dyncomp
RUN_2 (ISS orbit, spherical gravity, 28800s, 481 data points at 60s intervals).

**Phase 2 Tier 3 tests** (require reference CSVs from Docker):
- RUN_3A: 4x4 spherical harmonics gravity, 8-hour ISS orbit
- RUN_3B: 8x8 spherical harmonics gravity, 8-hour ISS orbit
- Test: `crates/jeod_gravity/tests/tier3_spherical_harmonics.rs`

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
