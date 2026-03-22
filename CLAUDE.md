# CLAUDE.md

## Project

Rust reimplementation of [NASA JEOD](https://github.com/nasa/jeod) (JSC Engineering
Orbital Dynamics, v5.4, 714 C++ source files) using Bevy ECS instead of NASA's Trick.
See [STRATEGY.md](STRATEGY.md) for architecture and [PLAN.md](PLAN.md) for tasking.

JEOD source is at `../jeod`. Set `JEOD_PATH` env var to override.

## Two-Layer Architecture (non-negotiable)

All physics lives in **`jeod_*`** crates (pure Rust, zero Bevy dependency).
Bevy wiring lives in **`bevy_jeod_*`** crates (thin glue: component derives, systems
that delegate to `jeod_*` pure functions, plugin registration).

Never put physics algorithms directly in a Bevy system function. The system queries
components, then calls a `jeod_*` function. This keeps physics portable to other ECS
frameworks, WASM, or standalone batch computation.

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
cargo test --workspace                  # all tests (needs JEOD_PATH for Tier 2+)
JEOD_PATH=../jeod cargo test            # explicit path
cargo test -p jeod_math                 # single crate
cargo test -p jeod_gravity -- verif     # gravity verification tests only
cargo test -p jeod_dynamics --test tier3_jeod_trajectory  # Tier 3 (needs test_data/)
```

## Generating Tier 3 Reference Data (Docker)

JEOD verification sims run inside a Rocky 9 container with Trick 25 + JEOD 5.4.
Trick is cloned at `../trick`, JEOD at `../jeod`.

```bash
# Build container (context is parent dir so trick/ and jeod/ are accessible)
docker build -f trick/Dockerfile -t jeod-trick ..

# Generate reference CSVs into test_data/
mkdir -p test_data
docker run --rm -v $(pwd)/test_data:/output jeod-trick
```

The container runs sims from the SIM root directory (not from SET_test/RUN_*/) because
JEOD's `input.py` files use paths relative to the SIM root. Output CSVs land in
`test_data/` and are consumed by `tier3_jeod_trajectory.rs`.

**Current results (Phase 1):** 0.4 m position error over 8 hours vs JEOD SIM_dyncomp
RUN_2 (ISS orbit, spherical gravity, 28800s, 481 data points at 60s intervals).

CSV column layout for `log_state_ASCII.csv`:
- Column 0: `sys.exec.out.time {s}`
- Columns 1,8,15: `composite_body.state.trans.position[0,1,2] {m}`
- Columns 2,9,16: `composite_body.state.trans.velocity[0,1,2] {m/s}`
- (interleaved with rotation matrix, quaternion, and angular velocity columns)

Test data files are gitignored. Tests skip gracefully when `test_data/` is absent.

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
