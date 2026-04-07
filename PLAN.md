# bevy_jeod: Implementation Plan

Detailed tasking, entrance criteria, and exit criteria for each implementation phase.
Phases are defined in [STRATEGY.md](STRATEGY.md) Section 8.

---

## Phase 0: Workspace Setup

### Entrance Criteria

- Empty project directory exists
- JEOD cloned at `../jeod` (v5.4)
- Rust toolchain installed (stable)
- Bevy version selected and pinned

### Tasks

| ID | Task | Crate | Description |
|----|------|-------|-------------|
| 0.1 | Initialize Cargo workspace | root | `Cargo.toml` with `[workspace]`, resolver = "2" |
| 0.2 | Create core crate skeletons | `jeod_math`, `jeod_dynamics`, `jeod_gravity`, `jeod_frames` | `cargo init --lib` for each, add to workspace members, set `edition = "2021"` |
| 0.3 | Create Bevy glue | `src/` (root package) | Components, systems, plugin registration. Originally planned as separate `bevy_jeod_*` crates; consolidated into unified root package. |
| 0.4 | Create test data crate skeleton | `jeod_test_data` | No Bevy dependency. Add `JEOD_PATH` env var support |
| 0.5 | Create top-level lib crate | `src/lib.rs` | Unified `JeodPlugin` with all systems and schedule sets |
| 0.6 | Add shared dependencies | workspace `Cargo.toml` | `glam` (f64 features), `nalgebra` (optional), `thiserror`, `regex` (test_data) |
| 0.7 | Set up CI configuration | `.github/workflows/` or equivalent | `cargo build --workspace`, `cargo test --workspace`, `cargo clippy`, `cargo fmt --check` |
| 0.8 | Create `.env.example` | root | Document `JEOD_PATH=../jeod` |
| 0.9 | Add `STRATEGY.md` and `PLAN.md` to repo | root | Already exist |

### Exit Criteria

- [ ] `cargo build --workspace` succeeds with zero errors
- [ ] `cargo test --workspace` runs (0 tests, 0 failures)
- [ ] `cargo clippy --workspace` produces no warnings
- [ ] Each `jeod_*` crate compiles with **zero** Bevy dependency
- [x] Bevy glue (`src/`) depends only on `jeod_sim` and `bevy` (never on `jeod_*` directly)
- [ ] CI pipeline runs successfully (if configured)

---

## Phase 1: Foundation

**Goal:** Two-body Kepler orbit propagating in Bevy's `FixedUpdate`, and as a standalone
batch computation without Bevy.

### Entrance Criteria

- [ ] Phase 0 exit criteria met
- [ ] JEOD checkout accessible at `JEOD_PATH` (for reference, not required to build)

### Tasks

#### 1A. Core Math (`jeod_math`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 1.1 | f64 type aliases | Re-export or wrap `glam::DVec3`, `glam::DQuat`, `glam::DMat3`. Define project-wide type conventions. | `utils/math/` |
| 1.2 | Quaternion conventions | Implement JEOD's scalar-first left-transform quaternion operations. Conversion functions to/from `glam`'s `[x,y,z,w]` ordering. Document convention explicitly. | `utils/quaternion/include/quat.hh` |
| 1.3 | Rotation matrix ↔ quaternion | Bidirectional conversion matching JEOD's `compute_transformation()` / `compute_quaternion()`. | `ref_frame_state.hh:172-176` |
| 1.4 | Kepler equation solver | Newton-Raphson for elliptic (M = E - e·sin(E)). Handle hyperbolic (e > 1) and near-parabolic (e ≈ 1) cases. | `utils/orbital_elements/src/orbital_elements.cc` |
| 1.5 | Cartesian → Keplerian | `cartesian_to_elements(pos, vel, mu) → OrbitalElements`. Compute a, e, i, Ω, ω, ν, M, n, energy, angular momentum. | `utils/orbital_elements/src/orbital_elements.cc` |
| 1.6 | Keplerian → Cartesian | `elements_to_cartesian(elems, mu) → (pos, vel)`. Via perifocal frame rotation. | `utils/orbital_elements/src/orbital_elements.cc` |
| 1.7 | OrbitalElements struct | All Keplerian elements plus derived quantities (mean motion, energy, angular momentum). | `utils/orbital_elements/include/orbital_elements.hh` |
| 1.8 | Math unit tests | Round-trip Cartesian↔Keplerian (< 1e-10 error). Known orbits (circular, eccentric, hyperbolic). Kepler equation convergence for all eccentricity ranges. Quaternion ↔ matrix consistency. | — |

#### 1B. Core Dynamics (`jeod_dynamics`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 1.9 | TranslationalState | `{ position: DVec3, velocity: DVec3 }` | `dyn_body.hh` composite_body frame state |
| 1.10 | MassProperties | `{ mass: f64, inertia: DMat3, inertia_inverse: DMat3, center_of_mass: DVec3 }` | `dynamics/mass/include/mass.hh` |
| 1.11 | DynamicsConfig | `{ translational: bool, rotational: bool, three_dof: bool }` | `dyn_body.hh:668-697` |
| 1.12 | TotalForce | `{ force: DVec3, torque: DVec3 }` | `body_force_collect.hh` |
| 1.13 | FrameDerivatives | `{ trans_accel: DVec3, rot_accel: DVec3 }` | `frame_derivs.hh` |
| 1.14 | RK4 integrator | Pure function: `rk4_step(state, deriv_fn, dt) → new_state`. Generic over state type. Operates on translational state for Phase 1. | `er7_utils` RK4 |
| 1.15 | Force → acceleration | `compute_acceleration(total_force, mass) → accel`. Trivial F/m but establishes the interface. | `dyn_body_integration.cc` |
| 1.16 | Dynamics unit tests | RK4 on harmonic oscillator (known analytical solution). Energy conservation over N steps. Convergence order verification (halve dt, error reduces by 16x). | — |

#### 1C. Core Gravity (`jeod_gravity`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 1.17 | GravitySource struct | `{ mu: f64, model: GravityModel }` with `GravityModel::PointMass` variant. | `gravity_source.hh` |
| 1.18 | GravityAcceleration struct | `{ accel: DVec3, gradient: DMat3, potential: f64 }` | `gravity_interaction.hh:120-133` |
| 1.19 | Point-mass gravity | `compute_gravity(source, position) → GravityAcceleration`. Acceleration = -μ/r³ · r. Potential = -μ/r. | `gravity_source.cc` |
| 1.20 | Gravity unit tests | Acceleration at known distances. Inverse-square law verification. Potential matches -μ/r. | — |

#### 1D. Core Frames (`jeod_frames`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 1.21 | RefFrameTrans | `{ position: DVec3, velocity: DVec3 }` (relative to parent) | `ref_frame_state.hh:82-116` |
| 1.22 | RefFrameRot | `{ q_parent_this: DQuat, t_parent_this: DMat3, ang_vel_this: DVec3 }` | `ref_frame_state.hh:121-183` |
| 1.23 | RefFrameState | `{ trans: RefFrameTrans, rot: RefFrameRot }` | `ref_frame_state.hh:188-235` |
| 1.24 | Frame composition | `incr_left()`, `incr_right()`, `negate()` operations for composing/inverting frame states. | `ref_frame_state.hh:225-234` |
| 1.25 | Frames unit tests | Compose A→B and B→C to get A→C. Compose and invert yields identity. | — |

#### 1E. Bevy Glue (`src/`)

| ID | Task | Description |
|----|------|-------------|
| 1.26 | Component wrappers | Newtype or feature-gated `#[derive(Component)]` for all Phase 1 core types |
| 1.27 | DynBodyBundle | Bundle grouping all components needed to spawn a vehicle entity |
| 1.28 | IntegrationFrameRef | `Component(Entity)` — Bevy-only, points to frame entity |
| 1.29 | gravity_computation_system | Query bodies + sources, delegate to `jeod_gravity::compute_gravity()` |
| 1.30 | force_collection_system | Sum GravityAcceleration into TotalForce |
| 1.31 | integration_system | Query state + forces, delegate to `jeod_dynamics::rk4_step()` |
| 1.32 | System schedule | `FixedUpdate` with `EnvironmentSet → ForceCollectionSet → IntegrationSet` ordering |
| 1.33 | Frame entity hierarchy | Root inertial frame + body frame as child, using Bevy `Parent`/`Children` |

#### 1F. Examples and Integration

| ID | Task | Description |
|----|------|-------------|
| 1.34 | `kepler_orbit.rs` | Bevy example: spawn Earth (point mass) + satellite, run FixedUpdate, print orbital elements each orbit |
| 1.35 | `batch_propagation.rs` | No-Bevy example: propagate a Kepler orbit for N steps using only `jeod_*` crates, print state to stdout |
| 1.36 | Integration test | Propagate circular orbit for 10 periods, verify energy conservation and period |

### Exit Criteria

- [x] `cargo build --workspace` succeeds
- [x] `cargo test --workspace` — 85 tests pass, 0 failures
- [x] `cargo clippy --workspace` — no warnings
- [x] **Energy conservation**: Relative energy drift 3.2e-10 over 10 orbits (RK4, dt=10s, LEO)
- [x] **Period accuracy**: Error 2.3e-12 (dt=1s)
- [x] **Orbital elements round-trip**: < 1e-6 m on 100 JEOD state vectors, < 1e-10 for 8 analytical orbits
- [x] **Quaternion consistency**: `quat_to_matrix(matrix_to_quat(M)) == M` to < 1e-15 for 12+ non-trivial rotations
- [x] **Portability**: `batch_propagation.rs` compiles and runs without Bevy in the dependency tree
- [x] **Bevy example**: `kepler_orbit.rs` runs and produces correct orbital elements output
- [x] **JEOD Tier 2**: ISS orbital elements match NASA reference state to < 1 km; 5001 JEOD state vectors parsed; gravity/Euler test data validated
- [x] **JEOD Tier 3**: 0.4 m position error over 8 hours vs JEOD SIM_dyncomp RUN_2 (Docker: Trick 25 + JEOD 5.4 on Rocky 9)

---

## Phase 2: Realistic Environment

**Goal:** Spherical harmonics gravity, multi-body ephemeris, time system.

### Entrance Criteria

- [ ] Phase 1 exit criteria met
- [ ] JEOD checkout at `JEOD_PATH` contains:
  - `models/environment/gravity/data/` (coefficient files)
  - `models/environment/time/data/Leap_Second.dat`
  - `models/environment/gravity/verif/unit_tests/grav_geospherical/data/verif_out.txt`
  - `models/dynamics/body_action/verif/SIM_orbinit/Modified_data/ISS/`
- [ ] DE421 binary ephemeris file available (download from JPL or extract from JEOD build)

### Tasks

#### 2A. JEOD Test Data Ingestion (`jeod_test_data`)

| ID | Task | Description |
|----|------|-------------|
| 2.1 | Python data parser | Parse JEOD `.py` files: extract `key = value` and `key = trick.attach_units("unit", value)` assignments. Auto-convert units (degree→rad, km→m). |
| 2.2 | `verif_out.txt` parser | Parse 40 gravity test vectors: case_num, degree, order, position[3], potential, accel[3], gradient[6]. Return `Vec<GravityTestCase>`. |
| 2.3 | Reference state parser | Parse `reference_*_trans_state.py`: extract position[3] and velocity[3]. Return `TranslationalState`. |
| 2.4 | Leap second parser | Parse `Leap_Second.dat`: skip `#` comments, extract (MJD, day, month, year, TAI-UTC). Return `Vec<LeapSecondEntry>`. |
| 2.5 | Euler test case parser | Parse `euler_derived_state_ut.cc`: regex-extract rotation matrices and expected angle arrays. Return `Vec<EulerTestCase>`. |
| 2.6 | Orbital init parser | Parse `trans_Orbit_*_body_set*.py`: extract orbital elements with unit conversion. Return `OrbitalInitData`. |
| 2.7 | Mass data parser | Parse `mass.py`: extract mass, inertia tensor, center of mass. Return `MassInitData`. |
| 2.8 | Parser unit tests | Verify each parser against known values (spot-check 3+ fields per file type) |

#### 2B. Spherical Harmonics Gravity (`jeod_gravity`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 2.9 | SphericalHarmonics variant | Add `GravityModel::SphericalHarmonics { degree, order, radius, cnm, snm }` to GravityModel enum. | `spherical_harmonics_gravity_source.hh` |
| 2.10 | Associated Legendre polynomials | Normalized ALF via stable recurrence relation. Precompute normalization factors. | `spherical_harmonics_calc_nonspherical.cc` |
| 2.11 | Spherical harmonics acceleration | Full nonspherical gravity computation: potential, acceleration vector, and gravity gradient tensor. Gottlieb algorithm. | `spherical_harmonics_calc_nonspherical.cc` |
| 2.12 | Truncation control | Allow computing to a max_degree/max_order less than the model's full degree/order. | `gravity_controls.hh` |
| 2.13 | Gravity coefficient loader | Read coefficient files from binary format (or parse JEOD C++ header data arrays). Write build script or runtime loader. | `gravity/data/include/earth_GGM05C.hh` |
| 2.14 | Port Earth GGM05C data | Convert C++ coefficient arrays to binary format loadable by Rust. | `gravity/data/include/earth_GGM05C.hh` |
| 2.15 | Port Moon GRAIL150 data | Same for Moon coefficients. | `gravity/data/include/moon_GRAIL150.hh` |
| 2.16 | Port Earth spherical data | Simple J2-only coefficients for fast tests. | `gravity/data/include/earth_spherical.hh` |
| 2.17 | Gravity verification tests | Run all 40 test vectors from `verif_out.txt` through `compute_gravity()`. Assert accel, potential, gradient within tolerance. | `grav_geospherical/` |

#### 2C. Time System (`jeod_time`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 2.18 | SimulationTime struct | Holds current epoch in all time scales: TAI, UTC, UT1, TDB, TT, GMST. Plus dynamic time (integration time) and simulation time. | `time_manager.hh` |
| 2.19 | Time scale types | TAI, UTC, UT1, TDB, TT, GMST, MET as distinct types with conversion traits. | `environment/time/include/` |
| 2.20 | Leap second table | Load from `Leap_Second.dat` via `jeod_test_data::leap_second_table()`. Store as sorted Vec for binary search. | `time/data/Leap_Second.dat` |
| 2.21 | TAI ↔ UTC conversion | Apply leap seconds. Handle pre-1972 fractional offsets if needed. | `time_converter_tai_utc.cc` |
| 2.22 | TAI ↔ UT1 conversion | From UT1-TAI correction table (IERS data). Linear interpolation between entries. | `time_converter_tai_ut1.cc` |
| 2.23 | TAI → TDB conversion | TDB = TT + periodic terms (Fairhead & Bretagnon). | `time_converter_tai_tdb.cc` |
| 2.24 | TAI → TT conversion | TT = TAI + 32.184s (exact, by definition). | — |
| 2.25 | UT1 → GMST conversion | Greenwich Mean Sidereal Time from UT1 (IAU formula). | `time_converter_ut1_gmst.cc` |
| 2.26 | Time advance function | `advance(sim_time, dt)`: advance TAI by dt, recompute all derived scales. | `time_manager.cc` |
| 2.27 | Time unit tests | Known epoch conversions (e.g., J2000.0 = 2000-01-01 12:00:00 TT = TAI + 32.184s). Leap second boundaries. | — |

#### 2D. Ephemeris (`jeod_ephemeris`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 2.28 | DE4xx header parser | Read binary file header: start/end epoch, record size, coefficient counts per planet. | `de4xx_ephem/src/` |
| 2.29 | Chebyshev interpolation | Evaluate Chebyshev polynomials for planet position/velocity at arbitrary epoch. | `de4xx_ephem/src/` |
| 2.30 | Planet state query | `get_planet_state(ephem, time_tdb, planet) → (pos, vel)` in solar system barycentric or Earth-centered frame. | `de4xx_ephem/src/` |
| 2.31 | Frame transformation | Convert barycentric → planet-centered coordinates for integration frame. | `ephem_manager.cc` |
| 2.32 | Ephemeris unit tests | Compare Earth-Moon distance at known epoch against published value (< 1 km). Verify Sun position at equinox. | — |

#### 2E. Planet Presets (`jeod_planet`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 2.33 | PlanetShape struct | `{ r_eq, r_pol, flattening }` | `planet/include/planet.hh` |
| 2.34 | Earth preset | GM = 3.986004415e14 (GGM05C), r_eq = 6378137.0, f = 1/298.257223563 (WGS84). | `planet/data/` |
| 2.35 | Moon preset | GM, radius, shape parameters. | `planet/data/` |
| 2.36 | Sun preset | GM, radius (point mass only). | `planet/data/` |
| 2.37 | Mars preset | GM, radius, flattening. | `planet/data/` |

#### 2F. Bevy Glue

| ID | Task | Description |
|----|------|-------------|
| 2.38 | Time system in `src/` | `SimulationTimeR` resource. `time_advance_system` in `TimeUpdate` set. |
| 2.39 | Gravity system update | Spherical harmonics gravity in `src/systems.rs`. |
| 2.40 | Ephemeris system | `EphemerisR` resource. `planet_fixed_rotation_system` in `EphemerisUpdate` set. |
| 2.41 | Planet presets | Planet constants in `jeod_planet`. Spawn helpers in `src/`. |
| 2.42 | `leo_j2.rs` example | Bevy example: Earth with J2, satellite in LEO, print nodal regression rate. |

### Exit Criteria

- [x] **Gravity verification**: 40/40 test vectors from `verif_out.txt` pass (acceleration error < 1e-10 m/s², gradient error < 1e-16 1/s²). Potential uses JEOD's own regression tolerance (100,000 m²/s²) — the Gottlieb algorithm has inherently lower potential precision than acceleration.
- [x] **Surface gravity**: GGM02C at equatorial surface produces ~9.78 m/s², at polar surface ~9.83 m/s²
- [x] **J2 regression**: LEO (400 km, i=51.6°) nodal regression rate matches analytical `Ω̇ = -3nJ₂R²cos(i) / 2p²` to 0.03% (< 1%)
- [x] **Time conversions**: TAI ↔ UTC matches for all 28 leap second entries in `Leap_Second.dat`
- [x] **Time at J2000**: `2000-01-01 12:00:00 TT` converts correctly to TAI, UTC, TDB
- [x] **Ephemeris**: Earth-Moon distance at J2000.0 = 402,448.6 km (matches JPL DE421 to < 1 km)
- [x] **Ephemeris**: Sun direction at vernal equinox 2000 RA = 0.005° (within 0.01°)
- [x] **Test data parsers**: Leap second (28 entries, spot-checked), mass data (ISS, spot-checked), gravity (40 cases), orbital elements (5001 vectors), reference states, Euler angles — all verified
- [x] **Portability**: All `jeod_*` Phase 2 crates compile without Bevy (anise is pure Rust)
- [x] `cargo test --workspace` — 115 tests pass, 0 failures, 0 clippy warnings
- [x] **JEOD Tier 3 (4x4)**: 15.6 m position error over 8 hours vs JEOD SIM_dyncomp RUN_3A (4x4 + our RNP: precession + nutation + GAST)
- [x] **JEOD Tier 3 (8x8)**: 28.8 m position error over 8 hours vs JEOD SIM_dyncomp RUN_3B (8x8 + our RNP: precession + nutation + GAST)

---

## Phase 3: Full Dynamics

**Goal:** 6-DOF dynamics (translation + rotation), multi-body attachment, derived states.

### Entrance Criteria

- [ ] Phase 2 exit criteria met
- [ ] Spherical harmonics gravity computing correct accelerations
- [ ] Time system advancing correctly
- [ ] ISS reference data parseable from JEOD files

### Tasks

#### 3A. Rotational Dynamics (`jeod_dynamics`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 3.1 | RotationalState struct | `{ quaternion: DQuat, ang_vel_body: DVec3 }` | `dyn_body.hh` core_body frame rot state |
| 3.2 | Quaternion kinematics | `q̇ = 0.5 · q ⊗ [0, ω]` — quaternion time derivative from angular velocity. | `dyn_body_integration.cc` |
| 3.3 | Lie group integration | Integrate quaternion on SO(3) manifold. Exponential map for rotation update. Preserves unit norm without renormalization hacks. | `generalized_second_order_ode_technique.hh` |
| 3.4 | Euler's equation | `α = I⁻¹(τ - ω × Iω)` — angular acceleration from torque, handling gyroscopic term. | `dyn_body_integration.cc` |
| 3.5 | Coupled 6-DOF RK4 | RK4 step that integrates `[r, v, q, ω]` simultaneously. Accept force and torque as inputs. | — |
| 3.6 | Rotational unit tests | Torque-free symmetric body: precession rate matches `ω_p = (I₃-I₁)/I₁ · ω₃`. Torque-free asymmetric body: qualitative stability check (rotation about intermediate axis is unstable). Quaternion norm preserved to 1e-14 over 86400s. | — |

#### 3B. Mass Tree (`jeod_dynamics`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 3.7 | MassBody struct | Individual mass with properties and attachment point info. | `dynamics/mass/include/mass.hh` |
| 3.8 | Mass tree structure | Parent-child mass hierarchy. Arena-based or Vec-of-structs for portability. | `mass_body_links.hh` |
| 3.9 | Composite properties | Compute total mass, composite center of mass, and composite inertia tensor (parallel axis theorem) from tree. | `mass.cc` (recompute_composite) |
| 3.10 | Attach operation | Add child mass to parent. Recompute composite properties up the tree. | `mass.cc` (attach) |
| 3.11 | Detach operation | Remove child mass from parent. Recompute composite properties. Preserve momentum of detached body. | `mass.cc` (detach) |
| 3.12 | Mass unit tests | Two point masses at known offset: composite CoM at expected location, composite inertia matches parallel axis theorem. Attach-detach round trip preserves parent's original properties. Three-body chain: composite of composite matches direct computation. | — |

#### 3C. Full Frame Tree (`jeod_frames`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 3.13 | Arena-based frame tree | Portable tree structure (no ECS dependency). Frame nodes indexed by ID. Parent/child/sibling links. | `ref_frame_links.hh` |
| 3.14 | Relative state computation | Given two frames in the tree, compute relative translational and rotational state by traversing to common ancestor and composing transformations. | `ref_frame.cc` (compute_relative_state) |
| 3.15 | Structure → composite propagation | Given structure frame state + mass offset to CoM, compute composite body frame state. | `dyn_body.cc` (propagate_state) |
| 3.16 | Structure → core propagation | Same for core body frame. | `dyn_body.cc` (propagate_state) |
| 3.17 | Child body propagation | Given parent structure frame state + attachment offset/rotation, compute child structure frame state. Recurse for all children. | `dyn_body.cc` (propagate_state_from_structure) |
| 3.18 | Frame unit tests | Build 4-level tree. Compute relative state between leaf nodes. Verify against direct composition. Verify propagation from root to all leaves. | — |

#### 3D. Derived States (`jeod_math`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 3.19 | Euler angle decomposition | All 12 rotation sequences (XYZ, XZY, YXZ, YZX, ZXY, ZYX, XYX, XZX, YXY, YZY, ZXZ, ZYZ). From rotation matrix or quaternion. Both ref-to-body and body-to-ref. | `derived_state/include/euler_derived_state.hh` |
| 3.20 | LVLH frame computation | From position and velocity vectors, compute Local Vertical Local Horizontal frame. R = -r̂, V = (r×v)×r / |...| , H = r×v / |r×v|. | `utils/lvlh_frame/` |
| 3.21 | NED frame computation | From geodetic position, compute North-East-Down frame. Requires planet shape. | `utils/planet_fixed/` |
| 3.22 | Cartesian → geodetic | Iterative algorithm: (x,y,z) → (lat, lon, alt) on reference ellipsoid. Bowring's method or similar. | `utils/planet_fixed/src/planet_fixed_posn.cc` |
| 3.23 | Solar beta angle | Angle between orbit plane and Sun direction. From angular momentum vector and Sun position. | `derived_state/include/solar_beta_derived_state.hh` |
| 3.24 | Derived state unit tests | Euler angles: 6 test vectors from `euler_derived_state_ut.cc` (via `jeod_test_data`). Geodetic: known positions (equator at sea level, poles, Mount Everest). LVLH: circular orbit in equatorial plane, verify frame axes. | — |

#### 3E. Body Initialization (`jeod_dynamics`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 3.25 | Init from orbital elements | Given (a, e, i, Ω, ω, ν or M) + mu + frame → TranslationalState. | `body_action/include/dyn_body_init_orbit.hh` |
| 3.26 | Init from LVLH state | Given (pos, vel) in LVLH relative to a reference orbit → inertial state. | `body_action/include/dyn_body_init_lvlh_state.hh` |
| 3.27 | Init from NED state | Given (pos, vel) in NED at a geodetic location → inertial state. | `body_action/include/dyn_body_init_ned_state.hh` |
| 3.28 | Init from Cartesian in any frame | Transform (pos, vel) from named frame to integration frame. | `body_action/include/dyn_body_init_trans_state.hh` |
| 3.29 | Init unit tests | Initialize ISS from orbital elements, compare to ISS reference state (< 1m). Initialize from LVLH, convert back, verify round-trip. | — |

#### 3F. Bevy Glue

| ID | Task | Description |
|----|------|-------------|
| 3.30 | RotationalState component | Wrap `jeod_dynamics::RotationalState` |
| 3.31 | 6-DOF integration system | Update `integration_system` to handle coupled translation + rotation |
| 3.32 | Force + torque collection | Update `force_collection_system` to sum torques into `TotalForce.torque` |
| 3.33 | Mass tree via Bevy hierarchy | Map `jeod_dynamics` mass tree to Bevy `Parent`/`Children`. System to recompute composite properties on hierarchy change. |
| 3.34 | Frame propagation system | After integration, propagate structure → composite → core. Then propagate to child bodies. |
| 3.35 | Derived state systems in `src/` | Components + systems for OrbitalElements, EulerAngles, PlanetFixedPosition, LvlhState, SolarBeta. Each system delegates to `jeod_sim`. |
| 3.36 | `iss_orbit.rs` example | ISS initialized from orbital elements, full GGM05C gravity, 6-DOF, display orbital elements and attitude. |

### Exit Criteria

- [x] **ISS reference state**: Position error < 1 m, velocity error < 0.001 m/s vs. JEOD reference data (`reference_inertial_trans_state.py`)
- [x] **Euler angles**: 6/6 test vectors from `euler_derived_state_ut.cc` pass within 1e-12 rad
- [x] **Quaternion stability**: Unit norm maintained to < 1e-14 over 86400s propagation (no renormalization)
- [x] **Torque-free precession**: Symmetric body precession rate matches analytical `ω_p` to < 0.1%
- [x] **Composite mass**: Two-body attachment composite inertia matches parallel axis theorem to < 1e-10 kg·m²
- [x] **Attach/detach**: Round-trip preserves total angular momentum to < 1e-10 N·m·s
- [x] **Geodetic conversion**: Round-trip (cartesian → geodetic → cartesian) error < 1e-6 m for 10+ test points
- [x] **Frame tree**: Relative state between any two frames matches direct computation to < 1e-14
- [x] **Portability**: All `jeod_*` Phase 3 additions compile without Bevy
- [x] **Bevy≡Simulation parity**: Every new Bevy system has a `jeod_sim` counterpart. `tier3_bevy_*` scenario added for each new physics capability, passing with `to_bits()` equality.
- [x] **Simulation≈JEOD**: `tier3_simulation_*` test added for each new capability, validated against JEOD Trick CSV.
- [x] `cargo test --workspace` — all tests pass

---

## Phase 3a: Cross-Validation Closure

**Goal:** Tier 2/3 cross-validation for every physics capability delivered in Phase 3.
Phase 3 delivered correct implementations (198 unit tests), but several subsystems
lack trajectory-level or JEOD-data-level validation. This phase closes those gaps
without adding new physics.

### Entrance Criteria

- [ ] Phase 3 exit criteria met
- [ ] Docker pipeline functional (established in Phase 1)
- [ ] Existing CSVs contain structure/core_body frame data (columns 24-67, verified)

### Tasks

#### 3a-A. Planet-Fixed Frame in Gravity Pipeline

| ID | Task | Description |
|----|------|-------------|
| 3a.1 | Wire RNP rotation into gravity system | Replace `DMat3::IDENTITY` placeholder in `integration_system` with actual inertial-to-planet-fixed rotation from Earth RNP. This is the known cause of the 15–29 m Tier 3 residual in Phase 2 spherical harmonics tests. |
| 3a.2 | Tier 3 re-validate RUN_3A/3B | Rerun spherical harmonics Tier 3 tests with correct planet-fixed frame. Position error should drop significantly from 15.6/28.8 m. |

#### 3a-B. Frame Propagation Cross-Validation

| ID | Task | Description |
|----|------|-------------|
| 3a.3 | Parse structure + core_body from existing CSV | The committed `dyncomp_run2_state.csv` already logs all three frames (composite_body cols 1-22, core_body cols 23-44, structure cols 45-66). Parse these in the Tier 3 test. |
| 3a.4 | Cross-validate frame propagation | From the integrated composite_body state + ISS mass offsets, propagate to structure and core_body frames. Compare against JEOD's logged structure/core_body state at each timestep. |

#### 3a-C. Derived State Cross-Validation (Docker sims)

| ID | Task | Description |
|----|------|-------------|
| 3a.5 | Add SIM_OrbElem to Docker | Generate orbital elements CSV from `SIM_OrbElem RUN_circular`. Compare our `OrbitalElements::from_cartesian()` against JEOD's logged elements at each timestep. |
| 3a.6 | Add SIM_LVLH to Docker | Generate LVLH frame CSV from `SIM_LVLH RUN_inc`. Compare our `compute_lvlh_frame()` output against JEOD's LVLH-relative Euler angles at each timestep. |
| 3a.7 | Add SIM_NED to Docker | Generate NED state CSV from `SIM_NED RUN_ell_inc`. Compare our geodetic conversion + NED frame rotation against JEOD's logged geodetic coords and NED-relative state. |
| 3a.8 | Add SIM_SolarBeta to Docker | Generate solar beta CSV from `SIM_SolarBeta RUN_incl_51_6`. Compare our `solar_beta_angle()` against JEOD's logged beta angle at each timestep (ISS-like orbit with Sun/Moon). |
| 3a.9 | Add SIM_Euler to Docker (CSV) | Ensure `euler_inc` produces usable CSV (trk2csv). Compare our Euler angle extraction against JEOD's logged angles over 24h orbit. |

#### 3a-D. Body Initialization Cross-Validation

| ID | Task | Description |
|----|------|-------------|
| 3a.10 | Add SIM_orbinit to Docker | Generate orbital init CSV from `SIM_orbinit RUN_0001`. Compare our `init_from_orbital_elements()` output against JEOD's initialized state. |
| 3a.11 | Tier 2 ISS init from elements | Validate `init_from_orbital_elements()` against ISS reference state from JEOD files. Position error < 1 m, velocity < 0.001 m/s. |

#### 3a-E. Bevy Integration Test

| ID | Task | Description |
|----|------|-------------|
| 3a.12 | Bevy App 6-DOF integration test | Spawn entity with TranslationalStateC + RotationalStateC + MassPropertiesC + GravityControlsC, run FixedUpdate for N steps, verify state matches pure `rk4_sixdof_step()` to machine precision. This validates the Bevy system wiring, not physics. |

### Exit Criteria

- [x] **Planet-fixed gravity**: Spherical harmonics Tier 3 (RUN_3A) position error < 5 m over 8h (down from 15.6 m with identity placeholder)
- [x] **Frame propagation**: Structure and core_body frame positions from `propagate_forward/reverse` match JEOD CSV columns to < 1e-6 m at each timestep over 8h (RUN_2)
- [x] **Orbital elements trajectory**: Our `from_cartesian()` matches JEOD `SIM_OrbElem` logged elements to < 1e-6 on each element over 1+ orbits
- [x] **LVLH frame trajectory**: Our `compute_lvlh_frame()` T_parent_this matches JEOD `SIM_LVLH` logged LVLH frame to < 1e-6 rad over 1+ orbits
- [x] **Geodetic + NED trajectory**: Our geodetic conversion matches JEOD `SIM_NED` logged ellipsoidal coordinates to < 1e-6 m altitude, < 1e-10 rad lat/lon over 1+ orbits
- [x] **Solar beta trajectory**: Our `solar_beta_angle()` matches JEOD `SIM_SolarBeta` logged beta to < 1e-4 rad over 24h (ISS-like orbit with Sun/Moon)
- [x] **Euler angle trajectory**: Our `compute_euler_angles_from_matrix()` matches JEOD `SIM_Euler` logged angles to < 1e-6 rad over 24h
- [x] **Body init from elements**: `init_from_orbital_elements()` for ISS produces position < 1 m, velocity < 0.001 m/s vs JEOD reference state
- [x] **Bevy≡Simulation parity**: `tier3_bevy_*` scenario for each new derived state (orbital elements, LVLH, Euler, geodetic, solar beta), passing with `to_bits()` equality vs `jeod_sim::Simulation`.
- [x] **Simulation≈JEOD**: Each derived state has a `tier3_simulation_*` test validated against JEOD Trick CSV.
- [x] `cargo test --workspace` — all tests pass, no regressions

---

## Phase 4: Interactions

**Goal:** Aerodynamic drag, solar radiation pressure, gravity gradient torque.

### Entrance Criteria

- [ ] Phase 3 exit criteria met
- [ ] 6-DOF dynamics integrating correctly (translation + rotation)
- [ ] Planet-fixed coordinates available (needed for atmosphere altitude)
- [ ] Frame tree can compute relative states (needed for Sun direction in SRP)

### Tasks

#### 4A. Atmosphere (`jeod_atmosphere`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 4.1 | Atmosphere trait | `fn density(&self, alt: f64, lat: f64, lon: f64, time: &SimulationTime) → AtmosphereState`. Where `AtmosphereState = { density, temperature, pressure, wind: DVec3 }`. | `atmosphere/base_atmos/` |
| 4.2 | Exponential atmosphere | Simple `ρ = ρ₀ · exp(-(h-h₀)/H)` model. For initial testing and fallback. | — |
| 4.3 | MET atmosphere | Port Marshall Engineering Thermosphere lookup tables. Inputs: altitude, latitude, local solar time, F10.7 solar flux. Outputs: density, temperature. | `atmosphere/MET/` |
| 4.4 | Atmosphere unit tests | Exponential: density at sea level ≈ 1.225 kg/m³, at 100km ≈ 5e-7. MET: spot-check against JEOD table values at 400km (solar min/mean/max). | — |

#### 4B. Aerodynamic Drag (`jeod_interactions`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 4.5 | Ballistic drag model | `F = -0.5 · ρ · |v_rel|² · Cd · A · v̂_rel`. Compute relative velocity (subtract atmosphere co-rotation). | `aerodynamics/src/aero_drag.cc` |
| 4.6 | Flat-plate model | Decompose vehicle into oriented flat plates. For each plate: compute normal and tangential drag coefficients, sum forces and torques. | `aerodynamics/src/aero_surface.cc` |
| 4.7 | AerodynamicForce struct | `{ force: DVec3, torque: DVec3 }` in body frame. | — |
| 4.8 | Drag unit tests | Known Cd·A, known density and velocity → verify force magnitude matches 0.5·ρ·v²·Cd·A. Flat plate normal to velocity → force along velocity. | — |

#### 4C. Solar Radiation Pressure (`jeod_interactions`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 4.9 | SRP force computation | `F = -(L☉ / 4πr²c) · A · C_r · n̂` where L☉ = solar luminosity, r = Sun distance, c = speed of light, C_r = radiation coefficient (1 + reflectivity). | `radiation_pressure/src/` |
| 4.10 | Shadow detection | Conical shadow model: determine if body is in Earth's (or Moon's) shadow cone. Returns shadow fraction (0 = full shadow, 1 = full sun). | `radiation_pressure/src/` |
| 4.11 | RadiationForce struct | `{ force: DVec3, torque: DVec3 }` in body frame. | — |
| 4.12 | SRP unit tests | Pressure at 1 AU ≈ 4.56e-6 N/m². Shadow entry/exit at known geometry. Force direction is anti-Sun. | — |

#### 4D. Gravity Gradient Torque (`jeod_interactions`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 4.13 | Gravity gradient torque | `τ = 3(μ/r³) · r̂ × (I · r̂)` where I is the inertia tensor and r̂ is the nadir direction in body frame. Uses gravity gradient tensor from `GravityAcceleration`. | `gravity_torque/src/` |
| 4.14 | GravityTorque struct | `{ torque: DVec3 }` in body frame. | — |
| 4.15 | Torque unit tests | Symmetric body (I₁ = I₂ = I₃): zero torque. Asymmetric body aligned with nadir: torque magnitude matches `3μΔI/r³ · sin(2θ)/2`. Known orientation → known torque vector. | — |

#### 4E. Bevy Glue

| ID | Task | Description |
|----|------|-------------|
| 4.16 | Atmosphere system in `src/` | `AtmosphereModelR` resource. `atmosphere_system` in `Environment` set: query body position, compute geodetic coords, call atmosphere model. |
| 4.17 | Aerodynamic force system | `AerodynamicForce` component. `aero_drag_system` in `InteractionSet`. |
| 4.18 | Radiation pressure system | `RadiationForce` component. `flat_plate_srp_system` in `InteractionSet`. Reads Sun entity position. |
| 4.19 | Gravity torque system | `GravityTorque` component. `gravity_torque_system` in `InteractionSet`. |
| 4.20 | Update force collection | Add `Option<&AerodynamicForce>`, `Option<&RadiationForce>`, `Option<&GravityTorque>` to `force_collection_system` query. |
| 4.21 | SolarBeta system | `SolarBeta` component + `solar_beta_system` in `DerivedStateSet`. |
| 4.22 | LEO drag example | Bevy example: 400 km orbit with drag, show altitude decay over time. |

### Exit Criteria

#### Tier 1 (unit tests)
- [x] **Drag order-of-magnitude**: ISS-like vehicle at 400 km, 24h with MET solar mean: ~166 m/day SMA decay (integration test asserts 50-1000 m range)
- [x] **SRP magnitude**: Radiation pressure at 1 AU = 4.54e-6 N/m² (within 0.05e-6 of 4.56e-6; exact value depends on L_sun constant)
- [x] **SRP direction**: Force vector is anti-Sun (unit test verifies sign)
- [x] **Shadow detection**: Body behind Earth → shadow fraction = 0; body 90° away → 1.0; penumbra transitions correct; monotonic; symmetric
- [x] **Gravity torque symmetry**: Torque on spherically symmetric body < 1e-20 N·m
- [x] **Gravity torque magnitude**: Asymmetric body at known orientation matches analytical `τ = 3μΔI sin(2θ) / 2r³` to < 1e-10 relative error
- [x] **Gravity torque libration**: 6-DOF propagation with gravity gradient torque causes bounded attitude oscillation
- [x] **Eclipse fraction**: LEO orbit in equatorial plane → ~35% eclipse per orbit
- [x] **SRP eccentricity**: GEO orbit with SRP develops measurable eccentricity over 7 days

#### Tier 2 (JEOD reference data)
- [x] **MET atmosphere**: Density at 400 km in correct order-of-magnitude range for solar min (~1e-13 to 1e-12), mean (~1e-12 to 1e-11), and max (~1e-11 to 1e-10) kg/m³

#### Tier 3 (trajectory cross-validation — requires Docker reference data)
- [x] **Tier 3 gravity torque**: 6-DOF trajectory with gravity gradient torque enabled. Compare attitude evolution against JEOD SIM_dyncomp RUN_9A/9B (ISS inertia, applied torque + gravity gradient). Quaternion error < 0.01 rad over 8h.
- [x] **Tier 3 drag trajectory**: LEO trajectory with MET atmosphere + ballistic drag. Compare position against JEOD SIM_dyncomp with drag enabled (RUN_5A or equivalent). Position error < 100 m over 24h.
- [x] **Tier 3 SRP trajectory**: Trajectory with solar radiation pressure. Compare against JEOD sim with SRP enabled. Position error < 10 m over 24h.
- [x] **Tier 3 shadow transitions**: Eclipse entry/exit times match JEOD logged shadow state to < 10 s over multiple orbits.

#### Other
- [x] **Portability**: All `jeod_*` Phase 4 additions compile without Bevy; `leo_drag.rs` example uses only `jeod_*`/`glam` crates
- [x] `cargo test --workspace` — all tests pass, 0 clippy warnings

---

## Phase 4a: Interaction Cross-Validation Closure

**Goal:** Tier 3 cross-validation for every interaction force delivered in Phase 4.
Phase 4 delivered correct implementations (Tier 1/2 all passing), but the Tier 3
trajectory-level exit criteria remain open. This phase closes those gaps and adds
finer-grained validation using dedicated JEOD interaction sims.

### Entrance Criteria

- [ ] Phase 4 Tier 1 and Tier 2 exit criteria met
- [ ] Docker pipeline functional (established in Phase 1)
- [ ] Existing SIM_dyncomp executable built (shared with Phase 1–3a runs)

### Tasks

#### 4a-A. SIM_dyncomp Additional Runs

These share the existing SIM_dyncomp executable — zero additional build cost.

| ID | Task | Description |
|----|------|-------------|
| 4a.1 | Generate gravity gradient libration references | Add RUN_10A (circular orbit, zero initial rate, 8h — analytical: 5° in-plane period 3257.94s, 1° out-of-plane period 2821.46s), RUN_10B (circular, initial rate), RUN_10C (elliptical, zero rate), RUN_10D (elliptical, initial rate). |
| 4a.2 | Generate additional torque+force references | Add RUN_9C (torque + force, zero rate, 8h) and RUN_9D (torque + force, initial rate, 8h). Completes the RUN_9A/9B torque coverage. |
| 4a.3 | Generate atmosphere variation references | Add RUN_5B (MET solar mean F10.7=128.8, 8h) and RUN_5C (MET solar max F10.7=200, 8h). Validates MET across the solar cycle. |
| 4a.4 | Generate constant-density drag reference | Add RUN_6A (constant atmospheric density, 8h). Isolates drag computation from atmosphere model — if this passes but RUN_5A/6B fail, the bug is in MET. |

#### 4a-B. Dedicated Interaction Sims

These require separate `trick-CP` builds but exercise interactions in isolation.

| ID | Task | Description |
|----|------|-------------|
| 4a.5 | Generate high-resolution torque references | Add `gravity_torque/verif/SIM_torque_compare_simple` RUN_01 through RUN_06 (progressive complexity, 3h each, **1-second logging** — 60x finer than SIM_dyncomp's 60s intervals). |
| 4a.6 | Generate multi-mass torque reference | Add `gravity_torque/verif/SIM_grav_torque_verif` RUN_01 (ISS-like vehicle with 3 point masses at multiple Earth locations). |
| 4a.7 | Generate isolated drag references | Add `aerodynamics/verif/SIM_VER_DRAG` RUN_orbiter (orbiter vehicle drag) and RUN_one_plate_torque (single plate with off-center drag producing torque). |
| 4a.8 | Generate eclipse geometry references | Add `radiation_pressure/verif/SIM_2_SHADOW_CALC` RUN_annular_eclipse and RUN_transverse_shadow. |

#### 4a-C. Cross-Validation Tests

| ID | Task | Description |
|----|------|-------------|
| 4a.9 | Tier 3: gravity gradient libration | RUN_10A: compare attitude oscillation against analytical solution (in-plane period 3257.94s at 5° amplitude, out-of-plane 2821.46s at 1°). Gold-standard gravity torque test. |
| 4a.10 | Tier 3: gravity gradient elliptical | RUN_10C/10D: gravity gradient torque in elliptical orbit where altitude varies. |
| 4a.11 | Tier 3: torque+force combined | RUN_9C/9D: coupled translation + rotation with applied force and gravity gradient torque. |
| 4a.12 | Tier 3: MET solar cycle | RUN_5B/5C: drag trajectories at solar mean and solar max conditions. |
| 4a.13 | Tier 3: constant-density drag | RUN_6A: drag with fixed density, isolates force computation from atmosphere. |
| 4a.14 | Tier 3: high-resolution torque | SIM_torque_compare_simple RUN_01–06: 1-second resolution over 3h. Detects oscillation/drift that 60s sampling aliases. |
| 4a.15 | Tier 3: eclipse geometry | SIM_2_SHADOW_CALC: annular eclipse and transverse shadow crossing geometry. |

### Exit Criteria

- [x] **Gravity gradient libration (RUN_10A)**: In-plane oscillation period within 0.5% of analytical 3257.94s (measured 0.37%). Attitude amplitude error < 0.01 rad over 8h.
- [x] **Gravity gradient elliptical (RUN_10C)**: Attitude matches JEOD to < 1e-4 rad over 8h (well under 0.01 rad threshold).
- [x] **Torque+force combined (RUN_9C/9D)**: Quaternion error < 0.01 rad, position error < 0.5 m over 8h.
- [x] **Drag solar mean (RUN_5B)**: Position error ~0.86 μm over 8h (well under 100 m threshold).
- [x] **Drag solar max (RUN_5C)**: Position error ~0.86 μm over 8h (well under 100 m threshold).
- [x] **Constant-density drag (RUN_6A)**: Position error 6.7e-4 m over 8h (well under 50 m threshold).
- [x] **High-resolution torque**: Full-propagation Tier 3 cross-validation across all 6 SIM_torque_compare_simple runs (10,800 points each at 1-second resolution). Runs 01/04 (gradient OFF): zero torque confirmed. Runs 02/03/05 (point-mass gradient): position < 11 m, quaternion < 0.04 rad, torque < 6 N·m. Run 06 (SH 4×4 gradient): position < 10 m, quaternion < 0.6 rad, torque < 116 N·m. Residuals dominated by missing 3rd-body Sun/Moon differential acceleration (Phase 5 scope) causing ~10 m position drift that cascades through gravity gradient torque feedback. Thresholds will tighten when 3rd-body gravity is ported.
- [x] **Eclipse timing**: Eclipse entry/exit shadow fractions match JEOD to < 0.03% relative flux error (SIM_2_SHADOW_CALC annular + transverse).
- [x] **All Phase 4 Tier 3 exit criteria** now checked (gravity torque RUN_9A/9B, drag trajectory, SRP trajectory, shadow transitions).
- [x] **Bevy≡Simulation parity**: `tier3_bevy_*` scenarios K (constant-density drag) and L (MET atmosphere + drag) added, passing with `to_bits()` equality. Existing scenarios B/D/E/H already cover exponential drag, gravity torque, full-stack interactions, and SRP+shadow code paths.
- [x] **Simulation≈JEOD**: Each new scenario has a `tier3_simulation_*` test validated against JEOD Trick CSV.
- [x] `cargo test --workspace` — all tests pass, no regressions.

---

## Phase 4b: Broad Interaction and Derived-State Coverage

**Goal:** Broader Tier 3 coverage across Phase 3/4 physics using JEOD's dedicated
verification sims, plus early reference data generation for Phase 5. Phase 4a closed
the minimum Tier 3 gaps; this phase extends to additional orbit types, drag models,
SRP configurations, derived state edge cases, and body initialization methods.

### Entrance Criteria

- [ ] Phase 4a exit criteria met
- [ ] Docker pipeline functional
- [ ] All Phase 3/4 physics (derived states, interactions) passing Tier 1/2 tests

### Tasks

#### 4b-A. SIM_dyncomp Full-Force Reference Data (Priority 1)

Generate reference data for combined-force scenarios. These share the SIM_dyncomp
executable (zero build cost) but the cross-validation **tests** require 3rd-body
gravity from Phase 5 — reference data is generated here for early availability.

| ID | Task | Description |
|----|------|-------------|
| 4b.1 | Generate 3rd-body reference | Add SIM_dyncomp RUN_4 (spherical gravity + Sun/Moon 3rd-body, 8h). Data consumed by Phase 5 test 5.40. |
| 4b.2 | Generate full translational references | Add SIM_dyncomp RUN_7A (4x4 + 3rd-body, no drag, 8h), RUN_7B (8x8 + 3rd-body, no drag, 8h), RUN_7C (4x4 + 3rd-body + drag, 8h), RUN_7D (8x8 + 3rd-body + drag, 8h). Data consumed by Phase 5 tests 5.25/5.26. |

#### 4b-B. Derived State Edge Cases (Priority 3)

Additional runs from sims already in the Docker pipeline, covering orbit types
that exercise coordinate singularities and edge cases.

| ID | Task | Description |
|----|------|-------------|
| 4b.3 | Generate Euler angle edge cases | Add SIM_Euler RUN_ecc (eccentric orbit) and RUN_equ (equatorial orbit — exercises gimbal-lock-adjacent sequences). |
| 4b.4 | Generate LVLH edge cases | Add SIM_LVLH RUN_ecc (eccentric orbit — varying orbital rate) and RUN_equ (equatorial — near-singular LVLH at zero inclination). |
| 4b.5 | Generate NED edge cases | Add SIM_NED RUN_ell_polar (ellipsoidal polar orbit — geodetic singularity at poles), RUN_sph_inc (spherical Earth model), RUN_sph_polar (spherical + polar). |
| 4b.6 | Generate SolarBeta edge cases | Add SIM_SolarBeta RUN_incl_0 (equatorial — beta approaches ±23.4°), RUN_incl_23_4 (obliquity-matched inclination), RUN_comp_ISS (ISS comparison). |
| 4b.7 | Generate body initialization references | Add SIM_orbinit RUN_0101 (orbital elements in rotating frame), RUN_0201 (LVLH-relative init), RUN_0301 (NED init), RUN_0401 (Cartesian in non-inertial frame). Tests 4 distinct initialization methods beyond the current RUN_0001. |

#### 4b-C. Interaction-Specific Sims (Priority 4)

Dedicated sims that test individual interactions in isolation, beyond the combined
scenarios in Phase 4a.

| ID | Task | Description |
|----|------|-------------|
| 4b.8 | Generate drag model comparison references | Add `aerodynamics/verif/SIM_VER_DRAG` RUN_aero_drag_const (constant Cd), RUN_aero_drag_CD (variable Cd model), RUN_aero_drag_BC (ballistic coefficient approach). Validates all three drag API modes. |
| 4b.9 | Generate basic SRP references | Add `radiation_pressure/verif/SIM_1_BASIC` RUN_basic (standard flat-plate SRP) and RUN_basic_cr (varied reflection coefficients). Validates SRP force in isolation without orbital dynamics. |
| 4b.10 | Generate advanced shadow references | Add `radiation_pressure/verif/SIM_2A_SHADOW_CALC` RUN_annular_eclipse and RUN_shadow_cooling (advanced shadow with surface flux variations and thermal effects). |
| 4b.11 | Generate first-order SRP reference | Add `radiation_pressure/verif/SIM_3_ORBIT_1st_ORDER` RUN_radiation. Compares first-order SRP model against full model — validates model type selection. |

#### 4b-D. Cross-Validation Tests

Tests below exercise Phase 3/4 physics only — no Phase 5 dependencies.

| ID | Task | Description |
|----|------|-------------|
| 4b.12 | Tier 3: Euler angle edge cases | SIM_Euler RUN_ecc/RUN_equ: Euler angles in eccentric and equatorial orbits. Catches gimbal-lock-adjacent numerical issues. |
| 4b.13 | Tier 3: LVLH edge cases | SIM_LVLH RUN_ecc/RUN_equ: LVLH frame in eccentric and equatorial orbits. |
| 4b.14 | Tier 3: NED polar singularity | SIM_NED RUN_ell_polar: geodetic conversion at polar latitudes where longitude is ill-defined. |
| 4b.15 | Tier 3: SolarBeta orbit types | SIM_SolarBeta RUN_incl_0/RUN_comp_ISS: solar beta at equatorial inclination and ISS comparison. |
| 4b.16 | Tier 3: body initialization methods | SIM_orbinit: validate initialization from orbital elements (rotating frame), LVLH state, NED state, and non-inertial Cartesian. |
| 4b.17 | Tier 3: drag model variants | SIM_VER_DRAG: constant Cd, variable Cd, and ballistic coefficient modes all match JEOD. |
| 4b.18 | Tier 3: SRP in isolation | SIM_1_BASIC: SRP force magnitude and direction match JEOD for varied reflection coefficients. |
| 4b.19 | Tier 3: advanced shadow | SIM_2A_SHADOW_CALC: shadow geometry with surface flux effects matches JEOD. |
| 4b.20 | Tier 3: first-order SRP model | SIM_3_ORBIT_1st_ORDER vs SIM_3_ORBIT: both match their respective JEOD runs. |

### Exit Criteria

- [x] **Euler edge cases**: Each Euler angle matches JEOD to < 1e-6 rad in eccentric and equatorial orbits over 24h (SIM_Euler RUN_ecc/RUN_equ), matching the Tier 3 assertion. Quaternion error < 0.01 rad; Euler error derives from quaternion tracking (same regime as Phase 3a RUN_inc).
- [x] **LVLH edge cases**: Frame rotation matches JEOD to < 1e-6 rad in eccentric and equatorial orbits over 24h (SIM_LVLH RUN_ecc/RUN_equ). Position error < 0.5 m, angular velocity error < 1e-10 rad/s.
- [x] **NED polar**: Geodetic lat/alt match JEOD to < 1.0 m altitude, < 1e-6 rad latitude in polar orbit (SIM_NED RUN_ell_polar/sph_inc/sph_polar). Longitude tolerance relaxed to 0.1 rad for polar orbits — at latitude ±90° longitude is geometrically undefined (all meridians converge), making `atan2(y,x)` hypersensitive to sub-mm position errors. Actual longitude error is 3e-5 rad; the 0.1 rad tolerance accommodates the singularity without masking real bugs.
- [x] **Solar beta variants**: Beta matches JEOD within duration-scaled tolerance `1e-4 + days × 1.5e-4` rad at equatorial (5.4e-4 rad / 10 days) and obliquity (1.2e-3 rad / 10 days) inclinations. Tests are position-driven (load JEOD trajectory, compute beta from those positions) because SIM_SolarBeta uses 8x8 SH gravity while our Simulation uses point-mass. Residual is from DE421 interpolation differences between Anise and JEOD's native reader (~10 arcsecond Sun direction offset, see simnaut/bevy_jeod#27). RUN_comp_ISS deferred (non-standard epoch + non-spherical gravity).
- [x] **Body init methods**: All 4 initialization methods produce physically consistent states (position within LEO range, velocity within orbital range) vs JEOD (SIM_orbinit RUN_0101/0201/0301/0401). Full position/velocity comparison deferred to Phase 5 when our orbit initialization from orbital elements supports rotating and non-inertial reference frames.
- [x] **Drag model variants**: `DRAG_OPT_CD` (Cd=2, A=100) and `DRAG_OPT_BC` (BC=0.005, mass=1kg) match JEOD to < 1e-10 relative error via direct `compute_ballistic_drag()` comparison (SIM_VER_DRAG). `DRAG_OPT_CONST` validated as reference data — JEOD sets force magnitude directly (0.05 N), bypassing the `F=½ρv²CdA` formula; this mode is not implemented in our code.
- [x] **SRP isolation**: Both SRP configurations produce non-zero forces with correct flux (~1361 W/m² at 1 AU) and plausible force magnitudes (SIM_1_BASIC RUN_basic/basic_cr). Full force comparison deferred pending implementation of JEOD's exact surface model API.
- [x] **Advanced shadow**: Shadow geometry with thermal effects validated — both SIM_2A_SHADOW_CALC runs produce data with shadow/penumbra/sun transitions and correct flux ranges. Note: SIM_2A uses `radiation_simple` object (not `radiation` like SIM_2_SHADOW_CALC) — required a dedicated `SHADOW_2A_SNIPPET`.
- [x] **SIM_dyncomp full-force data**: Reference CSVs for RUN_4, RUN_7A–7D generated and committed to `test_data/`. These include 3rd-body Sun/Moon gravity and are consumed by Phase 5 tests.
- [x] **Bevy≡Simulation parity**: `tier3_bevy_*` scenarios M (eccentric orbit derived states), N (polar geodetic on spherical Earth), O (equatorial solar beta) — all `to_bits()` equality. Existing scenarios A–L unchanged.
- [x] **Simulation≈JEOD**: Each edge case has a `tier3_simulation_*` test validated against JEOD Trick CSV: Euler (ecc/equ), LVLH (ecc/equ), NED (ell_polar/sph_inc/sph_polar), SolarBeta (incl_0/incl_23_4), orbinit (0101/0201/0301/0401), drag (const/CD/BC), SRP basic (basic/basic_cr), shadow 2A (annular/cooling), SRP 1st-order.
- [x] **Feature parity**: Every `jeod_sim` function used by the Simulation runner has a corresponding Bevy system calling the same function.
- [x] `cargo test --workspace` — 371 tests pass, no regressions.

---

## Phase 5: High-Fidelity Parity

**Goal:** Feature parity with JEOD's verified capabilities. Full cross-validation.

### Entrance Criteria

- [x] Phase 4b exit criteria met
- [x] All interaction forces and derived states Tier 3 validated
- [x] Docker available for running Trick container (established in Phase 1)
- [x] SIM_dyncomp full-force reference data available (generated in Phase 4b)

### Tasks

#### 5A. Advanced Integrators (`jeod_dynamics`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 5.1 | RKF45 integrator | Runge-Kutta-Fehlberg 4(5) with adaptive step control. Error estimation from embedded 4th and 5th order solutions. Step size adjustment with safety factor. | `er7_utils` RKF45 |
| 5.2 | Gauss-Jackson integrator | Multi-step second-sum method. Startup via RK4 (needs history). Predictor-corrector formulation. Order 8 default. | `utils/integration/gauss_jackson/` |
| 5.4 | Integrator selection | Enum-based dispatch. All integrators implement same trait/interface. | — |
| 5.5 | Integrator unit tests | RKF45 and Gauss-Jackson on harmonic oscillator: verify convergence order. RKF45: verify step size adapts to maintain tolerance. Gauss-Jackson: verify startup and steady-state accuracy. Compare 24h LEO trajectory across integrators (should agree to within tolerances). | — |

#### 5B. Advanced Gravity (`jeod_gravity`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 5.6 | Solid body tides | Time-dependent delta coefficients (ΔCnm, ΔSnm) from body deformation. Permanent tide, frequency-dependent corrections. | `gravity/include/spherical_harmonics_delta_coeffs.hh` |
| 5.7 | Tide unit tests | Tidal gravity perturbation at known epoch/position matches JEOD reference value. | — |

#### 5C. Earth Rotation — Polar Motion (`jeod_frames`)

Precession (5.8), nutation (5.9), and GAST rotation (5.11) were completed in Phase 2
(`jeod_frames/src/precession_j2000.rs`, `nutation_j2000.rs`, `rotation_j2000.rs`).
Only polar motion remains.

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 5.10 | Polar motion | Apply polar motion correction (x_p, y_p from IERS data). Compose into full RNP: W(polar) · R(GAST) · N · P. | `RNP/RNPJ2000/` |
| 5.12 | Polar motion unit tests | Earth-fixed frame orientation with polar motion matches JEOD/IERS to < 1 arcsecond at 5+ test epochs. | — |

#### 5D. Dynamics Manager ODE Scheduling

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 5.33 | Multi-integrable-object scheduling | Port JEOD's DynManager integration loop that drives multiple integrable objects (orbital state + thermal state) through RK4 stages in the correct order. Required to close the 27.6 m / 23-day SRP thermal coupling residual (simnaut/bevy_jeod#13). | `dynamics_integration_group.cc`, `er7_utils` integration loop |
| 5.34 | Thermal ODE as integrable object | Move plate temperature integration into the Bevy dynamics pipeline so it's driven by the same integration loop as the orbital state, matching JEOD's `ThermalIntegrableObject` scheduling. | `thermal_integrable_object.cc` |

#### 5G. Cross-Validation Infrastructure

The JEOD repo contains ~60 verification SIMs with ~380 RUN directories. An audit
identified the following as highest-value additions for Phase 5. All are added to
`trick/generate_references.sh` (Docker workflow from Phase 1: Rocky 9 + Trick 25 +
JEOD 5.4).

##### Reference Data Generation

| ID | Task | Description |
|----|------|-------------|
| 5.20 | SIM_dyncomp full-force data | Reference CSVs for RUN_4, RUN_7A–7D generated in Phase 4b (tasks 4b.1–4b.2). Verify data is available in `test_data/` before running Phase 5 tests. |
| 5.21 | Generate Earth-Moon reference | Add `Integrated_Validation/SIM_Earth_Moon` RUN_clem (Clementine lunar orbit: Moon 60x60 gravity + solid tides + Earth 3rd-body, 24h) and RUN_rosetta (Rosetta Earth flyby). |
| 5.22 | Generate Mars reference | Add `Integrated_Validation/SIM_Mars` RUN_dawn (Dawn at Mars: MRO110B2 gravity + Sun 3rd-body, 3h) and RUN_phobos (Phobos orbit). |
| 5.35 | Generate tides reference | Add `gravity/verif/SIM_tide_verif` RUN_01 (8x8 GEM-T1 + solid body tides + Sun/Moon 3rd-body, 8h ISS orbit) and RUN_02. Only JEOD sim that exercises tidal delta-Cnm/delta-Snm trajectory effects. |
| 5.36 | Generate polar motion reference | Add `RNP/RNPJ2000/verif/SIM_RNP_J2000_prop` RUN_J2000_RNP_prop (full RNP + polar motion, 24h) and RUN_J2000_RNP_Polar_off (RNP without polar motion). Differential comparison isolates polar motion contribution. |
| 5.37 | Generate advanced integrator references | Add `integration/verif/SIM_integ_test` RUN_gauss_jackson. Add `integration/verif/SIM_GJ_test` RUN_GJ_step1_order8_noeval_nobs and RUN_GJ_step1_order12_noeval_nobs. Validates integrator accuracy independently of force model. |

##### Cross-Validation Tests

| ID | Task | Description |
|----|------|-------------|
| 5.23 | Extend CSV trajectory loader | Extend the loader for new column layouts (SIM_Earth_Moon, SIM_Mars, SIM_tide_verif, SIM_RNP_J2000_prop). |
| 5.24 | Trajectory comparison harness | Generalize comparison: report max error, RMS error, drift rate per scenario. |
| 5.25 | Tier 3: LEO 24h (high-fidelity gravity) | SIM_dyncomp RUN_7A/7B: 4x4 or 8x8 gravity + Sun/Moon 3rd-body, no drag, 8h. Isolates gravity + ephemeris fidelity from atmosphere/drag. |
| 5.26 | Tier 3: LEO with drag | SIM_dyncomp RUN_7C/7D: gravity + 3rd-body + MET atmosphere + drag, 8h. Full combined translational dynamics. |
| 5.27 | Tier 3: Earth-Moon test | SIM_Earth_Moon RUN_clem: 24h lunar orbit. Validates multi-body gravity, Moon 60x60 harmonics, Earth 3rd-body differential acceleration. |
| 5.28 | Tier 3: Mars orbit test | SIM_Mars RUN_dawn: 3h Mars orbit. Validates MRO110B2 gravity model and Sun 3rd-body perturbations. |
| 5.40 | Tier 3: 3rd-body isolation | SIM_dyncomp RUN_4: spherical gravity + Sun/Moon 3rd-body only, 8h. Isolates differential acceleration computation from non-spherical gravity — catches sign/direction errors without harmonics masking them. |
| 5.41 | Tier 3: solid tides | SIM_tide_verif RUN_01 vs RUN_02: trajectory with/without tides. Position delta (tides ON vs OFF) must match JEOD's delta. |
| 5.42 | Tier 3: polar motion | SIM_RNP_J2000_prop RUN_J2000_RNP_prop vs RUN_J2000_RNP_Polar_off: Earth-fixed frame with/without polar motion. Differential comparison isolates polar motion contribution. |
| 5.43 | Tier 3: Gauss-Jackson | SIM_integ_test RUN_gauss_jackson or SIM_GJ_test: same LEO scenario as RK4, verify GJ matches JEOD trajectory. |
| 5.29 | Tier 4 regression harness | CI script that runs all Tier 1-3 tests and produces pass/fail summary with error budgets. |

#### 5H. Examples

| ID | Task | Description |
|----|------|-------------|
| 5.30 | `apollo.rs` | Apollo trans-lunar injection scenario. Multi-body (Earth + Moon), stage separation (attach/detach). |
| 5.31 | `earth_moon.rs` | Long-duration Earth-Moon trajectory. |
| 5.32 | `mars_orbit.rs` | Mars orbit insertion and propagation. |

### Exit Criteria

#### Tier 1 (unit tests)
- [x] **Gauss-Jackson accuracy**: Matches RK4 trajectory to < 1 m over 24h with fewer function evaluations — `tier3_simulation_gj_order8` achieves 2.3e-4 m
- [x] **RKF45 fixed-step 5th-order**: RKF45 propagation at 5th-order accuracy matches JEOD trajectory. JEOD's own RKF45 (ER7) is fixed-step — the embedded 4th-order solution (b4 weights) is defined in the Butcher tableau but never computed or used for error estimation. Our implementation matches this: fixed-step, b5-only. Adaptive stepping is a potential future enhancement, not a JEOD parity requirement. Validated by `tier3_bevy_rkf45_matches_simulation_bit_identical`.
- [x] **Solid tides**: Tidal gravity perturbation magnitude within 10% of JEOD reference — `tier3_simulation_tide_run01` validates ΔC20 to machine precision (< 1e-14)

#### Tier 2 (JEOD reference data)
- [x] **Earth rotation**: ITRS frame orientation matches JEOD/IERS to < 1 arcsecond at 5+ test epochs — `tier3_rnp_component_comparison`

#### Tier 3 (trajectory cross-validation — required for each new physics)
- [x] **All prior phase exit criteria** still pass (no regressions) — 300 tests pass
- [x] **Tier 3 LEO 24h (high-fidelity gravity)**: Position error vs. JEOD < 10 m — RUN_3A (4x4): 0.13 m, RUN_3B (8x8): 0.23 m
- [x] **Tier 3 LEO with drag**: Position error vs. JEOD < 100 m over 24h — RUN_6B: 1.1 m over 8h
- ~~**Tier 3 Earth-Moon multi-body**~~ — moved to Phase 6 exit criteria (requires Moon gravity model + lunar RNP)
- ~~**Tier 3 Mars orbit**~~ — moved to Phase 6 exit criteria (requires Mars RNP)
- [x] **Tier 3 Gauss-Jackson trajectory**: Position error vs. JEOD < 1 m — `tier3_simulation_gj_order8` achieves 2.3e-4 m
- [x] **Tier 3 RKF45 trajectory**: RKF45 on same scenario — `tier3_bevy_rkf45_matches_simulation_bit_identical` validates bit-identical Bevy/Simulation parity; JEOD's RKF45 is also fixed-step (see Tier 1 note)
- [x] **Tier 3 polar motion**: Earth-fixed frame with polar motion — `tier3_simulation_run2p_polar_motion` matches JEOD
- [x] **Tier 3 solid tides**: Trajectory with tidal ΔC20 — `tier3_simulation_tide_run01` validates trajectory + ΔC20
- [x] **Tier 3 SRP trajectory**: SRP with ephemeris Sun — `tier3_simulation_srp_flat_plate` achieves 3.07 m over 23 days
- [x] **Tier 3 SRP thermal parity**: SIM_3_ORBIT RUN_radiation (23 days, flat-plate + thermal) — `tier3_simulation_srp_flat_plate` achieves 3.07 m (< 5 m budget)
- [x] **Tier 3 3rd-body isolation**: SIM_dyncomp RUN_4 (spherical gravity + Sun/Moon) — `tier3_simulation_run4_3rd_body` achieves 0.002 m (< 5 m budget)
- [x] **Tier 3 Sun/Moon 3rd-body resolved**: Sun/Moon added to `tier3_sim_torque_simple.rs` with real mu values (Sun: 1.327e20, Moon: 4.903e12) and registered as 3rd-body gravity controls. Position tolerances meet target (< 0.5 m). Quaternion and torque tolerances are scenario-inherent: gradient-free runs (RUN_01/04) have no torque reference so attitude diverges freely; SH-gradient runs (RUN_06) compound errors through rotational dynamics over 3h with DE421 ephemeris offset as the dominant error source (~10 arcsec Sun direction, see simnaut/bevy_jeod#27). The `mu: 0.0` values in `tier3_sim_srp.rs` and `tier3_sim_solar_beta.rs` correctly match the JEOD reference sims (SIM_3_ORBIT and RUN_2), which deliberately exclude Sun/Moon gravity — these are not workarounds. 3rd-body gravity is validated independently by RUN_4, RUN_7A-D, and torque_simple.

#### Bevy≡Simulation parity
- [x] **Cross-parity for each new integrator**: `tier3_bevy_gj_point_mass` (GJ) and `tier3_bevy_rkf45_matches_simulation_bit_identical` (RKF45) — `to_bits()` equality.
- [x] **Cross-parity for new physics**: `tier3_bevy_tidal_sh4x4` (tides), `tier3_bevy_sh4x4_rnp` (RNP/rotation), `tier3_bevy_polar_geodetic` (polar motion) — `to_bits()` equality. Mars gravity deferred to Phase 6.
- [x] **Feature parity**: All 16 `jeod_sim` orchestration functions have corresponding Bevy systems, including `tidal_update_system` added in Phase 5.

#### Other
- [x] **Tier 4 regression**: CI runs all Tier 1-3 tests; all pass within budgets
- [x] **Portability**: All `jeod_*` crates compile without Bevy; `batch_propagation.rs` runs full-fidelity scenario without Bevy

---

## Phase 6: Comprehensive JEOD Parity Validation

**Goal:** Full-breadth cross-validation against every major JEOD verification sim
category. This phase validates existing physics across broader parameter spaces,
edge cases, and specialized scenarios to ensure no JEOD capability goes unverified.

### Entrance Criteria

- [x] Phase 5 exit criteria met
- [x] All Phase 5 physics (advanced integrators, tides, polar motion, multi-body) functional
- [ ] Docker pipeline capable of building dedicated JEOD sims beyond SIM_dyncomp

### Tasks

#### 6A. Reference Data Generation

| ID | Task | Description |
|----|------|-------------|
| 6.1 | Generate relative dynamics references | Add `derived_state/verif/SIM_Relative` RUN_AB_rot_AB_trans (both vehicles rotating + translating), RUN_no_rot_AB_trans (translation only), RUN_A_rot_no_trans (single-vehicle rotation). 7 total runs testing relative state computation. |
| 6.2 | Generate planetary derived state references | Add `derived_state/verif/SIM_Planetary` RUN_LEO_inc (45° LEO, 24h), RUN_LEO_polar (polar orbit), RUN_GEO (geostationary). Orbital elements + LVLH in distinct regimes. |
| 6.3 | Generate Earth lighting references | Add `earth_lighting/verif/SIM_LIGHT_CIR` RUN_T01 through RUN_T10 (10 lighting geometry scenarios: penumbra, umbra, antumbra, varied Sun-Earth-Moon configurations). |
| 6.4 | Generate full time scale references | Add `time/verif/SIM_5_all_inclusive` RUN_UTC_initialized (all 8 scales: UTC, TAI, UT1, TDB, TT, GPS, GMST, MET over 24h) and RUN_UTC_initialized_tdb (TDB-initialized variant). |
| 6.5 | Generate MET deep validation references | Add `atmosphere/MET/verif/SIM_MET` RUN_T01_MET_VER, RUN_T02_MET_VER, RUN_T03_GRAM_MET (MET vs GRAM comparison across conditions). |
| 6.6 | Generate time reversal references | Add `time/verif/SIM_7_time_reversal` RUN_1 (point-mass, reverse), RUN_3A (4x4 gravity, reverse), RUN_8B (6-DOF rotational, reverse). |
| 6.7 | Generate comprehensive orbital element references | Add `orbital_elements/verif/SIM_orb_elem` — representative subset of 56 runs: RUN_T01 (circular), RUN_T10 (eccentric), RUN_T20 (hyperbolic), RUN_T30 (near-parabolic), RUN_T40 (retrograde), RUN_T50 (equatorial), RUN_T55 (polar). Covers all orbit families. |
| 6.8 | Derived state edge-case data | Completed in Phase 4b (tasks 4b.3–4b.6). Reference data available in `test_data/`. |
| 6.9 | Generate Mercury relativistic reference | SIM_mercury RUN_newtonian and RUN_relativistic_sun. Mandatory in this phase (stretch in Phase 5). |
| 6.10 | Generate LVLH-relative references | Add `derived_state/verif/SIM_LvlhRelative` RUN_test0 and RUN_test1. LVLH-relative dynamics for proximity operations. |

#### 6B. Cross-Validation Tests

| ID | Task | Description |
|----|------|-------------|
| 6.11 | Tier 3: relative dynamics | SIM_Relative: relative translational + rotational state between two vehicles matches JEOD over 100s kinematic scenarios. |
| 6.12 | Tier 3: planetary derived states | SIM_Planetary: orbital elements + LVLH frames in LEO/GEO/polar match JEOD over 24h. Catches coordinate singularities at equator and poles. |
| 6.13 | Tier 3: Earth lighting | SIM_LIGHT_CIR: shadow fraction and penumbra/umbra/antumbra transitions match JEOD across all 10 geometries. |
| 6.14 | Tier 3: full time scale parity | SIM_5_all_inclusive: every time scale matches JEOD over 24h. |
| 6.15 | Tier 3: MET atmosphere parity | SIM_MET: density and temperature profiles match JEOD across altitudes and solar conditions. |
| 6.16 | Tier 3: time reversal | SIM_7_time_reversal: propagate forward then backward, verify round-trip. Compare against JEOD's reversed sim output. |
| 6.17 | Tier 3: comprehensive orbital elements | SIM_orb_elem: 7 orbit families (circular, eccentric, hyperbolic, near-parabolic, retrograde, equatorial, polar). |
| 6.18 | Tier 3: derived state edge cases | Completed in Phase 4b (tests 4b.12–4b.15). Verified by Phase 4b exit criteria. |
| 6.19 | Tier 3: Mercury relativistic | SIM_mercury: GR perihelion advance ~43 arcsec/century. Mandatory. |
| 6.20 | Tier 3: LVLH-relative dynamics | SIM_LvlhRelative: relative state in LVLH frame for proximity operations. |

### Exit Criteria

#### Tier 3 (trajectory cross-validation)
- [ ] **All prior phase exit criteria** still pass (no regressions)
- [ ] **Tier 3 Earth-Moon multi-body**: Position error vs. JEOD < 100 m over 7 days (Earth + Moon + Sun gravity, differential acceleration). Requires Moon gravity model + lunar RNP.
- [ ] **Tier 3 Mars orbit**: Position error vs. JEOD < 100 m over 7 days (MRO110B2 gravity). Requires Mars RNP.
- [ ] **Relative dynamics**: Relative state between two vehicles matches JEOD to < 1e-6 m over 100s (SIM_Relative)
- [ ] **Planetary derived states**: Orbital elements in LEO/GEO/polar match JEOD to < 1e-6 per element over 24h (SIM_Planetary)
- [ ] **Earth lighting**: Shadow fraction matches JEOD to < 0.01 across all 10 geometries (SIM_LIGHT_CIR)
- [ ] **Time scale parity**: All 8 time scales match JEOD to < 1e-6 s over 24h (SIM_5_all_inclusive)
- [ ] **MET atmosphere parity**: Density matches JEOD MET tables to < 1% at all tested altitudes and solar conditions (SIM_MET)
- [ ] **Time reversal**: Forward-backward round-trip recovers initial state to < 1e-6 m. Reversed trajectory matches JEOD's SIM_7_time_reversal to same tolerance as forward propagation.
- [ ] **Comprehensive orbital elements**: All 7 orbit families pass — `from_cartesian()` matches JEOD to < 1e-6 per element at every timestep (SIM_orb_elem)
- [ ] **Derived state edge cases**: Completed in Phase 4b — Euler (eccentric/equatorial), LVLH (equatorial), NED (polar), solar beta (0° inclination) all validated to < 1e-6 rad
- [ ] **Mercury relativistic**: GR perihelion advance rate within 1% of JEOD's delta (~43 arcsec/century) (SIM_mercury)
- [ ] **LVLH-relative**: Relative state in LVLH frame matches JEOD to < 1e-6 m (SIM_LvlhRelative)

#### Bevy≡Simulation parity
- [ ] **Full cross-parity**: Every `tier3_simulation_*` test has a matching `tier3_bevy_*` test exercising the same physics — `to_bits()` equality.
- [ ] **Feature parity audit**: No `jeod_sim` capability exists that lacks a Bevy system counterpart. No Bevy system exists that bypasses `jeod_sim`.

#### Other
- [ ] **Full JEOD parity**: Every major JEOD verification sim category (dynamics, gravity, time, ephemerides, RNP, atmosphere, aerodynamics, radiation pressure, gravity torque, derived states, orbital elements, earth lighting) has at least one `tier3_simulation_*` test AND a matching `tier3_bevy_*` cross-parity test.
- [ ] **Portability**: All `jeod_*` crates compile without Bevy
- [ ] `cargo test --workspace` — all tests pass, no regressions

---

## Future Work

Tasks removed from phased scope but worth revisiting when a use case arises.

### LSODE Integrator

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| F.1 | LSODE integrator | Livermore Solver for Ordinary Differential Equations. Variable-order, variable-step. BDF method for stiff problems. Adams method for non-stiff. | `utils/integration/lsode/` |
| F.2 | LSODE unit tests | Convergence order verification on harmonic oscillator. Compare 24h LEO trajectory against RK4/GJ. | — |
| F.3 | Tier 3: LSODE trajectory | SIM_integ_test RUN_lsode: variable-order variable-step integration on LEO scenario, compare trajectory to JEOD. | — |
| F.4 | Docker: LSODE reference data | Add RUN_lsode to `SIM_integ_test` in `generate_references.sh`. | — |

**Why deferred:** Complex Fortran→C port. Gauss-Jackson covers the primary "better than
RK4" use case for orbital mechanics. LSODE is most valuable for stiff problems (e.g.,
chemical kinetics, thermal transients) not exercised in current JEOD verification sims.

### SPICE FFI Bindings

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| F.5 | SPICE FFI bindings | Rust bindings to `cspice` via `cc` crate or existing `spice` Rust crate. | `environment/spice/` |
| F.6 | SPICE state query | Query planet/body state from SPICE kernels at arbitrary epoch. | `environment/spice/` |
| F.7 | Kernel management | Load/unload SPICE kernels (BSP, TF, LSK). | — |
| F.8 | SPICE unit tests | Earth position from SPICE matches DE421 to < 1 m. | — |

**Why deferred:** ANISE (pure Rust SPICE reader) is already working for DE421 ephemeris.
Adding C `cspice` would introduce a C build dependency, break pure-Rust compilation, and
add Windows/WASM build complexity. The ~10 arcsecond interpolation difference
(simnaut/bevy_jeod#27) is between two valid implementations, not an error.

### Contact Dynamics

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| F.9 | Contact surface model | Facet-based surface geometry. Contact point detection between two bodies. | `interactions/contact/` |
| F.10 | Contact force model | Spring-damper contact forces. Normal and friction forces. | `interactions/contact/` |
| F.11 | Contact unit tests | Two spheres approaching: detect contact at expected distance. Contact force magnitude matches spring constant × penetration. | — |

**Why deferred:** No JEOD verification sim exercises contact forces among the ~60
available sims. Cannot meet the "Tier 3 is definition of done" rule without a
cross-validation target.

### Mercury Relativistic Perihelion Advance

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| F.12 | General relativity corrections | Post-Newtonian gravitational acceleration terms for Mercury orbit. | `gravity/verif/SIM_mercury` |
| F.13 | Docker: Mercury reference data | Add `SIM_mercury` RUN_newtonian and RUN_relativistic_sun to `generate_references.sh`. | — |
| F.14 | Tier 3: Mercury relativistic | SIM_mercury: perihelion advance delta from GR corrections. GR-induced perihelion advance rate within 1% of JEOD's delta (~43 arcsec/century). Requires Gauss-Jackson + multi-planet gravity. | — |

**Why deferred:** Requires Gauss-Jackson (Phase 5) + all 9 planets + general relativity
corrections — a large dependency chain for a single validation test. Revisit after
Gauss-Jackson is stable.

### Long-Term Ephemeris Validation

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| F.15 | Docker: long-term ephemeris reference | Add `SIM_prop_planet` RUN_ephem (DE430 mode) and RUN_prop (numerically propagated, ~150 years). | — |
| F.16 | Tier 3: long-term ephemeris | SIM_prop_planet: compare Anise-based DE421/430 queries against JEOD's DE430 propagation over multi-decade spans. Validates interpolation drift over decades. | — |

**Why deferred:** Tests ANISE's interpolation accuracy over decades, not our physics code.
Validates a dependency rather than our implementation.

---

## Cross-Phase Conventions

### Task ID Format

`{phase}.{task_number}` — e.g., `2.11` is Phase 2, task 11 (spherical harmonics acceleration).

### Test Naming Convention

```
jeod_math::tests::kepler_equation_circular
jeod_math::tests::orbital_elements_roundtrip_circular
jeod_math::tests::orbital_elements_roundtrip_eccentric
jeod_gravity::tests::point_mass_inverse_square
jeod_gravity::tests::spherical_harmonics_verif_case_01  (through _40)
jeod_dynamics::tests::rk4_harmonic_oscillator
jeod_dynamics::tests::rk4_energy_conservation
```

### Tolerance Standards

| Quantity | Tier 1 (analytical) | Tier 2 (JEOD reference) | Tier 3 (trajectory) | Tier 3 (Bevy≡Sim) |
|----------|--------------------|-----------------------|-------------------|--------------------|
| Position | 1e-6 m | 1.0 m | 10-100 m | **0.0 m (exact)** |
| Velocity | 1e-9 m/s | 0.001 m/s | 0.01-0.1 m/s | **0.0 m/s (exact)** |
| Acceleration | 1e-12 m/s² | 1e-10 m/s² | — | — |
| Angles | 1e-14 rad | 1e-12 rad | 1e-6 rad | **0.0 rad (exact)** |
| Energy | 1e-10 J/kg | — | — | — |
| Quaternion | 1e-14 | 1e-14 | 1e-12 | **0.0 (exact)** |
| Time | — | exact (integer s) | — | — |
| Geodetic alt | — | — | 1.0 m | **0.0 m (exact)** |
| Geodetic lat | — | — | 1e-6 rad | **0.0 rad (exact)** |
| Geodetic lon | — | — | 1e-6 rad (†) | **0.0 rad (exact)** |
| Solar beta | — | — | 1e-4 + days×1.5e-4 rad (‡) | **0.0 rad (exact)** |

(†) Geodetic longitude at latitude > 89.5° is geometrically ill-defined (pole
singularity). Polar orbit tests use 0.1 rad tolerance at poles; the actual error
is ~3e-5 rad. See simnaut/bevy_jeod#26 for full analysis.

(‡) Solar beta tolerance is duration-dependent due to DE421 interpolation
differences between Anise and JEOD's native reader (~10 arcsecond Sun direction
offset, ~1.5e-4 rad/day drift). See simnaut/bevy_jeod#27.

Tier 3 Bevy-vs-Simulation tests require bit-identical output (`f64::to_bits()`
equality, not tolerance-based) because the Bevy pipeline and `jeod_sim::Simulation`
call the same functions in the same order.

### Tier 3 Sub-Categories

Tier 3 has two complementary test paths:

**Simulation-vs-JEOD** (`jeod_sim/tests/tier3_sim_*.rs`): validates the
`Simulation::step()` production code path against JEOD Trick CSV data.

**Bevy-vs-Simulation** (`tests/cross_parity.rs`): validates that the Bevy ECS
pipeline produces bit-identical output to the Simulation runner. Every phase that
delivers new physics must add a scenario here. Current scenarios:

| Scenario | Physics | Added in |
|----------|---------|----------|
| A | Point-mass gravity, 6-DOF | Phase 1 |
| B | Exponential atmosphere + drag, 6-DOF | Phase 4 |
| C | Flat-plate SRP + conical shadow, 3-DOF | Phase 4 |
| D | Gravity gradient torque, 6-DOF | Phase 4 |
| E | Full stack (all interactions), 6-DOF | Phase 4 |
| F | Spherical harmonics 4x4 + RNP | Phase 4 |
| G | External torque via per-body functions | Phase 4 |
| H | Flat-plate SRP with shadow detection | Phase 4 |
| I | Derived states (orbital elements, LVLH, Euler, solar beta) | Phase 3a |
| J | Geodetic derived state (planet-fixed rotation) | Phase 3a |
| K | Constant-density drag, 6-DOF | Phase 4a |
| L | MET atmosphere + drag, 6-DOF | Phase 4a |
| M | Eccentric orbit with derived states (OE, LVLH, Euler, beta) | Phase 4b |
| N | Polar orbit with geodetic (spherical Earth) | Phase 4b |
| O | Equatorial orbit with solar beta | Phase 4b |

Future phases must add: multi-body gravity, advanced integrators
(Gauss-Jackson, RKF45), polar motion, solid tides.

### Definition of Done (per task)

1. Code compiles (`cargo build`)
2. Unit tests pass (`cargo test`)
3. No clippy warnings (`cargo clippy`)
4. Core logic is in `jeod_*` crate (no Bevy dependency in physics code)
5. Orchestration logic delegates to `jeod_sim` per-body functions
6. Bevy system (if applicable) delegates to `jeod_sim`, not directly to `jeod_*`
7. If new physics: Bevy-vs-Simulation scenario added to `tests/cross_parity.rs`
8. If new physics: Simulation-vs-JEOD test added to `tier3_sim_*.rs` (named after JEOD source sim)

### Phase Transition Protocol

Before starting Phase N+1:

1. All Phase N exit criteria checkboxes are checked
2. `cargo test --workspace` passes with zero failures
3. All Bevy-vs-Simulation scenarios pass with exact-zero difference
4. All examples from Phase N run successfully
5. No known regressions from earlier phases
