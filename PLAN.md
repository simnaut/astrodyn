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
| 0.3 | Create Bevy glue crate skeletons | `bevy_jeod_dynamics`, `bevy_jeod_gravity`, `bevy_jeod_frames` | Add `bevy` dependency, depend on corresponding `jeod_*` crate |
| 0.4 | Create test data crate skeleton | `jeod_test_data` | No Bevy dependency. Add `JEOD_PATH` env var support |
| 0.5 | Create top-level lib crate | `src/lib.rs` | Re-export all `bevy_jeod_*` plugins as `JeodPlugin` |
| 0.6 | Add shared dependencies | workspace `Cargo.toml` | `glam` (f64 features), `nalgebra` (optional), `thiserror`, `regex` (test_data) |
| 0.7 | Set up CI configuration | `.github/workflows/` or equivalent | `cargo build --workspace`, `cargo test --workspace`, `cargo clippy`, `cargo fmt --check` |
| 0.8 | Create `.env.example` | root | Document `JEOD_PATH=../jeod` |
| 0.9 | Add `STRATEGY.md` and `PLAN.md` to repo | root | Already exist |

### Exit Criteria

- [ ] `cargo build --workspace` succeeds with zero errors
- [ ] `cargo test --workspace` runs (0 tests, 0 failures)
- [ ] `cargo clippy --workspace` produces no warnings
- [ ] Each `jeod_*` crate compiles with **zero** Bevy dependency
- [ ] Each `bevy_jeod_*` crate depends on its corresponding `jeod_*` crate and on `bevy`
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

#### 1E. Bevy Glue (`bevy_jeod_*`)

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
| 2.34 | Earth preset | GM = 3.986004418e14, r_eq = 6378137.0, f = 1/298.257223563 (WGS84). | `planet/data/` |
| 2.35 | Moon preset | GM, radius, shape parameters. | `planet/data/` |
| 2.36 | Sun preset | GM, radius (point mass only). | `planet/data/` |
| 2.37 | Mars preset | GM, radius, flattening. | `planet/data/` |

#### 2F. Bevy Glue

| ID | Task | Description |
|----|------|-------------|
| 2.38 | `bevy_jeod_time` plugin | `SimulationTime` as `Resource`. `time_advance_system` in `TimeUpdateSet`. |
| 2.39 | `bevy_jeod_gravity` update | Replace point-mass system with spherical harmonics. Load coefficients via `AssetServer` or embed. |
| 2.40 | `bevy_jeod_ephemeris` plugin | `EphemerisData` resource. `ephemeris_update_system` in `EphemerisUpdateSet`. Updates planet frame positions each step. |
| 2.41 | `bevy_jeod_planet` plugin | `Planet` marker component. Preset spawning functions (`spawn_earth()`, etc.). |
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
| 3.35 | `bevy_jeod_derived` plugin | Components + systems for OrbitalElements, EulerAngles, PlanetFixedPosition, LvlhState, SolarBeta. Each system calls corresponding `jeod_math` pure function. |
| 3.36 | `iss_orbit.rs` example | ISS initialized from orbital elements, full GGM05C gravity, 6-DOF, display orbital elements and attitude. |

### Exit Criteria

- [ ] **ISS reference state**: Position error < 1 m, velocity error < 0.001 m/s vs. JEOD reference data (`reference_inertial_trans_state.py`)
- [ ] **Euler angles**: 6/6 test vectors from `euler_derived_state_ut.cc` pass within 1e-12 rad
- [ ] **Quaternion stability**: Unit norm maintained to < 1e-14 over 86400s propagation (no renormalization)
- [ ] **Torque-free precession**: Symmetric body precession rate matches analytical `ω_p` to < 0.1%
- [ ] **Composite mass**: Two-body attachment composite inertia matches parallel axis theorem to < 1e-10 kg·m²
- [ ] **Attach/detach**: Round-trip preserves total angular momentum to < 1e-10 N·m·s
- [ ] **Geodetic conversion**: Round-trip (cartesian → geodetic → cartesian) error < 1e-6 m for 10+ test points
- [ ] **Frame tree**: Relative state between any two frames matches direct computation to < 1e-14
- [ ] **Portability**: All `jeod_*` Phase 3 additions compile without Bevy
- [ ] `cargo test --workspace` — all tests pass

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

- [ ] **Planet-fixed gravity**: Spherical harmonics Tier 3 (RUN_3A) position error < 5 m over 8h (down from 15.6 m with identity placeholder)
- [ ] **Frame propagation**: Structure and core_body frame positions from `propagate_forward/reverse` match JEOD CSV columns to < 1e-6 m at each timestep over 8h (RUN_2)
- [ ] **Orbital elements trajectory**: Our `from_cartesian()` matches JEOD `SIM_OrbElem` logged elements to < 1e-6 on each element over 1+ orbits
- [ ] **LVLH frame trajectory**: Our `compute_lvlh_frame()` T_parent_this matches JEOD `SIM_LVLH` logged LVLH frame to < 1e-6 rad over 1+ orbits
- [ ] **Geodetic + NED trajectory**: Our geodetic conversion matches JEOD `SIM_NED` logged ellipsoidal coordinates to < 1e-6 m altitude, < 1e-10 rad lat/lon over 1+ orbits
- [ ] **Solar beta trajectory**: Our `solar_beta_angle()` matches JEOD `SIM_SolarBeta` logged beta to < 1e-4 rad over 24h (ISS-like orbit with Sun/Moon)
- [ ] **Euler angle trajectory**: Our `compute_euler_angles_from_matrix()` matches JEOD `SIM_Euler` logged angles to < 1e-6 rad over 24h
- [ ] **Body init from elements**: `init_from_orbital_elements()` for ISS produces position < 1 m, velocity < 0.001 m/s vs JEOD reference state
- [ ] **Bevy system parity**: Bevy App propagation matches pure `rk4_sixdof_step()` to < 1e-8 m position, < 1e-11 m/s velocity, < 1e-14 quaternion/ω over 100 steps
- [ ] `cargo test --workspace` — all tests pass, no regressions

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
| 4.1 | Atmosphere trait | `fn density(&self, alt: f64, lat: f64, lon: f64, time: &SimulationTime) → AtmosphericState`. Where `AtmosphericState = { density, temperature, pressure, wind: DVec3 }`. | `atmosphere/base_atmos/` |
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
| 4.16 | `bevy_jeod_atmosphere` plugin | `AtmosphericState` component. `atmosphere_update_system` in `EnvironmentSet`: query body position, compute geodetic coords, call `Atmosphere::density()`. |
| 4.17 | Aerodynamic force system | `AerodynamicForce` component. `aero_drag_system` in `InteractionSet`. |
| 4.18 | Radiation pressure system | `RadiationForce` component. `radiation_pressure_system` in `InteractionSet`. Reads Sun entity position. |
| 4.19 | Gravity torque system | `GravityTorque` component. `gravity_torque_system` in `InteractionSet`. |
| 4.20 | Update force collection | Add `Option<&AerodynamicForce>`, `Option<&RadiationForce>`, `Option<&GravityTorque>` to `force_collection_system` query. |
| 4.21 | SolarBeta system | `SolarBeta` component + `solar_beta_system` in `DerivedStateSet`. |
| 4.22 | LEO drag example | Bevy example: 400 km orbit with drag, show altitude decay over time. |

### Exit Criteria

#### Tier 1 (unit tests)
- [ ] **Drag order-of-magnitude**: ISS-like vehicle (Cd·A/m ≈ 0.01 m²/kg) at 400 km loses ~100-300 m/day altitude (matches empirical expectation)
- [ ] **SRP magnitude**: Radiation pressure at 1 AU = 4.56 ± 0.01 μN/m²
- [ ] **SRP direction**: Force vector is anti-Sun to < 0.001°
- [ ] **Shadow detection**: Body at known position behind Earth correctly returns shadow fraction = 0; body 90° away returns 1.0
- [ ] **Gravity torque symmetry**: Torque on spherically symmetric body < 1e-20 N·m
- [ ] **Gravity torque magnitude**: Asymmetric body at known orientation matches analytical `τ = 3μΔI sin(2θ) / 2r³` to < 1%

#### Tier 2 (JEOD reference data)
- [ ] **MET atmosphere**: Density at 400 km matches JEOD's MET tables to < 5% for solar min, mean, and max conditions

#### Tier 3 (trajectory cross-validation — required for each new physics)
- [ ] **Tier 3 gravity torque**: 6-DOF trajectory with gravity gradient torque enabled. Compare attitude evolution against JEOD SIM_dyncomp RUN_9A/9B (ISS inertia, applied torque + gravity gradient). Quaternion error < 0.01 rad over 8h.
- [ ] **Tier 3 drag trajectory**: LEO trajectory with MET atmosphere + ballistic drag. Compare position against JEOD SIM_dyncomp with drag enabled (RUN_5A or equivalent). Position error < 100 m over 24h.
- [ ] **Tier 3 SRP trajectory**: Trajectory with solar radiation pressure. Compare against JEOD sim with SRP enabled. Position error < 10 m over 24h.
- [ ] **Tier 3 shadow transitions**: Eclipse entry/exit times match JEOD logged shadow state to < 10 s over multiple orbits.

#### Other
- [ ] **Portability**: All `jeod_*` Phase 4 additions compile without Bevy
- [ ] `cargo test --workspace` — all tests pass

---

## Phase 5: High-Fidelity Parity

**Goal:** Feature parity with JEOD's verified capabilities. Full cross-validation.

### Entrance Criteria

- [ ] Phase 4 exit criteria met
- [ ] All basic forces (gravity, drag, SRP, gravity torque) producing correct results
- [ ] Docker available for running Trick container (established in Phase 1)
- [ ] Additional JEOD reference trajectories generated for Phase 5 scenarios

### Tasks

#### 5A. Advanced Integrators (`jeod_dynamics`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 5.1 | RKF45 integrator | Runge-Kutta-Fehlberg 4(5) with adaptive step control. Error estimation from embedded 4th and 5th order solutions. Step size adjustment with safety factor. | `er7_utils` RKF45 |
| 5.2 | Gauss-Jackson integrator | Multi-step second-sum method. Startup via RK4 (needs history). Predictor-corrector formulation. Order 8 default. | `utils/integration/gauss_jackson/` |
| 5.3 | LSODE integrator | Livermore Solver for Ordinary Differential Equations. Variable-order, variable-step. BDF method for stiff problems. Adams method for non-stiff. | `utils/integration/lsode/` |
| 5.4 | Integrator selection | Enum-based dispatch. All integrators implement same trait/interface. | — |
| 5.5 | Integrator unit tests | All integrators on harmonic oscillator: verify convergence order. RKF45: verify step size adapts to maintain tolerance. Gauss-Jackson: verify startup and steady-state accuracy. Compare 24h LEO trajectory across all integrators (should agree to within tolerances). | — |

#### 5B. Advanced Gravity (`jeod_gravity`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 5.6 | Solid body tides | Time-dependent delta coefficients (ΔCnm, ΔSnm) from body deformation. Permanent tide, frequency-dependent corrections. | `gravity/include/spherical_harmonics_delta_coeffs.hh` |
| 5.7 | Tide unit tests | Tidal gravity perturbation at known epoch/position matches JEOD reference value. | — |

#### 5C. Earth Rotation (`jeod_planet`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 5.8 | Precession model | IAU 2006 precession: compute precession matrix from epoch. | `RNP/RNPJ2000/` |
| 5.9 | Nutation model | IAU 2000A or 2000B nutation: compute nutation angles (Δψ, Δε) and nutation matrix. | `RNP/RNPJ2000/` |
| 5.10 | Polar motion | Apply polar motion correction (x_p, y_p from IERS data). | `RNP/RNPJ2000/` |
| 5.11 | Full RNP composition | GCRS → ITRS: W(polar) · R(Earth rotation) · N(nutation) · P(precession). | `RNP/GenericRNP/` |
| 5.12 | RNP unit tests | Earth-fixed frame orientation at J2000.0 matches IERS reference to < 1 arcsecond. Sidereal day rate matches expected value. | — |

#### 5D. SPICE Integration (`jeod_ephemeris`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 5.13 | SPICE FFI bindings | Rust bindings to `cspice` via `cc` crate or existing `spice` Rust crate. | `environment/spice/` |
| 5.14 | SPICE state query | Query planet/body state from SPICE kernels at arbitrary epoch. | `environment/spice/` |
| 5.15 | Kernel management | Load/unload SPICE kernels (BSP, TF, LSK). | — |
| 5.16 | SPICE unit tests | Earth position from SPICE matches DE421 to < 1 m. | — |

#### 5E. Contact Dynamics (`jeod_interactions`)

| ID | Task | Description | JEOD Reference |
|----|------|-------------|----------------|
| 5.17 | Contact surface model | Facet-based surface geometry. Contact point detection between two bodies. | `interactions/contact/` |
| 5.18 | Contact force model | Spring-damper contact forces. Normal and friction forces. | `interactions/contact/` |
| 5.19 | Contact unit tests | Two spheres approaching: detect contact at expected distance. Contact force magnitude matches spring constant × penetration. | — |

#### 5F. Cross-Validation Infrastructure

| ID | Task | Description |
|----|------|-------------|
| 5.20 | Generate JEOD reference trajectories | Add new sims to `trick/generate_references.sh`. Docker workflow established in Phase 1 (Rocky 9 + Trick 25 + JEOD 5.4). Key runs: RUN_7A (4x4 harmonics), RUN_1A (full gravity). |
| 5.21 | Generate Earth-Moon reference | Add `SIM_Earth_Moon` (Clementine/Rosetta) to generate script. Export to CSV. |
| 5.22 | Generate Mars reference | Add `SIM_Mars` (Dawn/Phobos) to generate script. Export to CSV. |
| 5.23 | Extend CSV trajectory loader | Extend the loader in `tier3_jeod_trajectory.rs` for new column layouts. |
| 5.24 | Trajectory comparison harness | Generalize comparison: report max error, RMS error, drift rate per scenario. |
| 5.25 | Tier 3: LEO 24h test | RK4, GGM05C degree 20, compare position to JEOD at each timestep. |
| 5.26 | Tier 3: LEO with drag test | 24h with MET atmosphere, compare trajectory. |
| 5.27 | Tier 3: Earth-Moon test | 7-day multi-body trajectory, compare to JEOD. |
| 5.28 | Tier 3: Mars orbit test | Mars orbit with MRO110B2 gravity, compare to JEOD. |
| 5.29 | Tier 4 regression harness | CI script that runs all Tier 1-3 tests and produces pass/fail summary with error budgets. |

#### 5G. Examples

| ID | Task | Description |
|----|------|-------------|
| 5.30 | `apollo.rs` | Apollo trans-lunar injection scenario. Multi-body (Earth + Moon), stage separation (attach/detach). |
| 5.31 | `earth_moon.rs` | Long-duration Earth-Moon trajectory. |
| 5.32 | `mars_orbit.rs` | Mars orbit insertion and propagation. |

### Exit Criteria

#### Tier 1 (unit tests)
- [ ] **Gauss-Jackson accuracy**: Matches RK4 trajectory to < 1 m over 24h with fewer function evaluations
- [ ] **RKF45 adaptivity**: Step size varies by > 2x between perigee and apogee on eccentric orbit
- [ ] **Solid tides**: Tidal gravity perturbation magnitude within 10% of JEOD reference

#### Tier 2 (JEOD reference data)
- [ ] **Earth rotation**: ITRS frame orientation matches JEOD/IERS to < 1 arcsecond at 5+ test epochs

#### Tier 3 (trajectory cross-validation — required for each new physics)
- [ ] **All prior phase exit criteria** still pass (no regressions)
- [ ] **Tier 3 LEO 24h (high-fidelity gravity)**: Position error vs. JEOD < 10 m (RK4, GGM05C deg 20, Earth rotation + polar motion)
- [ ] **Tier 3 LEO with drag**: Position error vs. JEOD < 100 m over 24h (MET atmosphere + ballistic drag)
- [ ] **Tier 3 Earth-Moon multi-body**: Position error vs. JEOD < 100 m over 7 days (Earth + Moon + Sun gravity, differential acceleration)
- [ ] **Tier 3 Mars orbit**: Position error vs. JEOD < 100 m over 7 days (MRO110B2 gravity)
- [ ] **Tier 3 Gauss-Jackson trajectory**: Gauss-Jackson integrator on same scenario as RK4 Tier 3, position error vs. JEOD < 1 m over 24h (demonstrating integrator fidelity, not just accuracy)
- [ ] **Tier 3 RKF45 trajectory**: RKF45 on same scenario, position error vs. JEOD < 10 m over 24h with adaptive stepping
- [ ] **Tier 3 polar motion**: Earth-fixed frame with polar motion enabled matches JEOD to < 0.1 arcsecond over 24h
- [ ] **Tier 3 solid tides**: Trajectory with tidal ΔCnm/ΔSnm corrections. Position difference (tides ON vs OFF) matches JEOD's difference to < 10% over 24h

#### Other
- [ ] **Tier 4 regression**: CI runs all scenarios automatically; all pass within budgets
- [ ] **Portability**: All `jeod_*` crates compile without Bevy; `batch_propagation.rs` runs full-fidelity scenario without Bevy

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

| Quantity | Tier 1 (analytical) | Tier 2 (JEOD reference) | Tier 3 (trajectory) |
|----------|--------------------|-----------------------|-------------------|
| Position | 1e-6 m | 1.0 m | 10-100 m |
| Velocity | 1e-9 m/s | 0.001 m/s | 0.01-0.1 m/s |
| Acceleration | 1e-12 m/s² | 1e-10 m/s² | — |
| Angles | 1e-14 rad | 1e-12 rad | 1e-6 rad |
| Energy | 1e-10 J/kg | — | — |
| Quaternion norm | 1e-14 | 1e-14 | 1e-12 |
| Time | — | exact (integer s) | — |

### Definition of Done (per task)

1. Code compiles (`cargo build`)
2. Unit tests pass (`cargo test`)
3. No clippy warnings (`cargo clippy`)
4. Core logic is in `jeod_*` crate (no Bevy dependency in physics code)
5. Bevy system (if applicable) delegates to core function

### Phase Transition Protocol

Before starting Phase N+1:

1. All Phase N exit criteria checkboxes are checked
2. `cargo test --workspace` passes with zero failures
3. All examples from Phase N run successfully
4. No known regressions from earlier phases
