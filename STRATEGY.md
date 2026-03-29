# bevy_jeod: Reimplementation Strategy

Reimplementing [NASA JEOD](https://github.com/nasa/jeod) (JSC Engineering Orbital Dynamics)
in Rust, using the [Bevy](https://bevyengine.org/) game engine's Entity Component System
as the simulation framework — replacing NASA's Trick.

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. ECS Architecture Mapping](#2-ecs-architecture-mapping)
- [3. Component Design](#3-component-design)
- [4. System Pipeline](#4-system-pipeline)
- [5. Plugin Architecture](#5-plugin-architecture)
- [6. Verification Strategy](#6-verification-strategy)
- [6b. JEOD Invariant Tracking](#6b-jeod-invariant-tracking)
- [7. JEOD Data Ingestion](#7-jeod-data-ingestion)
- [8. Implementation Phases](#8-implementation-phases)
- [9. Key Architectural Decisions](#9-key-architectural-decisions)
- [10. Risks and Mitigations](#10-risks-and-mitigations)

---

## 1. Project Overview

### What is JEOD?

JEOD 5.4 is a C++ orbital dynamics library developed at NASA JSC. It models:

- **Dynamics**: 6-DOF rigid body propagation, multi-body attachment/detachment, mass trees
- **Environment**: Spherical harmonics gravity (GGM05C, GRAIL150, MRO110B2), JPL DE4xx
  ephemerides, MET atmosphere, time scales (TAI/UTC/UT1/TDB/TT/GMST/MET)
- **Interactions**: Aerodynamic drag, solar radiation pressure, gravity gradient torque,
  contact mechanics
- **Utilities**: Reference frame trees, integration methods (RK4, RKF45, Gauss-Jackson,
  LSODE), quaternion/orbital-element math

JEOD runs inside NASA's **Trick** simulation framework, which provides job scheduling, data
recording, checkpoint/restart, and Python-based configuration.

### What are we building?

A Rust reimplementation where **Bevy's ECS replaces Trick** as the simulation framework.
Bevy provides:

- **Entities** in place of Trick simulation objects
- **Components** in place of C++ class member data
- **Systems** in place of Trick scheduled jobs
- **Resources** in place of global manager state
- **Schedules** in place of Trick's job ordering
- **Plugins** in place of Trick's S_modules

### Portability Goal

While Bevy is the primary executor, **the physics and math must not depend on Bevy**.
The codebase is split into two layers:

- **`jeod_*` crates** — Pure Rust libraries containing all physics, math, algorithms,
  data models, and domain types. Zero Bevy dependency. These crates define plain structs,
  pure functions, and traits. They are usable from any Rust ECS (hecs, legion, shipyard,
  flecs), a custom simulation loop, a WASM module, or no ECS at all.

- **`bevy_jeod_*` crates** — Thin Bevy integration layers. These add `#[derive(Component)]`
  and `#[derive(Resource)]` to core types (via newtype wrappers or feature-gated derives),
  define Bevy systems that call into `jeod_*` functions, and register plugins.

This separation means:

1. **Portability**: Swap Bevy for another ECS by writing a new thin glue layer. The
   physics code doesn't change.
2. **Testability**: Core algorithms are tested as pure functions — no need to spin up a
   Bevy `App` for unit tests.
3. **Embeddability**: `jeod_*` crates can be used in non-ECS contexts (batch trajectory
   computation, optimization loops, Monte Carlo analysis) without pulling in Bevy.
4. **Stability**: Physics crates are insulated from Bevy's rapid release cycle and
   breaking API changes.

### Why Bevy?

| Concern | Trick | Bevy |
|---------|-------|------|
| Language | C++/Python | Rust (memory safety, no UB) |
| Architecture | OOP with manager objects | Data-oriented ECS |
| Parallelism | Manual thread management | Automatic system parallelism |
| Ecosystem | NASA-internal | Open source, active community |
| Visualization | External tools | Built-in rendering, egui integration |
| Distribution | Complex build chain | `cargo build` |

---

## 2. ECS Architecture Mapping

### The Core Translation

JEOD is built on deep OOP hierarchies with manager god-objects. The translation is not
mechanical — it requires rethinking how state flows through the simulation.

```
JEOD (OOP)                          Bevy (ECS)
─────────────────────────────────   ─────────────────────────────────
DynBody class (1200 lines)      →   ~10 focused Components on an Entity
DynManager.gravitation()         →   gravity_computation_system
GravityManager (singleton)       →   Resource + System
TimeManager (singleton)          →   Resource + System
RefFrame tree (pointer graph)    →   Entity hierarchy (Parent/Children)
BodyAction subclasses            →   Events or Commands
Virtual dispatch (GravitySource) →   Trait objects or enum dispatch
Method call ordering             →   System ordering constraints
```

### Pattern-by-Pattern Mapping

**Manager Pattern → Resource + Systems**

JEOD's `DynManager`, `GravityManager`, `TimeManager`, and `EphemerisManager` are
singletons that coordinate subsystems. In ECS:

- Manager **state** becomes a `Resource` (e.g., `SimulationTime`, `EphemerisData`)
- Manager **behavior** becomes one or more `System`s
- Manager **coordination** becomes system ordering via `configure_sets()`

**Class Hierarchy → Component Composition**

JEOD: `DynBody : RefFrameOwner, IntegrableObject` — deep inheritance tree.
ECS: An entity gets the components it needs. No inheritance. A "DynBody" is just an entity
that has `TranslationalState` + `RotationalState` + `MassProperties` + etc.

**Tree Structures → Bevy's Entity Hierarchy**

JEOD's `RefFrame` tree and `MassBody` tree use raw pointers. Bevy has built-in
`Parent`/`Children` components that give us the same tree structure with safe entity
references.

**Virtual Dispatch → Enum or Trait Objects**

JEOD uses virtual base classes (`GravitySource`, `Atmosphere`, etc.) for extensibility.
In Rust: use an enum for the closed set of known models, or `Box<dyn Trait>` for
user-extensible models. Prefer enums where the model set is fixed (gravity, atmosphere).

**Core vs. Glue Separation**

All of the above mappings happen in two layers:

```
jeod_dynamics        (plain Rust structs, pure functions)
    ↕ used by
bevy_jeod_dynamics   (derives Component/Resource, defines systems, registers plugin)
```

The `jeod_*` crate defines the data types and algorithms. The `bevy_jeod_*` crate wraps
them for Bevy. Switching to another ECS means writing new `{ecs}_jeod_*` glue crates —
the physics code is untouched.

---

## 3. Component Design

### 3.1 Core vs. Bevy Split

Every data type exists first as a **plain Rust struct** in a `jeod_*` crate, then gets
wrapped or re-derived as a Bevy component in the `bevy_jeod_*` crate.

```rust
// ── jeod_dynamics/src/state.rs (pure Rust, no Bevy) ─────────────
/// Translational state in the integration frame.
pub struct TranslationalState {
    pub position: DVec3,    // m
    pub velocity: DVec3,    // m/s
}

// ── bevy_jeod_dynamics/src/components.rs (Bevy glue) ────────────
use bevy::prelude::*;
use jeod_dynamics::TranslationalState;

// Option A: newtype wrapper
#[derive(Component, Deref, DerefMut)]
pub struct TranslationalStateComponent(pub TranslationalState);

// Option B: feature-gated derive (if we control the core crate)
//   #[derive(Component)]  // behind #[cfg(feature = "bevy")]
//   pub struct TranslationalState { ... }
```

Option B (feature-gated derive) is preferred when practical — it avoids wrapper
boilerplate. Option A is the fallback when the core type can't carry Bevy derives
(e.g., when it contains non-`Reflect` fields).

### 3.2 Core State Types

These are the ECS equivalent of `DynBody`'s member data, decomposed by access pattern —
components that are read/written together by the same systems stay together.

The structs below live in `jeod_dynamics` (pure Rust). In the Bevy layer they gain
`#[derive(Component)]`.

```rust
// ── jeod_dynamics/src/state.rs ──────────────────────────────────
// All types are plain Rust. In the Bevy layer they gain #[derive(Component)].

/// Translational state in the integration frame.
pub struct TranslationalState {
    pub position: DVec3,    // m
    pub velocity: DVec3,    // m/s
}

/// Rotational state (body orientation and angular velocity).
pub struct RotationalState {
    pub quaternion: DQuat,       // left transformation, parent-to-body
    pub ang_vel_body: DVec3,     // rad/s, expressed in body frame
}

/// Rigid body mass properties.
pub struct MassProperties {
    pub mass: f64,                // kg
    pub inertia: DMat3,           // kg*m^2, in body frame
    pub inertia_inverse: DMat3,   // precomputed I^-1
    pub center_of_mass: DVec3,    // m, in structural frame
}

/// Dynamics configuration flags.
pub struct DynamicsConfig {
    pub translational: bool,      // integrate translation?
    pub rotational: bool,         // integrate rotation?
    pub three_dof: bool,          // translation-only mode?
}
```

Note: `IntegrationFrameRef(Entity)` only exists in the Bevy layer since `Entity` is a
Bevy type. In a non-ECS context, the integration frame is identified by name or index.

### 3.3 Force/Torque Types

Each interaction system writes its own force output. The force collection system reads all
of them and produces `TotalForce`. All types live in `jeod_dynamics` (pure Rust).

```rust
// ── jeod_dynamics/src/forces.rs ─────────────────────────────────

/// Gravitational acceleration and gradient at the body's position.
pub struct GravityAcceleration {
    pub accel: DVec3,       // m/s^2, in integration frame
    pub gradient: DMat3,    // 1/s^2, tidal gradient tensor
    pub potential: f64,     // m^2/s^2
}

/// Aerodynamic drag force and torque.
pub struct AerodynamicForce {
    pub force: DVec3,       // N, in body frame
    pub torque: DVec3,      // N*m, in body frame
}

/// Solar radiation pressure force and torque.
pub struct RadiationForce {
    pub force: DVec3,       // N, in body frame
    pub torque: DVec3,      // N*m, in body frame
}

/// Gravity gradient torque on an extended body.
pub struct GravityTorque {
    pub torque: DVec3,      // N*m, in body frame
}

/// Sum of all forces and torques acting on the body.
pub struct TotalForce {
    pub force: DVec3,       // N, in integration frame
    pub torque: DVec3,      // N*m, in body frame
}

/// Computed accelerations (output of F=ma).
pub struct FrameDerivatives {
    pub trans_accel: DVec3,  // m/s^2
    pub rot_accel: DVec3,    // rad/s^2
}
```

### 3.4 Reference Frame Types

JEOD's `RefFrame` tree is the backbone of all coordinate transformations. The state
types live in `jeod_frames` (pure Rust). The tree structure is ECS-specific.

```rust
// ── jeod_frames/src/state.rs (pure Rust) ────────────────────────

/// State of a reference frame relative to its parent frame.
/// Mirrors JEOD's RefFrameState = RefFrameTrans + RefFrameRot.
pub struct RefFrameState {
    pub trans: RefFrameTrans,
    pub rot: RefFrameRot,
}

pub struct RefFrameTrans {
    pub position: DVec3,    // m, in parent frame
    pub velocity: DVec3,    // m/s, in parent frame
}

pub struct RefFrameRot {
    pub q_parent_this: DQuat,       // left transformation quaternion
    pub t_parent_this: DMat3,       // transformation matrix (redundant with quat)
    pub ang_vel_this: DVec3,        // rad/s, in this frame
}

/// Frame identity and classification.
pub struct RefFrameInfo {
    pub name: String,               // "Earth.inertial", "ISS.structure", etc.
    pub kind: RefFrameKind,
}

pub enum RefFrameKind {
    Inertial,       // non-rotating, valid as integration frame
    PlanetFixed,    // rotating with planet
    Body,           // attached to a dynamic body
}
```

```rust
// ── bevy_jeod_frames/src/components.rs (Bevy glue) ──────────────

// The frame tree uses Bevy's built-in Parent/Children hierarchy.
// Marker components identify frame types for queries.

// Frame tree example:
//
// Sun.inertial (root)
//   +-- Earth.inertial [EphemerisFrame]
//   |     +-- Earth.pfix [PlanetFixedFrame]
//   |     +-- ISS.composite_body [BodyFrame(iss_entity)]
//   |     +-- ISS.structure [BodyFrame(iss_entity)]
//   |     +-- ISS.core_body [BodyFrame(iss_entity)]
//   +-- Moon.inertial [EphemerisFrame]
//   +-- Mars.inertial [EphemerisFrame]
```

In a non-Bevy ECS, the tree would use that ECS's hierarchy mechanism (e.g., `hecs`
parent/child relations, or a standalone arena-based tree from `jeod_frames`).

### 3.5 Planet Types

Planets and vehicles share the same state types (TranslationalState, RotationalState) —
differentiated by marker components in the ECS layer.

```rust
// ── jeod_planet/src/lib.rs (pure Rust) ──────────────────────────

/// Planetary shape parameters (reference ellipsoid).
pub struct PlanetShape {
    pub r_eq: f64,          // equatorial radius, m
    pub r_pol: f64,         // polar radius, m
    pub flattening: f64,    // flattening coefficient (1/298.257 for Earth)
}

// ── jeod_gravity/src/source.rs (pure Rust) ──────────────────────

/// Gravity source definition.
pub struct GravitySource {
    pub mu: f64,            // gravitational parameter, m^3/s^2
    pub model: GravityModel,
}

pub enum GravityModel {
    PointMass,
    SphericalHarmonics {
        degree: usize,
        order: usize,
        radius: f64,                   // reference radius, m
        cnm: Vec<Vec<f64>>,            // cosine coefficients [n][m]
        snm: Vec<Vec<f64>>,            // sine coefficients [n][m]
    },
}

// ── jeod_gravity/src/compute.rs (pure Rust) ─────────────────────

/// Pure function: compute gravity acceleration at a position.
/// No ECS dependency — callable from any context.
pub fn compute_gravity(
    source: &GravitySource,
    position: DVec3,         // in source-centered frame
) -> GravityAcceleration { ... }
```

### 3.6 Gravity Controls (Vehicle-to-Planet Link)

Each vehicle specifies which planets affect it and how. The core type lives in
`jeod_gravity` and uses a generic identifier for the source (string name or index).
The Bevy layer maps this to `Entity`.

```rust
// ── jeod_gravity/src/controls.rs (pure Rust) ────────────────────

/// Per-vehicle specification of gravitational interactions.
pub struct GravityControls<SourceId = String> {
    pub controls: Vec<GravityControl<SourceId>>,
}

pub struct GravityControl<SourceId = String> {
    pub source_id: SourceId,       // planet identifier (generic)
    pub spherical_only: bool,      // point-mass vs full harmonics
    pub max_degree: Option<usize>, // truncation override
    pub max_order: Option<usize>,
    pub compute_gradient: bool,    // tidal gradient needed?
}

// ── bevy_jeod_gravity/src/components.rs (Bevy glue) ─────────────

/// In Bevy, SourceId = Entity for efficient queries.
pub type BevyGravityControls = GravityControls<Entity>;
```

### 3.7 Derived State Types

Optional data that computes secondary state representations. The **computation functions**
live in `jeod_math` (pure Rust). The ECS layer attaches these as components and runs
systems that call the pure functions.

```rust
// ── jeod_math/src/orbital_elements.rs (pure Rust) ───────────────

pub struct OrbitalElements {
    pub semi_major_axis: f64,      // m
    pub eccentricity: f64,
    pub inclination: f64,          // rad
    pub raan: f64,                 // rad, right ascension of ascending node
    pub arg_periapsis: f64,        // rad
    pub true_anomaly: f64,         // rad
    pub mean_anomaly: f64,         // rad
    pub mean_motion: f64,          // rad/s
    pub orbital_energy: f64,       // m^2/s^2
    pub ang_momentum: f64,         // m^2/s
}

/// Pure function — no ECS dependency.
pub fn cartesian_to_elements(pos: DVec3, vel: DVec3, mu: f64) -> OrbitalElements { ... }
pub fn elements_to_cartesian(elems: &OrbitalElements, mu: f64) -> (DVec3, DVec3) { ... }

// ── jeod_math/src/orientation.rs (pure Rust) ────────────────────

pub struct EulerAngles {
    pub sequence: EulerSequence,
    pub ref_body_angles: DVec3,        // rad
    pub body_ref_angles: DVec3,        // rad
}

pub fn decompose_euler(matrix: &DMat3, seq: EulerSequence) -> EulerAngles { ... }

// ── jeod_math/src/planet_fixed.rs (pure Rust) ───────────────────

pub struct PlanetFixedPosition {
    pub latitude: f64,       // rad, geodetic
    pub longitude: f64,      // rad
    pub altitude: f64,       // m, above ellipsoid
}

pub fn cartesian_to_geodetic(pos: DVec3, shape: &PlanetShape) -> PlanetFixedPosition { ... }
```

### 3.8 Bevy Bundles

Bundles group components for convenient entity spawning. These exist only in the
`bevy_jeod_*` layer.

```rust
// ── bevy_jeod_dynamics/src/bundles.rs (Bevy-only) ───────────────

#[derive(Bundle)]
pub struct DynBodyBundle {
    pub name: Name,
    pub trans_state: TranslationalStateComponent,
    pub rot_state: RotationalStateComponent,
    pub mass: MassPropertiesComponent,
    pub dynamics_config: DynamicsConfigComponent,
    pub integ_frame: IntegrationFrameRef,   // Bevy Entity reference
    pub gravity_accel: GravityAccelerationComponent,
    pub gravity_controls: BevyGravityControls,
    pub total_force: TotalForceComponent,
    pub derivs: FrameDerivativesComponent,
}
```

---

## 4. System Pipeline

### Execution Schedule

JEOD's integration loop translates to Bevy's `FixedUpdate` schedule with ordered system
sets. The ordering matches JEOD's `DynManager` sequencing exactly.

```
FixedUpdate
 |
 |-- TimeUpdateSet
 |     '-- time_advance_system            // advance TAI, compute UTC/UT1/TDB/GMST
 |
 |-- EphemerisUpdateSet                   // .after(TimeUpdateSet)
 |     |-- ephemeris_update_system        // update planet positions from DE4xx
 |     '-- planet_rotation_system         // update planet-fixed frame rotations (RNP)
 |
 |-- EnvironmentSet                       // .after(EphemerisUpdateSet)
 |     |-- gravity_computation_system     // for each body: spherical harmonics accel
 |     '-- atmosphere_update_system       // compute density at body positions
 |
 |-- InteractionSet                       // .after(EnvironmentSet)
 |     |-- aerodynamic_force_system       // F_drag = 0.5 * rho * v^2 * Cd * A
 |     |-- radiation_pressure_system      // solar radiation pressure
 |     '-- gravity_torque_system          // gravity gradient torque
 |
 |-- ForceCollectionSet                   // .after(InteractionSet)
 |     '-- force_collection_system        // sum all force components -> TotalForce
 |
 |-- IntegrationSet                       // .after(ForceCollectionSet)
 |     |-- integration_system             // propagate state via RK4/GJ/etc.
 |     '-- frame_propagation_system       // update attached body & child frames
 |
 '-- DerivedStateSet                      // .after(IntegrationSet)
       |-- orbital_elements_system        // Cartesian -> Keplerian
       |-- euler_angles_system            // quaternion -> Euler angles
       |-- planet_fixed_system            // inertial -> geodetic coords
       '-- lvlh_system                    // compute LVLH frame state
```

### System Ordering in Code

```rust
app.configure_sets(FixedUpdate, (
    TimeUpdateSet,
    EphemerisUpdateSet.after(TimeUpdateSet),
    EnvironmentSet.after(EphemerisUpdateSet),
    InteractionSet.after(EnvironmentSet),
    ForceCollectionSet.after(InteractionSet),
    IntegrationSet.after(ForceCollectionSet),
    DerivedStateSet.after(IntegrationSet),
));
```

### Multi-Stage Integration

JEOD uses multi-stage integrators (e.g., RK4 has 4 stages per timestep, each requiring a
fresh force evaluation). This is handled with a resource tracking stage state:

```rust
#[derive(Resource)]
pub struct IntegrationState {
    pub method: IntegrationMethod,
    pub current_stage: usize,
    pub total_stages: usize,
    pub dt: f64,
}

pub enum IntegrationMethod {
    Rk4,                    // 4 stages, fixed step
    Rkf45 { tol: f64 },    // adaptive step
    GaussJackson { order: usize },  // multi-step
}
```

The `integration_system` runs the full multi-stage loop internally: for each stage it
re-evaluates forces, computes derivatives, and advances the stage. This keeps the
multi-stage logic contained rather than spreading it across the schedule.

### Key System Signatures

Bevy systems are thin wrappers that query components and delegate to `jeod_*` pure
functions. This keeps the physics testable without Bevy.

```rust
// ── bevy_jeod_gravity/src/systems.rs ────────────────────────────

fn gravity_computation_system(
    mut bodies: Query<(&TranslationalState, &BevyGravityControls, &mut GravityAcceleration)>,
    sources: Query<(&GravitySource, &RefFrameState), With<Planet>>,
) {
    for (state, controls, mut accel) in &mut bodies {
        // Delegate to pure function from jeod_gravity
        *accel = jeod_gravity::compute_all_gravity(
            state.position, controls, |entity| sources.get(entity),
        );
    }
}

// ── bevy_jeod_dynamics/src/systems.rs ───────────────────────────

fn integration_system(
    mut bodies: Query<(
        &TotalForce, &MassProperties, &DynamicsConfig,
        &mut TranslationalState, &mut RotationalState,
        &mut FrameDerivatives,
    )>,
    integ_state: Res<IntegrationState>,
) {
    for (force, mass, config, mut trans, mut rot, mut derivs) in &mut bodies {
        // Delegate to pure function from jeod_dynamics
        jeod_dynamics::integrate_step(
            &force, mass, config, &mut trans, &mut rot, &mut derivs,
            integ_state.method, integ_state.dt,
        );
    }
}
```

---

## 5. Plugin Architecture

### Crate Organization

The workspace has two layers: **core crates** (`jeod_*`) with zero Bevy dependency, and
**Bevy glue crates** (`bevy_jeod_*`) that add ECS integration. This separation is the
key to portability — see [Section 1: Portability Goal](#portability-goal).

```
bevy_jeod/                               # workspace root
|
+-- crates/
|   |
|   | ── CORE LAYER (pure Rust, no Bevy dependency) ───────────────
|   |
|   +-- jeod_math/                       # f64 math, quaternions, orbital elements
|   |   +-- src/
|   |       +-- lib.rs
|   |       +-- quaternion.rs            # JEOD quaternion conventions (scalar-first)
|   |       +-- orbital_elements.rs      # Cartesian <-> Keplerian, Kepler equation
|   |       +-- orientation.rs           # Euler angle decomposition, rotation matrices
|   |       +-- planet_fixed.rs          # geodetic coordinate conversions
|   |       +-- lvlh.rs                  # LVLH frame computation
|   |
|   +-- jeod_time/                       # Time scales and conversions
|   |   +-- src/
|   |       +-- lib.rs
|   |       +-- scales.rs               # TAI, UTC, UT1, TDB, TT, GMST, MET types
|   |       +-- converters.rs           # time scale conversions (leap seconds, UT1-TAI)
|   |       +-- sim_time.rs             # SimulationTime state struct
|   |
|   +-- jeod_frames/                     # Reference frame state and transformations
|   |   +-- src/
|   |       +-- lib.rs
|   |       +-- state.rs                # RefFrameState, RefFrameTrans, RefFrameRot
|   |       +-- transform.rs           # relative state computation, frame composition
|   |       +-- tree.rs                 # arena-based frame tree (for non-ECS use)
|   |
|   +-- jeod_gravity/                    # Gravity models and computation
|   |   +-- src/
|   |       +-- lib.rs
|   |       +-- source.rs              # GravitySource, GravityModel
|   |       +-- controls.rs            # GravityControls<SourceId>
|   |       +-- compute.rs             # compute_gravity() pure function
|   |       +-- spherical_harmonics.rs  # Legendre polynomials, coefficient evaluation
|   |   +-- data/                       # coefficient files (binary or RON)
|   |       +-- earth_ggm05c.bin
|   |       +-- moon_grail150.bin
|   |       +-- mars_mro110b2.bin
|   |
|   +-- jeod_ephemeris/                  # Ephemeris readers
|   |   +-- src/
|   |       +-- lib.rs
|   |       +-- de4xx.rs               # JPL DE4xx binary reader (Chebyshev interpolation)
|   |
|   +-- jeod_atmosphere/                 # Atmosphere models
|   |   +-- src/
|   |       +-- lib.rs
|   |       +-- model.rs              # Atmosphere trait
|   |       +-- met.rs                # Marshall Engineering Thermosphere tables
|   |
|   +-- jeod_dynamics/                   # State types, integration methods, force collection
|   |   +-- src/
|   |       +-- lib.rs
|   |       +-- state.rs              # TranslationalState, RotationalState
|   |       +-- mass.rs               # MassProperties, composite mass computation
|   |       +-- forces.rs             # GravityAcceleration, AeroForce, TotalForce, etc.
|   |       +-- integration.rs        # RK4, RKF45, Gauss-Jackson, LSODE (pure functions)
|   |       +-- body_action.rs        # initialization from orbital elements, LVLH, NED
|   |
|   +-- jeod_interactions/               # Force/torque computation
|   |   +-- src/
|   |       +-- lib.rs
|   |       +-- aerodynamics.rs       # drag computation (pure function)
|   |       +-- radiation.rs          # SRP computation (pure function)
|   |       +-- gravity_torque.rs     # gradient torque (pure function)
|   |
|   +-- jeod_planet/                     # Planet data and presets
|   |   +-- src/
|   |       +-- lib.rs
|   |       +-- shape.rs             # PlanetShape
|   |       +-- presets.rs           # Earth, Moon, Mars, Sun constants
|   |       +-- rnp.rs              # precession, nutation, polar motion
|   |
|   | ── ORCHESTRATION LAYER (ECS-agnostic pipeline) ──────────────
|   |
|   +-- jeod_sim/                        # Pipeline orchestration, Simulation runner
|   |   +-- src/
|   |       +-- lib.rs
|   |       +-- atmosphere.rs            # AtmosphereModel, AtmosphereConfig, evaluate_atmosphere()
|   |       +-- gravity.rs              # accumulate_gravity()
|   |       +-- forces.rs              # collect_and_resolve_forces()
|   |       +-- integration.rs         # integrate_body()
|   |       +-- validation.rs          # validate_body() -> Result<(), Vec<ValidationError>>
|   |       +-- pipeline.rs            # PipelineStage enum, PIPELINE_ORDER
|   |       +-- simulation.rs          # Simulation, SimBody, GravitySourceEntry
|   |
|   | ── BEVY GLUE LAYER (thin, delegates to jeod_sim) ───────────
|   |
|   +-- bevy_jeod_time/                  # Bevy plugin: time resource + system
|   +-- bevy_jeod_frames/                # Bevy plugin: frame components + propagation system
|   +-- bevy_jeod_gravity/               # Bevy plugin: gravity components + system
|   +-- bevy_jeod_ephemeris/             # Bevy plugin: ephemeris resource + update system
|   +-- bevy_jeod_atmosphere/            # Bevy plugin: atmosphere components + system
|   +-- bevy_jeod_dynamics/              # Bevy plugin: state components + integration system
|   +-- bevy_jeod_interactions/          # Bevy plugin: force components + systems
|   +-- bevy_jeod_derived/               # Bevy plugin: derived state components + systems
|   +-- bevy_jeod_planet/                # Bevy plugin: planet components + presets
|   |
|   | ── TEST INFRASTRUCTURE ──────────────────────────────────────
|   |
|   +-- jeod_test_data/                  # JEOD file parsers (no Bevy dependency)
|       +-- src/
|           +-- lib.rs
|           +-- py_data.rs              # Python data file parser (trick.attach_units)
|           +-- dat_parser.rs           # whitespace-delimited numeric tables
|           +-- gravity_verif.rs        # parses verif_out.txt -> GravityTestCase
|           +-- reference_state.rs      # parses reference_*_trans_state.py
|           +-- leap_seconds.rs         # parses Leap_Second.dat
|
+-- src/
|   +-- lib.rs                           # top-level JeodPlugin composing all Bevy plugins
|
+-- examples/
|   +-- kepler_orbit.rs                  # simple two-body orbit (Bevy)
|   +-- leo_j2.rs                        # LEO with J2 perturbation (Bevy)
|   +-- iss_orbit.rs                     # ISS full-fidelity (Bevy)
|   +-- apollo.rs                        # Apollo mission (Bevy)
|   +-- batch_propagation.rs             # no-ECS batch trajectory (jeod_* only)
|
+-- data/                                # runtime data assets
    +-- gravity/                         # spherical harmonics coefficient files
    +-- time/                            # leap seconds, UT1 corrections
    +-- ephemeris/                        # DE421.bsp binary files
```

**Dependency graph:**
```
jeod_math  <── jeod_dynamics  <── jeod_interactions
   ^              ^                      |
   |              |                      v
jeod_time    jeod_gravity          jeod_atmosphere
   ^              ^
   |              |
jeod_frames  jeod_ephemeris    jeod_planet
                                     |
     All jeod_* are pure Rust        |
     ────────────────────────────────┘
              |
              v
     jeod_sim (orchestration: composes jeod_* functions, zero Bevy dep)
              |
              v
     bevy_jeod_* crates (thin Bevy glue, delegates to jeod_sim)
              |
              v
     bevy_jeod (top-level plugin)
```

### Three-Layer Architecture

The codebase has three layers:

1. **`jeod_*` crates** — Pure physics algorithms and data types. Zero Bevy dependency.
   Define per-function operations (gravity evaluation, RK4 step, drag computation, etc.).

2. **`jeod_sim` crate** — ECS-agnostic orchestration. Zero Bevy dependency. Composes
   `jeod_*` functions into pipeline stages and provides:
   - **Per-body functions** (primary API for ECS adapters): `accumulate_gravity()`,
     `evaluate_atmosphere()`, `collect_and_resolve_forces()`, `integrate_body()`,
     `validate_body()`. All borrow-based — the ECS world remains the source of truth.
   - **`Simulation` runner** (for non-ECS use): standalone struct for batch propagation,
     scripting, and tests. Owns state internally.
   - **`PipelineStage` enum** and `PIPELINE_ORDER`: canonical stage ordering that any
     adapter must respect.

3. **`bevy_jeod_*` crates** — Thin Bevy glue. Each system function queries components
   and delegates to `jeod_sim` per-body functions. Component definitions, plugin
   registration, and system scheduling live here.

**Why three layers?** The original two-layer design (`jeod_*` + `bevy_jeod_*`) kept
physics portable, but the orchestration logic — pipeline ordering, gravity accumulation,
frame transform composition, force contribution assembly, integration routing, and
validation — lived exclusively in `bevy_jeod_*` code. A non-Bevy ECS user would have
had to reverse-engineer ~10 systems across 8 Bevy crates to build a working simulation
loop. The `jeod_sim` layer extracts this orchestration into a single, Bevy-free crate
that any ECS (or no ECS) can use directly.

Each `bevy_jeod_*` crate depends on its corresponding `jeod_*` crate, on `jeod_sim`,
and on `bevy`. The `jeod_*` and `jeod_sim` crates have **no** Bevy dependency and can
be used standalone.

### Top-Level Plugin Composition

```rust
pub struct JeodPlugin;

impl Plugin for JeodPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            JeodTimePlugin,
            JeodFramesPlugin,
            JeodGravityPlugin,
            JeodEphemerisPlugin,
            JeodAtmospherePlugin,
            JeodDynamicsPlugin,
            JeodInteractionsPlugin,
            JeodDerivedStatePlugin,
            JeodPlanetPlugin,
        ));
    }
}
```

Users can also add individual plugins for a minimal setup:

```rust
// Minimal: just gravity and dynamics, no atmosphere or interactions
app.add_plugins((
    JeodTimePlugin,
    JeodFramesPlugin,
    JeodGravityPlugin,
    JeodDynamicsPlugin,
));
```

---

## 6. Verification Strategy

### Five-Tier Verification Plan

### Tier 0: Cross-Parity (Bevy vs Simulation Runner)

Verify that the Bevy ECS pipeline and `jeod_sim::Simulation` produce **bit-identical**
output from the same initial conditions. This guarantees that a non-Bevy ECS user gets
exactly the same numerical results as the Bevy pipeline.

Cross-parity tests set up identical scenarios in both execution paths, run the same
number of steps at the same dt, and assert exact equality on all state variables
(position, velocity, quaternion, angular velocity). The test file
`tests/cross_parity.rs` covers five physics scenarios:

| Scenario | Physics | DOF |
|----------|---------|-----|
| A | Point-mass gravity | 6-DOF |
| B | Exponential atmosphere + ballistic drag | 6-DOF |
| C | Solar radiation pressure | 3-DOF |
| D | Gravity gradient torque | 6-DOF |
| E | Full stack (drag + SRP + gravity torque + wind) | 6-DOF |

All five produce `0.00e0` difference — not just within tolerance, but **exactly zero**.
This is possible because both paths call the same `jeod_sim` per-body functions,
which call the same `jeod_*` pure functions, in the same order, with the same inputs.

**Rule:** Every phase that delivers new physics must add a corresponding cross-parity
scenario. The cross-parity test is part of the definition of done alongside Tier 3.

### Tier 1: Analytical Unit Tests

Test pure math with known exact solutions. No JEOD data needed. Implement these alongside
each module.

```rust
#[test]
fn kepler_equation_circular() {
    // For circular orbit (e=0), mean anomaly = eccentric anomaly
    assert_f64_eq!(solve_kepler(0.0, PI / 4.0), PI / 4.0);
}

#[test]
fn orbital_elements_roundtrip() {
    let state = CartesianState { r: dvec3(...), v: dvec3(...) };
    let elems = OrbitalElements::from_cartesian(&state, MU_EARTH);
    let back = elems.to_cartesian(MU_EARTH);
    assert_dvec3_near!(state.r, back.r, 1e-10);
}

#[test]
fn quaternion_rotation_matrix_consistency() { ... }
#[test]
fn frame_composition_is_identity_for_self() { ... }
#[test]
fn kepler_orbit_conserves_energy() { ... }
#[test]
fn kepler_orbit_period_matches_analytical() {
    // T = 2*pi*sqrt(a^3/mu) — must match to integrator precision
}
```

### Tier 2: Component Tests Against JEOD Reference Values

Use JEOD's own test data files (read unmodified from `../jeod`) via the `jeod_test_data`
crate. See [Section 7](#7-jeod-data-ingestion) for parser details.

```rust
use jeod_test_data::{reference_states, gravity_test_cases, orbital_init_data};

#[test]
fn iss_reference_inertial_state() {
    // Source: jeod/models/dynamics/body_action/verif/SIM_orbinit/
    //         Modified_data/ISS/reference_inertial_trans_state.py
    let expected = reference_states("../jeod", "ISS", "inertial");
    // expected.position = [1244540.53, 5655938.85, 3425643.22]
    // expected.velocity = [-6003.833051, -1469.496044, 4590.511776]

    let init = orbital_init_data("../jeod", "ISS", "trans_Orbit_inertial_body_set01");
    let state = propagate_from_elements(&init);
    assert_dvec3_near!(state.position, expected.position, 1.0);  // 1m tolerance
}

#[test]
fn earth_gravity_at_known_positions() {
    // Source: jeod/models/environment/gravity/verif/unit_tests/
    //         grav_geospherical/data/verif_out.txt
    for case in gravity_test_cases("../jeod") {
        let result = gravity_source.compute(case.position, case.degree, case.order);
        assert_dvec3_near!(result.accel, case.expected_accel, 1e-12);
        assert_near!(result.potential, case.expected_potential, 1e-6);
    }
}

#[test]
fn euler_angle_decomposition() {
    // Source: jeod/models/dynamics/derived_state/verif/unit_tests/
    //         euler_derived_state_ut.cc
    // 6 test cases with rotation matrix -> expected Euler angles
    for case in euler_test_cases("../jeod") {
        let result = euler_decompose(case.matrix, case.sequence);
        assert_dvec3_near!(result, case.expected_angles, 1e-14);
    }
}
```

### Tier 3: Trajectory Cross-Validation

Generate reference trajectories by running JEOD's verification sims inside a Rocky 9
Docker container with Trick 25, then compare against bevy_jeod propagation from
identical initial conditions.

**Docker workflow:**

```bash
# Build the container (from bevy_jeod root, with trick/ and jeod/ as siblings)
docker build -f trick/Dockerfile -t jeod-trick ..

# Generate reference CSVs (runs JEOD sims, exports to test_data/)
mkdir -p test_data
docker run --rm -v $(pwd)/test_data:/output jeod-trick
```

The container builds Trick and JEOD from source using the exact package list from
Trick's CI (`test_linux.yml` Rocky 9 matrix entry), runs verification sims, and
exports ASCII CSV trajectories. The CSV files are gitignored — they are generated
locally and consumed by `cargo test`.

**Cross-validation test (implemented):**

```rust
#[test]
fn tier3_cross_validate_against_jeod_dyncomp() {
    let csv_path = Path::new("../../test_data/dyncomp_run2_state.csv");
    assert!(csv_path.exists(), "Generate with: docker run ...");

    let jeod_trajectory = load_jeod_trajectory(&csv_path);
    let mut state = TranslationalState {
        position: jeod_trajectory[0].position,
        velocity: jeod_trajectory[0].velocity,
    };

    for record in &jeod_trajectory[1..] {
        // Propagate using OUR ported code — never JEOD outputs
        state = propagate_to(state, record.time, dt);
        let pos_error = (state.position - record.position).length();
        assert!(pos_error < tolerance);
    }
}
```

**Key rules:**
- Tests assert on missing data — never skip gracefully.
- All computation (gravity, Earth rotation, time conversion) is our own ported code.
  JEOD CSV data is used **only** for comparison, never as input to our computation.
- Tier 3 tests are part of the **definition of done** for every phase, not optional.
- Tier 3 tests call `jeod_*` pure functions directly (no Bevy, no `jeod_sim`).
  Tier 0 cross-parity tests separately prove that the Bevy pipeline and `jeod_sim`
  Simulation produce identical output. Together, they guarantee: any ECS adapter
  using `jeod_sim` will match both Bevy and JEOD to the same tolerances.

**Results:**

| Phase | Run | Gravity | Pos Error (8h) | Attitude |
|-------|-----|---------|----------------|----------|
| 1 | RUN_2 | Point-mass | 0.4 m | — |
| 2 | RUN_3A | 4×4 harmonics + our RNP | 15.6 m | — |
| 2 | RUN_3B | 8×8 harmonics + our RNP | 28.8 m | — |
| 3 | RUN_2 | Point-mass, 6-DOF | 0.32 m | 4.21e-8 rad |

**Available JEOD sims for cross-validation:**

| Sim | Run | Duration | Gravity | Validates |
|-----|-----|----------|---------|-----------|
| SIM_dyncomp | RUN_2 | 28800s | Spherical | Phase 1: point-mass dynamics |
| SIM_dyncomp | RUN_7A | 28800s | 4x4 harmonics | Phase 2: spherical harmonics |
| SIM_orbinit | RUN_0001 | instant | — | Orbital element initialization |
| SIM_Euler | RUN_inc | 86400s | GGM05C | Phase 3: Euler angles |
| SIM_integ_test | RUN_rk4 | 28800s | — | Integrator accuracy |
| SIM_Earth_Moon | RUN_clem | days | multi-body | Phase 5: Earth-Moon dynamics |

### Tier 4: Regression Suite

Automated CI that tracks error budgets across all scenarios:

```
Scenario                         | Quantity  | Tolerance | Phase | Status
---------------------------------|----------|-----------|-------|--------
Kepler 2-body (1 orbit)          | position | 1e-6 m    |   1   | [x] 0.017 m
Energy conservation (10 orbits)  | energy   | 1e-8 rel  |   1   | [x] 3.2e-10
Period accuracy                  | period   | 1e-4 rel  |   1   | [x] 2.3e-12
ISS 24h point-mass               | altitude | 1 km      |   1   | [x] exact
JEOD trajectory (8h spherical)   | position | 5 km      |   1   | [x] 0.4 m
Orbital elements roundtrip       | position | 1e-6 m    |   1   | [x] <1e-6
Euler angle decomposition        | angles   | 1e-12 rad |   1   | [x] <1e-15
Gravity acceleration             | accel    | 1e-12 m/s²|   1   | [x] exact
LEO + J2 (24h)                   | position | 1.0 m     |   2   | [ ]
ISS full gravity (24h)           | position | 10.0 m    |   2   | [ ]
Earth-Moon 3-body (7 days)       | position | 100.0 m   |   5   | [ ]
```

---

## 6b. JEOD Invariant Tracking

JEOD's C++ architecture enforces ~120 invariants through `MessageHandler::fail()` (fatal),
`MessageHandler::error()` (non-fatal auto-correction), structural guarantees (value members,
deleted copy constructors), and flag-gated code paths. In ECS, components are optional and
can be added/removed freely, so these invariants must be tracked and enforced explicitly.

### Three-part system

**1. Catalog** — `docs/JEOD_invariants.md`

A table of every known JEOD invariant, organized by section (DB=DynBody, MA=Mass,
GV=Gravity, etc.). Each row has:

| Field | Purpose |
|-------|---------|
| Tag | Unique ID like `GV.04` for cross-referencing |
| Invariant | What the invariant requires |
| Enforcement | How JEOD enforces it (fatal, error, structural, flag-gate) |
| Category | When it applies (initialization, runtime, structural, consistency, ordering) |
| Our Status | How we enforce it (`enforced`, `partial`, `deferred`, `n/a`, `structural`) |

**2. Source tags** — `// JEOD_INV: XX.YY` comments

Every enforcement site in our Rust source is tagged:

```rust
// JEOD_INV: GV.04 — degree <= source degree
assert!(
    self.degree <= data.degree,
    "Gravity field degree requested ({}) exceeds source ({})",
    self.degree, data.degree
);
```

Tag text must describe what **our code** does. When we diverge from JEOD's approach,
note the divergence:

```rust
// JEOD_INV: DB.18 — F=ma via precomputed inverse_mass (matches JEOD MassPointState.inverse_mass)
```

**3. CI coverage** — `tests/invariant_coverage.rs`

Bidirectional consistency test:
- Every catalog entry marked `enforced`, `partial`, or `structural` (with a file reference)
  must have at least one `// JEOD_INV:` tag in source.
- Every source tag must reference a valid catalog entry.
- No duplicate IDs in the catalog.

### Workflow: adding a new invariant

When reading JEOD source and encountering a `MessageHandler::fail()`, `error()`, assert,
or structural guarantee not already in the catalog:

1. Add a row to `docs/JEOD_invariants.md` with the next tag in the section (e.g., `DB.28`).
2. Add `// JEOD_INV: DB.28 — description` at the enforcement site in our code, or mark
   the catalog entry `deferred`/`n/a` if we don't enforce it yet.
3. Run `cargo test --test invariant_coverage` to verify consistency.

### Workflow: tagging an untagged enforcement site

If our code already enforces a JEOD invariant but lacks a `// JEOD_INV` tag:

1. Find or create the catalog entry.
2. Add the tag at the enforcement site.
3. Run the coverage test.

### Current state

The catalog has 118 invariants across 10 sections (DB, MA, DM, GV, TM, RF, EP, AT, IN, FD).
62 are tagged and enforced in source across 86 tag sites, 29 are deferred to Phase 5,
and 25 are n/a for ECS architecture. The remaining 2 are not yet enforced.

---

## 7. JEOD Data Ingestion

### The `jeod_test_data` Crate

A standalone crate (no Bevy dependency) that reads JEOD's original files unmodified from
a local JEOD checkout. It provides parsed, typed Rust structs for use in tests.

### File Parsability Assessment

JEOD's data files fall into three categories:

#### Directly parseable (no modification needed)

| File | Location | Format | Parser |
|------|----------|--------|--------|
| `Leap_Second.dat` | `models/environment/time/data/` | `# comments` + whitespace columns | Line parser, skip `#` |
| `verif_out.txt` | `models/environment/gravity/verif/unit_tests/grav_geospherical/data/` | 18 space-separated numeric fields, 40 rows | `sscanf`-equivalent |
| `reference_*_trans_state.py` | `models/dynamics/body_action/verif/SIM_orbinit/Modified_data/ISS/` | `vehicle.expected_state.trans.position = [x, y, z]` | Regex on RHS arrays |
| `iss_rate_def.py` | same ISS directory | `return [0.002, 0.006, -0.003]` | Regex on `return` literal |
| `lvlh_rate_def.py` | same ISS directory | `return -0.06556131568278` | Regex on `return` literal |
| `earth_discrep.txt` | `models/environment/spice/verif/compare/` | `Angle = 1.26e-07Axis = -0.92 0.38 0.07` | Regex |

#### Parseable with `trick.attach_units()` stripping

Orbital element, mass property, and attitude files wrap numeric data in Trick calls.
A single regex handles all of them:

```python
# What the files look like:
vehicle_reference.orb_init.inclination = trick.attach_units("degree", 51.670450765)
vehicle_reference.orb_init.semi_major_axis = trick.attach_units("km", 6732.90120152)
vehicle_reference.orb_init.eccentricity = 0.00129073350
vehicle_reference.mass_init.properties.mass = 100000.0
vehicle_reference.mass_init.properties.inertia[0] = [7e12, 0.0, 0.0]
vehicle_reference.att_init.orientation.euler_angles = trick.attach_units("degree", [77.59, -30.60, -46.10])
```

The parser extracts (dotted_key, unit_or_none, value) tuples:

```rust
pub struct JeodPyValue {
    pub key: String,              // "orb_init.inclination"
    pub unit: Option<String>,     // Some("degree")
    pub value: JeodValue,         // Scalar(51.670450765) or Vec([...])
}

pub enum JeodValue {
    Scalar(f64),
    Vec(Vec<f64>),
    Str(String),
    Bool(bool),
}
```

**Files covered by this parser:**

| Pattern | Count | Content |
|---------|-------|---------|
| `trans_Orbit_*_body_set*.py` | ~20 | Orbital elements (a, e, i, RAAN, omega, tp) |
| `mass.py` | per vehicle | Mass, inertia tensor, center of mass, attach points |
| `att_RotState_*.py` | ~10 | Euler angles, quaternions |
| `rate_RotState_*.py` (some) | ~5 | Angular velocity |

**Unit conversions** are trivial — the parser applies them automatically:

| JEOD Unit | Conversion |
|-----------|------------|
| `"degree"` | multiply by `PI/180` |
| `"km"` | multiply by `1000` |
| `"s"` | no conversion |

#### Not parseable (orchestration logic)

~30% of files contain `exec()` chains, `eval()`, or complex control flow. These are
**orchestration files** that wire together the data files above — they don't contain
unique data.

| File | Why unparseable |
|------|-----------------|
| `single_vehicle_run.py` | `exec()` chains, `eval("set_" + name + "_mass(...)")` |
| `earth.py` | Method calls: `set_date_and_time(2005, 7, 28, 10, 9, 59.0)` |
| `system.py` | Pure Trick API calls |
| `run_files.py` | Dynamic file loading with `exec()` |

**These don't need to be parsed.** The data they reference lives in the parseable files.
The scenario configuration (start date, integration method, stop time) can be hardcoded
in Rust test functions since there are a finite number of scenarios.

### C++ Unit Test Extraction

Of JEOD's 262 C++ unit test files, only 2 contain extractable numerical test vectors:

| Source | Test Cases | Content |
|--------|------------|---------|
| `euler_derived_state_ut.cc` | 6 | Rotation matrix → expected Euler angles |
| `verif_out.txt` | 40 | Position → expected gravity acceleration, gradient, potential |

The rest are structural tests (empty bodies, mock verification, boolean checks) with no
hardcoded numerical assertions. Not worth parsing.

For `euler_derived_state_ut.cc`, values can be extracted with:
```
regex: double\s+(\w+)\[3\]\s*=\s*\{([^}]+)\}
```

### Complete Parser Inventory

```rust
// jeod_test_data/src/lib.rs

/// Parse JEOD Python data files with optional trick.attach_units() stripping.
/// Works for orbital elements, mass properties, attitude, rate definitions.
pub fn parse_py_data(path: &Path) -> Vec<JeodPyValue>;

/// Parse reference state vectors from reference_*_trans_state.py files.
/// Returns position[3] and velocity[3].
pub fn reference_states(jeod_root: &str, vehicle: &str, frame: &str)
    -> TranslationalState;

/// Parse orbital initialization data from trans_Orbit_*.py files.
/// Returns orbital elements with units already converted (deg->rad, km->m).
pub fn orbital_init_data(jeod_root: &str, vehicle: &str, init_name: &str)
    -> OrbitalInitData;

/// Parse gravity verification test cases from verif_out.txt.
/// Returns 40 test cases with (position, degree, order, expected accel/grad/pot).
pub fn gravity_test_cases(jeod_root: &str) -> Vec<GravityTestCase>;

/// Parse Leap_Second.dat into a leap second table.
/// Returns Vec<(mjd, tai_minus_utc)>.
pub fn leap_second_table(jeod_root: &str) -> Vec<LeapSecondEntry>;

/// Parse Euler angle test vectors from euler_derived_state_ut.cc.
/// Returns 6 test cases with (matrix, expected angles, sequence).
pub fn euler_test_cases(jeod_root: &str) -> Vec<EulerTestCase>;
```

---

## 8. Implementation Phases

### Phase 1: Foundation

**Goal:** A dot orbiting a point mass in Bevy's `FixedUpdate`.

**Core crates:** `jeod_math`, `jeod_dynamics` (minimal), `jeod_gravity` (point mass only),
`jeod_frames` (minimal)

**Bevy crates:** `bevy_jeod_dynamics`, `bevy_jeod_gravity`, `bevy_jeod_frames`

**Deliver:**
- `DVec3`/`DQuat`/`DMat3` math operations (using `glam` f64 types)
- Orbital element ↔ Cartesian conversions with Kepler equation solver
- `TranslationalState`, `MassProperties`, `TotalForce` types (core) + components (Bevy)
- RK4 integrator as pure function + Bevy integration system
- Point-mass gravity computation + Bevy gravity system
- Minimal reference frame hierarchy (inertial root + one body frame)
- `OrbitalElements` derived state
- `batch_propagation.rs` example using `jeod_*` crates with no Bevy

**Verify with:**
- Kepler orbit conserves energy and angular momentum to machine precision
- Orbital period matches analytical `T = 2*pi*sqrt(a^3/mu)`
- Orbital elements round-trip test

### Phase 2: Realistic Environment

**Goal:** J2+ spherical harmonics gravity, time system, basic ephemeris.

**Core crates:** `jeod_gravity` (spherical harmonics), `jeod_time`, `jeod_ephemeris`,
`jeod_planet`, `jeod_test_data`

**Bevy crates:** `bevy_jeod_gravity`, `bevy_jeod_time`, `bevy_jeod_ephemeris`,
`bevy_jeod_planet`

**Deliver:**
- Full spherical harmonics gravity engine (port of `spherical_harmonics_calc_nonspherical.cc`)
- Earth GGM05C, Moon GRAIL150 coefficient data
- TAI, UTC, UT1, TDB, TT time scales with converters
- Leap second table (from JEOD data)
- DE421 binary ephemeris reader
- Planet position updates from ephemeris
- Earth, Moon, Sun planet presets with shapes and gravity

**Verify with:**
- Tier 2 gravity tests: 40 test vectors from `verif_out.txt`
- LEO + J2 nodal regression rate matches analytical prediction
- Time conversion tests against known epochs

### Phase 3: Full Dynamics

**Goal:** 6-DOF dynamics with rotational state and multi-body attachment.

**Core crates:** `jeod_dynamics` (full), `jeod_frames` (full), `jeod_math` (derived states)

**Bevy crates:** `bevy_jeod_dynamics`, `bevy_jeod_frames`, `bevy_jeod_derived`

**Deliver:**
- Rotational integration (Lie group technique for quaternion propagation)
- Force and torque collection system
- Mass tree with composite property updates on attach/detach
- Full reference frame propagation (structure → composite → core body)
- Body initialization actions (orbital elements, LVLH, NED)
- Euler angles, LVLH, NED, planet-fixed derived states

**Verify with:**
- Tier 2 ISS reference state tests
- Tier 2 Euler angle decomposition tests (6 vectors from `euler_derived_state_ut.cc`)
- Tier 3 cross-validation against JEOD's `SIM_dyncomp` (6-DOF attitude: 4.21e-8 rad/8h)

### Phase 3a: Cross-Validation Closure

**Goal:** Tier 2/3 cross-validation for every Phase 3 capability. No new physics.

**Deliver:**
- Wire planet-fixed frame into gravity pipeline (fix 15–29 m RNP residual)
- Cross-validate structure/core_body frame propagation against existing CSV data
- Docker sims: SIM_OrbElem, SIM_LVLH, SIM_NED, SIM_SolarBeta, SIM_Euler, SIM_orbinit
- Trajectory-level validation of orbital elements, LVLH, geodetic, NED, solar beta, Euler angles
- Bevy system integration test (wiring parity)

**Verify with:**
- Tier 3 spherical harmonics with correct RNP (target < 5 m, down from 15.6 m)
- Tier 3 frame propagation (structure/core match JEOD CSV)
- Tier 3 derived states (each validated against its own JEOD sim)
- Tier 2 body initialization (ISS reference state < 1 m)

### Phase 4: Interactions

**Goal:** Aerodynamic drag, radiation pressure, gravity gradient torque.

**Core crates:** `jeod_atmosphere`, `jeod_interactions`

**Bevy crates:** `bevy_jeod_atmosphere`, `bevy_jeod_interactions`

**Deliver:**
- MET atmosphere model (density/temperature/pressure tables)
- Aerodynamic drag system (ballistic coefficient and flat-plate models)
- Solar radiation pressure system
- Gravity gradient torque system
- Solar beta angle derived state

**Verify with:**
- Tier 1: LEO with drag orbital decay rate matches expected behavior
- Tier 1: SRP magnitude matches analytical `P = L_sun / (4*pi*r^2*c)`
- Tier 1: Gravity torque on known inertia tensor matches analytical gradient torque
- Tier 2: MET atmosphere density at 400 km matches JEOD tables to < 5%
- Tier 3: Gravity torque trajectory (RUN_9A/9B) attitude < 0.01 rad/8h
- Tier 3: Drag trajectory (SIM_dyncomp with drag) position < 100 m/24h
- Tier 3: SRP trajectory position < 10 m/24h
- Tier 3: Eclipse entry/exit times match JEOD to < 10 s

### Phase 5: High-Fidelity Parity

**Goal:** Feature parity with JEOD's verified capabilities.

**Crates:** All — advanced features added to existing crates

**Deliver:**
- Advanced integrators: Gauss-Jackson, LSODE, RKF45 (adaptive step)
- Solid body tides in gravity model
- Full RNP model for Earth rotation (precession, nutation, polar motion)
- SPICE integration (via FFI to cspice, or native Rust reader)
- Contact dynamics
- Full regression suite against JEOD (Tier 4)
- Multi-body scenarios: Apollo trans-lunar, Earth-Moon, Mars

**Verify with (Tier 3 required for each new physics):**
- Tier 3 LEO 24h high-fidelity gravity (GGM05C deg 20 + polar motion) < 10 m
- Tier 3 LEO with drag (MET + ballistic drag) < 100 m/24h
- Tier 3 Earth-Moon multi-body (Sun/Moon differential accel) < 100 m/7d
- Tier 3 Mars orbit (MRO110B2 gravity) < 100 m/7d
- Tier 3 Gauss-Jackson trajectory < 1 m/24h
- Tier 3 RKF45 trajectory < 10 m/24h with adaptive stepping
- Tier 3 polar motion: Earth-fixed frame < 0.1 arcsecond/24h
- Tier 3 solid tides: ON vs OFF position delta matches JEOD delta to < 10%
- Tier 4 automated regression suite with error budget tracking

---

## 9. Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **ECS portability** | Three-layer crate split: `jeod_*` (pure physics) + `jeod_sim` (orchestration) + `bevy_jeod_*` (thin Bevy glue) | Physics algorithms in `jeod_*` are reusable anywhere. Pipeline orchestration in `jeod_sim` codifies stage ordering, gravity accumulation, force collection, and integration routing without ECS dependency. `bevy_jeod_*` systems delegate to `jeod_sim` per-body functions. A non-Bevy ECS writes its own thin glue calling the same `jeod_sim` functions, guaranteed bit-identical by cross-parity tests. |
| **Floating-point precision** | `f64` everywhere via custom components (not Bevy's `Transform`) | Orbital mechanics requires ~15 significant digits. `f32` loses km-scale accuracy at Earth-orbit distances. |
| **Math library** | `glam` with f64 features (`DVec3`, `DQuat`, `DMat3`) + `nalgebra` for NxN matrices | `glam` provides f64 types with no Bevy dependency (it's a standalone crate). `nalgebra` is better for variable-size matrices needed by spherical harmonics coefficient arrays. Both work in `jeod_*` crates. |
| **Reference frame tree** | `jeod_frames` provides an arena-based tree; `bevy_jeod_frames` maps it to Bevy's `Parent`/`Children` | Core tree is portable. Bevy layer adds ECS hierarchy for efficient queries. Other ECS layers can use their own hierarchy mechanism. |
| **Integration loop** | Custom inner loop within `FixedUpdate` with stage-tracking resource | Multi-stage integrators (RK4 = 4 stages) need multiple force evaluations per timestep. An inner loop keeps this self-contained. |
| **Gravity coefficient data** | Binary asset files loaded at runtime via Bevy's `AssetServer` (or direct file I/O in non-Bevy contexts) | Keeps multi-MB coefficient arrays out of the compiled binary. Enables runtime model swapping (e.g., switch from GGM05C to GEMT1). `jeod_gravity` provides a `load_from_file()` function independent of Bevy's asset system. |
| **Ephemeris data** | Standard JPL DE421 binary files | Well-documented format. Existing parsers available. Same files JEOD uses. `jeod_ephemeris` reads them directly; `bevy_jeod_ephemeris` wraps via `AssetServer`. |
| **Plugin granularity** | One core + one glue crate per model category | Users opt into only what they need. A simple Kepler simulation doesn't pull in atmosphere code. Parallel compilation. Non-Bevy users depend only on `jeod_*` crates. |
| **Quaternion convention** | JEOD's left-quaternion, scalar-first `[q0, q1, q2, q3]` | Must match JEOD exactly for verification. Document any conversions needed at the `glam` boundary (`glam` uses `[x, y, z, w]` ordering). |
| **Testing approach** | `#[cfg(test)]` unit tests + integration test binaries + `criterion` benchmarks | Core physics tested as pure functions (no Bevy `App` needed). Bevy integration tested separately. Matches JEOD's tiered verification. |
| **JEOD data access** | Read from `../jeod` at test time via `jeod_test_data` crate; `JEOD_PATH` env var override | Avoids duplicating or modifying JEOD files. Tests skip gracefully if JEOD checkout is absent. |

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Numerical precision drift vs. JEOD** | Tests fail despite correct implementation | Use relative tolerances scaled by magnitude. Document known precision differences between GCC and Rust's LLVM backend. Start with generous tolerances, tighten as confidence grows. |
| **Spherical harmonics performance** | Degree-2190 GGM05C is computationally expensive | Implement with cache-friendly memory layout. Benchmark early. Provide degree/order truncation as a runtime option. Consider SIMD for inner loops. |
| **JEOD verification data requires Trick** | Cannot produce Tier 3 baseline trajectories without Trick installed | Start with Tier 1 (analytical) and Tier 2 (reference values) — these need no Trick. Generate Tier 3 baselines once, store as CSV. |
| **Bevy's `FixedUpdate` assumes fixed timestep** | Adaptive integrators (RKF45, LSODE) need variable dt | Use inner sub-stepping loop within `FixedUpdate`. The outer schedule provides a maximum dt; the integrator may take smaller steps internally. |
| **Quaternion convention mismatch** | Subtle rotation bugs that pass simple tests but fail complex scenarios | Document JEOD's convention (scalar-first, left-transform) explicitly. Write conversion functions at the `glam` boundary. Test with non-trivial rotations (not just identity or 90-degree). |
| **Mass tree / attachment complexity** | Rigid body attachment/detachment is intricate and error-prone | Implement incrementally: single body first (Phase 1-2), then parent-child attachment (Phase 3), then multi-level trees (Phase 5). Test each level before proceeding. |
| **Scope creep** | JEOD has 714 source files; reimplementing everything is years of work | Strict phasing. Each phase is independently useful and verifiable. Phase 1 alone enables two-body mission analysis. Resist adding features ahead of schedule. |
| **`glam` vs `nalgebra` friction** | Two math libraries with different conventions, conversion overhead | Standardize on `glam` for 3-vectors and quaternions (hot path). Use `nalgebra` only for NxN matrices in gravity coefficients and similar. Define clear boundary types. |
| **Two-layer crate overhead** | More crates to maintain, potential API duplication | Each `bevy_jeod_*` crate is intentionally thin (~100-200 lines): component derives, system functions that delegate to `jeod_*`, and a plugin registration. The physics code only exists once. The overhead pays for itself in testability and portability. |
| **Bevy breaking changes** | Bevy's rapid release cycle breaks the glue layer | Only `bevy_jeod_*` crates need updating. Physics code in `jeod_*` is untouched. Pin Bevy version in workspace; upgrade glue crates as a batch when a new Bevy release lands. |
