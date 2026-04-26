# The bevy_jeod Type System

This document is the contributor primer for the typed quantity layer added by
the type-system refactor (#101, Phases 0–11). It serves two audiences:

- **`bevy_jeod` internal contributors** adding a new frame, time scale,
  dimension, or recipe.
- **Mission-crate authors** decoding compiler errors and understanding the
  conventions encoded in the type system.

If you are writing mission code (downstream of `bevy_jeod`), start with
`examples/typed_mission.rs` and the `## Building a Mission Crate` section in
`CLAUDE.md`. Use this document as the reference when you need to know *why*
the compiler refused something.

## 1. Why a typed layer

`bevy_jeod` reimplements NASA JEOD orbital mechanics. JEOD's C++ API uses
naked `double`s and `double[3]`s carrying conventions in field names and
comments — sign conventions, frame conventions, time scales, quaternion
layouts. Those conventions are not compile-checked; getting one wrong produces
code that compiles, passes trivial tests, and silently gives wrong answers
at scale.

The motivating incident (catalogued in `CLAUDE.md` "JEOD Convention Rule"):
an agent guessed `M = 2π − n·t` for the JEOD `time_periapsis` → mean anomaly
formula. The correct convention is `M = n·t`. The bug produced **11,668 km
error against NASA flight data** and was hidden for multiple commits because
a broken test path silently skipped the validation. Reading
`models/dynamics/body_action/src/dyn_body_init_orbit.cc` would have given the
correct formula immediately, but no compile-time check could fire.

The type-system refactor moves this class of bug from runtime/discipline to
compile time. Frame mismatches, time-scale mismatches, scalar-vs-vector
quaternion confusion, and unit-dimensional errors are now compile errors —
in **physics language**, via custom `#[diagnostic::on_unimplemented]`
messages.

## 2. The three-layer facade

```
┌──────────────────────────────────────────────────────────┐
│ Facade  (bevy_jeod::prelude, jeod_sim::recipes)          │
│   F64Ext: 400.0.km(), 51.6.deg(), 420_000.0.kg()         │
│   Concrete Component wrappers (no visible generics)      │
│   Custom #[diagnostic::on_unimplemented] messages        │
├──────────────────────────────────────────────────────────┤
│ Typed jeod_* siblings                                     │
│   Position<F: Frame>, SecondsSince<S: TimeScale>,        │
│   Quat<L, T>, NormalizedQuat, FrameTransform<From, To>   │
├──────────────────────────────────────────────────────────┤
│ jeod_quantities  (bottom of dep graph)                   │
│   uom re-exports, Qty3<D, F>, phantom frames/scales,     │
│   F64Ext / Vec3Ext / Array3Ext                           │
└──────────────────────────────────────────────────────────┘
```

- **`jeod_quantities`** sits at the bottom of the workspace dependency graph.
  Every other `jeod_*` crate depends on it. It defines the typed primitives:
  - `Qty3<D, F>` — componentwise 3-vector with `uom` dimension `D` and
    phantom frame tag `F`. Aliases: `Position<F>`, `Velocity<F>`,
    `Acceleration<F>`, `Force<F>`, `Torque<F>`, etc.
  - `SecondsSince<S>` — a `uom` `Time` carrying a phantom time-scale tag `S`.
  - `Quat<L, T>` — quaternion carrying phantom layout (`ScalarFirst`/`ScalarLast`)
    and transform-convention (`LeftTransform`/`RightTransform`) tags.
    `NormalizedQuat<L, T>` is a constructor-gated witness.
  - `FrameTransform<From, To>` — typed transform that composes only when
    inner frames match.
  - `F64Ext` — the facade trait that lets mission code write
    `400.0.km()` instead of `Length::new::<kilometer>(400.0)`.

- **Typed `jeod_*` siblings** — every public physics function in `jeod_*`
  has a typed entry point (the f64 forms were deleted in Phase 10). Each
  typed function takes typed inputs, calls `.raw_si()` to drop into the
  shared kernel for arithmetic density, and re-wraps on exit.

- **Facade** — `bevy_jeod::prelude` and `jeod_sim::recipes` re-export
  concrete typed Components and recipe functions so mission code never sees
  `PhantomData` or `uom::si::*` paths. The typestate `VehicleBuilder`
  (`NeedsState → NeedsMass → HasIntegrator → Ready`) gates construction
  ordering at compile time.

## 3. Phantom tags catalog

### Frames (`crates/jeod_quantities/src/frame.rs`)

| Tag | Meaning |
|-----|---------|
| `Inertial` | Earth-centered inertial (J2000) |
| `Ecef` | Earth-centered Earth-fixed (rotates with Earth) |
| `PlanetFixed<P>` | Generic planet-fixed frame parameterized by planet `P` (`Earth`, `Moon`, `Mars`, `Sun`, …) |
| `BodyFrame<V>` | Body-fixed frame of a vehicle `V` |
| `StructuralFrame<V>` | Structural reference frame of a vehicle `V` (used for sensor placement, attachment) |
| `Lvlh<Chief>` | Local Vertical / Local Horizontal frame relative to a chief body |
| `Ned<Chief>` | North-East-Down topocentric frame relative to a chief body |
| `SelfRef` | "Same frame as the carrier" — used by Components that store a quantity expressed in their own entity's frame |

Frame phantoms only exist at the type level; they have no runtime representation.

### Time scales (`crates/jeod_quantities/src/time_scale.rs`)

`TAI`, `UTC`, `UT1`, `TT`, `TDB`, `GPS`, `GMST`. `SecondsSince<S>` carries
the phantom; `TimeConverter<From, To>` is the explicit conversion entry point
(e.g., `TAI_TO_TT`, `GPS_TO_TAI`).

### Quaternion conventions (`crates/jeod_quantities/src/quat.rs`)

| Axis | Tags |
|------|------|
| Layout | `ScalarFirst` (JEOD's convention: `[q0, q1, q2, q3]`), `ScalarLast` (glam's: `[x, y, z, w]`) |
| Transform | `LeftTransform` (JEOD: `r' = q r q⁻¹`), `RightTransform` |
| Normalization | `Quat<L, T>` (raw), `NormalizedQuat<L, T>` (witness; constructed via `NormalizedQuat::new(q)?` or `NormalizedQuat::renormalize(q)`) |

JEOD-internal physics uses `Quat<ScalarFirst, LeftTransform>`. Conversion to
`glam::DQuat` (which is `ScalarLast`) happens at module boundaries.

## 4. Adding a new frame / time scale / quantity

### A new frame tag

1. Add the marker type to `crates/jeod_quantities/src/frame.rs`:
   ```rust
   pub struct MyFrame;
   impl Frame for MyFrame {}
   ```
2. Re-export from `prelude.rs` if it should be visible to mission code.
3. Add a `FrameTransform<MyFrame, Existing>` (and inverse) constructor where
   appropriate — typically in the crate that owns the physics translating
   between the frames.
4. Add a tier-1 unit test verifying the transform round-trips.

### A new time scale

1. Add the marker to `crates/jeod_quantities/src/time_scale.rs`:
   ```rust
   pub struct MyScale;
   impl TimeScale for MyScale {}
   ```
2. Define `TimeConverter::<MyScale, From>` and the inverse with the actual
   physics in `jeod_time::time_<myscale>`.
3. Re-export from `prelude.rs`.
4. Add tier-1 round-trip + tier-2 reference-vector tests.

### A new dimensional quantity

1. If `uom` already has the dimension (most common), add a type alias to
   `crates/jeod_quantities/src/aliases.rs` and a `Qty3` alias for the
   3-vector form:
   ```rust
   pub type AngularMomentum<F> = Qty3<dims::AngularMomentum, F>;
   ```
2. If the dimension is new, add it to `crates/jeod_quantities/src/dims.rs`
   using `uom`'s dimension macros.
3. Add `F64Ext` constructor methods (e.g., `.kg_m2_per_s()`).
4. Re-export from `prelude.rs`.

## 5. Reading compiler errors

The custom diagnostics in `crates/jeod_quantities/src/diagnostics.rs` are
zero-cost marker traits whose only purpose is to carry a tailored
`#[diagnostic::on_unimplemented]` message that fires when the marker bound
fails. The error then renders in physics language instead of as a generic
"trait not implemented" wall.

### Frame mismatch on `+` / `-`

```rust
let a: Position<Inertial> = ...;
let b: Position<Ecef> = ...;
let _ = a + b; // ← error
```

```text
error: cannot combine values in frame `Inertial` with values in frame `Ecef`
   = note: apply a `FrameTransform<Ecef, Inertial>` (or its inverse) to bring
           both operands into the same frame before combining
```

### Bare f64 where a typed quantity is expected

```rust
fn altitude_check(altitude: Length) { ... }
altitude_check(400_000.0); // ← error
```

```text
error: bare `f64` is not a `Length` — attach a unit with `F64Ext`
   = note: use `.m()`, `.km()`, `.cm()`, `.mm()`, `.ft()`, `.mi()`, or `.nmi()`
           to produce a `Length`
```

Fix: `altitude_check(400.0.km())`.

### Quaternion convention mismatch

```rust
let q_glam: Quat<ScalarLast, LeftTransform> = ...;
let q_jeod: Quat<ScalarFirst, LeftTransform> = q_glam; // ← error
```

```text
error: quaternion layout mismatch: expected `ScalarFirst`, found `ScalarLast`
   = note: use `.to_scalar_first()` or `.to_scalar_last()` to convert
           between layouts
```

### Vector × Vector ambiguity

```rust
let r: Position<Inertial> = ...;
let v: Velocity<Inertial> = ...;
let _ = r * v; // ← error: ambiguous
```

```text
error: two `Qty3`s cannot be multiplied componentwise
   = note: use `.dot(other)` for scalar product (returns a scalar) or
           `.cross(other)` for vector product (returns a `Qty3`)
```

The full set of custom diagnostics is in
`crates/jeod_quantities/src/diagnostics.rs`. New ones should follow the same
pattern: zero-cost marker trait, `#[diagnostic::on_unimplemented]` message in
physics language, the `note:` suggesting the corrective API.

## 6. Runtime escape hatches

The type system has two documented escape hatches. Both are deliberate; both
require justification in the PR description that introduces a use.

### `_unchecked` constructors

`Qty3::from_raw_si_unchecked(DVec3)`, `NormalizedQuat::from_raw_unchecked(DQuat)`,
etc. These bypass the typed constructor's validation (e.g., normalization).
They follow Rust convention: the `_unchecked` suffix is grep-able and
recognizable. Use only when:

- Constructing a typed quantity from a value that is *known* by construction
  to satisfy the invariant (e.g., reading the t=0 row of a JEOD reference
  CSV that has already been validated upstream).
- The validating constructor would be redundant or measurably impact a hot
  path.

### `// allowed:` comments

The escape-hatch CI guard (`scripts/check_no_escape_hatches.sh`) refuses
`#[doc(hidden)]` and the `tag_as_inertial!` macro in `crates/` and `src/`
unless the line carries a `// allowed: <reason>` opt-out comment. Each
exemption is reviewed at PR time.

If a future contributor needs to bypass the type system at a public surface,
the answer is almost always to extend the type system to express the missing
case — not to widen the escape hatches. The escape hatches exist for legacy
boundaries (JEOD CSV ingestion, `glam::DQuat` interop) and should not grow.

## 7. References

- **Source**:
  - `crates/jeod_quantities/src/lib.rs` — crate root with module-level docs.
  - `crates/jeod_quantities/src/diagnostics.rs` — full custom-diagnostic catalog.
  - `crates/jeod_quantities/src/frame.rs`, `time_scale.rs`, `quat.rs` —
    phantom-tag definitions.
  - `crates/jeod_quantities/src/qty3.rs` — `Qty3<D, F>` and its operations.

- **Worked examples**:
  - `examples/typed_mission.rs` — canonical mission-crate composition.
  - `examples/kepler_orbit.rs` — minimal orbit propagator.
  - `crates/jeod_runner/examples/{apollo,earth_moon,mars_orbit,leo_drag,batch_propagation}.rs` —
    larger scenario demonstrations.

- **Architecture**:
  - `STRATEGY.md` §8 "Phase 8: Type-System Refactor (#101)".
  - `CLAUDE.md` "Precision" and "Building a Mission Crate" sections.

- **Phase issues** (closed): #102, #103, #104, #105, #106, #107, #108, #109,
  #110, #111, #112, #113. Parent: #101.

- **Future work** (deferred trackers filed at Phase 11 close-out): #150
  (session types), #151 (capability tokens), #152 (branded simulation
  lifetimes), #153 (Docker CSV pipeline), #154 (Bevy Reflect), #155
  (FrameTransform Component erasure), #156 (`pre_step` hook), #157
  (`EvaluationCase` shape).
