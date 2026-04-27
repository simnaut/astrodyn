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

The `Frame` and `TimeScale` traits are **sealed** (`Sealed + 'static`) and
require a `const NAME: &'static str`. The catalog of frame *kinds*
(`Inertial`, `Ecef`, `BodyFrame<V>`, …) and time *scales*
(`TAI`, `TT`, …) is fixed inside `jeod_quantities`; downstream crates
cannot mint new ones. Per-vehicle and per-planet *parameter* tags are
the exception — those are intentionally extensible via macros so a
mission crate can give each vehicle a distinct `Vehicle` marker.

### A new vehicle or planet marker (downstream extensible)

Mission crates that model multiple vehicles (e.g., a chief + deputy
formation, or the ISS plus a visiting Soyuz) need distinct
compile-time `Vehicle` markers so `Position<BodyFrame<Iss>>` and
`Position<BodyFrame<Soyuz>>` are type-distinct. Use the
[`define_vehicle!`] / [`define_planet!`] macros, which are the only
way to extend the `Vehicle` / `Planet` catalog from outside
`jeod_quantities`:

```rust
use bevy_jeod::prelude::*;

define_vehicle!(Iss);
define_vehicle!(Soyuz);
define_planet!(Pluto);

// Each generates a zero-sized marker type with a sealed `Vehicle`
// (or `Planet`) impl.
let _iss_pos: Position<BodyFrame<Iss>> = Qty3::zero();
let _soyuz_pos: Position<BodyFrame<Soyuz>> = Qty3::zero();
// `Position<BodyFrame<Iss>> + Position<BodyFrame<Soyuz>>` is a
// compile error with the standard frame-mismatch diagnostic.
```

The macros generate `pub struct $name;`, the sealed `Sealed` impl, and
the `Vehicle`/`Planet` impl with `const NAME: &'static str =
stringify!($name)`. The seal is preserved because the macro reaches
`Sealed` via a private re-export inside `jeod_quantities` that
downstream code cannot name directly.

`Frame::NAME` is still the *kind* (`"BodyFrame"`, `"PlanetFixed"`),
not the per-vehicle identifier — `const &'static str` cannot splice
`V::NAME` at compile time. For diagnostics that need the fully-qualified
name (e.g., `Iss` rather than `BodyFrame`), use
`std::any::type_name::<F>()`, which `Qty3`'s `Debug` impl already does.

### A new frame *kind* (in-crate only)

Adding a new frame kind (something on par with `Inertial` / `Ecef` /
`BodyFrame`, not a per-vehicle parameter) requires editing
`crates/jeod_quantities/src/frame.rs`:

```rust
// Inside crate jeod_quantities:
use crate::sealed::Sealed;

#[derive(Debug, Clone, Copy)]
pub struct MyFrame;
impl Sealed for MyFrame {}
impl Frame for MyFrame {
    const NAME: &'static str = "MyFrame";
}
```

Then:

1. Re-export from `prelude.rs` (and from `jeod_sim::lib.rs` so the root
   `bevy_jeod::prelude` picks it up via `jeod_sim`'s re-export chain).
2. Add a `FrameTransform<MyFrame, Existing>` (and inverse) constructor where
   appropriate — typically in the crate that owns the physics translating
   between the frames.
3. Add a tier-1 unit test verifying the transform round-trips.

For a parametric frame kind (planet- or vehicle-tagged), use the
`PlanetFixed<P>` / `BodyFrame<V>` patterns already in `frame.rs` as
templates — they wrap a `PhantomData<P>` and impl `Sealed`/`Frame` with
a generic bound.

[`define_vehicle!`]: https://docs.rs/jeod_quantities/latest/jeod_quantities/macro.define_vehicle.html
[`define_planet!`]: https://docs.rs/jeod_quantities/latest/jeod_quantities/macro.define_planet.html

### A new time scale

Add to `crates/jeod_quantities/src/time_scale.rs`:

```rust
use crate::sealed::Sealed;

#[derive(Debug, Clone, Copy)]
pub struct MyScale;
impl Sealed for MyScale {}
impl TimeScale for MyScale {
    const NAME: &'static str = "MyScale";
}
```

Then:

1. Define `TimeConverter::<MyScale, From>` and the inverse with the actual
   physics in `jeod_time::time_<myscale>`.
2. Re-export from `prelude.rs`.
3. Add tier-1 round-trip + tier-2 reference-vector tests.

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

### Raw / `_unchecked` constructors

The crate provides constructors that bypass invariant validation when the
caller has external proof that the invariant holds. They follow two
conventions depending on the invariant:

- **Trusted SI-unit boundary**: `Qty3::from_raw_si(DVec3)` accepts a raw
  `glam::DVec3` interpreted in SI base units. There is no separate
  `_unchecked` variant — the choice of *which frame phantom* to attach is
  the caller's responsibility, and the only "unchecked" aspect is that
  the SI interpretation is taken on faith. Use at JEOD-CSV / `glam`
  boundary code (e.g., reading the t=0 row of a reference CSV).
- **Genuine `_unchecked` skip**: `InertiaTensor::from_dmat3_unchecked(DMat3)`
  bypasses the symmetry check that `InertiaTensor::from_dmat3` enforces.
  Use only when the symmetry of the source matrix is guaranteed by
  construction (e.g., rotating a verified-symmetric tensor through an
  orthogonal change-of-basis: `R^T · I · R` preserves symmetry up to
  floating-point noise).

For quaternion validity, use `NormalizedQuat::new(q)?` (which validates the
norm against `NormalizedQuat::DEFAULT_TOLERANCE = 1e-12` and returns
`Err(NotNormalized)` if it fails) or `NormalizedQuat::renormalize(q)`
(which forces `|q| = 1` and returns `Option`). There is no
`from_raw_unchecked` variant; callers that need the witness without a
runtime check should renormalize.

### `// allowed:` comments

The escape-hatch CI guard (`scripts/check_no_escape_hatches.sh`) refuses
`#[doc(hidden)]` and the `tag_as_inertial!` macro in `crates/` and `src/`
unless the line carries a `// allowed: <reason>` opt-out comment. (The
`tag_as_inertial!` macro itself does not currently exist — the script
greps defensively to keep the door closed against future re-introduction.)
Each `// allowed:` exemption is reviewed at PR time.

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
