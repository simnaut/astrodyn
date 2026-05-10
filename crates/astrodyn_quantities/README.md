# astrodyn_quantities

Dimensional-analysis and phantom-tag foundation for the
[`astrodyn_bevy`](https://github.com/simnaut/astrodyn) workspace.

Sits at the bottom of the workspace dependency graph. Every other
`astrodyn_*` crate, plus `astrodyn`, `astrodyn_runner`, and the `astrodyn_bevy`
Bevy glue, depends on `astrodyn_quantities` for typed quantities and the
phantom frame / time-scale tags.

## Three-layer facade

```
┌──────────────────────────────────────────────────────────┐
│ Facade  (astrodyn_bevy::prelude, astrodyn::recipes)          │
│   F64Ext: 400.0.km(), 51.6.deg(), 420_000.0.kg()         │
│   Concrete Component wrappers (no visible generics)      │
│   Custom #[diagnostic::on_unimplemented] messages        │
├──────────────────────────────────────────────────────────┤
│ Typed astrodyn_* siblings                                     │
│   Position<F: Frame>, SecondsSince<S: TimeScale>,        │
│   Quat<L, T>, NormalizedQuat, FrameTransform<From, To>   │
├──────────────────────────────────────────────────────────┤
│ astrodyn_quantities  (you are here)                          │
│   uom re-exports, Qty3<D, F>, phantom frames/scales,     │
│   F64Ext / Vec3Ext / Array3Ext                           │
└──────────────────────────────────────────────────────────┘
```

Mission-crate code consumes the facade layer and never sees
`PhantomData` or `uom::si::*` paths. Internal physics kernels drop
down to raw `glam::DVec3` / `f64` for arithmetic density via
`.raw_si()` and re-wrap on exit.

## Public surface

- Reference-frame phantom markers — three kind-distinct inertial
  flavors plus rotating / vehicle frames:
  - `RootInertial` — the simulation's unique root inertial frame.
  - `PlanetInertial<P: Planet>` — a particular planet's inertial frame
    (centered on `P`'s CoM, non-rotating).
  - `IntegrationFrame` — a body's integration frame; only convertible
    to `RootInertial` via the `IntegOrigin` shift API (issue #255 /
    `RF.10`).
  - Plus `Ecef`, `PlanetFixed<P>`, `BodyFrame<V>`, `StructuralFrame<V>`,
    `Lvlh<Chief>`, `Ned<Chief>`.
- Time-scale phantom markers (`TAI`, `TT`, `TDB`, `UT1`, `UTC`, …).
- `uom`-backed componentwise 3-vectors `Qty3<D, F>` with aliases
  `Position<F>`, `Velocity<F>`, `Acceleration<F>`, `Force<F>`,
  `Torque<F>`, …
- Quaternion convention tags (`ScalarFirst`/`ScalarLast`,
  `LeftTransform`/`RightTransform`) plus the `NormalizedQuat`
  constructor-gated witness.
- Typed `FrameTransform<From, To>` composing only when inner frames
  match.
- The `F64Ext` facade (`400.0.km()`, `51.6.deg()`, `420_000.0.kg()`).
- Compiler error messages in physics language via
  `#[diagnostic::on_unimplemented]`.

## See also

- [Type-System wiki page](https://github.com/simnaut/astrodyn/wiki/Type-System)
  — contributor primer (phantom-tag pattern, adding a new
  frame/scale/quantity, escape hatches).
- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture.
- Rendered rustdoc:
  <https://docs.rs/astrodyn_quantities>
