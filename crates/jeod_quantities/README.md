# jeod_quantities

Dimensional-analysis and phantom-tag foundation for the
[`bevy_jeod`](https://github.com/simnaut/bevy_jeod) workspace.

Sits at the bottom of the workspace dependency graph. Every other
`jeod_*` crate, plus `jeod_sim`, `jeod_runner`, and the `bevy_jeod`
Bevy glue, depends on `jeod_quantities` for typed quantities and the
phantom frame / time-scale tags.

## Three-layer facade

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
│ jeod_quantities  (you are here)                          │
│   uom re-exports, Qty3<D, F>, phantom frames/scales,     │
│   F64Ext / Vec3Ext / Array3Ext                           │
└──────────────────────────────────────────────────────────┘
```

Mission-crate code consumes the facade layer and never sees
`PhantomData` or `uom::si::*` paths. Internal physics kernels drop
down to raw `glam::DVec3` / `f64` for arithmetic density via
`.raw_si()` and re-wrap on exit.

## Public surface

- Reference-frame and time-scale phantom markers (`Inertial`, `Ecef`,
  `PlanetFixed<P>`, `BodyFrame<V>`, `Lvlh<Chief>`, `TAI`, `TT`, …).
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

- [Type-System wiki page](https://github.com/simnaut/bevy_jeod/wiki/Type-System)
  — contributor primer (phantom-tag pattern, adding a new
  frame/scale/quantity, escape hatches).
- [Project README](../../README.md) and
  [`CLAUDE.md`](../../CLAUDE.md) — workspace-level architecture.
- Rendered rustdoc:
  <https://simnaut.github.io/bevy_jeod/jeod_quantities/>
