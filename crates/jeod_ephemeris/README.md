# jeod_ephemeris

Planetary ephemerides backed by JPL DE-series SPK files for the
[`bevy_jeod`](https://github.com/simnaut/bevy_jeod) workspace.

Ports
[`models/environment/ephemerides/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/ephemerides/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod). Where JEOD links a
hand-rolled binary loader to JPL DE405 / DE421 kernels, this crate delegates
the file format and Chebyshev evaluation to the
[`anise`](https://crates.io/crates/anise) crate (a pure-Rust SPICE/NAIF
reimplementation) and exposes a thin, frame-tagged API on top.

## Layered architecture

```
bevy_jeod        (Bevy ECS adapter, mission code)
   ↓
jeod_sim         (orchestration, recipes, single API surface)
   ↓
jeod_ephemeris   ←  this crate (pure Rust, zero Bevy)
   ↓
jeod_quantities  (Position<Inertial>, Velocity<Inertial>, frame tags)
```

`jeod_ephemeris` is part of the `jeod_*` physics layer — pure Rust with no
Bevy dependency. The Bevy-side glue lives in `bevy_jeod::*`; the pipeline
that drives it lives in `jeod_sim`. See the project
[Strategy wiki page](https://github.com/simnaut/bevy_jeod/wiki/Strategy)
for the layered architecture.

## Public surface

- `Ephemeris` — owns an `anise::Almanac` and answers position/velocity
  queries from `.bsp` files (e.g., `de421.bsp`, `de440.bsp`).
- `EphemerisBody` — body-identifier enum mapping JEOD's body IDs to
  NAIF integer IDs.
- `EphemerisError` — fail-loudly error type for missing files,
  unsupported bodies, or out-of-range epochs.

## See also

- [`docs/JEOD_invariants.md`](../../docs/JEOD_invariants.md) — `EP.*`
  tags catalog the ephemeris invariants this crate enforces.
- [Project README](../../README.md) and
  [`CLAUDE.md`](../../CLAUDE.md) — workspace-level architecture and
  conventions.
- Rendered rustdoc:
  <https://simnaut.github.io/bevy_jeod/jeod_ephemeris/>
