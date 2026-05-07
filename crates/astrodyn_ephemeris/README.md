# astrodyn_ephemeris

Planetary ephemerides backed by JPL DE-series SPK files for the
[`astrodyn_bevy`](https://github.com/simnaut/astrodyn) workspace.

Ports
[`models/environment/ephemerides/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/ephemerides/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod). Where JEOD links a
hand-rolled binary loader to JPL DE405 / DE421 kernels, this crate delegates
the file format and Chebyshev evaluation to the
[`anise`](https://crates.io/crates/anise) crate (a pure-Rust SPICE/NAIF
reimplementation) and exposes a thin, frame-tagged API on top.

## Layered architecture

```
astrodyn_bevy        (Bevy ECS adapter, mission code)
   ↓
astrodyn         (orchestration, recipes, single API surface)
   ↓
astrodyn_ephemeris   ←  this crate (pure Rust, zero Bevy)
   ↓
astrodyn_quantities  (Position<RootInertial>, Velocity<RootInertial>, frame tags)
```

`astrodyn_ephemeris` is part of the `astrodyn_*` physics layer — pure Rust with no
Bevy dependency. The Bevy-side glue lives in `astrodyn_bevy::*`; the pipeline
that drives it lives in `astrodyn`. See the project
[Strategy wiki page](https://github.com/simnaut/astrodyn/wiki/Strategy)
for the layered architecture.

## Public surface

- `Ephemeris` — owns an `anise::Almanac` and answers position/velocity
  queries from `.bsp` files (e.g., `de421.bsp`, `de440.bsp`).
- `EphemerisBody` — body-identifier enum mapping JEOD's body IDs to
  NAIF integer IDs.
- `EphemerisError` — fail-loudly error type for missing files,
  unsupported bodies, or out-of-range epochs.

## See also

- [`docs/JEOD_invariants.md`](https://github.com/simnaut/astrodyn/blob/main/docs/JEOD_invariants.md) — `EP.*`
  tags catalog the ephemeris invariants this crate enforces.
- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture and
  conventions.
- Rendered rustdoc:
  <https://simnaut.github.io/astrodyn_bevy/astrodyn_ephemeris/>
