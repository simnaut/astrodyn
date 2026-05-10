# astrodyn_atmosphere

Neutral-atmosphere density, temperature, pressure, and wind models for the
[`astrodyn_bevy`](https://github.com/simnaut/astrodyn) workspace.

Ports
[`models/environment/atmosphere/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/atmosphere/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod), including the
Marshall Engineering Thermosphere (MET) implementation in
[`models/environment/atmosphere/MET/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/atmosphere/MET/).

## Layered architecture

```
astrodyn_bevy        (Bevy ECS adapter, mission code)
   ↓
astrodyn         (orchestration, recipes, single API surface)
   ↓
astrodyn_atmosphere  ←  this crate (pure Rust, zero Bevy)
   ↓
astrodyn_quantities  (typed mass density, pressure, velocity)
```

`astrodyn_atmosphere` is part of the `astrodyn_*` physics layer — pure Rust with
no Bevy dependency.

## Public surface

- `AtmosphereState` — density / temperature / pressure / inertial-frame
  wind, returned by every model evaluation.
- `exponential` — `rho = rho_0 * exp(-(h - h_0) / H)`. Single-parameter
  scale-height fallback and unit-test baseline.
- `met` — Marshall Engineering Thermosphere model (Jacchia 1970/1971).
  Production model for LEO drag.
- `compute_corotation_wind` / `compute_corotation_wind_typed` — `omega ×
  r` co-rotation wind for planets whose angular velocity points along
  the inertial Z axis.

## See also

- [`docs/JEOD_invariants.md`](https://github.com/simnaut/astrodyn/blob/main/docs/JEOD_invariants.md) — `AT.*`
  invariants this crate enforces.
- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture.
- Rendered rustdoc:
  <https://docs.rs/astrodyn_atmosphere>
