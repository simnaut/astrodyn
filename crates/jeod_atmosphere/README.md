# jeod_atmosphere

Neutral-atmosphere density, temperature, pressure, and wind models for the
[`bevy_jeod`](https://github.com/simnaut/bevy_jeod) workspace.

Ports
[`models/environment/atmosphere/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/atmosphere/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod), including the
Marshall Engineering Thermosphere (MET) implementation in
[`models/environment/atmosphere/MET/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/atmosphere/MET/).

## Layered architecture

```
bevy_jeod        (Bevy ECS adapter, mission code)
   ↓
jeod_sim         (orchestration, recipes, single API surface)
   ↓
jeod_atmosphere  ←  this crate (pure Rust, zero Bevy)
   ↓
jeod_quantities  (typed mass density, pressure, velocity)
```

`jeod_atmosphere` is part of the `jeod_*` physics layer — pure Rust with
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

- [`docs/JEOD_invariants.md`](https://github.com/simnaut/bevy_jeod/blob/main/docs/JEOD_invariants.md) — `AT.*`
  invariants this crate enforces.
- [Project README](https://github.com/simnaut/bevy_jeod/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/bevy_jeod/blob/main/CLAUDE.md) — workspace-level architecture.
- Rendered rustdoc:
  <https://simnaut.github.io/bevy_jeod/jeod_atmosphere/>
