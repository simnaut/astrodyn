# astrodyn_atmosphere

Neutral-atmosphere density, temperature, pressure, and wind models for the
[`astrodyn_bevy`](https://github.com/simnaut/astrodyn) workspace.

Ports
[`models/environment/atmosphere/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/atmosphere/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod), including the
Marshall Engineering Thermosphere (MET) implementation in
[`models/environment/atmosphere/MET/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/atmosphere/MET/).

## When to use

- **Aerodynamic drag** — pair an `AtmosphereState` evaluation with
  `astrodyn_interactions::aero_drag` (scalar Cd) or `flat_plate_aero`
  (per-facet panel method) to compute drag force on a LEO vehicle.
- **Density / temperature / pressure profiles** for thermal,
  out-gassing, or trajectory-energy diagnostics at sub-1000-km
  altitudes.
- **Inertial-frame wind** (`compute_corotation_wind`) — every drag
  computation needs the vehicle velocity *relative to the
  co-rotating atmosphere*, not relative to the bare inertial frame.

This crate stays pure; orchestration (pulling the body position, the
current epoch, the planet's angular velocity, and feeding `aero_drag`)
lives in `astrodyn`.

## Key concepts

Every atmosphere evaluation returns one type — `AtmosphereState` —
holding density (kg/m³), temperature (K), pressure (Pa), and the
inertial-frame wind velocity (m/s). Consumers that need only density
or only wind read the matching field; there is no "fast path" that
elides the rest, by design, because the cost of computing them
together is already amortized inside the MET kernel.

Two models are provided. `exponential` is the
`rho = rho_0 * exp(-(h - h_0) / H)` single-scale-height fallback,
intended as a unit-test baseline and as a stand-in for studies that
don't need MET fidelity. `met` is the JEOD-faithful port of the
Marshall Engineering Thermosphere — Jacchia 1970/1971 temperature
profiles integrated via Gauss quadrature plus seasonal-latitude
density corrections — and is the production choice for LEO drag.

Co-rotation wind specializes to planets whose angular velocity points
along the inertial Z axis (Earth, Mars, Venus to good approximation):
`wind = omega × r`. JEOD makes the same simplification; bodies with
significant axis tilt would require a per-step rotation-axis input.

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
