# jeod_interactions

Surface interactions — aerodynamic drag, solar radiation pressure,
gravity-gradient torque, shadow geometry, contact, thermal — for the
[`bevy_jeod`](https://github.com/simnaut/bevy_jeod) workspace.

Ports
[`models/interactions/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/interactions/)
and the surface-model utilities under
[`models/utils/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod).

## Layered architecture

```
bevy_jeod        (Bevy ECS adapter, mission code)
   ↓
jeod_sim         (orchestration; sums forces into TotalForce)
   ↓
jeod_interactions ←  this crate (pure Rust, zero Bevy)
   ↓
jeod_atmosphere, jeod_quantities
```

Each module produces a force, torque, or environmental scalar at a
vehicle position. The orchestration that sums them into a body's
`jeod_dynamics::TotalForce` lives in `jeod_sim`.

## Public surface

- `aero_drag` (scalar-Cd) and `flat_plate_aero` (per-facet) —
  aerodynamic drag.
- `radiation_pressure` — solar flux against a surface model with
  absorption / specular / diffuse coefficients.
- `shadow` — umbra / penumbra geometry. `earth_lighting` extends with
  Earth-albedo and IR contributions.
- `gravity_torque` — cross product of inertia tensor and
  gravity-gradient tensor through body attitude.
- `surface_model` — `SurfaceFacet`, `ArticulatedFacet`, `SurfaceShape`
  — per-facet geometry shared by aero / SRP / contact / thermal.
- `contact` — collision / contact response.
- `thermal_rider` — per-facet power balance.

## See also

- [`docs/JEOD_invariants.md`](../../docs/JEOD_invariants.md) — `AE.*`,
  `RP.*`, `GG.*`, `SH.*`, `CT.*`, `TH.*` invariants.
- [Project README](../../README.md) and
  [`CLAUDE.md`](../../CLAUDE.md) — workspace-level architecture.
- Rendered rustdoc:
  <https://simnaut.github.io/bevy_jeod/jeod_interactions/>
