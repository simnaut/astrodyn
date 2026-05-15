# astrodyn_interactions

Surface interactions — aerodynamic drag, solar radiation pressure,
gravity-gradient torque, shadow geometry, contact, thermal — for the
[`astrodyn_bevy`](https://github.com/simnaut/astrodyn) workspace.

Ports
[`models/interactions/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/interactions/)
and the surface-model utilities under
[`models/utils/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod).

## When to use

- **Aerodynamic drag** — scalar-Cd ballistic-coefficient
  (`aero_drag`) or per-facet panel method (`flat_plate_aero`)
  against an atmospheric state from `astrodyn_atmosphere`.
- **Solar radiation pressure** — `radiation_pressure` against a
  surface model with absorption / specular / diffuse coefficients,
  shadow-corrected via umbra / penumbra geometry and optionally
  Earth-albedo / IR contributions for Earth-orbit vehicles.
- **Gravity-gradient torque** — `compute_gravity_torque` consumes
  the gradient tensor from `astrodyn_gravity` and projects through
  the body's inertia tensor and attitude.
- **Surface models** — `SurfaceFacet`, `ArticulatedFacet`,
  `SurfaceShape` are the shared per-facet geometry inputs that
  aero, SRP, contact, and thermal all read from.
- **Contact and thermal** — `contact` (collision response) and
  `thermal_rider` (per-facet power balance) for articulated
  spacecraft and EVA / robotic-arm studies.

The orchestration that sums these contributions into a body's
`astrodyn_dynamics::TotalForce` lives in `astrodyn`; this crate keeps
each model side-effect-free so it composes cleanly.

## Key concepts

Every interaction module follows the same shape: it takes a vehicle's
state (position, velocity, attitude), an environmental input (an
`AtmosphereState`, a Sun position, a gravity-gradient tensor), and a
geometry input (a `SurfaceFacet` for panel methods, a ballistic
coefficient + cross-section for scalar models), and returns a force,
torque, or scalar without mutating its inputs. That separation is
what lets the orchestration layer drive aero + SRP + gravity-gradient
in any order or in parallel.

The surface model `SurfaceFacet` is shared across aero, SRP,
contact, and thermal — one facet definition, four physical responses
— with `ArticulatedFacet` adding a hinge so robotic arms / solar
panels / antenna gimbals can sweep through their commanded angles.
Shadow geometry (`shadow`) is a separate module from
`radiation_pressure` because shadow is also consumed by
`earth_lighting` and by thermal calculations; the umbra / penumbra
calculation runs once per step regardless of how many interactions
depend on it.

## Layered architecture

```
astrodyn_bevy        (Bevy ECS adapter, mission code)
   ↓
astrodyn         (orchestration; sums forces into TotalForce)
   ↓
astrodyn_interactions ←  this crate (pure Rust, zero Bevy)
   ↓
astrodyn_atmosphere, astrodyn_quantities
```

Each module produces a force, torque, or environmental scalar at a
vehicle position. The orchestration that sums them into a body's
`astrodyn_dynamics::TotalForce` lives in `astrodyn`.

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

- [`docs/JEOD_invariants.md`](https://github.com/simnaut/astrodyn/blob/main/docs/JEOD_invariants.md) — `AE.*`,
  `RP.*`, `GG.*`, `SH.*`, `CT.*`, `TH.*` invariants.
- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture.
- Rendered rustdoc:
  <https://docs.rs/astrodyn_interactions>
