# astrodyn_gravity

Gravity computation for the
[`astrodyn_bevy`](https://github.com/simnaut/astrodyn) workspace —
point-mass, spherical harmonics (Gottlieb algorithm), tides, and
post-Newtonian relativistic corrections.

Ports
[`models/environment/gravity/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/gravity/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod). The core spherical-
harmonics kernel is a faithful port of
[`spherical_harmonics_calc_nonspherical.cc`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/gravity/src/spherical_harmonics_calc_nonspherical.cc),
a numerically stable normalized Legendre recursion that scales to high
degree and order without the underflow / overflow problems of the
classical formulation.

## Layered architecture

```
astrodyn_bevy        (Bevy ECS adapter, mission code)
   ↓
astrodyn         (orchestration, recipes, single API surface)
   ↓
astrodyn_gravity     ←  this crate (pure Rust, zero Bevy)
   ↓
astrodyn_dynamics, astrodyn_math, astrodyn_quantities
```

`astrodyn_gravity` is part of the `astrodyn_*` physics layer — pure Rust with
no Bevy dependency.

## Public surface

- `calc_spherical`, `gravitation`, `gravitation_with_scratch` — point-
  mass and dispatched gravity computation.
- `calc_nonspherical*`, `GottliebScratch` — the Gottlieb spherical-
  harmonics kernel + reusable scratch buffers.
- `GravitySource`, `GravityModel`, `SphericalHarmonicsData` — per-body
  μ + coefficient payload.
- `GravityControl`, `GravityControls` — per-source selectors (degree /
  order, gradient, third-body / Battin / relativistic toggles).
- `tides`, `relativistic` — small-correction terms.

Coefficient files at JEOD source paths like
[`earth_GGM05C.hh`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/gravity/data/include/earth_GGM05C.hh)
are parsed by `astrodyn_test_data::jeod_cc` into the binary fixtures
committed under `test_data/gravity/`; production gravity does not parse
JEOD source.

## See also

- [`docs/JEOD_invariants.md`](https://github.com/simnaut/astrodyn/blob/main/docs/JEOD_invariants.md) — `GV.*`
  invariants this crate enforces.
- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture.
- Rendered rustdoc:
  <https://simnaut.github.io/astrodyn_bevy/astrodyn_gravity/>
