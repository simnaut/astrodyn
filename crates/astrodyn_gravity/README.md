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

## When to use

- **Evaluating gravitational acceleration** at a body's inertial
  position against one or more sources — point-mass through the
  full Gottlieb spherical-harmonics expansion at the source's full
  degree / order.
- **Computing the gravity-gradient tensor** for tidal forces,
  gravity-gradient torque (consumed by `astrodyn_interactions`), or
  diagnostics.
- **Configuring a vehicle's gravity controls** — choosing which
  sources to include, whether to add Battin / relativistic
  corrections, and whether to evaluate the gradient.
- **Loading JEOD coefficient files** (`earth_GGM05C.hh`,
  `moon_GRAIL150.hh`, …) into the binary fixture format consumed at
  runtime — via the `extract_grav_coeffs` regen binary.

Production pipelines never parse JEOD `.cc` files at runtime;
`extract_grav_coeffs` runs offline and writes binary fixtures under
`test_data/gravity/` that the runtime loads.

## Key concepts

The Gottlieb algorithm replaces the classical
unnormalized-associated-Legendre recursion with a **normalized**
recursion that pushes the dynamic-range problem out of the polynomial
tail. This is the only formulation that stays numerically valid past
degree ~30; JEOD uses it for GGM05C (up to 360×360) and we port it
verbatim, with the same row-major coefficient layout and the same
recursion order so test vectors match bit-for-bit.

`GravityControl` is **per-source**: a vehicle can pull spherical
harmonics from Earth while taking the Sun and Moon as point masses,
and toggle Battin third-body or post-Newtonian relativistic
corrections independently. `GottliebScratch` holds the recursion's
intermediate buffers so they can be reused across timesteps without
reallocating. Gravity acceleration here **excludes** the integration
frame's own acceleration toward the source — third-body contributions
arrive as the differential acceleration (vehicle-toward-Sun minus
Earth-toward-Sun), which is the JEOD convention and the only choice
that keeps Earth-centered inertial integration consistent.

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
are parsed by `astrodyn_gravity::jeod_cc` into the binary fixtures
committed under `test_data/gravity/`; production gravity does not parse
JEOD source.

## See also

- [`docs/JEOD_invariants.md`](https://github.com/simnaut/astrodyn/blob/main/docs/JEOD_invariants.md) — `GV.*`
  invariants this crate enforces.
- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture.
- Rendered rustdoc:
  <https://docs.rs/astrodyn_gravity>
