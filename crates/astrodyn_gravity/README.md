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
are parsed by `astrodyn_gravity::jeod_cc` into the binary fixtures
committed under `test_data/gravity/`; production gravity does not parse
JEOD source.

## Fixture provenance

Every committed `.bin` under `test_data/gravity/` carries a sidecar
`<name>.json` (schema 2) recording the upstream source path, JEOD
version, JEOD commit SHA at extraction time, generation timestamp, and
the SHA-256 + byte count of the produced binary. Verbatim-mirrored text
fixtures (e.g. `grav_geospherical_verif_out.txt`) carry a parallel
`<name>.meta.json`. The workspace-level `tests/fixture_metadata.rs`
asserts that every committed fixture has a matching sidecar with
size + SHA-256 fields that match the file bytes, so a regen run that
desynchronises the metadata fails CI. Re-run
`cargo run -p astrodyn_gravity --bin extract_grav_coeffs` (or
`extract_mars_data`) against a JEOD checkout to refresh the fixtures
and sidecars together.

## See also

- [`docs/JEOD_invariants.md`](https://github.com/simnaut/astrodyn/blob/main/docs/JEOD_invariants.md) — `GV.*`
  invariants this crate enforces.
- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture.
- Rendered rustdoc:
  <https://docs.rs/astrodyn_gravity>
