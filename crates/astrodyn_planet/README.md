# astrodyn_planet

Reference-ellipsoid parameters and standard preset bodies for the
[`astrodyn_bevy`](https://github.com/simnaut/astrodyn) workspace.

Ports
[`models/environment/planet/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/planet/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod) — the per-body
shape parameters (gravitational parameter, equatorial and polar radii,
flattening) consumed by gravity, geodetic, atmospheric, and
frame-rotation code.

## When to use

- **Building a gravity source** — every `astrodyn_gravity` source
  needs `mu`; `PlanetShape::mu` is the canonical accessor.
- **Geodetic conversions** — `astrodyn_math::geodetic` needs
  `r_eq`, `r_pol`, and the flattening for the ellipsoidal latitude
  / longitude / altitude solver.
- **Atmosphere models** — MET and the exponential fallback both
  need the reference radius to compute altitude above the
  ellipsoid.
- **Reusing canonical bodies** — `presets::EARTH` / `MOON` /
  `SUN` / `MARS` give the JEOD-matched parameter blocks so
  cross-validation tests don't drift from JEOD numerics.

Most mission code reaches the preset constants via the recipe layer
in `astrodyn::recipes::earth` / `moon` / `mars`; this crate is the
pure parameter source those recipes wrap.

## Key concepts

`PlanetShape` is a value type — `name`, `mu` (m³/s²), `r_eq` (m),
`r_pol` (m), `flat_coeff` — plus a small handful of derived helpers
(`flat_inv`, `e_ellipsoid`). It carries no state and no allocations,
so it is freely `Copy`-passable through the physics pipeline. There
is no separate "live planet" struct: the gravity source, the geodetic
solver, and the frame-rotation models all read from `PlanetShape` and
combine it with their own model-specific data (gravity coefficients,
nutation tables) downstream.

`presets::EARTH` follows the GGM05C `mu = 3.986004415e14 m³/s²`
rather than IERS 2010, which differs by ~3e6 m³/s² — we follow JEOD's
choice so Tier 3 cross-validation against JEOD trajectories doesn't
chase a 7-ppm `mu` offset for hundreds of orbits. The other presets
similarly track the JEOD source data files
(`planet/data/src/<body>.cc`) verbatim.

## Layered architecture

```
astrodyn_bevy        (Bevy ECS adapter, mission code)
   ↓
astrodyn         (orchestration, recipes, single API surface)
   ↓
astrodyn_planet      ←  this crate (pure Rust, zero Bevy)
   ↓
astrodyn_quantities  (typed length / GravParam quantities)
```

`astrodyn_planet` is part of the `astrodyn_*` physics layer — pure Rust with no
Bevy dependency. See the project
[Strategy wiki page](https://github.com/simnaut/astrodyn/wiki/Strategy)
for the layered architecture.

## Public surface

- `PlanetShape` — reference-ellipsoid parameter block (`mu`, `r_eq`,
  `r_pol`, `flat_coeff`) with derived helpers and typed accessors.
- `presets::EARTH` / `MOON` / `SUN` / `MARS` — canonical constants
  matching the JEOD `planet/data/src/<body>.cc` files plus the
  corresponding gravity-model `mu` (Earth uses GGM05C, Moon GRAIL150,
  Mars MRO110B2, Sun spherical).

Earth's `mu` follows the GGM05C value `3.986004415e14 m³/s²` rather
than IERS 2010, to keep cross-validation faithful to JEOD source.

## See also

- [`docs/JEOD_invariants.md`](https://github.com/simnaut/astrodyn/blob/main/docs/JEOD_invariants.md) — `PL.*`
  invariants this crate enforces.
- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture.
- Rendered rustdoc:
  <https://docs.rs/astrodyn_planet>
