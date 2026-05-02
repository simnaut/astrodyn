# jeod_planet

Reference-ellipsoid parameters and standard preset bodies for the
[`bevy_jeod`](https://github.com/simnaut/bevy_jeod) workspace.

Ports
[`models/environment/planet/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/planet/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod) — the per-body
shape parameters (gravitational parameter, equatorial and polar radii,
flattening) consumed by gravity, geodetic, atmospheric, and
frame-rotation code.

## Layered architecture

```
bevy_jeod        (Bevy ECS adapter, mission code)
   ↓
jeod_sim         (orchestration, recipes, single API surface)
   ↓
jeod_planet      ←  this crate (pure Rust, zero Bevy)
   ↓
jeod_quantities  (typed length / GravParam quantities)
```

`jeod_planet` is part of the `jeod_*` physics layer — pure Rust with no
Bevy dependency. See the project
[Strategy wiki page](https://github.com/simnaut/bevy_jeod/wiki/Strategy)
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

- [`docs/JEOD_invariants.md`](https://github.com/simnaut/bevy_jeod/blob/main/docs/JEOD_invariants.md) — `PL.*`
  invariants this crate enforces.
- [Project README](https://github.com/simnaut/bevy_jeod/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/bevy_jeod/blob/main/CLAUDE.md) — workspace-level architecture.
- Rendered rustdoc:
  <https://simnaut.github.io/bevy_jeod/jeod_planet/>
