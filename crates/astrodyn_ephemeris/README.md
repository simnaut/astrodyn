# astrodyn_ephemeris

Planetary ephemerides backed by JPL DE-series SPK files for the
[`astrodyn_bevy`](https://github.com/simnaut/astrodyn) workspace.

Ports
[`models/environment/ephemerides/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/ephemerides/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod). Where JEOD links a
hand-rolled binary loader to JPL DE405 / DE421 kernels, this crate delegates
the file format and Chebyshev evaluation to the
[`anise`](https://crates.io/crates/anise) crate (a pure-Rust SPICE/NAIF
reimplementation) and exposes a thin, frame-tagged API on top.

## When to use

- **Third-body gravity** — query Sun/Moon/Mars/Venus positions in
  J2000 ICRF to feed `astrodyn_gravity` for third-body
  perturbations on an Earth orbit, or for an interplanetary
  trajectory's central-body switch.
- **Radiation pressure and shadow geometry** — every SRP step
  needs the Sun's inertial position; every Earth-orbit lighting
  computation needs the Sun, Moon, and Earth in the same frame.
- **Mission-design epoch queries** — read body positions at a
  specific TDB epoch for plotting, mission planning, or comparing
  against published trajectories.

The crate's default `fetch` feature downloads the required `.bsp`
kernels from the project's GitHub Releases on first use; for
air-gapped builds, set `$ASTRODYN_EPHEMERIS_KERNELS_DIR` to a directory
holding pre-downloaded kernels and disable the feature.

## Key concepts

Every position / velocity returned by `Ephemeris` is wrapped as
`Position<RootInertial>` / `Velocity<RootInertial>` from
`astrodyn_quantities`, in meters and m/s, in J2000 ICRF. There is no
"raw f64 array" public surface — the frame phantom is the contract
that lets gravity, SRP, and lighting code treat the ephemeris output
as a drop-in source position without re-tagging.

`EphemerisBody` is the workspace's body-identifier enum; it maps
JEOD's `EphemerisBody` constants to `anise`'s NAIF integer IDs so the
two codebases agree on what "Mars barycenter" means. `EphemerisError`
is a fail-loudly type — missing files, unsupported bodies, and
out-of-range epochs all panic with a diagnostic that names the
mismatch and the kernel that would resolve it, rather than returning
a silent zero.

## Layered architecture

```
astrodyn_bevy        (Bevy ECS adapter, mission code)
   ↓
astrodyn         (orchestration, recipes, single API surface)
   ↓
astrodyn_ephemeris   ←  this crate (pure Rust, zero Bevy)
   ↓
astrodyn_quantities  (Position<RootInertial>, Velocity<RootInertial>, frame tags)
```

`astrodyn_ephemeris` is part of the `astrodyn_*` physics layer — pure Rust with no
Bevy dependency. The Bevy-side glue lives in `astrodyn_bevy::*`; the pipeline
that drives it lives in `astrodyn`. See the project
[Strategy wiki page](https://github.com/simnaut/astrodyn/wiki/Strategy)
for the layered architecture.

## Public surface

- `Ephemeris` — owns an `anise::Almanac` and answers position/velocity
  queries from `.bsp` files (e.g., `de421.bsp`, `de440.bsp`).
- `EphemerisBody` — body-identifier enum mapping JEOD's body IDs to
  NAIF integer IDs.
- `EphemerisError` — fail-loudly error type for missing files,
  unsupported bodies, or out-of-range epochs.

## See also

- [`docs/JEOD_invariants.md`](https://github.com/simnaut/astrodyn/blob/main/docs/JEOD_invariants.md) — `EP.*`
  tags catalog the ephemeris invariants this crate enforces.
- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture and
  conventions.
- Rendered rustdoc:
  <https://docs.rs/astrodyn_ephemeris>
