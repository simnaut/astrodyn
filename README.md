# bevy_jeod

A Rust port of [NASA JEOD](https://github.com/nasa/jeod) (JSC Engineering
Orbital Dynamics, v5.4) with [Bevy ECS](https://bevy.org) wiring on top.

`bevy_jeod` reimplements JEOD's spacecraft dynamics — spherical-harmonics
gravity, Earth rotation (precession/nutation/polar motion), atmospheric
drag, solar radiation pressure, gravity-gradient torque, multi-step
integrators, time-scale conversion (TAI/UTC/UT1/TDB/TT/GMST), DE4xx
ephemerides — as pure Rust crates, then exposes them through a thin Bevy
adapter so they slot into any Bevy app.

**Status:** pre-1.0. Tier 3 cross-validated against JEOD Trick simulations
(see the [Tier3-Regeneration wiki page](https://github.com/simnaut/bevy_jeod/wiki/Tier3-Regeneration)).
API may change before 1.0.

## Architecture

Three layers, separated by hard dependency rules:

- `jeod_*` — pure Rust physics crates, **zero Bevy dependency**. Math,
  integrators, frame transforms, gravity, time scales, ephemerides.
- `jeod_sim` — orchestration and recipes. Composes `jeod_*` into a
  pipeline; the single API surface for any ECS adapter. Zero Bevy
  dependency.
- `bevy_jeod` (this crate) — thin Bevy glue. Components, systems that
  delegate to `jeod_sim`, plugin registration. Depends only on
  `jeod_sim` + `bevy`.

See the [Strategy](https://github.com/simnaut/bevy_jeod/wiki/Strategy)
and [Type-System](https://github.com/simnaut/bevy_jeod/wiki/Type-System)
wiki pages for architecture detail and the typed-quantity layer.

## Quick start

```toml
[dependencies]
bevy = "0.18"
bevy_jeod = "0.1"
```

```rust
use bevy::prelude::*;
use bevy_jeod::prelude::*;
use bevy_jeod::recipes::{earth, orbital_elements, vehicle};

fn setup(mut commands: Commands) {
    let earth_recipe = earth::point_mass();
    let mu = earth_recipe.source.mu.m3_per_s2();
    let earth_entity = commands
        .spawn((
            GravitySourceC(earth_recipe.source),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
        ))
        .id();

    let cfg = VehicleBuilder::new()
        .from_orbital_elements(orbital_elements::iss(), mu)
        .three_dof_point_mass(vehicle::iss_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, false))
        .build();

    cfg.spawn_bevy(&mut commands, &[earth_entity]);
}

fn main() {
    App::new()
        .add_plugins((MinimalPlugins, JeodPlugin))
        .insert_resource(Time::<Fixed>::from_seconds(10.0))
        .add_systems(Startup, setup)
        .run();
}
```

The typestate `VehicleBuilder` rejects misuse at compile time
(no integrator chosen, no state set, mismatched coordinate frames).
Errors render in physics language — *"expected `Position<Inertial>`,
found `Position<Ecef>` — apply a `FrameTransform<Ecef, Inertial>` first"* —
not as `PhantomData` mismatches.

A full worked example lives in
[`examples/typed_mission.rs`](examples/typed_mission.rs).

## Verification

Three test tiers, all part of the definition of done for every release:

- **Tier 1** — unit tests on pure functions (round-trips, convergence).
- **Tier 2** — comparison against static reference vectors extracted from
  JEOD source files (gravity test cases, Euler angle tables).
- **Tier 3** — end-to-end trajectory cross-validation: propagate from the
  same initial conditions as a JEOD Trick simulation and compare position,
  velocity, attitude, and angular velocity over hours or days. Reference
  CSVs are committed to the repo.

```bash
cargo nextest run --workspace -E 'not test(tier3_)'   # fast: Tier 1 + 2
cargo nextest run --workspace -E 'test(tier3_)'       # Tier 3
```

Set `JEOD_HOME` to a JEOD source checkout for tests that load JEOD data
files. See [`CLAUDE.md`](CLAUDE.md) for the full build and test workflow.

## Documentation

Most docs live on the [project wiki](https://github.com/simnaut/bevy_jeod/wiki):
architecture and phase history ([Strategy](https://github.com/simnaut/bevy_jeod/wiki/Strategy)),
typed-quantity primer ([Type-System](https://github.com/simnaut/bevy_jeod/wiki/Type-System)),
Tier 3 regeneration recipe ([Tier3-Regeneration](https://github.com/simnaut/bevy_jeod/wiki/Tier3-Regeneration)),
the JEOD↔bevy_jeod [capability matrix](https://github.com/simnaut/bevy_jeod/wiki/JEOD-Capability-Matrix),
[per-SIM coverage map](https://github.com/simnaut/bevy_jeod/wiki/JEOD-Sim-Coverage),
and [audit findings](https://github.com/simnaut/bevy_jeod/wiki/Audit-Findings).

The one exception that stays in the repo is
[`docs/JEOD_invariants.md`](docs/JEOD_invariants.md) — the catalog of JEOD
C++ invariants and where each is enforced in our Rust port. It lives next
to the code because tags like `// JEOD_INV: XX.YY` in source are
consistency-checked against the catalog.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.

NASA JEOD itself is distributed under NASA's open-source license and is
not redistributed by this project.
