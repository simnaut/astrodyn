# astrodyn_bevy

A Rust port of [NASA JEOD](https://github.com/nasa/jeod) (JSC Engineering
Orbital Dynamics, v5.4) with [Bevy ECS](https://bevy.org) wiring on top.

`astrodyn_bevy` reimplements JEOD's spacecraft dynamics — spherical-harmonics
gravity, Earth rotation (precession/nutation/polar motion), atmospheric
drag, solar radiation pressure, gravity-gradient torque, multi-step
integrators, time-scale conversion (TAI/UTC/UT1/TDB/TT/GMST), DE4xx
ephemerides — as pure Rust crates, then exposes them through a thin Bevy
adapter so they slot into any Bevy app.

**Status:** pre-1.0. Tier 3 cross-validated against JEOD Trick simulations
(see the [Tier3-Regeneration wiki page](https://github.com/simnaut/astrodyn/wiki/Tier3-Regeneration)).
API may change before 1.0.

## When to use

- **Building a Bevy-based mission simulation** — Earth-orbit
  constellation, lunar / Mars approach, station-keeping study,
  rendezvous-and-proximity scenarios — where the Bevy `App` is the
  runtime and the ECS world is the single source of truth for
  state.
- **Composing a scenario** — sources, bodies, ephemeris, mass tree,
  integrator config — with `SimulationBuilder::populate_app::<P>`
  and the typestate `VehicleBuilder` (canonical entry point per
  CLAUDE.md).
- **Inserting a single vehicle** into an existing `App` (a smaller
  example, a follow-up insert during a running sim) via
  `VehicleConfig::spawn_bevy`.

For non-Bevy use — Tier 3 cross-validation tests, batch propagation,
offline studies — `astrodyn_runner` is the parallel arena-state
consumer of the same `astrodyn` pipeline. For pure physics math
(coordinate conversions, attitude algebra, gravity evaluation) without
either runtime, depend on `astrodyn` directly.

## Key concepts

The crate is the **thin glue** layer of the three-layer architecture
(see below): components are typed newtypes around `astrodyn`
quantities, systems delegate to `astrodyn` pipeline functions, and
the `AstrodynPlugin` wires the seven `AstrodynSet` variants into
`FixedUpdate` in JEOD-step order. There is no physics in this crate
— a CI lint refuses any `astrodyn_*`-physics-crate direct dep, so
the adapter cannot quietly reimplement what the gateway re-exports.

The typestate `VehicleBuilder` makes vehicle construction a compile-
time refusal of misconfigurations: `.three_dof_point_mass(...)` is
unavailable until a state is set, `.rk4()` until mass is set,
`.build()` until an integrator is chosen. Frame mismatches at
`spawn_bevy` boundaries surface as physics-language errors
(*"expected `Position<RootInertial>`, found `Position<Ecef>` — apply
a `FrameTransform<Ecef, RootInertial>` first"*) via
`#[diagnostic::on_unimplemented]`, not as raw `PhantomData`
type-mismatch walls.

## Architecture

Three layers, separated by hard dependency rules:

- `astrodyn_*` — pure Rust physics crates, **zero Bevy dependency**. Math,
  integrators, frame transforms, gravity, time scales, ephemerides.
- `astrodyn` — orchestration and recipes. Composes `astrodyn_*` into a
  pipeline; the single API surface for any ECS adapter. Zero Bevy
  dependency.
- `astrodyn_bevy` (this crate) — thin Bevy glue. Components, systems that
  delegate to `astrodyn`, plugin registration. Depends only on
  `astrodyn` + `bevy`.

See the [Strategy](https://github.com/simnaut/astrodyn/wiki/Strategy)
and [Type-System](https://github.com/simnaut/astrodyn/wiki/Type-System)
wiki pages for architecture detail and the typed-quantity layer.

## Quick start

```toml
[dependencies]
bevy = "0.19"
astrodyn_bevy = "0.1"
```

```rust,no_run
use bevy::prelude::*;
use astrodyn_bevy::prelude::*;
use astrodyn_bevy::recipes::{earth, orbital_elements, vehicle};
use astrodyn::{Earth, FrameUid, PlanetInertial};

fn setup(mut commands: Commands) {
    let earth_recipe = earth::point_mass();
    let mu = earth_recipe.source.mu.m3_per_s2();
    commands.spawn((
        // The source's inertial-frame identity (issue #668): gravity
        // controls reference it by this value, in every host.
        FrameUidC(FrameUid::of::<PlanetInertial<Earth>>()),
        GravitySourceC(earth_recipe.source),
        SourceInertialPositionC::default(),
        TranslationalStateC::<Earth>::default(),
    ));

    let cfg = VehicleBuilder::new()
        .vehicle_named("iss")
        .from_orbital_elements(orbital_elements::iss(), mu)
        .three_dof_point_mass(vehicle::iss_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(
            FrameUid::of::<PlanetInertial<Earth>>(),
            GravityGradient::Skip,
        ))
        .build();

    cfg.spawn_bevy::<Earth>(&mut commands);
}

fn main() {
    App::new()
        .add_plugins(MinimalPlugins)
        .add_astrodyn(10.0)
        .add_systems(Startup, setup)
        .run();
}
```

The typestate `VehicleBuilder` rejects misuse at compile time
(no integrator chosen, no state set, mismatched coordinate frames).
Errors render in physics language — *"expected `Position<RootInertial>`,
found `Position<Ecef>` — apply a `FrameTransform<Ecef, RootInertial>` first"* —
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

`cargo nextest run --workspace` works on a fresh clone without
`$JEOD_HOME` — every tier (unit, Tier 2, Tier 3) reads from committed
fixtures under `test_data/`. `$JEOD_HOME` is required only when
regenerating those fixtures via the `extract_*` binaries under
`crates/astrodyn_verif_jeod/src/bin/` or refreshing the verbatim mirror at
`crates/astrodyn_verif_jeod/test_data/jeod_inputs/`. See
[`CLAUDE.md`](../../CLAUDE.md) for the full build / test / regen workflow.

## Documentation

Most docs live on the [project wiki](https://github.com/simnaut/astrodyn/wiki):
architecture and phase history ([Strategy](https://github.com/simnaut/astrodyn/wiki/Strategy)),
typed-quantity primer ([Type-System](https://github.com/simnaut/astrodyn/wiki/Type-System)),
Tier 3 regeneration recipe ([Tier3-Regeneration](https://github.com/simnaut/astrodyn/wiki/Tier3-Regeneration)),
the JEOD↔astrodyn_bevy [capability matrix](https://github.com/simnaut/astrodyn/wiki/JEOD-Capability-Matrix),
[per-SIM coverage map](https://github.com/simnaut/astrodyn/wiki/JEOD-Sim-Coverage),
and [audit findings](https://github.com/simnaut/astrodyn/wiki/Audit-Findings).

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
