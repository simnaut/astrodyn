# astrodyn_runner

Standalone arena-state simulation harness for the
[`astrodyn`](https://github.com/simnaut/astrodyn) orbital-dynamics
pipeline.

`astrodyn_runner` is a parallel non-Bevy consumer of `astrodyn`. It owns
its own state and drives the same pipeline functions the Bevy adapter
runs from system schedules.

## When to use

- **Tier 3 cross-validation tests** propagating from JEOD initial
  conditions and comparing against Trick reference CSVs without
  standing up a Bevy `App` — `astrodyn_verif_jeod`'s 70+
  `tier3_*` tests drive `Simulation` end-to-end.
- **Batch propagation, scripting, offline studies, and Jupyter-style
  exploration** that don't need ECS scheduling, parallelism, or Bevy
  plugins.
- **Lockstep parity verification** — `astrodyn_verif_parity` runs
  identical scenarios through both `astrodyn_runner` and
  `astrodyn_bevy` and asserts every component is bit-identical, the
  guard that gives Bevy adapter Tier 3 coverage by transitivity.

For the production target — Bevy ECS as the runtime, with mission
code expressed as systems and components — reach for `astrodyn_bevy`
instead. Don't reach for `astrodyn_runner` from a mission crate; its
state-container shape is incompatible with the ECS world.

## Key concepts

`Simulation` is an arena-state struct: it owns one
`SimulationContext` (time, ephemeris, gravity sources) and one
contiguous `Vec` of body states, and steps everything in lockstep.
The arena layout is deliberate — it makes the runner trivial to
checkpoint (snapshot the `Vec`), trivial to compare against Trick
output (the body index is stable across the run), and free of the
ECS scheduling and change-detection machinery that the Bevy adapter
takes on.

The pipeline functions it drives — `pre_step`, the per-step
`TimeUpdate / EphemerisUpdate / Environment / Interaction /
ForceCollection / Integration / DerivedState` sets — live in
`astrodyn`, not here, so any improvement that lands there benefits
both consumers identically. The runner is **not** a dependency of
`astrodyn_bevy`; both crates depend on `astrodyn` and only
`astrodyn` for physics, and a CI lint
(`scripts/check_no_bypass_deps.sh`) refuses any
`astrodyn_*`-physics-crate direct dep on the four consumer crates.

## Layered architecture

```
                  astrodyn
                  /       \
        astrodyn_bevy          astrodyn_runner    ←  this crate
        (Bevy adapter)     (plain Rust harness)
```

`astrodyn_runner` and `astrodyn_bevy` sit *next to* each other in the dep
graph — both depend on `astrodyn` and the wider `astrodyn_*` family;
neither depends on the other. Mission code targeting the production
Bevy runtime depends on `astrodyn_bevy`, never on `astrodyn_runner`.

## Quick start

```toml
[dependencies]
astrodyn = "0.1"
astrodyn_runner = "0.1"
```

```rust,no_run
use astrodyn_runner::SimulationBuilderExt;
use astrodyn::recipes::Mission;

let mut sim = Mission::iss_leo().into_builder().build().unwrap();
sim.step_n(10);
let body = sim.body(0);
println!("altitude: {:.0} m", body.trans.position.raw_si().length() - 6_378_137.0);
```

`Mission::iss_leo()` is one of the canned scenarios in
`astrodyn::recipes::scenarios`; see that module for the full list.
For Tier 3 verification scaffolding (Trick CSV diff, JEOD-source-backed
initial conditions), reach for `astrodyn_verif_jeod` in addition to this
crate.

## See also

- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture, Tier
  conventions, regen workflow.
- Rendered rustdoc:
  <https://docs.rs/astrodyn_runner>
