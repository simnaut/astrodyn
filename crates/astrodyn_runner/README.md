# astrodyn_runner

Standalone arena-state simulation harness for the
[`astrodyn`](https://github.com/simnaut/astrodyn) orbital-dynamics
pipeline.

`astrodyn_runner` is a parallel non-Bevy consumer of `astrodyn`. It owns
its own state and drives the same pipeline functions the Bevy adapter
runs from system schedules. Used for:

- **Tier 3 cross-validation tests** (`tests/tier3_*.rs`) — propagating
  from JEOD initial conditions and comparing against Trick reference
  CSVs without standing up a Bevy `App`.
- **Batch propagation, scripting, and offline studies** that don't
  need ECS scheduling, parallelism, or Bevy plugins.

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
