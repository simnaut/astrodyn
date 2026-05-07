# astrodyn_runner

Standalone simulation runner and Tier 3 verification harness for the
[`astrodyn_bevy`](https://github.com/simnaut/astrodyn) workspace.

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

## Features

- `verification` (default) — pulls in `astrodyn_test_data` and exposes the
  `run_verification::*` Tier-3 case machinery. `--no-default-features`
  drops the JEOD-source-backed fixtures and gives a smaller runner.

## See also

- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture, Tier
  conventions, regen workflow.
- Rendered rustdoc:
  <https://simnaut.github.io/astrodyn_bevy/astrodyn_runner/>
