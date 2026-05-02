# jeod_runner

Standalone simulation runner and Tier 3 verification harness for the
[`bevy_jeod`](https://github.com/simnaut/bevy_jeod) workspace.

`jeod_runner` is a parallel non-Bevy consumer of `jeod_sim`. It owns
its own state and drives the same pipeline functions the Bevy adapter
runs from system schedules. Used for:

- **Tier 3 cross-validation tests** (`tests/tier3_*.rs`) — propagating
  from JEOD initial conditions and comparing against Trick reference
  CSVs without standing up a Bevy `App`.
- **Batch propagation, scripting, and offline studies** that don't
  need ECS scheduling, parallelism, or Bevy plugins.

## Layered architecture

```
                  jeod_sim
                  /       \
        bevy_jeod          jeod_runner    ←  this crate
        (Bevy adapter)     (plain Rust harness)
```

`jeod_runner` and `bevy_jeod` sit *next to* each other in the dep
graph — both depend on `jeod_sim` and the wider `jeod_*` family;
neither depends on the other. Mission code targeting the production
Bevy runtime depends on `bevy_jeod`, never on `jeod_runner`.

## Features

- `verification` (default) — pulls in `jeod_test_data` and exposes the
  `run_verification::*` Tier-3 case machinery. `--no-default-features`
  drops the JEOD-source-backed fixtures and gives a smaller runner.

## See also

- [Project README](../../README.md) and
  [`CLAUDE.md`](../../CLAUDE.md) — workspace-level architecture, Tier
  conventions, regen workflow.
- Rendered rustdoc:
  <https://simnaut.github.io/bevy_jeod/jeod_runner/>
