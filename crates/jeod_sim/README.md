# jeod_sim

Pipeline orchestration, the typestate `VehicleBuilder`, and the
recipes module — the single API surface that any consumer (the Bevy
adapter or a non-Bevy runner) depends on.

`jeod_sim` composes the
[`jeod_*`](https://github.com/simnaut/bevy_jeod/tree/main/crates) physics
crates into pipeline stages and re-exports their types so a downstream
crate only needs `jeod_sim` to access the entire physics surface.
Pure Rust, zero Bevy dependency.

## Layered architecture

```
bevy_jeod        (Bevy ECS adapter)
   ↓
jeod_sim         ←  this crate (pure Rust, zero Bevy)
   ↓
jeod_dynamics, jeod_gravity, jeod_time, jeod_frames,
jeod_atmosphere, jeod_interactions, jeod_ephemeris,
jeod_planet, jeod_math, jeod_quantities
```

The titular simulation environment is `bevy_jeod` (the workspace root
package). `jeod_sim` is also exercised directly by the standalone
`jeod_runner` Tier 3 harness; both consumers share the same API
surface. See the
[Strategy wiki page](https://github.com/simnaut/bevy_jeod/wiki/Strategy)
for the layered-architecture rules.

## Public surface

- `VehicleBuilder` — typestate builder that refuses `.build()` until
  state, mass, and integrator are set.
- `recipes::*` — `earth`, `orbital_elements`, `vehicle`, `scenarios`,
  `verification` (the last gated behind the in-repo-only
  `jeod-source` feature).
- Per-stage pipeline functions (`accumulate_gravity`,
  `validate_body`, …) that mirror the `JeodSet` schedule slot the
  Bevy adapter exposes.

## See also

- [Project README](https://github.com/simnaut/bevy_jeod/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/bevy_jeod/blob/main/CLAUDE.md) — workspace-level architecture.
- [`examples/typed_mission.rs`](https://github.com/simnaut/bevy_jeod/blob/main/examples/typed_mission.rs) —
  canonical worked example.
- Rendered rustdoc:
  <https://simnaut.github.io/bevy_jeod/jeod_sim/>
