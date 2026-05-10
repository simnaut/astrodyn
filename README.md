# astrodyn

Pipeline orchestration, the typestate `VehicleBuilder`, and the
recipes module — the single API surface that any consumer (the Bevy
adapter or a non-Bevy runner) depends on.

`astrodyn` composes the
[`astrodyn_*`](https://github.com/simnaut/astrodyn/tree/main/crates) physics
crates into pipeline stages and re-exports their types so a downstream
crate only needs `astrodyn` to access the entire physics surface.
Pure Rust, zero Bevy dependency.

**Status:** pre-1.0. Tier 3 cross-validated against JEOD Trick simulations
(see the [Tier3-Regeneration wiki page](https://github.com/simnaut/astrodyn/wiki/Tier3-Regeneration)).
API may change before 1.0.

## Quick start

Most users want one of the consumer crates rather than this orchestration
layer directly:

- `astrodyn_bevy` for the Bevy-ECS production runtime,
- `astrodyn_runner` for plain-Rust batch propagation and Tier 3 tests.

If you are building a custom adapter on top of the pipeline, depend on
`astrodyn` directly:

```toml
[dependencies]
astrodyn = "0.1"
```

```rust,no_run
use astrodyn::{
    recipes::{earth, orbital_elements, vehicle},
    F64Ext, GravityControl, VehicleBuilder,
};

let mu = earth::point_mass().source.mu.m3_per_s2();
let cfg = VehicleBuilder::new()
    .from_orbital_elements(orbital_elements::iss(), mu)
    .three_dof_point_mass(vehicle::iss_mass())
    .rk4()
    .gravity(GravityControl::new_spherical(0_usize, false))
    .build();
// `cfg` is a `VehicleConfig` ready to hand to either
// `astrodyn_bevy::spawn_bevy::<Earth>(...)` or
// `astrodyn_runner::Simulation::add_vehicle(...)`.
# let _ = cfg;
```

The typestate `VehicleBuilder` rejects misuse at compile time
(no integrator chosen, no state set, mismatched coordinate frames).

## Layered architecture

```
astrodyn_bevy        (Bevy ECS adapter)
   ↓
astrodyn         ←  this crate (pure Rust, zero Bevy)
   ↓
astrodyn_dynamics, astrodyn_gravity, astrodyn_time, astrodyn_frames,
astrodyn_atmosphere, astrodyn_interactions, astrodyn_ephemeris,
astrodyn_planet, astrodyn_math, astrodyn_quantities
```

`astrodyn` (this crate) sits at the workspace root. The Bevy-ECS
adapter is `astrodyn_bevy` (`crates/astrodyn_bevy/`), and `astrodyn` is
also exercised directly by the standalone `astrodyn_runner` Tier 3
harness; both consumers share the same API surface. See the
[Strategy wiki page](https://github.com/simnaut/astrodyn/wiki/Strategy)
for the layered-architecture rules.

## Public surface

- `VehicleBuilder` — typestate builder that refuses `.build()` until
  state, mass, and integrator are set.
- `recipes::*` — `earth`, `moon`, `mars`, `sun`, `orbital_elements`,
  `vehicle`, `scenarios`. Mission-facing only; the JEOD-source-backed
  Tier 3 verification scaffolding lives in `astrodyn_verif_jeod`.
- Per-stage pipeline functions (`accumulate_gravity`,
  `validate_body`, …) that mirror the `AstrodynSet` schedule slot the
  Bevy adapter exposes.
- Frame-tree orchestration helpers shared by `astrodyn_bevy` and
  `astrodyn_runner`: `SourceFrameIds` (root + per-source frame IDs),
  `sync_pfix_rotation` (writes a planet-fixed child's rotation +
  angular velocity into a `FrameTree`),
  `evaluate_and_apply_frame_switch::<SourceId, F>` (generic
  on-approach/on-departure switch driver), and the source-state
  mutators `set_source_position` / `set_source_state`. Lifted out
  of `astrodyn_runner` in #71 so both consumers share one implementation.

## See also

- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture.
- [`crates/astrodyn_bevy/examples/typed_mission.rs`](https://github.com/simnaut/astrodyn/blob/main/crates/astrodyn_bevy/examples/typed_mission.rs)
  — canonical worked example.
- Rendered rustdoc:
  <https://docs.rs/astrodyn>
