# astrodyn

A pure-Rust orbital-dynamics simulation framework — a port of the physics in
[NASA JEOD](https://github.com/nasa/jeod) v5.4 (the JSC Engineering Orbital
Dynamics package). It models what a high-fidelity space-mission propagator
needs: spherical-harmonics gravity, Earth rotation (RNP precession/nutation),
multiple time scales, reference-frame trees, atmosphere and drag, solar
radiation pressure, third-body and gravity-gradient effects, and multi-body
rigid-body dynamics with a family of integrators.

The framework is **engine-agnostic**: the physics is plain Rust with no game
engine, ECS, or async runtime baked in, so it can be driven from a batch
propagator, a custom integrator loop, an ECS, or any other host. It is split
into a stack of small, single-responsibility `astrodyn_*` physics crates (each
independently publishable), an orchestration **gateway** crate (`astrodyn`,
this crate) that any host depends on, and optional reference consumers. See
[The astrodyn workspace](#the-astrodyn-workspace) below for the full crate map.

## This crate (`astrodyn`)

`astrodyn` is the orchestration gateway: it composes the `astrodyn_*` physics
crates into pipeline stages and re-exports their types, so a host needs only
`astrodyn` to reach the entire physics surface. It also provides the typestate
`VehicleBuilder` and the `recipes` module of mission presets. The per-stage
pipeline functions are borrow-based and storage-agnostic — the host owns state
and decides how to store it. No game-engine or runtime dependency.

**Status:** pre-1.0. Tier 3 cross-validated against JEOD Trick simulations
(see the [Tier3-Regeneration wiki page](https://github.com/simnaut/astrodyn/wiki/Tier3-Regeneration)).
API may change before 1.0.

## The astrodyn workspace

The framework is organized into architectural layers. Every consumer of
physics reads through `astrodyn` and only `astrodyn`; the `astrodyn_*` crates
below it are plain Rust and can also be used à la carte.

**Foundations**

| Crate | Responsibility |
| --- | --- |
| [`astrodyn_quantities`](https://docs.rs/astrodyn_quantities) | Phantom-tagged typed quantities (`Position<F>`, `Velocity<F>`, `Quat<L,T>`, …) — frame/unit safety at API boundaries |
| [`astrodyn_math`](https://docs.rs/astrodyn_math) | Quaternion, Euler, geodetic, orbital-element, and LVLH math kernels |
| [`astrodyn_time`](https://docs.rs/astrodyn_time) | Time scales (TAI/UTC/UT1/TDB/TT/GMST) and converters |

**Environment & physics**

| Crate | Responsibility |
| --- | --- |
| [`astrodyn_frames`](https://docs.rs/astrodyn_frames) | Reference-frame tree and Earth rotation (RNP, nutation, precession) |
| [`astrodyn_planet`](https://docs.rs/astrodyn_planet) | Planet definitions and presets (Earth, Moon, Sun, Mars) |
| [`astrodyn_ephemeris`](https://docs.rs/astrodyn_ephemeris) | DE4xx (SPICE) binary ephemeris reader |
| [`astrodyn_gravity`](https://docs.rs/astrodyn_gravity) | Spherical-harmonics gravity (Gottlieb), tides, and third-body |
| [`astrodyn_atmosphere`](https://docs.rs/astrodyn_atmosphere) | Atmospheric density models (exponential, MET) |
| [`astrodyn_interactions`](https://docs.rs/astrodyn_interactions) | Aerodynamic drag, SRP, gravity-gradient torque, shadow, and contact |
| [`astrodyn_dynamics`](https://docs.rs/astrodyn_dynamics) | Rigid-body dynamics, integrators (RK4, RKF45, GJ, ABM4), mass tree, body initialization |

**Orchestration & consumers**

| Crate | Responsibility |
| --- | --- |
| [`astrodyn`](https://docs.rs/astrodyn) | **This crate.** Pipeline orchestration, `VehicleBuilder`, recipes — the single API surface every host depends on |
| [`astrodyn_runner`](https://docs.rs/astrodyn_runner) | Standalone arena-state harness that owns all state and drives the pipeline — batch propagation and the Tier 3 test harness |
| [`astrodyn_bevy`](https://docs.rs/astrodyn_bevy) | Optional adapter for hosts built on the [Bevy](https://bevyengine.org/) ECS: component derives, systems, plugin registration |

`astrodyn_runner` and `astrodyn_bevy` are the two reference consumers shipped
in this workspace; a host with different storage or scheduling needs depends on
`astrodyn` directly and supplies its own state container.

**Verification** (not published to crates.io). The `astrodyn_verif_jeod`,
`astrodyn_verif_jeod_fixtures`, `astrodyn_verif_nesc`, and
`astrodyn_verif_parity` crates hold the JEOD Tier 3 cross-validation rigs, the
NESC GN&C Lunar Check Cases track, and the parity tests asserting the two
reference consumers stay bit-identical. They live in the
[workspace](https://github.com/simnaut/astrodyn/tree/main/crates) but stay
in-tree as `publish = false`.

## Quick start

Many users want one of the reference consumer crates rather than this
orchestration layer directly:

- `astrodyn_runner` for plain-Rust batch propagation and Tier 3 tests,
- `astrodyn_bevy` if your host is built on the Bevy ECS.

If you are wiring the pipeline into your own host (a batch tool, an integrator
loop, a different ECS, …), depend on `astrodyn` directly:

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
// `cfg` is a `VehicleConfig` ready to hand to any host — e.g.
// `astrodyn_runner::Simulation::add_vehicle(...)` for batch propagation,
// or `astrodyn_bevy::spawn_bevy::<Earth>(...)` for a Bevy host.
# let _ = cfg;
```

The typestate `VehicleBuilder` rejects misuse at compile time
(no integrator chosen, no state set, mismatched coordinate frames).

## Layered architecture

```
host  (astrodyn_runner, astrodyn_bevy, or your own)
   ↓
astrodyn         ←  this crate (the single gateway, no engine dependency)
   ↓
astrodyn_dynamics, astrodyn_gravity, astrodyn_time, astrodyn_frames,
astrodyn_atmosphere, astrodyn_interactions, astrodyn_ephemeris,
astrodyn_planet, astrodyn_math, astrodyn_quantities
```

`astrodyn` (this crate) sits at the workspace root and is the only physics
dependency a host needs. `astrodyn_runner` is the standalone arena-state host
used by the Tier 3 harness; `astrodyn_bevy` is the optional Bevy-ECS adapter.
Any host shares the same API surface. See the
[Strategy wiki page](https://github.com/simnaut/astrodyn/wiki/Strategy)
for the layered-architecture rules.

## Public surface

- `VehicleBuilder` — typestate builder that refuses `.build()` until
  state, mass, and integrator are set.
- `recipes::*` — `earth`, `moon`, `mars`, `sun`, `orbital_elements`,
  `vehicle`, `scenarios`. Mission-facing only; the JEOD-source-backed
  Tier 3 verification scaffolding lives in `astrodyn_verif_jeod`.
- Per-stage pipeline functions (`accumulate_gravity`,
  `validate_body`, …), one per stage of the per-step pipeline, that a host
  schedules in order. (The Bevy adapter maps these onto its `AstrodynSet`
  schedule slots; a batch host just calls them in sequence.)
- Frame-tree orchestration helpers shared by the reference consumers:
  `SourceFrameIds` (root + per-source frame IDs),
  `sync_pfix_rotation` (writes a planet-fixed child's rotation +
  angular velocity into a `FrameTree`),
  `evaluate_and_apply_frame_switch::<SourceId, F>` (generic
  on-approach/on-departure switch driver), and the source-state
  mutators `set_source_position` / `set_source_state`. Lifted out
  of `astrodyn_runner` in #71 so both consumers share one implementation.

## Performance toolkit

Three criterion microbenches measure the hot path under `cargo bench`:

- `cargo bench -p astrodyn_gravity --bench accumulate` — spherical-harmonics
  kernel at degree 4 / 20 / 60.
- `cargo bench -p astrodyn_gravity --bench integration` — RK4 6-DOF
  step, with and without realistic Moon LP150Q gravity.
- `cargo bench -p astrodyn_verif_jeod --bench step` — full `Simulation::step`
  for the Earth–Moon Clementine scenario.

Flamegraph SVGs land under `target/criterion/<group>/<bench>/profile/`
via `pprof`'s criterion integration.

For steady-state per-step measurement with JSON output (used by CI's
`perf-baseline-track` job), the `tier3_perf_runner` binary wraps the
canonical scenarios:

```bash
cargo xtask perf-baseline                     # default Earth–Moon run
cargo xtask perf-baseline --phase-timing      # adds per-phase µs/step
cargo xtask perf-baseline --help              # see all options
```

Direct invocation (skipping the xtask wrapper) is also available:

```bash
cargo run --profile release-with-debug \
    -p astrodyn_verif_jeod --bin tier3_perf_runner -- \
    --scenario earth_moon_clem --steps 100000 --warmup 1000 --repeat 5
```

## Minimum supported Rust version

astrodyn requires **Rust 1.89 or newer**. The `rust-version` field is
declared in `[workspace.package]` so every published member crate
inherits it; users on older toolchains get a clean
`error: package <name> requires Rust 1.89` from cargo rather than a
deep dependency-tree compile error.

The binding constraint is the workspace's `bevy = "0.18"` dependency (used
only by the optional `astrodyn_bevy` adapter — `astrodyn` and the `astrodyn_*`
physics crates do not depend on it): bevy 0.18.1 declares its own
`rust-version = "1.89"`, which MSRV-aware resolution in cargo 1.85+ enforces
transitively, and the floor is shared workspace-wide via `[workspace.package]`.
Our own direct usage of recent stdlib features (`u{32,usize}::is_multiple_of`,
stabilized in 1.87; `#[diagnostic::on_unimplemented]`; recent const generics)
sits comfortably below this floor.

**Bump policy.** Raising the MSRV is treated as a minor-version event,
not a patch. The project tracks the last two to three stable Rust
releases as the supported window; updates land when a dependency
forces the floor higher or when a stdlib feature with no clean polyfill
becomes load-bearing. Bumps are called out in the changelog.

**Source of truth.** The `msrv` job in
[`.github/workflows/tooling.yml`](.github/workflows/tooling.yml) pins
`dtolnay/rust-toolchain@1.89` and runs
`cargo check --workspace --all-targets`. That CI gate — not the
`rust-version` field alone — is what every PR has to clear. Stdlib drift
beyond `clippy::incompatible_msrv` (cfg-gated syntax, transitive-dep MSRV
bumps) surfaces here.

## License and attribution

Dual-licensed under either [`LICENSE-MIT`](LICENSE-MIT) or
[`LICENSE-APACHE`](LICENSE-APACHE) at your option (SPDX: `MIT OR Apache-2.0`).

The repository also redistributes a verbatim mirror of NASA JEOD v5.4 source
files under `crates/astrodyn_verif_jeod/test_data/jeod_inputs/` as
verification fixtures only. Those files remain governed by NASA's Open Source
Agreement (NOSA) v1.3, not astrodyn's dual license. See
[`NOTICE.md`](NOTICE.md) for the full attribution chain and the architectural
distinction (astrodyn is an independent reimplementation, not a JEOD fork).

## See also

- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture.
- [`crates/astrodyn_runner/examples/batch_propagation.rs`](https://github.com/simnaut/astrodyn/blob/main/crates/astrodyn_runner/examples/batch_propagation.rs)
  — engine-agnostic worked example (plain-Rust host).
- [`crates/astrodyn_bevy/examples/typed_mission.rs`](https://github.com/simnaut/astrodyn/blob/main/crates/astrodyn_bevy/examples/typed_mission.rs)
  — the same pipeline driven from a Bevy ECS host.
- Rendered rustdoc:
  <https://docs.rs/astrodyn>
