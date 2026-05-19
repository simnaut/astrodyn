# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-19

### Changed

- **MSRV bumped to 1.89** (was 1.87). Forced by `bevy 0.18.1` declaring
  its own `rust-version = "1.89"`; cargo's MSRV-aware resolution refuses
  to build the workspace on older toolchains. README § "Minimum
  supported Rust version" updated.

### Breaking

- `astrodyn::source_frames::SourceFrameIds` gained a `pub central: bool`
  field (originally landed in #568 without a version bump). Downstream
  code constructing this struct via struct literal must add the new
  field. Detected and gated by the new `cargo-semver-checks` CI job.

### Tooling

Bundled as part of [#527](https://github.com/simnaut/astrodyn/issues/527):

- `cargo-deny` supply-chain fence (`deny.toml`, advisories + licenses +
  bans + sources). Caught and cleared two real `rustls-webpki`
  vulnerabilities (RUSTSEC-2026-0099 / -0104) during initial wiring.
- Dependabot weekly cadence on `cargo` and `github-actions`, grouped
  minor/patch.
- MSRV CI gate using `dtolnay/rust-toolchain@1.89`.
- `cargo-semver-checks` gating the public `astrodyn` surface against
  the crates.io baseline.
- `cargo-hack` feature-powerset on `astrodyn_ephemeris` to keep the
  `--no-default-features` air-gapped build path honest.
- CI ergonomics: cancel-superseded `concurrency:` blocks on both
  `ci.yml` and the new `tooling.yml`, plus `RUST_BACKTRACE=1` so
  Tier 3 / parity panics emit stack traces.

## [0.1.0] - 2026-04-28

Initial public release. The original phased implementation plan
(Phases 1–7) closed in April 2026; ongoing work tracks as GitHub issues.

### Added

- **Typed quantities** (`astrodyn_quantities`): phantom-tagged
  `Position<F>`, `Velocity<F>`, `Acceleration<F>`,
  `SecondsSince<S>`, `Quat<L, T>`, `NormalizedQuat`,
  `FrameTransform<From, To>`, and the `F64Ext` facade
  (`400.0.km()`, `51.6.deg()`, `420_000.0.kg()`). Compile-time
  rejection of cross-frame mismatches and unit-dimensional errors.
- **Mission API** (`astrodyn_bevy::prelude` + `astrodyn_bevy::recipes`):
  typestate `VehicleBuilder` (no-state / no-integrator gates),
  recipe modules for Earth/Moon/Sun/Mars, common orbital elements
  (ISS, GEO, GTO, Molniya), and vehicle masses.
- **Pipeline** (`astrodyn`): seven-stage integration loop —
  `TimeUpdate`, `EphemerisUpdate`, `Environment` (gravity +
  atmosphere), `Interaction` (aero, SRP, gravity torque, contact,
  shadow), `ForceCollection`, `Integration` (with frame propagation
  as the integrator's post-step), `DerivedState`. Single API surface
  for ECS adapters.
- **Bevy adapter** (`astrodyn_bevy`): `AstrodynPlugin`, `AstrodynSet` system-set
  enum, component bundles, and a thin systems layer that delegates
  to `astrodyn`.
- **Standalone runner** (`astrodyn_runner`): `Simulation` propagator that
  consumes the same `VehicleConfig` as the Bevy adapter — mission
  code can swap between Bevy and the runner without rebuilding the
  configuration.
- **Physics models**: spherical-harmonics gravity (Gottlieb
  algorithm), tides, third-body, RNP Earth rotation
  (precession/nutation/polar motion), MET atmosphere, exponential
  atmosphere, drag, SRP, gravity-gradient torque, contact, eclipse
  shadow, Earth lighting.
- **Time scales**: TAI, UTC, UT1, TDB, TT, GMST with leap-second
  table from JEOD source.
- **Integrators**: RK4, RKF45, ABM4, Gauss-Jackson 8th-order.
- **Ephemerides**: DE4xx binary reader via `anise`.
- **Verification**: three test tiers — Tier 1 unit tests, Tier 2
  static reference vectors from JEOD source, Tier 3 trajectory
  cross-validation against JEOD Trick simulations (committed
  reference CSVs, Docker-based regeneration workflow).

### Crates

Fourteen workspace crates at this version:

- `astrodyn` (workspace root, gateway / orchestration), `astrodyn_quantities`,
  `astrodyn_math`, `astrodyn_frames`, `astrodyn_time`, `astrodyn_planet`,
  `astrodyn_ephemeris`, `astrodyn_gravity`, `astrodyn_atmosphere`,
  `astrodyn_dynamics`, `astrodyn_interactions`, `astrodyn_runner`,
  `astrodyn_bevy` (Bevy ECS adapter).
- Plus two `publish = false` verification crates: `astrodyn_verif_jeod`
  (JEOD-source parsers + Tier 3 cross-validation tests +
  `run_verification` scenario rigs) and `astrodyn_verif_parity`
  (runner ↔ Bevy bit-identical parity tests).

[0.2.0]: https://github.com/simnaut/astrodyn/releases/tag/v0.2.0
[0.1.0]: https://github.com/simnaut/astrodyn/releases/tag/v0.1.0
