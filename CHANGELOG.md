# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Bevy 0.18 → 0.19** ([#706](https://github.com/simnaut/astrodyn/issues/706)).
  Routine dependency bump; no 0.19 feature changes astrodyn's physics or
  pipeline. Compile-level touch-points, all confined to test helpers: the
  `ExecutorKind` API was removed (now
  `Schedule::set_executor(SingleThreadedExecutor::new())`), and
  `SystemState::get`/`get_mut` return a `Result` (SystemParam validation
  moved to fetch time).
- **MSRV bumped to 1.95** (was 1.89). Forced by `bevy 0.19` declaring its
  own `rust-version = "1.95"`; cargo's MSRV-aware resolution refuses to
  build the workspace on older toolchains. README § "Minimum supported
  Rust version" and the `tooling.yml` MSRV gate updated.
- **Unified `glam` to 0.32** ([#707](https://github.com/simnaut/astrodyn/issues/707)).
  Bumped the workspace `glam` 0.30 → 0.32 to match Bevy 0.19's math stack,
  eliminating the benign duplicate (`glam 0.30.10` for astrodyn physics vs
  `0.32.1` for Bevy's internal math) the Bevy bump introduced. A one-line
  dependency change: glam 0.31/0.32 ship no numeric-implementation changes
  to the operations astrodyn uses (only `DVec3`/`DMat3`/`DQuat`), so the
  full Tier 3 cross-validation and bit-identity parity suites pass with
  unchanged tolerances and baselines.

### Removed

- Dropped the unused `bevy_reflect` dependency and the vestigial
  `#[derive(Reflect)]` on `FrameAttachedC` — the only `Reflect` in the
  workspace, with no `register_type`/registry consumer. bevy_reflect 0.19
  moved to glam 0.32 (the workspace was still on glam 0.30 at the time; it
  was unified to 0.32 in #707), so the derive no longer resolved; since
  nothing introspected the type, the derive and dependency were removed
  rather than bumping glam across the physics core in the same PR.

## [0.2.0] - 2026-06-08

Second release. Headlined by the **frame-identity** work
([#659](https://github.com/simnaut/astrodyn/issues/659) / #660–#664 /
#668): every reference frame now carries a stable `FrameUid`, every
cross-source reference resolves through it, and a new
`astrodyn_frame_doc` crate captures that identity in a self-describing,
replayable wire schema.

### Added

- **`astrodyn_frame_doc` crate** (#663) — frame-document schema: a
  self-describing serialized form of a reference-frame tree (snapshot
  `FrameDocument` + replay-series `FrameSeries`) carrying identity,
  topology, origin, and epoch on every record, with bit-exact `f64`
  round-trips. Per-record validators (`validate_header`,
  `validate_uid_table`, `validate_record`) support independent
  streaming consumers (#659). The gateway exposes it behind a new,
  off-by-default `frame-doc` feature (`astrodyn::frame_doc` +
  `astrodyn::frame_doc_io`), keeping the production build serde-free.
- **Frame identity vocabulary** (#660) — `FrameUid` plus
  `Frame::DESCRIPTOR` on the sealed `Frame` trait.
- **`CartesianState<F>`** (#650) — typed position + velocity state
  record with opt-in serde (the gateway's `serde` feature forwards to
  it).
- **New frames and presets**: Moon mean-Earth (ME) frame for DEM
  georeferencing (#652); site-anchored topocentric ENU frame (#651);
  generic IAU body-fixed rotation beyond Moon/Mars (#653); a typed
  `Ephemeris` rotation accessor (#648); Jupiter + Saturn shape presets
  (#649).
- **LSODE integrator family** (#615/#616/#617) — non-stiff Adams and
  stiff BDF (Newton corrector + Jacobian) integrators, wired through
  the Bevy adapter with a parity test.

### Changed

- **MSRV bumped to 1.89** (was 1.87). Forced by `bevy 0.18.1` declaring
  its own `rust-version = "1.89"`; cargo's MSRV-aware resolution refuses
  to build the workspace on older toolchains. README § "Minimum
  supported Rust version" updated.
- Docs and package metadata reframed to describe astrodyn as an
  engine-agnostic framework rather than a Bevy-first library (#647).

### Breaking

- **Frame identity is now required throughout** (#661/#664/#668).
  `RefFrameKind` was removed; `FrameUid` is required on `FrameTree`
  nodes behind a checked typed boundary (`validate()` + namespace
  rules); and the former `SourceId` collapsed so every cross-source
  reference is a `FrameUid`. Downstream code that built frames without
  identities, or referenced frames via the old `SourceId` /
  `RefFrameKind` types, must migrate to `FrameUid`.
- `astrodyn::source_frames::SourceFrameIds` gained a `pub central: bool`
  field (originally landed in #568 without a version bump). Downstream
  code constructing this struct via struct literal must add the new
  field. Detected and gated by the `cargo-semver-checks` CI job.

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
- Claude Code review + `@claude` mention workflows (#666).

### Verification

- Large Tier 2 / Tier 3 cross-validation expansion against JEOD Trick
  sims: SIM_orbinit (full 46/46 init-variant matrix), SIM_verif_attach_mass,
  SIM_dyncomp, SIM_MET, SIM_tide_verif, SIM_7_time_reversal, SIM_VER_DRAG
  (with the `DRAG_OPT_CONST` aero option), SIM_RNP_J2000_prop, and the
  SIM_csr_compare gravity-acceleration octant sweep. Added Rosetta
  swing-by and Phobos mission benchmarks (#203/#204).

### Crates

Fourteen publishable workspace crates at this version — the thirteen
from 0.1.0 plus the new `astrodyn_frame_doc`. Four `publish = false`
verification crates: `astrodyn_verif_jeod`, the new
`astrodyn_verif_jeod_fixtures` (pure JEOD-data parsers, no pipeline
dependency) and `astrodyn_verif_nesc` (NESC GN&C Lunar Check Cases
track), and `astrodyn_verif_parity`.

## [0.1.1] - 2026-05-11

Same-day patch release following the initial 0.1.0 publish.

### Changed

- **Ephemeris kernels are distributed via GitHub Releases** (the
  `kernels-v1` tag) rather than bundled in the published crate (#476),
  keeping the `astrodyn_ephemeris` `.crate` small. See the
  [Environment](https://github.com/simnaut/astrodyn/wiki/Environment)
  wiki for kernel handling.

### Added

- `force_torque_response` recipe extracted as a reusable surface, with a
  `bevy_parity` wrapper (#477).
- `bevy_parity` wrappers for `drag_verif` and `drag_rot_verif` (#475).

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

[0.2.0]: https://github.com/simnaut/astrodyn/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/simnaut/astrodyn/releases/tag/v0.1.1
[0.1.0]: https://github.com/simnaut/astrodyn/releases/tag/v0.1.0
