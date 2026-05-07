# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **Pipeline** (`astrodyn`): nine-stage integration loop —
  TimeUpdate, EphemerisUpdate, EnvironmentSet (gravity, atmosphere),
  InteractionSet (aero, SRP, gravity torque, contact, shadow),
  ForceCollectionSet, IntegrationSet, DerivedStateSet. Single API
  surface for ECS adapters.
- **Bevy adapter** (`astrodyn_bevy`): `JeodPlugin`, `JeodSet` system-set
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

Thirteen crates published at this version:

- `astrodyn_quantities`, `astrodyn_math`, `astrodyn_frames`, `astrodyn_time`,
  `astrodyn_planet`, `astrodyn_ephemeris`, `astrodyn_gravity`, `astrodyn_atmosphere`,
  `astrodyn_dynamics`, `astrodyn_interactions`, `astrodyn`, `astrodyn_runner`,
  `astrodyn_test_data`, plus the root `astrodyn_bevy` crate.

[0.1.0]: https://github.com/simnaut/astrodyn/releases/tag/v0.1.0
