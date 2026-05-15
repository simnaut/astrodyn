# astrodyn_time

Time scales, leap seconds, calendar dates, and the time manager for the
[`astrodyn_bevy`](https://github.com/simnaut/astrodyn) workspace.

Ports
[`models/environment/time/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/time/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod), including the
[`Leap_Second.dat`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/time/data/Leap_Second.dat)
table.

## When to use

- **Driving the per-step simulation clock** — `SimulationTime`
  holds the current epoch in every registered scale; the
  integration loop advances it once per step and downstream
  consumers (gravity, ephemeris, atmosphere) read from it rather
  than keeping their own clocks.
- **Converting between scales** — TAI ↔ UTC ↔ UT1 ↔ TT ↔ TDB ↔ GPS
  via the registered `TimeConverter_*` pipeline, plus
  UT1 → GMST for Earth body-fixed rotation.
- **Mission elapsed time / user-defined epoch** —
  `MissionElapsedTime` is the relative-time scale most operators
  log against; `UserDefinedEpoch` is the per-sim zero.
- **Leap-second-aware date arithmetic** — `CalendarDate` +
  `LeapSecondTable` handle UTC's leap-second discontinuities
  rather than papering over them.

## Key concepts

Time scales are typed at the `astrodyn_quantities` boundary as
`SecondsSince<S: TimeScale>` (`TAI`, `TT`, `TDB`, `UT1`, `UTC`, `GPS`,
…), so any function that takes "seconds since TAI epoch" cannot
accidentally be called with "seconds since UTC epoch" — a class of
sign-and-offset bug that JEOD catches via runtime checks and which
the typed surface elides at compile time. `TimeManager` keeps the
currently registered scales and answers conversions by composing
per-pair `TimeConverter_*` functions, mirroring JEOD's class layout.

Leap seconds are not optional. UTC → TAI is a piecewise-constant
discontinuity at every leap-second boundary, and silently smearing
it (the "UTC seconds since epoch" mistake) produces 1-second
position errors that cascade through every downstream conversion.
`LeapSecondTable` parses JEOD's `Leap_Second.dat` verbatim and the
`time_converter_*` family threads it through every relevant
conversion. GMST in particular drives Earth body-fixed rotation in
`astrodyn_frames`, so any GMST error propagates directly into ECEF
positions.

## Layered architecture

```
astrodyn_bevy        (Bevy ECS adapter, mission code)
   ↓
astrodyn         (orchestration, recipes, single API surface)
   ↓
astrodyn_time        ←  this crate (pure Rust, zero Bevy)
   ↓
astrodyn_quantities  (typed time scales, SecondsSince<S>)
```

`astrodyn_time` is part of the `astrodyn_*` physics layer — pure Rust with no
Bevy dependency.

## Public surface

- `TimeManager`, `TimeScaleId` — orchestrator for registered scales.
- `SimulationTime` — per-step time-state resource (gravity, ephemeris,
  atmosphere read from this).
- `DynamicTime` — dynamics-frame time passed through the integrator.
- `LeapSecondTable` — JEOD `Leap_Second.dat` parser / lookup.
- `CalendarDate`, `UTC_EPOCH_TAI_TJT` — Gregorian calendar.
- `UserDefinedEpoch`, `MissionElapsedTime` — sim-defined epoch + MET.
- `GpsTimeComponents`, `TAI_GPS_OFFSET` — GPS week / time-of-week.
- Per-pair converters: `time_converter_tai_tdb`,
  `time_converter_tai_tt`, `time_converter_ut1_gmst`. GMST drives
  Earth body-fixed rotation in `astrodyn_frames`.

## See also

- [`docs/JEOD_invariants.md`](https://github.com/simnaut/astrodyn/blob/main/docs/JEOD_invariants.md) — `TM.*`,
  `LS.*` invariants this crate enforces.
- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture.
- Rendered rustdoc:
  <https://docs.rs/astrodyn_time>
