# jeod_time

Time scales, leap seconds, calendar dates, and the time manager for the
[`bevy_jeod`](https://github.com/simnaut/bevy_jeod) workspace.

Ports
[`models/environment/time/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/time/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod), including the
[`Leap_Second.dat`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/time/data/Leap_Second.dat)
table.

## Layered architecture

```
bevy_jeod        (Bevy ECS adapter, mission code)
   ↓
jeod_sim         (orchestration, recipes, single API surface)
   ↓
jeod_time        ←  this crate (pure Rust, zero Bevy)
   ↓
jeod_quantities  (typed time scales, SecondsSince<S>)
```

`jeod_time` is part of the `jeod_*` physics layer — pure Rust with no
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
  Earth body-fixed rotation in `jeod_frames`.

## See also

- [`docs/JEOD_invariants.md`](../../docs/JEOD_invariants.md) — `TM.*`,
  `LS.*` invariants this crate enforces.
- [Project README](../../README.md) and
  [`CLAUDE.md`](../../CLAUDE.md) — workspace-level architecture.
- Rendered rustdoc:
  <https://simnaut.github.io/bevy_jeod/jeod_time/>
