# astrodyn_time

Time scales, leap seconds, calendar dates, and the time manager for the
[`astrodyn_bevy`](https://github.com/simnaut/astrodyn) workspace.

Ports
[`models/environment/time/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/time/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod), including the
[`Leap_Second.dat`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/time/data/Leap_Second.dat)
table.

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
