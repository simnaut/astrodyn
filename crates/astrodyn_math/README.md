# astrodyn_math

JEOD-faithful math kernels — quaternions, Euler angles, orbital
elements, geodetic coordinates, the LVLH frame, the solar beta angle,
and the small but load-bearing collection of vector / matrix helpers
that mirror JEOD's `Vector3` / `Matrix3x3` inline functions.

Ports
[`models/utils/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod). Specifically:
[`orbital_elements/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/orbital_elements/),
[`orientation/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/orientation/),
[`planet_fixed/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/planet_fixed/),
[`lvlh_frame/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/lvlh_frame/),
[`quaternion/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/quaternion/),
and the inline helpers under
[`math/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/math/include/).

## When to use

- **Coordinate / orbital-element conversions** — Cartesian ↔ Keplerian
  (`OrbitalElements`), geodetic ↔ Cartesian, body-frame ↔ LVLH ↔ NED.
- **Attitude algebra** — composing or interpolating JEOD-convention
  quaternions, computing Euler-angle sequences, building rotation
  matrices that round-trip with JEOD source.
- **Lighting / pointing scalars** — solar beta angle for thermal /
  power budgeting against an orbit.

If you're an upstream physics crate (gravity, dynamics, frames,
interactions) that needs a battle-tested coordinate or attitude
primitive, reach here rather than rolling your own. If you're mission
code, the recipes / typed components in `astrodyn` and `astrodyn_bevy`
already wrap these — you rarely call `astrodyn_math` directly.

## Key concepts

The crate is organized around a single attitude type — `JeodQuat`, the
unified scalar-first, left-transformation `Quat<ScalarFirst,
LeftTransform>` re-exported from `astrodyn_quantities` — so there is
only one quaternion convention in the workspace, enforced by the type
system rather than by comments. Algebra (composition, conjugation,
interpolation), conversions (to/from Euler angles and rotation
matrices), and the witness-gated `NormalizedQuat` constructor all
operate on that one type.

Coordinate kernels follow JEOD's source layout faithfully:
`cartesian_to_geodetic` uses Borkowski's iterative latitude solver
(stable at the poles and at low altitude); `mat3_from_rows` matches
JEOD's row-major matrix convention; `OrbitalElements` keeps the same
true/mean/orbital anomaly trio JEOD carries. Angles are radians,
lengths meters, throughout. The typed `*_typed` siblings drop into
those raw kernels via `.raw_si()` and re-wrap on exit, so the public
surface stays frame- and unit-checked without taxing the inner loop.

## Layered architecture

```
astrodyn_bevy        (Bevy ECS adapter, mission code)
   ↓
astrodyn         (orchestration, recipes, single API surface)
   ↓
astrodyn_math        ←  this crate (pure Rust, zero Bevy)
   ↓
astrodyn_quantities  (typed quantities + JeodQuat re-export)
```

`astrodyn_math` is part of the `astrodyn_*` physics layer — pure Rust with no
Bevy dependency. After Phase 2 (#104) of the type-system refactor,
`JeodQuat` is a re-export of `astrodyn_quantities::JeodQuat`.

## Public surface

- `OrbitalElements` — Cartesian ↔ Keplerian conversion.
- `EulerSequence`, `compute_*_typed` helpers — Euler-angle algebra.
- `cartesian_to_geodetic_typed` / `geodetic_to_cartesian_typed`,
  `GeodeticState`, `GeodeticStateTyped` — ellipsoidal coordinates.
- `compute_lvlh_frame_typed`, `LvlhFrame` — local-vertical / local-
  horizontal frame.
- `solar_beta_angle_typed` — solar beta angle.
- `quaternion` — JEOD-convention algebra on the unified `JeodQuat`.
- `types::mat3_from_rows` — row-major constructor for `glam::DMat3`,
  matching JEOD source layout.

## JEOD conventions

- Quaternions are **scalar-first, left-transformation** (`JeodQuat ==
  Quat<ScalarFirst, LeftTransform>`).
- Geodetic conversions follow Borkowski's iterative latitude solver.
- All angles in radians, all lengths in meters.

## See also

- [`docs/JEOD_invariants.md`](https://github.com/simnaut/astrodyn/blob/main/docs/JEOD_invariants.md) — `OE.*`,
  `QT.*`, `EU.*`, `GD.*` invariants this crate enforces.
- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture and
  conventions.
- Rendered rustdoc:
  <https://docs.rs/astrodyn_math>
