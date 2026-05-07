# astrodyn_frames

Reference-frame tree and Earth/Mars/Moon rotation models for the
[`astrodyn_bevy`](https://github.com/simnaut/astrodyn) workspace.

Ports
[`models/utils/ref_frames/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/ref_frames/)
(the `RefFrame` tree that is the backbone of every coordinate system in
JEOD) and
[`models/environment/RNP/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/RNP/)
(precession, nutation, polar motion) from
[NASA JEOD v5.4.0](https://github.com/nasa/jeod).

## Layered architecture

```
astrodyn_bevy        (Bevy ECS adapter, mission code)
   ↓
astrodyn         (orchestration, recipes, single API surface)
   ↓
astrodyn_frames      ←  this crate (pure Rust, zero Bevy)
   ↓
astrodyn_quantities, astrodyn_math
```

`astrodyn_frames` is part of the `astrodyn_*` physics layer — pure Rust with no
Bevy dependency.

## Public surface

- `FrameTree` / `FrameNode` / `FrameId` — arena-based hierarchy
  mirroring JEOD's parent/child links. State is stored **relative to
  parent**.
- `RefFrameState` / `RefFrameTrans` / `RefFrameRot` — untyped storage
  form of a frame's translational + rotational state (RF.06 + RF.07).
- `RefFrameStateTyped<P, C>` — frame-tagged sibling for the public API
  boundary.
- `rotation_j2000`, `precession_j2000`, `nutation_j2000`,
  `data_nutation_j2000` — Earth precession + IAU-1980 nutation series.
- `rotation_mars`, `rotation_moon` — body-fixed rotation models for the
  Mars and Moon target bodies used by the JEOD verification sims.

## JEOD conventions

- Quaternions are **scalar-first, left-transformation** — see
  `astrodyn_math::JeodQuat`.
- Translational state is in **parent-frame coordinates**, not absolute /
  inertial.
- `t_parent_this` is a derived cache of `q_parent_this`; the quaternion
  is the canonical source of truth (RF.04).

## See also

- [`docs/JEOD_invariants.md`](https://github.com/simnaut/astrodyn/blob/main/docs/JEOD_invariants.md) — `RF.*`
  invariants this crate enforces.
- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture.
- Rendered rustdoc:
  <https://simnaut.github.io/astrodyn_bevy/astrodyn_frames/>
