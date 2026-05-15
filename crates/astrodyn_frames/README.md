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

## When to use

- **Storing a hierarchy of reference frames** — Earth inertial,
  Earth body-fixed, Moon body-fixed, vehicle structure, vehicle
  composite-body / core-body — with parent/child links and
  relative state at every node.
- **Computing Earth's body-fixed rotation** — precession (J2000 →
  mean-of-date), nutation (mean-of-date → true-of-date, IAU-1980
  106-term series), and polar motion / GMST-driven sidereal
  rotation, all of which must compose in JEOD's order to match
  reference trajectories.
- **Mars and Moon body-fixed rotation** for Mars / Apollo verification
  sims, including lunar libration models.

Mission code rarely touches `FrameTree` directly; the `astrodyn`
pipeline maintains the tree and the `astrodyn_bevy` adapter mirrors it
into ECS components. Reach in here when porting a new RNP series, when
adding a new target body's body-fixed rotation, or when implementing a
custom frame relative to an existing parent.

## Key concepts

Every node in `FrameTree` stores its state **relative to its
parent** (not absolute / inertial), matching JEOD's `RefFrame`
convention. Relative state between arbitrary frames is computed by
walking to the common ancestor and composing — typed
`RefFrameStateTyped<P, C>` siblings at the public boundary make those
compositions frame-checked, so the compiler refuses to combine a
"vehicle wrt Earth-inertial" with a "Moon wrt Sun" without explicitly
re-parenting. The tree is a flat `Vec` with parallel
`Option<FrameId>` parent pointers, which keeps lookups cache-friendly
and avoids the pointer chasing of JEOD's intrusive linked lists.

Earth rotation composes as `polar_motion · sidereal · nutation ·
precession`, with each factor a left-transformation,
scalar-first quaternion (`JeodQuat`). `t_parent_this` is a derived
cache of `q_parent_this`; the quaternion is the canonical source of
truth (RF.04), so any update writes the quaternion and refreshes the
matrix, never the reverse. Mars and Moon rotation models follow the
same pattern with body-specific pole-direction series.

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
  <https://docs.rs/astrodyn_frames>
