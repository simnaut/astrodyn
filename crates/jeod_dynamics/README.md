# jeod_dynamics

Rigid-body dynamics, integrators (RK4, RKF45, Gauss-Jackson, ABM4), the
mass tree, and body initialization for the
[`bevy_jeod`](https://github.com/simnaut/bevy_jeod) workspace.

Ports
[`models/dynamics/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/dynamics/)
and
[`models/utils/integration/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/integration/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod). Decomposes JEOD's
1200-line `DynBody` god-class into ~10 narrow components.

## Layered architecture

```
bevy_jeod        (Bevy ECS adapter, mission code)
   ↓
jeod_sim         (orchestration, recipes)
   ↓
jeod_dynamics    ←  this crate (pure Rust, zero Bevy)
   ↓
jeod_math, jeod_frames, jeod_quantities
```

## Public surface

- `state::TranslationalState`, `rotational::RotationalState`,
  `rotational::SixDofState` — body state in the integration frame.
- `mass::MassProperties` — mass, inertia, CoM offset, plus the
  parallel-axis (Steiner) composition for mass-tree subtrees.
- `forces::TotalForce`, `forces::FrameDerivatives`,
  `forces::DynamicsConfig`, `forces::GravityAcceleration` — per-step
  force / derivative accumulators.
- `integration::IntegrationMethod` — RK4 / RKF45 / GJ / ABM4 dispatch.
- `propagation`, `subtree`, `attach`, `body_init`, `constraints`,
  `mass_body` — the rest of the `DynBody` decomposition.

## JEOD conventions

- Translational state is stored in the **integration frame** (typically
  J2000 ECI), absolute (not parent-frame-relative).
- Quaternions are scalar-first left-transformation
  (`jeod_math::JeodQuat`).
- `inverse_mass` and `inverse_inertia` are pre-computed once per step
  to keep the inner loop multiply-only.

## See also

- [`docs/JEOD_invariants.md`](../../docs/JEOD_invariants.md) — `DB.*`,
  `MS.*`, `IN.*`, `RK.*`, `AB.*`, `GJ.*` invariants.
- [Project README](../../README.md) and
  [`CLAUDE.md`](../../CLAUDE.md) — workspace-level architecture.
- Rendered rustdoc:
  <https://simnaut.github.io/bevy_jeod/jeod_dynamics/>
