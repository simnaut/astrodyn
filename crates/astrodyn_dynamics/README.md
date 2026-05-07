# astrodyn_dynamics

Rigid-body dynamics, integrators (RK4, RKF45, Gauss-Jackson, ABM4), the
mass tree, and body initialization for the
[`astrodyn_bevy`](https://github.com/simnaut/astrodyn) workspace.

Ports
[`models/dynamics/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/dynamics/)
and
[`models/utils/integration/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/integration/)
from [NASA JEOD v5.4.0](https://github.com/nasa/jeod). Decomposes JEOD's
1200-line `DynBody` god-class into ~10 narrow components.

## Layered architecture

```
astrodyn_bevy        (Bevy ECS adapter, mission code)
   ↓
astrodyn         (orchestration, recipes)
   ↓
astrodyn_dynamics    ←  this crate (pure Rust, zero Bevy)
   ↓
astrodyn_math, astrodyn_frames, astrodyn_quantities
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
  J2000 ECI), absolute (not parent-frame-relative). Typed clients use
  `TranslationalStateTyped<F>`; the runner pins
  `F = astrodyn_quantities::IntegrationFrame` so consumers requiring
  root-inertial coordinates (gravity, SRP, solar beta, earth lighting)
  must apply the integration-origin shift via
  `body.trans.to_inertial(&integ_origin)` — a compile error otherwise.
  See issue #255 / `RF.10` in `docs/JEOD_invariants.md`.
- Quaternions are scalar-first left-transformation
  (`astrodyn_math::JeodQuat`).
- `inverse_mass` and `inverse_inertia` are pre-computed once per step
  to keep the inner loop multiply-only.

## See also

- [`docs/JEOD_invariants.md`](https://github.com/simnaut/astrodyn/blob/main/docs/JEOD_invariants.md) — `DB.*`,
  `MS.*`, `IN.*`, `RK.*`, `AB.*`, `GJ.*` invariants.
- [Project README](https://github.com/simnaut/astrodyn/blob/main/README.md) and
  [`CLAUDE.md`](https://github.com/simnaut/astrodyn/blob/main/CLAUDE.md) — workspace-level architecture.
- Rendered rustdoc:
  <https://simnaut.github.io/astrodyn_bevy/astrodyn_dynamics/>
