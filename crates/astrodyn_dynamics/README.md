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

## When to use

- **Propagating a rigid body's state** — translational (3-DOF) or
  full 6-DOF — through an RK4, RKF45, Adams-Bashforth-Moulton, or
  Gauss-Jackson integrator step.
- **Composing mass trees** with parallel-axis-theorem (Steiner)
  contributions from offset child masses, and re-deriving the
  composite center of mass / inertia tensor after attach / detach.
- **Initializing a body** from orbital elements, mean anomaly,
  time-since-periapsis, LVLH, or NED — all five JEOD `BodyAction`
  pathways are ported.
- **Pre-/post-step frame propagation** — moving state between a
  body's `structure`, `composite_body`, and `core_body` frames, and
  applying holonomic constraints (tethers, articulated joints).

Mission code rarely calls into this crate directly; the
`VehicleBuilder` typestate in `astrodyn` and the `astrodyn_bevy`
components wrap these primitives. Reach for `astrodyn_dynamics`
directly when porting JEOD `BodyAction` subclasses, when adding a new
integrator, or when implementing a custom constraint.

## Key concepts

Translational state is stored **absolute in the integration frame**
(typically J2000 ECI), not parent-relative — this is the JEOD
convention and it's enforced at the type level via
`TranslationalStateTyped<IntegrationFrame>`. Consumers that require
root-inertial coordinates (gravity, SRP, solar beta, Earth lighting)
must apply the integration-origin shift through
`body.trans.to_inertial(&integ_origin)`; the compiler refuses to pass
integration-frame state where root-inertial is required. See `RF.10`
in `docs/JEOD_invariants.md` for the structural rationale.

Mass properties carry both `mass` and the pre-computed
`inverse_mass` (similarly `inertia` and `inverse_inertia`), refreshed
once per step so the inner force-to-acceleration loop is
multiply-only. The `MassTree` keeps subtree composition in Steiner
form, so a leaf re-attach updates the composite CoM / inertia at the
root via the same algorithm JEOD uses. Integrators (`rk4_*`,
`rkf45`, `abm4`, `gauss_jackson`) all reduce to the same per-stage
derivative call (`compute_translational_derivatives` /
`compute_frame_derivatives`), so adding a new integrator is purely
adding a new dispatch arm to `IntegrationMethod`.

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
  <https://docs.rs/astrodyn_dynamics>
