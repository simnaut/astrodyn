# JEOD → ECS mapping

How the JEOD per-step pipeline and key C++ classes map onto our Bevy ECS
architecture. Useful when porting a new JEOD subsystem or reading the
`src/sets.rs` orchestration code.

## Integration loop → `AstrodynSet`

JEOD's per-step pipeline collapses to **seven** `AstrodynSet` variants
(`src/sets.rs`), not nine. Two adjacent JEOD steps share a single set
where the bundling is natural — gravity + atmosphere both run in
`Environment`, and frame propagation rides inside the `Integration`
system as its post-step rather than as a separate schedule pass.

```
JEOD step                        →  AstrodynSet variant
1. Time update                   →  AstrodynSet::TimeUpdate
2. Ephemeris update              →  AstrodynSet::EphemerisUpdate
3. Gravity computation         ┐
                               ├─→  AstrodynSet::Environment
4. Atmosphere update           ┘
5. Aero / SRP / gravity torque   →  AstrodynSet::Interaction
6. Force collection              →  AstrodynSet::ForceCollection
7. State integration           ┐
                               ├─→  AstrodynSet::Integration
8. Frame propagation           ┘    (post-step inside the integration system)
9. Derived states                →  AstrodynSet::DerivedState
```

Multi-stage integrators (RK4 = 4 stages) run as an inner loop within the
integration system, not as multiple schedule passes.

## JEOD key classes → ECS components / systems

```
DynBody (1200-line god-class)  ->  ~10 components on an entity
DynManager                     ->  system ordering + resources
GravityManager                 ->  gravity_computation_system
TimeManager                    ->  SimulationTime resource + time_advance_system
EphemerisManager               ->  EphemerisData resource + ephemeris_update_system
RefFrame tree                  ->  entities with Parent/Children hierarchy
BodyAction subclasses          ->  events or direct initialization functions
```

`DynBody` decomposes into: `TranslationalState`, `RotationalState`,
`MassProperties`, `DynamicsConfig`, `GravityAcceleration`, `GravityControls`,
`TotalForce`, `FrameDerivatives`, plus optional interaction components
(`AerodynamicForce`, `RadiationForce`, `GravityTorque`).

## See also

- [JEOD-Source-Data](JEOD-Source-Data) — navigating the JEOD source tree, the
  catalog of extractable verification data, and the per-crate `extract_*`
  binaries.
- [Architecture](Architecture) — the three-layer rule that determines where
  each component / system actually lives in the workspace.
