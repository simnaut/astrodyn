# JEOD Invariants Catalog

Exhaustive catalog of invariants enforced by JEOD's C++ architecture. Each has a
`Section.Item` tag (e.g., `DB.03`) for cross-referencing from our Rust source
with `// JEOD_INV: DB.03` comments. To find the JEOD source for any invariant,
grep the JEOD tree for the distinctive identifier in the invariant description
(function name, field name, error message text, etc.).

**Categories:**
- `initialization` — checked during `initialize_simulation()` or model init
- `runtime` — checked on every relevant call during simulation
- `structural` — enforced by C++ class layout (mandatory members, inheritance)
- `consistency` — state synchronization (A must match B after mutation)
- `ordering` — method A must be called before method B

**Enforcement legend:**
- `fatal` — `MessageHandler::fail()` (aborts simulation)
- `error` — `MessageHandler::error()` (logged, may auto-correct)
- `warn` — `MessageHandler::warn()` (logged, continues)
- `structural` — C++ value member, constructor, deleted copy ctor
- `assert` — runtime assertion
- `flag-gate` — boolean flag skips code path

**Our status:**
- `enforced` — we check this invariant (with tag location)
- `partial` — partially enforced
- `deferred` — requires future infrastructure (noted which phase)
- `n/a` — not applicable to ECS architecture
- `structural` — guaranteed by Rust type system or Bevy ECS

---

## Section DB: DynBody

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| DB.01 | Body must have non-empty name | fatal | initialization | n/a (entities use Entity IDs) |
| DB.02 | `integ_frame_name` must be non-empty | fatal | initialization | deferred (Phase 5) |
| DB.03 | `integ_frame` must resolve to valid integration frame | fatal | runtime | deferred (Phase 5) |
| DB.04 | Three frames (structure, composite_body, core_body) always exist | structural | structural | deferred (Phase 5) |
| DB.05 | `three_dof=true` prevents rotational integrator creation | flag-gate | structural | enforced (`validation.rs:66`) |
| DB.06 | `three_dof=true` AND `rotational_dynamics=true` is invalid | implicit | consistency | enforced (`validation.rs:67`) |
| DB.07 | `translational_dynamics` gates force collection and integration | flag-gate | runtime | partial (integration gated; force collection is unconditional) |
| DB.08 | `rotational_dynamics` gates torque collection and integration | flag-gate | runtime | partial (integration gated; torque collection is unconditional) |
| DB.09 | Quaternion normalized after every integration step | structural | consistency | enforced (`integration.rs:161`, `rotational.rs:113`) |
| DB.10 | T_parent_this recomputed from quaternion after normalization | structural | consistency | n/a (computed on demand) |
| DB.11 | `initialized_states` tracks which state components are set | structural | initialization | partial (`validation.rs:149` warns on zero state) |
| DB.12 | `integrated_frame` must be structure or composite_body | fatal | structural | deferred (Phase 5) |
| DB.13 | State propagation delegates to root body | structural | consistency | deferred (Phase 5) |
| DB.14 | Integration frame switch delegates to root body | structural | consistency | deferred (Phase 5) |
| DB.15 | `grav_interaction` always synchronized with integration frame | structural | consistency | deferred (Phase 5) |
| DB.16 | Child forces propagated to parent recursively | structural | ordering | deferred (Phase 5) |
| DB.17 | Only root body computes total acceleration | structural | structural | deferred (Phase 5) |
| DB.18 | `inverse_mass` used for F=ma (precomputed) | structural | consistency | enforced (`systems.rs:138`, `forces.rs:64`; we divide by mass at runtime instead of precomputing inverse) |
| DB.19 | `inverse_inertia` used for Euler equation | structural | consistency | enforced (`validation.rs:101`, `rotational.rs:46`) |
| DB.20 | Small rot_accel truncated to zero (< 1e-20) | structural | runtime | not enforced |
| DB.21 | Only unattached bodies integrate | flag-gate | runtime | deferred (Phase 5, no frame attachment yet) |
| DB.22 | DynBody not copyable | structural | structural | n/a (ECS components are Copy where needed) |
| DB.23 | `compute_inverse_inertia` enabled for DynBody | structural | structural | structural (`mass.rs:38`, always computed in `MassProperties::with_inertia`) |
| DB.24 | Default `integrated_frame` is composite_body | structural | structural | structural (`components.rs:9`, we integrate composite_body state) |
| DB.25 | DynBody name is reference to MassBody name | structural | structural | n/a (ECS entities, no name reference) |
| DB.26 | DynBody mass constructed with `this` as owner | structural | structural | n/a (ECS entities, no ownership reference) |
| DB.27 | State initialization order: attitude → rate → position → velocity | structural | ordering | deferred (Phase 5) |

## Section MA: Mass / MassBody / MassProperties

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| MA.01 | MassBody always present on DynBody (value member) | structural | structural | enforced (`validation.rs:81`, `systems.rs:139`) |
| MA.02 | mass > 0 for meaningful dynamics | conditional | consistency | enforced (`mass.rs:20,36`, `systems.rs:140`) |
| MA.03 | `inverse_mass` consistent with mass | conditional | consistency | n/a (no `inverse_mass` field; we divide by `mass` at runtime) |
| MA.04 | `inverse_inertia` consistent with inertia | structural | consistency | enforced (`mass.rs:39`, `validation.rs:102`) |
| MA.05 | Inverse inertia computed only for root bodies with positive mass | conditional | consistency | structural (`mass.rs:37`, all bodies compute inverse — intentional divergence) |
| MA.06 | Bottom-up mass property update (children first) | structural | ordering | enforced (`mass_body.rs:240`) |
| MA.07 | `needs_update` flag cleared after recomputation | structural | consistency | structural (`mass_body.rs:241`, always recomputes) |
| MA.08 | No cycle in mass tree | error | consistency | enforced (`mass_body.rs:164`) |
| MA.09 | MassPoint names unique within body | fatal | initialization | deferred (no mass points in ECS yet) |
| MA.10 | MassPoint names non-empty | fatal | initialization | deferred |
| MA.11 | core/composite attached to structure (internal tree) | structural | structural | deferred (Phase 5, three-frame model) |
| MA.12 | core_wrt_composite has identity orientation | structural | structural | deferred (Phase 5) |
| MA.13 | MassBody not copyable | structural | structural | n/a |
| MA.14 | MassProperties not copyable | structural | structural | n/a (our MassProperties is Copy) |
| MA.15 | Detach recomputes inverse inertia for new root | structural | consistency | enforced (`mass_body.rs:213`) |
| MA.16 | 180° yaw convention for attach-by-point | structural | structural | deferred (no attach-by-point yet) |
| MA.17 | Dynamic attachment conserves momentum | structural | consistency | deferred (Phase 5) |
| MA.18 | Partially initialized child state blocks attachment | warn | consistency | deferred (Phase 5) |
| MA.19 | No same-tree attachment (cycle prevention) | error | consistency | enforced (`mass_body.rs:165`) |
| MA.20 | Child integration frame synced to parent on attach | structural | consistency | deferred (Phase 5) |

## Section DM: DynManager

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| DM.01 | At most one GravityManager registered | error | structural | n/a (no GravityManager singleton in ECS) |
| DM.02 | GravityManager registered before `initialized=true` | error | ordering | n/a |
| DM.03 | `initialized` flag set last in init sequence | structural | initialization | partial (validation system uses `Local<bool>` for one-shot) |
| DM.04 | Init order: ephemerides → gravity controls → frame ownership → activate → update → gravity state → integ groups → dyn bodies | structural | ordering | partial (system set ordering: TimeUpdate → Ephemeris → Environment → ...) |
| DM.05 | All required states initialized before first integration | fatal | initialization | partial (`validation.rs` warns on zero state) |
| DM.06 | DynBody name unique across all bodies | error | structural | n/a (ECS entity IDs are unique) |
| DM.07 | DynBody name unique across MassBodies | error | structural | n/a |
| DM.08 | Gravitation requires initialized + gravity_manager | error | runtime | partial (no initialized gate; gravity source panic is enforced) |
| DM.09 | Init order: mass init → mass attach → mass update → state init | structural | ordering | n/a (user assembles entities directly) |
| DM.10 | Only root bodies get gravity computed | structural | runtime | deferred (Phase 5, no parent/child DynBody) |
| DM.11 | Only root bodies collect forces | structural | runtime | deferred (Phase 5) |
| DM.12 | Only root bodies are integrated | structural | runtime | deferred (Phase 5) |
| DM.13 | Ephemeris updated before gravity if needed | structural | ordering | enforced (EphemerisUpdate before Environment in system sets) |

## Section GV: Gravity

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| GV.01 | Gravity source name non-empty | fatal | initialization | n/a (sources referenced by Entity) |
| GV.02 | Gravity source name unique | fatal | structural | n/a (Entity IDs unique) |
| GV.03 | `check_validity()` called on every degree/order mutation | structural | runtime | partial (`validation.rs` at startup; JEOD also validates on every setter — our fields are public with no setter guards) |
| GV.04 | degree ≤ source degree | fatal | runtime | enforced (`gravity_controls.rs check_validity`) |
| GV.05 | order ≤ source order | fatal | runtime | enforced (`gravity_controls.rs check_validity`) |
| GV.06 | order ≤ degree | fatal | runtime | enforced (`gravity_controls.rs check_validity`) |
| GV.07 | degree=0 with spherical=false auto-corrects to spherical | error | consistency | enforced (`gravity_controls.rs check_validity`) |
| GV.08 | gradient_degree ≤ degree (clamped) | error | consistency | enforced (`gravity_controls.rs check_validity`) |
| GV.09 | gradient_degree ≠ 1 (reset to 0) | error | consistency | enforced (`gravity_controls.rs check_validity`) |
| GV.10 | gradient_order ≤ gradient_degree (clamped) | error | consistency | enforced (`gravity_controls.rs check_validity`) |
| GV.11 | gradient_order ≤ order (clamped) | error | consistency | enforced (`gravity_controls.rs check_validity`) |
| GV.12 | Gravity source must exist for control | error | initialization | enforced (`validation.rs` + `systems.rs` panic) |
| GV.13 | Gravity source must have inertial frame | error | initialization | enforced (`systems.rs` panics if nonspherical without PlanetFixedRotationC) |
| GV.14 | Third-body vs direct gravity classification | structural | initialization | deferred (Phase 5, requires frame tree ancestry) |
| GV.15 | `integ_frame_index` synchronized with body's integration frame | structural | consistency | deferred (Phase 5) |
| GV.16 | Active controls subscribe to inertial frame | structural | consistency | n/a (no frame subscription in ECS) |
| GV.17 | Active nonspherical controls subscribe to planet-fixed frame | structural | consistency | enforced (PlanetFixedRotationC required for nonspherical) |
| GV.18 | Gravity source name matches planet name | structural | consistency | n/a (matched by Entity in ECS) |

## Section TM: Time

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| TM.01 | Time type names unique | fatal | initialization | n/a (single SimulationTime resource) |
| TM.02 | Converter type-pair names required | fatal | initialization | n/a |
| TM.03 | Time types updated in dependency order | structural | ordering | structural (`SimulationTime::advance` updates all scales in order) |
| TM.04 | Init tree completeness (all types reachable from initializer) | fatal | initialization | structural (all scales hardcoded in `SimulationTime`) |
| TM.05 | Update tree completeness (all types reachable from TimeDyn) | fatal | initialization | structural |
| TM.06 | No duplicate converters between same pair | fatal | initialization | structural |
| TM.07 | `simtime` initialized to -1.0 (forces first update) | structural | initialization | structural (`SimulationTime` constructed with explicit epoch) |

## Section RF: Reference Frames

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| RF.01 | `compute_relative_state` requires same tree | fatal | runtime | deferred (Phase 5) |
| RF.02 | `compute_state_wrt_pred` requires valid predecessor | fatal | runtime | deferred (Phase 5) |
| RF.03 | Quaternion normalized after every composition | structural | consistency | structural (normalized in `incr_right`, `negate`, and integration) |
| RF.04 | T_parent_this recomputed after quaternion composition | structural | consistency | structural (T derived from normalized Q in both `incr_right` and `negate`) |
| RF.05 | `ang_vel_products` recomputed after angular velocity change | structural | consistency | n/a (we don't cache products) |
| RF.06 | Position/velocity in parent coordinates | structural | structural | structural (documented convention) |
| RF.07 | Q_parent_this is left-transformation quaternion | structural | structural | structural (documented, JeodQuat convention) |
| RF.08 | Frame names unique | error | initialization | n/a (Entity IDs) |
| RF.09 | Quaternion assumed normalized for `left_quat_to_transformation` | implicit | consistency | structural (`normalize_integ` called after every integration step) |

## Section EP: Ephemeris

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| EP.01 | Planet name required and unique | error | initialization | n/a (Entity IDs) |
| EP.02 | Ephemeris models registered in dependency order | structural | ordering | deferred (Phase 5) |
| EP.03 | Frame tree rebuilt on active-status change | structural | runtime | deferred (Phase 5) |
| EP.04 | `integ_frame_index` lookup must succeed | fatal | runtime | deferred (Phase 5) |

## Section AT: Atmosphere

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| AT.01 | `active` flag gates computation | flag-gate | runtime | structural (no atmosphere → no AtmosphericStateC) |
| AT.02 | Atmosphere model pointer non-null for update | structural | runtime | structural (AtmosphereModelR resource checked) |
| AT.03 | Planet-fixed position required for geodetic altitude | structural | runtime | enforced (`bevy_jeod_atmosphere/systems.rs` — panics if planet_entity set but PlanetFixedRotationC missing) |

## Section IN: Interactions

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| IN.01 | GravityTorque.subject_body required (non-null) | fatal | runtime | structural (system queries require all components) |
| IN.02 | GravityTorque.active gates computation | flag-gate | runtime | structural (no GravityTorqueC → no torque) |
| IN.03 | AerodynamicDrag.active gates computation | flag-gate | runtime | structural (no DragConfigC → no drag) |
| IN.04 | `aero_surface_ptr` required when `use_default_behavior=false` | fatal | runtime | n/a (only ballistic model implemented) |
| IN.05 | Ballistic coefficient non-zero for DRAG_OPT_BC | fatal | runtime | n/a (only DRAG_OPT_CD implemented) |
| IN.06 | RadiationPressure.active gates computation | flag-gate | runtime | structural (no SrpConfigC → no SRP) |
| IN.07 | RadiationThirdBody name required | fatal | initialization | n/a (shadow bodies by Entity) |
| IN.08 | RadiationThirdBody belongs to one model only | fatal | structural | n/a (function-based, no ownership) |
| IN.09 | RadiationSource planet must be found by DynManager | fatal | initialization | enforced (`bevy_jeod_interactions/systems.rs` — panics on multiple SunMarker; zero SunMarker = SRP not configured, early return like JEOD `active=false`) |
| IN.10 | RadiationSource.luminosity ≥ 1e-6 for flux computation | flag-gate | runtime | n/a (luminosity is a compile-time constant; `distance < 1.0` guard prevents division by near-zero) |
| IN.11 | RadiationThirdBody.radius > 0 | fatal | initialization | enforced (`shadow.rs` handles degenerate cases) |
| IN.12 | RadiationSource.radius > 0 | fatal | initialization | enforced (`shadow.rs` handles degenerate cases) |
| IN.13 | Shadow model: vehicle distance > 0 | error | runtime | enforced (`shadow.rs` returns 0.0 if r_mag2 <= 0) |
| IN.14 | `d_source_to_third` > 0 | error | runtime | enforced (`shadow.rs` returns 1.0 if d <= 0) |
| IN.15 | Aero drag requires body orientation (T_inertial_struct) | structural (mandatory fn parameter) | runtime | enforced (`bevy_jeod_dynamics/systems.rs` — panics if AerodynamicForceC present without RotationalStateC) |

## Section FD: FrameDerivatives

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| FD.01 | `trans_accel = non_grav_accel + grav_accel` | structural | consistency | enforced (`systems.rs` force_collection_system writes FrameDerivativesC) |
| FD.02 | `rot_accel = I^-1 * (tau - omega x I*omega)` | structural | consistency | enforced (`systems.rs` force_collection_system writes FrameDerivativesC) |
