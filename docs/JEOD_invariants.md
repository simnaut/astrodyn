# JEOD Invariants Catalog

Exhaustive catalog of invariants enforced by JEOD's C++ architecture. Each has a
`Section.Item` tag (e.g., `DB.03`) for cross-referencing from our Rust source
with `// JEOD_INV: DB.03` comments.

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

| Tag | Invariant | JEOD Source | Line(s) | Enforcement | Category | Our Status |
|-----|-----------|-------------|---------|-------------|----------|------------|
| DB.01 | Body must have non-empty name | `dyn_body/src/dyn_body_initialize_model.cc` | 46-55 | fatal | initialization | n/a (entities use Entity IDs) |
| DB.02 | `integ_frame_name` must be non-empty | `dyn_body/src/dyn_body_initialize_model.cc` | 71-81 | fatal | initialization | deferred (Phase 5) |
| DB.03 | `integ_frame` must resolve to valid integration frame | `dyn_body/src/dyn_body_integration.cc` | 125-136 | fatal | runtime | deferred (Phase 5) |
| DB.04 | Three frames (structure, composite_body, core_body) always exist | `dyn_body/src/dyn_body_initialize_model.cc` | 96-111 | structural | structural | deferred (Phase 5) |
| DB.05 | `three_dof=true` prevents rotational integrator creation | `dyn_body/src/dyn_body_integration.cc` | 223 | flag-gate | structural | enforced (`validation.rs:66`) |
| DB.06 | `three_dof=true` AND `rotational_dynamics=true` is invalid | `dyn_body/include/dyn_body.hh` | 697 | implicit | consistency | enforced (`validation.rs:67`) |
| DB.07 | `translational_dynamics` gates force collection and integration | `dyn_body/src/dyn_body_collect.cc` | 95-108 | flag-gate | runtime | enforced (`systems.rs:41,133`) |
| DB.08 | `rotational_dynamics` gates torque collection and integration | `dyn_body/src/dyn_body_collect.cc` | 110-124 | flag-gate | runtime | enforced (`systems.rs:42,206`) |
| DB.09 | Quaternion normalized after every integration step | `dyn_body/src/dyn_body_integration.cc` | 380 | structural | consistency | enforced (`integration.rs:161`, `rotational.rs:113`) |
| DB.10 | T_parent_this recomputed from quaternion after normalization | `dyn_body/src/dyn_body_integration.cc` | 383 | structural | consistency | n/a (computed on demand) |
| DB.11 | `initialized_states` tracks which state components are set | `dyn_body/include/dyn_body.hh` | 1128 | structural | initialization | partial (`validation.rs:149` warns on zero state) |
| DB.12 | `integrated_frame` must be structure or composite_body | `dyn_body/src/dyn_body_propagate_state.cc` | 147-154 | fatal | structural | deferred (Phase 5) |
| DB.13 | State propagation delegates to root body | `dyn_body/src/dyn_body_propagate_state.cc` | 529-533 | structural | consistency | deferred (Phase 5) |
| DB.14 | Integration frame switch delegates to root body | `dyn_body/src/dyn_body_integration.cc` | 148-152 | structural | consistency | deferred (Phase 5) |
| DB.15 | `grav_interaction` always synchronized with integration frame | `dyn_body/src/dyn_body_integration.cc` | 113 | structural | consistency | deferred (Phase 5) |
| DB.16 | Child forces propagated to parent recursively | `dyn_body/src/dyn_body_collect.cc` | 128-131 | structural | ordering | deferred (Phase 5) |
| DB.17 | Only root body computes total acceleration | `dyn_body/src/dyn_body_collect.cc` | 205-279 | structural | structural | deferred (Phase 5) |
| DB.18 | `inverse_mass` used for F=ma (precomputed) | `dyn_body/src/dyn_body_collect.cc` | 224 | structural | consistency | enforced (`systems.rs:138`, `forces.rs:64`) |
| DB.19 | `inverse_inertia` used for Euler equation | `dyn_body/src/dyn_body_collect.cc` | 264 | structural | consistency | enforced (`validation.rs:101`, `rotational.rs:46`) |
| DB.20 | Small rot_accel truncated to zero (< 1e-20) | `dyn_body/src/dyn_body_collect.cc` | 267 | structural | runtime | not enforced |
| DB.21 | Only unattached bodies integrate | `dyn_body/src/dyn_body_integration.cc` | 309 | flag-gate | runtime | deferred (Phase 5, no frame attachment yet) |
| DB.22 | DynBody not copyable | `dyn_body/include/dyn_body.hh` | 131-132 | structural | structural | n/a (ECS components are Copy where needed) |
| DB.23 | `compute_inverse_inertia` enabled for DynBody | `dyn_body/src/dyn_body.cc` | 76 | structural | structural | structural (`mass.rs:38`, always computed in `MassProperties::with_inertia`) |
| DB.24 | Default `integrated_frame` is composite_body | `dyn_body/src/dyn_body.cc` | 73 | structural | structural | structural (`components.rs:9`, we integrate composite_body state) |
| DB.25 | DynBody name is reference to MassBody name | `dyn_body/src/dyn_body.cc` | 63 | structural | structural | n/a (ECS entities, no name reference) |
| DB.26 | DynBody mass constructed with `this` as owner | `dyn_body/src/dyn_body.cc` | 62 | structural | structural | n/a (ECS entities, no ownership reference) |
| DB.27 | State initialization order: attitude → rate → position → velocity | `dyn_body/src/dyn_body_propagate_state.cc` | 388-516 | structural | ordering | deferred (Phase 5) |

## Section MA: Mass / MassBody / MassProperties

| Tag | Invariant | JEOD Source | Line(s) | Enforcement | Category | Our Status |
|-----|-----------|-------------|---------|-------------|----------|------------|
| MA.01 | MassBody always present on DynBody (value member) | `dyn_body/include/dyn_body.hh` | 617 | structural | structural | enforced (`validation.rs:81`, `systems.rs:139`) |
| MA.02 | mass > 0 for meaningful dynamics | `mass/src/mass_update.cc` | 63-69 | conditional | consistency | enforced (`mass.rs:20,36`, `systems.rs:140`) |
| MA.03 | `inverse_mass` consistent with mass | `mass/src/mass_update.cc` | 63-69 | structural | consistency | partial (`mass.rs:21`, computed at construction, not re-synced on mutation) |
| MA.04 | `inverse_inertia` consistent with inertia | `mass/src/mass_update.cc` | 117-125 | structural | consistency | enforced (`mass.rs:39`, `validation.rs:102`) |
| MA.05 | Inverse inertia computed only for root bodies with positive mass | `mass/src/mass_update.cc` | 117-125 | conditional | consistency | structural (`mass.rs:37`, all bodies compute inverse in our architecture) |
| MA.06 | Bottom-up mass property update (children first) | `mass/src/mass_update.cc` | 81-88 | structural | ordering | enforced (`mass_body.rs:240`) |
| MA.07 | `needs_update` flag cleared after recomputation | `mass/src/mass_update.cc` | 146 | structural | consistency | structural (`mass_body.rs:241`, always recomputes) |
| MA.08 | No cycle in mass tree | `mass/src/mass_attach.cc` | 373-389 | error | consistency | enforced (`mass_body.rs:164`) |
| MA.09 | MassPoint names unique within body | `mass/src/mass.cc` | 360-372 | fatal | initialization | deferred (no mass points in ECS yet) |
| MA.10 | MassPoint names non-empty | `mass/src/mass.cc` | 346-357 | fatal | initialization | deferred |
| MA.11 | core/composite attached to structure (internal tree) | `mass/src/mass.cc` | 86-87 | structural | structural | deferred (Phase 5, three-frame model) |
| MA.12 | core_wrt_composite has identity orientation | `mass/src/mass.cc` | 101 | structural | structural | deferred (Phase 5) |
| MA.13 | MassBody not copyable | `mass/include/mass.hh` | 128-129 | structural | structural | n/a |
| MA.14 | MassProperties not copyable | `mass/include/mass_properties.hh` | 123-124 | structural | structural | n/a (our MassProperties is Copy) |
| MA.15 | Detach recomputes inverse inertia for new root | `mass/src/mass_detach.cc` | 327-335 | structural | consistency | enforced (`mass_body.rs:213`) |
| MA.16 | 180° yaw convention for attach-by-point | `dyn_body/src/dyn_body_attach.cc` | 469-472 | structural | structural | deferred (no attach-by-point yet) |
| MA.17 | Dynamic attachment conserves momentum | `dyn_body/src/dyn_body_attach.cc` | 876-1117 | structural | consistency | deferred (Phase 5) |
| MA.18 | Partially initialized child state blocks attachment | `dyn_body/src/dyn_body_attach.cc` | 121-136 | warn | consistency | deferred (Phase 5) |
| MA.19 | No same-tree attachment (cycle prevention) | `dyn_body/src/dyn_body_attach.cc` | 72-87 | error | consistency | enforced (`mass_body.rs:165`) |
| MA.20 | Child integration frame synced to parent on attach | `dyn_body/src/dyn_body_attach.cc` | 791-795 | structural | consistency | deferred (Phase 5) |

## Section DM: DynManager

| Tag | Invariant | JEOD Source | Line(s) | Enforcement | Category | Our Status |
|-----|-----------|-------------|---------|-------------|----------|------------|
| DM.01 | At most one GravityManager registered | `dyn_manager/src/dyn_manager.cc` | 128-134 | error | structural | n/a (no GravityManager singleton in ECS) |
| DM.02 | GravityManager registered before `initialized=true` | `dyn_manager/src/dyn_manager.cc` | 137-145 | error | ordering | n/a |
| DM.03 | `initialized` flag set last in init sequence | `dyn_manager/src/initialize_simulation.cc` | 94 | structural | initialization | partial (validation system uses `Local<bool>` for one-shot) |
| DM.04 | Init order: ephemerides → gravity controls → frame ownership → activate → update → gravity state → integ groups → dyn bodies | `dyn_manager/src/initialize_simulation.cc` | 64-85 | structural | ordering | partial (system set ordering: TimeUpdate → Ephemeris → Environment → ...) |
| DM.05 | All required states initialized before first integration | `dyn_manager/src/initialize_dyn_bodies.cc` | 337-385 | fatal | initialization | partial (`validation.rs` warns on zero state) |
| DM.06 | DynBody name unique across all bodies | `dyn_manager/src/dyn_bodies_primitives.cc` | 118-126 | error | structural | n/a (ECS entity IDs are unique) |
| DM.07 | DynBody name unique across MassBodies | `dyn_manager/src/dyn_bodies_primitives.cc` | 129-151 | error | structural | n/a |
| DM.08 | Gravitation requires initialized + gravity_manager | `dyn_manager/src/gravitation.cc` | 125-167 | error | runtime | partial (no initialized gate; gravity source panic is enforced) |
| DM.09 | Init order: mass init → mass attach → mass update → state init | `dyn_manager/src/initialize_dyn_bodies.cc` | 55-90 | structural | ordering | n/a (user assembles entities directly) |
| DM.10 | Only root bodies get gravity computed | `dynamics_integration_group.cc` | 282-292 | structural | runtime | deferred (Phase 5, no parent/child DynBody) |
| DM.11 | Only root bodies collect forces | `dynamics_integration_group.cc` | 300-313 | structural | runtime | deferred (Phase 5) |
| DM.12 | Only root bodies are integrated | `dynamics_integration_group.cc` | 361-373 | structural | runtime | deferred (Phase 5) |
| DM.13 | Ephemeris updated before gravity if needed | `dynamics_integration_group.cc` | 275-278 | structural | ordering | enforced (EphemerisUpdate before Environment in system sets) |

## Section GV: Gravity

| Tag | Invariant | JEOD Source | Line(s) | Enforcement | Category | Our Status |
|-----|-----------|-------------|---------|-------------|----------|------------|
| GV.01 | Gravity source name non-empty | `gravity/src/gravity_manager.cc` | 113-119 | fatal | initialization | n/a (sources referenced by Entity) |
| GV.02 | Gravity source name unique | `gravity/src/gravity_manager.cc` | 122-130 | fatal | structural | n/a (Entity IDs unique) |
| GV.03 | `check_validity()` called on every degree/order mutation | `gravity/src/spherical_harmonics_gravity_controls.cc` | 248-317 | structural | runtime | partial (`validation.rs` at startup; JEOD also validates on every setter — our fields are public with no setter guards) |
| GV.04 | degree ≤ source degree | `gravity/src/spherical_harmonics_gravity_controls.cc` | 350-361 | fatal | runtime | enforced (`gravity_controls.rs check_validity`) |
| GV.05 | order ≤ source order | `gravity/src/spherical_harmonics_gravity_controls.cc` | 365-376 | fatal | runtime | enforced (`gravity_controls.rs check_validity`) |
| GV.06 | order ≤ degree | `gravity/src/spherical_harmonics_gravity_controls.cc` | 380-391 | fatal | runtime | enforced (`gravity_controls.rs check_validity`) |
| GV.07 | degree=0 with spherical=false auto-corrects to spherical | `gravity/src/spherical_harmonics_gravity_controls.cc` | 334-347 | error | consistency | enforced (`gravity_controls.rs check_validity`) |
| GV.08 | gradient_degree ≤ degree (clamped) | `gravity/src/spherical_harmonics_gravity_controls.cc` | 398-410 | error | consistency | enforced (`gravity_controls.rs check_validity`) |
| GV.09 | gradient_degree ≠ 1 (reset to 0) | `gravity/src/spherical_harmonics_gravity_controls.cc` | 413-420 | error | consistency | enforced (`gravity_controls.rs check_validity`) |
| GV.10 | gradient_order ≤ gradient_degree (clamped) | `gravity/src/spherical_harmonics_gravity_controls.cc` | 424-438 | error | consistency | enforced (`gravity_controls.rs check_validity`) |
| GV.11 | gradient_order ≤ order (clamped) | `gravity/src/spherical_harmonics_gravity_controls.cc` | 441-453 | error | consistency | enforced (`gravity_controls.rs check_validity`) |
| GV.12 | Gravity source must exist for control | `gravity/src/gravity_controls.cc` | 71-98 | error | initialization | enforced (`validation.rs` + `systems.rs` panic) |
| GV.13 | Gravity source must have inertial frame | `gravity/src/gravity_controls.cc` | 101-109 | error | initialization | enforced (`systems.rs` panics if nonspherical without PlanetFixedRotationC) |
| GV.14 | Third-body vs direct gravity classification | `gravity/src/gravity_source.cc` | 88-101 | structural | initialization | deferred (Phase 5, requires frame tree ancestry) |
| GV.15 | `integ_frame_index` synchronized with body's integration frame | `gravity/include/gravity_interaction.hh` | 102-107 | structural | consistency | deferred (Phase 5) |
| GV.16 | Active controls subscribe to inertial frame | `gravity/src/gravity_controls.cc` | 135-152 | structural | consistency | n/a (no frame subscription in ECS) |
| GV.17 | Active nonspherical controls subscribe to planet-fixed frame | `gravity/src/gravity_controls.cc` | 156-177 | structural | consistency | enforced (PlanetFixedRotationC required for nonspherical) |
| GV.18 | Gravity source name matches planet name | `gravity/include/gravity_source.hh` | 97-101 | structural | consistency | n/a (matched by Entity in ECS) |

## Section TM: Time

| Tag | Invariant | JEOD Source | Line(s) | Enforcement | Category | Our Status |
|-----|-----------|-------------|---------|-------------|----------|------------|
| TM.01 | Time type names unique | `time/src/time_manager.cc` | 354-365 | fatal | initialization | n/a (single SimulationTime resource) |
| TM.02 | Converter type-pair names required | `time/src/time_manager.cc` | 240-278 | fatal | initialization | n/a |
| TM.03 | Time types updated in dependency order | `time/src/time_manager.cc` | 397-420 | structural | ordering | structural (`SimulationTime::advance` updates all scales in order) |
| TM.04 | Init tree completeness (all types reachable from initializer) | `time/src/time_manager_init.cc` | 465-491 | fatal | initialization | structural (all scales hardcoded in `SimulationTime`) |
| TM.05 | Update tree completeness (all types reachable from TimeDyn) | `time/src/time_manager_init.cc` | 564-591 | fatal | initialization | structural |
| TM.06 | No duplicate converters between same pair | `time/src/time_manager_init.cc` | 291-304 | fatal | initialization | structural |
| TM.07 | `simtime` initialized to -1.0 (forces first update) | `time/include/time_manager.hh` | 104 | structural | initialization | structural (`SimulationTime` constructed with explicit epoch) |

## Section RF: Reference Frames

| Tag | Invariant | JEOD Source | Line(s) | Enforcement | Category | Our Status |
|-----|-----------|-------------|---------|-------------|----------|------------|
| RF.01 | `compute_relative_state` requires same tree | `ref_frames/src/ref_frame_compute_relative_state.cc` | 82-92 | fatal | runtime | deferred (Phase 5) |
| RF.02 | `compute_state_wrt_pred` requires valid predecessor | `ref_frames/src/ref_frame_compute_relative_state.cc` | 196-207 | fatal | runtime | deferred (Phase 5) |
| RF.03 | Quaternion normalized after every composition | `ref_frames/src/ref_frame_state.cc` | 270-271 | structural | consistency | structural (our composition functions normalize) |
| RF.04 | T_parent_this recomputed after quaternion composition | `ref_frames/src/ref_frame_state.cc` | 272-273 | structural | consistency | structural (our RefFrameState keeps T in sync) |
| RF.05 | `ang_vel_products` recomputed after angular velocity change | `ref_frames/src/ref_frame_state.cc` | 225 | structural | consistency | n/a (we don't cache products) |
| RF.06 | Position/velocity in parent coordinates | `ref_frames/include/ref_frame_state.hh` | 88-98 | structural | structural | structural (documented convention) |
| RF.07 | Q_parent_this is left-transformation quaternion | `ref_frames/include/ref_frame_state.hh` | 127-143 | structural | structural | structural (documented, JeodQuat convention) |
| RF.08 | Frame names unique | `ref_frames/src/ref_frame_manager.cc` | 91-99 | error | initialization | n/a (Entity IDs) |
| RF.09 | Quaternion assumed normalized for `left_quat_to_transformation` | `quaternion/src/quat_to_mat.cc` | 79 | implicit | consistency | structural (`normalize_integ` called after every integration step) |

## Section EP: Ephemeris

| Tag | Invariant | JEOD Source | Line(s) | Enforcement | Category | Our Status |
|-----|-----------|-------------|---------|-------------|----------|------------|
| EP.01 | Planet name required and unique | `ephem_manager/src/ephem_manager.cc` | 108-133 | error | initialization | n/a (Entity IDs) |
| EP.02 | Ephemeris models registered in dependency order | `ephem_manager/src/ephem_manager.cc` | 204-211 | structural | ordering | deferred (Phase 5) |
| EP.03 | Frame tree rebuilt on active-status change | `ephem_manager/src/ephem_manager.cc` | 91-94 | structural | runtime | deferred (Phase 5) |
| EP.04 | `integ_frame_index` lookup must succeed | `ephem_manager/src/ephem_manager.cc` | 516-527 | fatal | runtime | deferred (Phase 5) |

## Section AT: Atmosphere

| Tag | Invariant | JEOD Source | Line(s) | Enforcement | Category | Our Status |
|-----|-----------|-------------|---------|-------------|----------|------------|
| AT.01 | `active` flag gates computation | `base_atmos/include/atmosphere.hh` | 86 | flag-gate | runtime | structural (no atmosphere → no AtmosphericStateC) |
| AT.02 | Atmosphere model pointer non-null for update | `base_atmos/src/atmosphere_state.cc` | 92-96 | structural | runtime | structural (AtmosphereModelR resource checked) |
| AT.03 | Planet-fixed position required for geodetic altitude | `base_atmos/src/atmosphere_state.cc` | 110-113 | structural | runtime | enforced (`bevy_jeod_atmosphere/systems.rs` — panics if planet_entity set but PlanetFixedRotationC missing) |

## Section IN: Interactions

| Tag | Invariant | JEOD Source | Line(s) | Enforcement | Category | Our Status |
|-----|-----------|-------------|---------|-------------|----------|------------|
| IN.01 | GravityTorque.subject_body required (non-null) | `gravity_torque/src/gravity_torque.cc` | 72-78 | fatal | runtime | structural (system queries require all components) |
| IN.02 | GravityTorque.active gates computation | `gravity_torque/src/gravity_torque.cc` | 64-69 | flag-gate | runtime | structural (no GravityTorqueC → no torque) |
| IN.03 | AerodynamicDrag.active gates computation | `aerodynamics/src/aero_drag.cc` | 101-105 | flag-gate | runtime | structural (no DragConfigC → no drag) |
| IN.04 | `aero_surface_ptr` required when `use_default_behavior=false` | `aerodynamics/src/aero_drag.cc` | 143-151 | fatal | runtime | n/a (only ballistic model implemented) |
| IN.05 | Ballistic coefficient non-zero for DRAG_OPT_BC | `aerodynamics/src/default_aero.cc` | 74-83 | fatal | runtime | n/a (only DRAG_OPT_CD implemented) |
| IN.06 | RadiationPressure.active gates computation | `radiation_pressure/src/radiation_pressure.cc` | 99-102 | flag-gate | runtime | structural (no SrpConfigC → no SRP) |
| IN.07 | RadiationThirdBody name required | `radiation_pressure/src/radiation_third_body.cc` | 59-68 | fatal | initialization | n/a (shadow bodies by Entity) |
| IN.08 | RadiationThirdBody belongs to one model only | `radiation_pressure/src/radiation_pressure.cc` | 203-224 | fatal | structural | n/a (function-based, no ownership) |
| IN.09 | RadiationSource planet must exist (exactly one) | `radiation_pressure/src/radiation_source.cc` | 119-133 | fatal | initialization | enforced (`bevy_jeod_interactions/systems.rs` — panics on multiple SunMarker; zero SunMarker = SRP not configured, early return like JEOD `active=false`) |
| IN.10 | RadiationSource.luminosity > 0 for flux | `radiation_pressure/src/radiation_source.cc` | 74-78 | structural | runtime | enforced (`radiation_pressure.rs` returns zero for distance < 1) |
| IN.11 | RadiationThirdBody.radius > 0 | `radiation_pressure/src/radiation_third_body.cc` | 163-177 | fatal | initialization | enforced (`shadow.rs` handles degenerate cases) |
| IN.12 | RadiationSource.radius > 0 | `radiation_pressure/src/radiation_third_body.cc` | 102-114 | fatal | initialization | enforced (`shadow.rs` handles degenerate cases) |
| IN.13 | Shadow model: vehicle distance > 0 | `radiation_pressure/src/radiation_third_body.cc` | 213-227 | error | runtime | enforced (`shadow.rs` returns 0.0 if r_mag2 <= 0) |
| IN.14 | `d_source_to_third` > 0 | `radiation_pressure/src/radiation_third_body.cc` | 446-460 | error | runtime | enforced (`shadow.rs` returns 1.0 if d <= 0) |
| IN.15 | Aero drag requires body orientation (T_inertial_struct) | `aerodynamics/src/aero_drag.cc` | 83-87 | structural (mandatory fn parameter) | runtime | enforced (`bevy_jeod_dynamics/systems.rs` — panics if AerodynamicForceC present without RotationalStateC) |

## Section FD: FrameDerivatives

| Tag | Invariant | JEOD Source | Line(s) | Enforcement | Category | Our Status |
|-----|-----------|-------------|---------|-------------|----------|------------|
| FD.01 | `trans_accel = non_grav_accel + grav_accel` | `dyn_body/src/dyn_body_collect.cc` | 225 | structural | consistency | enforced (`systems.rs` force_collection_system writes FrameDerivativesC) |
| FD.02 | `rot_accel = I^-1 * (tau - omega x I*omega)` | `dyn_body/src/dyn_body_collect.cc` | 264 | structural | consistency | enforced (`systems.rs` force_collection_system writes FrameDerivativesC) |
