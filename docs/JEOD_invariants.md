# JEOD Invariants Catalog

Exhaustive catalog of invariants enforced by JEOD's C++ architecture. Each has a
`Section.Item` tag (e.g., `DB.03`) for cross-referencing from our Rust source
with `// JEOD_INV: DB.03` comments. To find the JEOD source for any invariant,
grep the JEOD tree for the distinctive identifier in the invariant description
(function name, field name, error message text, etc.).

> **Phase 11 audit (#113, 2026-04-26)** — All 68 rows with Category=`runtime`
> were audited end-to-end against post-refactor enforcement sites in `crates/`
> and `src/`. The catalog is current: invariants whose violations are now
> made unrepresentable by the type system (frame mismatches, time-scale
> mismatches, `NormalizedQuat` witness, typed entry points) are already
> classified `structural` or `n/a`; invariants that remain genuine runtime
> checks (NaN/finite guards, table-bounds checks, convergence checks,
> flag-gated config) are correctly classified `enforced` or `partial`. No
> reclassifications were needed at Phase 11; future invariants follow the
> same structural-vs-runtime decision tree on addition. See
> the [Type-System wiki page](https://github.com/simnaut/bevy_jeod/wiki/Type-System) for the type-system architecture and
> the [Strategy wiki page](https://github.com/simnaut/bevy_jeod/wiki/Strategy) §8 for the refactor history.

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
| DB.13 | State propagation delegates to root body | structural | consistency | enforced (attach-time: `bevy_jeod::staging_system` runs `combine_states_at_attach` on `AttachEvent` and writes the merged composite-body state to the parent — the integrated body now owns the whole subtree's translational + rotational state; `jeod_runner::Simulation::attach_subtree_aligned` is the runner equivalent. Per-step: `crates/jeod_dynamics/src/kinematic_propagation.rs`, `crates/jeod_sim/src/kinematic_propagation.rs`, `src/kinematic_propagation.rs` — a pre-order walk from each `MassChildOf` root via `propagate_state_via_storage` derives every kinematic child's `RotationalStateC` / `TranslationalStateC` from its parent's state composed with the link rotation + offset, mirroring JEOD `DynBody::propagate_state_from_structure`. Roots seed the walk verbatim — only their integrator-written state is the source of truth. Frame-tree side and child-frame derivation deferred to Phase 5 / #198.) |
| DB.14 | Integration frame switch delegates to root body | structural | consistency | enforced (`bevy_jeod::frame_switch_system` reparents the body's frame entity under the target source's frame entity via `commands.entity(...).insert(ChildOf(...))`, rewrites `TranslationalStateC` and the body frame entity's `FrameTransC` in the new integration frame's coordinates via `RelativeFrameState`, and flips `GravityControlsC` differentials so the new central source becomes non-differential; `bevy_jeod::staging_system` writes the post-attach composite-body inertial state to the parent's `TranslationalStateC` / `RotationalStateC`. Mirrors `jeod_runner::Simulation`'s `evaluate_and_apply_frame_switch` over the arena.) |
| DB.15 | `grav_interaction` always synchronized with integration frame | structural | consistency | deferred (Phase 5) |
| DB.16 | Child forces propagated to parent recursively | structural | ordering | enforced (`crates/jeod_dynamics/src/wrench.rs`, `crates/jeod_sim/src/wrench.rs`, `src/wrench.rs`; per-link parallel-axis shift via `shift_wrench_to_parent` driven by `aggregate_wrenches_via_storage` over `MassChildOf` chains, mirroring JEOD `dyn_body_collect.cc:138-202`) |
| DB.17 | Only root body computes total acceleration | structural | structural | enforced (`src/wrench.rs` aggregates child wrenches into the root, zeroes non-root `TotalForceC` / `FrameDerivativesC`, AND inserts `KinematicChildC` on every non-root chain member; `src/components.rs::KinematicChildC` and `src/systems.rs::integration_system`'s `Without<KinematicChildC>` filter together gate non-root nodes out of integration so they cannot drift under gravity. The sister kinematic-only propagation system (`src/kinematic_propagation.rs::propagate_state_from_root_system`, scheduled between `composite_mass_system` and `wrench_aggregation_system`) overwrites every kinematic child's `RotationalStateC` / `TranslationalStateC` each step from the parent's state composed with the link's `t_parent_child` rotation and offset, mirroring JEOD `DynBody::propagate_state_from_structure` — children stay synchronized with the root rather than freezing at attach-time values.) |
| DB.18 | `inverse_mass` used for F=ma (precomputed) | structural | consistency | enforced (`forces.rs:85,221`, `mass.rs:79`; `inverse_mass` precomputed, F=ma via multiplication matching JEOD `Vector3::scale`) |
| DB.19 | `inverse_inertia` used for Euler equation | structural | consistency | enforced (`validation.rs:101`, `rotational.rs:46`) |
| DB.20 | Small rot_accel truncated to zero (< 1e-20) | structural | runtime | enforced (`rotational.rs:66`, `zero_small` per-component) |
| DB.21 | Only unattached bodies integrate | flag-gate | runtime | partial (`bevy_jeod::DetachedSubtreeStateC` + `bevy_jeod::step_detached_system` advance detached subtrees ballistically while integrators run on the integrated body's composite state; `jeod_runner::Simulation::detached_subtrees` is the runner equivalent. Frame-attached body integration / IntegFrameIdC gating deferred to Phase 5 / #198) |
| DB.22 | DynBody not copyable | structural | structural | n/a (ECS components are Copy where needed) |
| DB.23 | `compute_inverse_inertia` enabled for DynBody | structural | structural | structural (`mass.rs:38`, always computed in `MassProperties::with_inertia`) |
| DB.24 | Default `integrated_frame` is composite_body | structural | structural | structural (`src/components.rs`, we integrate composite_body state) |
| DB.25 | DynBody name is reference to MassBody name | structural | structural | n/a (ECS entities, no name reference) |
| DB.26 | DynBody mass constructed with `this` as owner | structural | structural | n/a (ECS entities, no ownership reference) |
| DB.27 | State initialization order: attitude → rate → position → velocity | structural | ordering | deferred (Phase 5) |
| DB.28 | Forces collected in structural frame, rotated to inertial at root | structural | consistency | enforced (`systems.rs` force_collection_system: T_inertial_struct^T * struct_force) |
| DB.29 | Torques collected in structural frame, rotated to body at root | structural | consistency | enforced (`systems.rs` force_collection_system: T_struct_body * struct_torque) |
| DB.30 | DynBody.add_integrable_object warns on duplicate registration; remove_integrable_object warns when target is missing (`models/dynamics/dyn_body/src/dyn_body.cc:235`, `models/dynamics/dyn_body/src/dyn_body.cc:268`) | warn | runtime | n/a (er7_utils IntegrableObject framework not ported; in our ECS model, integrable state lives in components keyed by entity, so duplicate addition and missing-on-remove are unrepresentable by construction — see IN.32 for the integration-protocol contract we do enforce) |

## Section MA: Mass / MassBody / MassProperties

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| MA.01 | MassBody always present on DynBody (value member) | structural | structural | enforced (`validation.rs:81`, `systems.rs:139`) |
| MA.02 | mass > 0 for meaningful dynamics | conditional | consistency | enforced (`mass.rs:20,36`, `systems.rs:140`) |
| MA.03 | `inverse_mass` consistent with mass | conditional | consistency | enforced (`mass.rs:79`; `recompute_derived()` recomputes `inverse_mass = 1/mass`, called by `mass_update_system` each step) |
| MA.04 | `inverse_inertia` consistent with inertia | structural | consistency | enforced (`mass.rs:39`, `validation.rs:102`) |
| MA.05 | Inverse inertia computed only for root bodies with positive mass | conditional | consistency | structural (`mass.rs:37`, all bodies compute inverse — intentional divergence) |
| MA.06 | Bottom-up mass property update (children first) | structural | ordering | enforced (`mass_body.rs:240`) |
| MA.07 | Derived quantities recomputed after mutation | structural | consistency | enforced (`mass_body.rs:241`, `mass.rs:79`; `recompute_derived()` updates `inverse_mass`/`inverse_inertia`, `mass_update_system` calls it each step) |
| MA.08 | No cycle in mass tree | error | consistency | enforced (`mass_body.rs:164`) |
| MA.09 | MassPoint names unique within body | fatal | initialization | enforced (`mass_body.rs:add_mass_point`) |
| MA.10 | MassPoint names non-empty | fatal | initialization | enforced (`crates/jeod_dynamics/src/mass_body.rs` `add_mass_point`) |
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
| MA.21 | Named attachment points must exist on body for attach_aligned | error | structural | enforced (`mass_body.rs` attach_aligned) |

## Section DM: DynManager

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| DM.01 | At most one GravityManager registered | error | structural | n/a (no GravityManager singleton in ECS) |
| DM.02 | GravityManager registered before `initialized=true` | error | ordering | n/a |
| DM.03 | `initialized` flag set last in init sequence | structural | initialization | partial (validation system fires on `Added<GravityControlsC>` so bodies spawned mid-simulation are validated on the next tick) |
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
| GV.14 | Third-body vs direct gravity classification | structural | initialization | enforced (`GravityControl.differential` flag, set explicitly per control; JEOD derives from frame tree ancestry via `is_progeny_of`) |
| GV.15 | `integ_frame_index` synchronized with body's integration frame | structural | consistency | deferred (Phase 5) |
| GV.16 | Active controls subscribe to inertial frame | structural | consistency | n/a (no frame subscription in ECS) |
| GV.17 | Active nonspherical controls subscribe to planet-fixed frame | structural | consistency | enforced (PlanetFixedRotationC required for nonspherical) |
| GV.18 | Gravity source name matches planet name | structural | consistency | n/a (matched by Entity in ECS) |

## Section TM: Time

Source: `../jeod/models/environment/time/src/`, especially `time_manager.cc`, `time_manager_init.cc`, `time_standard.cc`, `time_ude.cc`, and the converter files. Error identifiers are declared in `include/time_messages.hh`.

Our port's time scales are hardcoded fields on `SimulationTime` (crates/jeod_time/src/simulation_time.rs), not a dynamic registry, so many of JEOD's registry-hygiene invariants become `structural` or `n/a` — the dependency graph is encoded in the function sequence of `SimulationTime::advance`, not in a runtime-built tree.

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| TM.01 | Time type names unique | fatal | initialization | n/a (single SimulationTime resource) |
| TM.02 | Converter type-pair names required | fatal | initialization | n/a |
| TM.03 | Time types updated in dependency order | structural | ordering | structural (`SimulationTime::advance` updates all scales in order) |
| TM.04 | Init tree completeness (all types reachable from initializer) | fatal | initialization | structural (all scales hardcoded in `SimulationTime`) |
| TM.05 | Update tree completeness (all types reachable from TimeDyn) | fatal | initialization | structural |
| TM.06 | No duplicate converters between same pair | fatal | initialization | structural |
| TM.07 | `simtime` initialized to -1.0 (forces first update) | structural | initialization | structural (`SimulationTime` constructed with explicit epoch) |
| TM.08 | Initializer time-type must exist in registry (`time_manager_init.cc:169`) | fatal | initialization | n/a (no dynamic registry; initializer is the TAI epoch passed to `SimulationTime::new`) |
| TM.09 | Exactly one initializer when multiple time types present (`time_manager_init.cc:182`) | fatal | initialization | n/a |
| TM.10 | Registry must not contain two instances of the same time-type class (`time_manager_init.cc` redundancy_error) | fatal | initialization | structural (each scale is a single named field on `SimulationTime`) |
| TM.11 | Converter registry: at most one converter registered per ordered type-pair (`time_manager_init.cc` redundancy_error) | fatal | initialization | structural (each direction is a direct function, e.g. `tai_to_tt`/`tt_to_tai`) |
| TM.12 | No cycles in init tree; A→B→A detected (`time_standard.cc`, `time_ude.cc` invalid_setup_error) | fatal | initialization | structural (init order is topologically hardcoded in `SimulationTime::new`/`recompute_derived`) |
| TM.13 | No cycles in update tree; A depends on B depends on A detected (`time__add_type_update.cc` invalid_setup_error) | fatal | initialization | structural (update order hardcoded in `SimulationTime::advance`) |
| TM.14 | Converter available for every parent→child edge in init and update trees (`time_standard.cc`, `time__add_type_update.cc` incomplete_setup_error) | fatal | initialization | structural (all conversions hardcoded) |
| TM.15 | Initializer cannot itself specify an `initialize_from` source (`time_standard.cc`) | fatal | initialization | n/a (initializer is a raw epoch, not a configurable time-type) |
| TM.16 | Converter double-parent: a time-type may not be re-parented after being placed in the tree (`time__add_type_update.cc` invalid_node) | fatal | initialization | structural (no dynamic tree editing) |
| TM.17 | `TimeConverter::master_ptr` / `sub_ptr` must be non-null (`time_converter.cc` invalid_setup_error) | fatal | initialization | n/a (no pointer-based registry) |
| TM.18 | Converter init: parent type must already be initialized before child converter runs (`time_converter.cc` initialization_error) | fatal | initialization | structural (`recompute_derived` traverses parent-first) |
| TM.19 | Converter `int_dir` / `conv_dir` must be in {-1, +1} for every converter; other values are invalid_setup (most converter files) | fatal | initialization | n/a (each converter is a named function; direction is not a runtime field) |
| TM.20 | `update_converter_direction` / `conv_dir` read at runtime must be in {-1, 0, +1}; other values emit memory_error (`time.cc`, `time_standard.cc`) | fatal | runtime | n/a |
| TM.21 | TAI↔UTC converter requires a leap-second lookup table unless `override_data_table` is set; when two tables are paired (UTC and UT1), both must use the same override setting (`time_converter_tai_utc.cc`, `time_manager_init.cc`) | fatal | initialization | partial (we always use the leap-second table via `LeapSecondTable::from_entries`; no override path, so both halves are `n/a`) |
| TM.22 | TAI↔UT1 converter requires a UT1 data lookup table; no override is permitted (`time_converter_tai_ut1.cc` invalid_data_error) | fatal | initialization | deferred (we do not currently load a UT1-UTC table; UT1 ≈ UTC until Phase 5 EOP tables) |
| TM.23 | Dyn→TAI and Dyn→TDB converters require `DynTime == 0` at init (`time_converter_dyn_tai.cc`, `time_converter_dyn_tdb.cc` initialization_error) | fatal | initialization | structural (our Dyn time starts at zero; `SimulationTime::new` establishes this) |
| TM.24 | `time_converter_dyn_ude`: no converter available for UDE→Dyn direction; only Dyn→UDE is legal (`time_converter_dyn_ude.cc` incomplete_setup_error) | fatal | initialization | n/a (no dynamic converter lookup) |
| TM.25 | UDE setup: must have `update_from_name` for initialization; no cycles in `update_from` chain; no cycles in `epoch_defined_in` chain; each edge needs a converter (`time_ude.cc` incomplete_setup_error / invalid_setup_error — several sites) | fatal | initialization | partial (our `UserDefinedEpoch::new` takes `epoch_in_parent` directly; the parent-scale relationship is structural, but multi-level UDE-on-UDE is not supported — JEOD forbids that too) |
| TM.26 | UDE cannot be overconstrained: setting initial_value AND epoch+time_since_epoch simultaneously is a redundancy_error (`time_ude.cc`) | fatal | initialization | n/a (our UDE constructor takes a single parent-epoch parameter) |
| TM.27 | UDE that is both the initializer AND updates from Dyn must not define an epoch (redundancy, `time_ude.cc`) | fatal | initialization | n/a |
| TM.28 | UDE as initializer: its epoch may not be defined in another UDE; must resolve to a standard or dynamic time (`time_ude.cc` invalid_setup_error) | fatal | initialization | n/a |
| TM.29 | UDE+Dyn initializer that pulls standard times into the sim: must be rejected because the initialization graph breaks connectivity to standard classes (`time_ude.cc`) | fatal | initialization | n/a |
| TM.30 | TimeDyn cannot be the initializer when any absolute (calendar-valued) time types are present (`time_dyn.cc` invalid_setup_error) | fatal | initialization | structural (TimeDyn in our arch is the monotonic simulation clock; `SimulationTime` always has an absolute TAI epoch alongside it) |
| TM.31 | Sim-start data must be non-zero for an absolute-time initializer: either calendar or decimal specified, matching `sim_start_format` (`time_standard.cc` incomplete/invalid_data_error — several sites) | fatal | initialization | partial (we require an explicit TAI epoch at construction; calendar-vs-decimal ambiguity does not exist) |
| TM.32 | If both calendar values and decimal values are defined, `sim_start_format` must disambiguate (`time_standard.cc` redundancy_error) | fatal | initialization | n/a |
| TM.33 | Calendar clock format must be a recognized enum value; other values fail (`time_standard.cc` invalid_data_error) | fatal | initialization | n/a |
| TM.34 | GPS time does not have a calendar; attempts to query calendar from GPS fail (`time_gps.cc` invalid_data_error, two sites) | fatal | runtime | n/a (we do not expose a calendar accessor on GPS time) |
| TM.35 | GMST does not have a calendar and has no valid Truncated Julian Time; attempts fail (`time_gmst.cc` invalid_data_error, two sites) | fatal | runtime | n/a (we do not expose calendar or TJT accessors on GMST) |
| TM.36 | Trunc Julian Time < 0 is allowed (pre-1968) but warned for the initializer (`time_standard.cc` invalid_data_error, warn severity) | warn | initialization | n/a (we accept any finite TAI TJT without warning) |
| TM.37 | JeodBaseTime default `add_to_initialization_tree` and `initialize_from_parent` methods must be overridden by subclasses; calling the base is fatal (`time.cc` invalid_setup_error) | fatal | structural | structural (our time scales are concrete types, not a polymorphic hierarchy — no default to fall through to) |
| TM.38 | Clock decomposition must carry correctly at the 60s/60min/24h boundaries; a `clock_resolution = 1e-6` tolerance rounds up near-boundary seconds (`time_ude.cc` clock_update, `time_utc.cc`, `time_ut1.cc`) | structural | consistency | enforced (`time_ude.rs:66-79`; mirrors JEOD's clock_update with 1e-6 tolerance) |
| TM.39 | Leap-second lookup table is non-empty and sorted by TJT at construction (`time_converter_tai_utc.cc` relies on a monotonic `when_vec`) | structural | initialization | enforced (`leap_second.rs:21-26`; two asserts at table construction) |
| TM.40 | Time advance inputs must be finite (JEOD assumes valid f64 via sim inputs; we assert) | runtime | runtime | enforced (`simulation_time.rs:125`; asserts finite `dt` at each `advance`) |

## Section RF: Reference Frames

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| RF.01 | `compute_relative_state` requires same tree | fatal | runtime | structural (`crates/jeod_frames/src/frame_tree.rs` — one tree per `FrameTree`; both `FrameId` arguments are indices into the same arena, so same-tree is guaranteed by type) |
| RF.02 | `compute_state_wrt_pred` requires valid predecessor | fatal | runtime | structural (`crates/jeod_frames/src/frame_tree.rs` — `parent()`/`get()` bounds-check `FrameId`; invalid ids panic) |
| RF.03 | Quaternion normalized after every composition | structural | consistency | structural (normalized in `incr_right`, `negate`, and integration) |
| RF.04 | T_parent_this recomputed after quaternion composition | structural | consistency | structural (T derived from normalized Q in both `incr_right` and `negate`) |
| RF.05 | `ang_vel_products` recomputed after angular velocity change | structural | consistency | n/a (we don't cache products) |
| RF.06 | Position/velocity in parent coordinates | structural | structural | structural (documented convention) |
| RF.07 | Q_parent_this is left-transformation quaternion | structural | structural | structural (documented, JeodQuat convention) |
| RF.08 | Frame names unique | error | initialization | n/a (Entity IDs) |
| RF.09 | Quaternion assumed normalized for `left_quat_to_transformation` | implicit | consistency | structural (`normalize_integ` called after every integration step) |
| RF.10 | Integration-frame state must be shifted to root-inertial via the integration-origin offset before passing to consumers that *mix it with root-inertial source positions* (gravity sources, Sun, Moon). Affected sites: gravity, relativistic, SRP (`sun_to_vehicle`, shadow), solar beta, earth lighting. Sites that operate within a single planet's inertial frame (atmosphere, drag velocity, LVLH, geodetic, orbital elements) are NOT shift sites — the body's integration frame IS that planet's inertial frame in realistic configs, and shifting would break them. | structural (shift sites) + convention (non-shift sites) | consistency | partial — structural for shift sites (the `Position<IntegrationFrame> + Position<RootInertial>` mismatch makes `body - sun_pos` refuse to compile, forcing `to_inertial(&o)`); convention for non-shift sites (consumer takes raw `DVec3`, frame correctness depends on runtime invariant `body.integ_source == consumer_planet_source`). The `PlanetInertial<P: Planet>` phantom is provided for mission-crate code that knows the planet at compile time; runner-internal consumers are runtime-checked. See issue #255. |
| RF.11 | Gravitational parameter μ must match the planet whose inertial-frame position/velocity it is paired with at typed orbital-mechanics consumers. A μ for the wrong central body silently produces an orbit / frame mismatch at the numeric level. | structural (typed surfaces) + convention (registry-side) | consistency | partial — structural end-to-end at the typed orbital-mechanics surfaces that pair `GravParam<P>` with `Position/Velocity<PlanetInertial<P>>` and return a `<P>`-tagged result: `OrbitalElements::from_cartesian_typed`, `jeod_sim::derived::compute_orbital_elements_typed`, `jeod_dynamics::body_init::init_from_orbital_elements_typed`, `jeod_sim::recipes::orbital_elements::*`, and `VehicleBuilder::from_orbital_elements`. The `mu_*()` constants in `jeod_sim::recipes::constants` return planet-pinned `GravParam<Earth>` / `<Sun>` / `<Moon>` / `<Mars>`, so a wrong-planet μ is rejected at the call site of these typed consumers. As of PR #306 the `<P = SelfPlanet>` default on both `GravParam<P>` and `OrbitalElements<P>` is removed: every call site must commit to a planet via turbofish, type ascription, or argument inference, and `<SelfPlanet>` is now a deliberate opt-in for the registry-side boundary code rather than a hidden fallback the compiler silently fills in. Other μ-using code paths are **not** structurally guarded by `GravParam<P>` yet and remain convention-only: `compute_relativistic_correction` (and `accumulate_relativistic_corrections`) take `mu: f64`, and the runner's gravity-source registry, `PlanetShape::mu_typed()`, `PlanetConfig::mu_typed()`, and `GravitySourceTyped::mu` all carry a `SelfPlanet`-tagged μ — the planet is determined at runtime by which entity references the source. (`solar_beta_angle_typed` does not consume μ at all, so it is outside this invariant's scope.) The `relabel::<P>` escape hatch is the explicit boundary between the dynamic-registry and static-typed worlds. Tightening the relativistic / gravity-evaluation surfaces to consume `GravParam<P>` (and pair with `PlanetInertial<P>`) and closing the registry side (planet-phantom-on-source) are the natural follow-ups under the type-safety umbrella tracked in #263. See issues #303, #306, #263. |

## Section EP: Ephemeris

Source: `../jeod/models/environment/ephemerides/` (`ephem_manager/`, `de4xx_ephem/`, `ephem_item/`, `ephem_interface/`, `propagated_planet/`). Error identifiers live in `ephemerides_messages.hh`.

Our port wraps ANISE (`crates/jeod_ephemeris`, 243 lines: `ephemeris.rs` + `bodies.rs`), which loads DE4xx `.bsp` kernels and serves body states. ANISE enforces file-integrity, time-range, and body-ID invariants internally. JEOD's dynamic `EphemeridesManager` registry with activation/deactivation state machines is not ported — most JEOD registry invariants map to `n/a` or `deferred (Phase 5 frame tree)`.

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| EP.01 | Planet name required and unique | error | initialization | n/a (Entity IDs) |
| EP.02 | Ephemeris models registered in dependency order | structural | ordering | deferred (Phase 5) |
| EP.03 | Frame tree rebuilt on active-status change | structural | runtime | deferred (Phase 5) |
| EP.04 | `integ_frame_index` lookup must succeed | fatal | runtime | deferred (Phase 5) |
| EP.05 | Planet and ephem-item registry must not contain duplicates by name or id (`ephem_manager.cc` duplicate_entry, three sites) | error | initialization | n/a (no dynamic registry; bodies are loaded from ANISE kernel by SPICE ID) |
| EP.06 | Items with the same name that are simultaneously enabled: keep one, disable the other and warn (`ephem_manager.cc` inconsistent_setup) | warn | initialization | n/a |
| EP.07 | Ephemeris models must be registered with the manager before use; premature registration warns (`ephem_manager.cc` single_ephem_mode) | warn | initialization | n/a |
| EP.08 | Frame used by an ephemeris query must be an ephemeris reference frame (`ephem_manager.cc`, two sites: internal_error and inconsistent_setup) | fatal | runtime | deferred (Phase 5 frame-tree ephemeris-frame classification) |
| EP.09 | Ephem-item type must match query (angle vs point) — `get_angle` on a point is an error (`ephem_manager.cc` invalid_item, two sites) | error | runtime | n/a (typed accessors on `Ephemeris`: `state_of`, `mat_*_to_*` — mismatch is a compile error) |
| EP.10 | `add_integration_frame` target must be a registered integration frame (`ephem_manager.cc` invalid_item) | fatal | initialization | deferred (Phase 5) |
| EP.11 | DE4xx file: `dlopen` must succeed and export `metaData`, `itemData`, `segmentData`, `segment_coeffs_0` symbols (`de4xx_file.cc`, `de4xx_file_init.cc` file_error, five sites) | fatal | initialization | n/a (ANISE loads SPK kernels directly; errors surface as `EphemerisError::LoadError` on `Ephemeris::from_bsp`) |
| EP.12 | DE4xx file: entry count must not exceed `De4xx_File_MaxEntries` (`de4xx_file.cc` file_error) | fatal | initialization | n/a (ANISE enforces its own SPK limits) |
| EP.13 | DE4xx file: header `DE#` value must parse to a recognised ephemeris release (`de4xx_file_init.cc` garbage_file) | fatal | initialization | n/a (ANISE validates SPK segment metadata) |
| EP.14 | Query time must lie within the loaded file's epoch range (`de4xx_file_init.cc` time_not_in_range) | fatal | runtime | enforced (ANISE's `SPK::translate_from_to` returns a range error → mapped to `EphemerisError::QueryError` at `ephemeris.rs:69`) |
| EP.15 | DE4xx: re-initialization of an already-initialized ephemeris model is fatal (`de4xx_file_init.cc` internal_error) | fatal | initialization | n/a (our `Ephemeris::from_bsp` returns a new value; no mutable "initialized" state) |
| EP.16 | DE4xx: must not query a file that is not open (`de4xx_file_update.cc` internal_error) | fatal | runtime | n/a (ANISE keeps the kernel open for the lifetime of `SPK`) |
| EP.17 | DE4xx: body ephemeris must be available for the requested body index (`de4xx_file_update.cc` item_not_in_file) | fatal | runtime | enforced (ANISE raises a SPK segment-not-found error; surfaces as `EphemerisError::QueryError`) |
| EP.18 | DE4xx activation: a previously deactivated model cannot be re-activated (`de4xx_ephem.cc` internal_error; also `propagated_planet.cc`, `simple_ephemerides.cc`) | error | runtime | n/a (no activation state machine; bodies are always "active" once loaded) |
| EP.19 | DE4xx: time type must be TT or TDB; other scales warn and are ignored (`de4xx_ephem.cc` inconsistent_setup) | warn | initialization | structural (our API takes a `TDB` epoch directly; no runtime time-type selection) |
| EP.20 | DE4xx: Terrestrial Time and Dynamic Time objects must both be resolvable at init (`de4xx_ephem.cc` inconsistent_setup) | fatal | initialization | structural (our `SimulationTime` always supplies TT and TDB) |
| EP.21 | DE4xx: Earth and Moon must be supplied by the same ephemeris model (`de4xx_ephem.cc` inconsistent_setup) | fatal | initialization | structural (we use a single ANISE kernel for the whole DE4xx body set) |
| EP.22 | PropagatedPlanet: DynamicTime must be resolvable; parent frame and planet must be registered; planet must target planet frames; cannot switch to ephemeris mode after propagation begins (`propagated_planet.cc`, five sites) | fatal | initialization | n/a (PropagatedPlanet pattern not ported; planets are either ephemeris-sourced or ECS components) |
| EP.23 | EphemItem: immutable fields (name, target-frame-type) cannot change once set (`ephem_item.cc` invalid_name / invalid_item, four sites) | fatal | initialization | structural (our ephem items are value types constructed once) |
| EP.24 | SinglePlanetEphemeris / EmptySpaceEphemeris: exactly one planet registered in single-planet mode (`simple_ephemerides.cc` inconsistent_setup) | fatal | initialization | n/a (no single-planet-mode infrastructure) |
| EP.25 | EphemerisError surface area: LoadError for kernel load failures; QueryError for out-of-range, missing body, or rotation-lookup failures (`EphemerisError` enum) | runtime | runtime | enforced (`crates/jeod_ephemeris/src/ephemeris.rs:186-191`; all ANISE errors mapped through these two variants) |

## Section AT: Atmosphere

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| AT.01 | `active` flag gates computation | flag-gate | runtime | structural (no atmosphere → no AtmosphericStateC) |
| AT.02 | Atmosphere model pointer non-null for update | structural | runtime | structural (AtmosphereModelR resource checked) |
| AT.03 | Planet-fixed position required for geodetic altitude | structural | runtime | enforced (`src/systems.rs` — panics if planet_entity set but PlanetFixedRotationC missing) |
| AT.04 | Wind velocity computed as omega × position (co-rotation) | structural | runtime | enforced (`crates/jeod_atmosphere/src/lib.rs` compute_corotation_wind, `src/systems.rs` atmosphere_update_system) |

## Section IN: Interactions

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| IN.01 | GravityTorque.subject_body required (non-null) | fatal | runtime | structural (system queries require all components) |
| IN.02 | GravityTorque.active gates computation | flag-gate | runtime | structural (no GravityTorqueC → no torque) |
| IN.03 | AerodynamicDrag.active gates computation | flag-gate | runtime | structural (no DragConfigC → no drag) |
| IN.04 | `aero_surface_ptr` required when `use_default_behavior=false` | fatal | runtime | n/a (only ballistic model implemented) |
| IN.05 | Ballistic coefficient non-zero for DRAG_OPT_BC | fatal | runtime | n/a (only DRAG_OPT_CD implemented) |
| IN.06 | RadiationPressure.active gates computation | flag-gate | runtime | structural (no FlatPlateConfigC → no SRP) |
| IN.07 | RadiationThirdBody name required | fatal | initialization | n/a (shadow bodies by Entity) |
| IN.08 | RadiationThirdBody belongs to one model only | fatal | structural | n/a (function-based, no ownership) |
| IN.09 | RadiationSource planet must be found by DynManager | fatal | initialization | enforced (`src/systems.rs` — panics on multiple SunMarker; zero SunMarker = SRP not configured, early return like JEOD `active=false`) |
| IN.10 | RadiationSource.luminosity ≥ 1e-6 for flux computation | flag-gate | runtime | n/a (luminosity is a compile-time constant; `distance < 1.0` guard prevents division by near-zero) |
| IN.11 | RadiationThirdBody.radius > 0 | fatal | initialization | enforced (`shadow.rs` handles degenerate cases) |
| IN.12 | RadiationSource.radius > 0 | fatal | initialization | enforced (`shadow.rs` handles degenerate cases) |
| IN.13 | Shadow model: vehicle distance > 0 | error | runtime | enforced (`shadow.rs` returns 0.0 if r_mag2 <= 0) |
| IN.14 | `d_source_to_third` > 0 | error | runtime | enforced (`shadow.rs` returns 1.0 if d <= 0) |
| IN.15 | Aero drag requires body orientation (T_inertial_struct) | structural (mandatory fn parameter) | runtime | enforced (`src/systems.rs` — panics if AerodynamicForceC present without RotationalStateC) |
| IN.16 | RadiationThirdBody requires inertial frame pointer | fatal | initialization | n/a (stateless function takes positions directly) |
| IN.17 | RadiationSurface requires at least one facet (`num_facets > 0`) | fatal | initialization | deferred (caller passes plate slice; empty slice produces zero force) |
| IN.18 | `power_emit` must be non-negative (thermal radiation) | fatal | runtime | structural (`power_emit = rad_constant * t_pow4`; both factors non-negative by construction) |
| IN.19 | RadiationDefaultSurface reflectance spec: either `rad_coeff` alone (range 1.0–1.44444, i.e. 13/9), or `albedo` and `diffuse` both in [0, 1] — never both; never neither (`radiation_default_surface.cc` four invalid_setup_error sites) | fatal | initialization | deferred (our components accept albedo and diffuse as separate fields; no range-validation pass yet — user-responsibility gap to close in a future audit pass) |
| IN.20 | RadiationDefaultSurface: exactly one of `cx_area` / `surface_area` must be specified, and the chosen value must be non-zero (`radiation_default_surface.cc` two invalid_setup_error sites) | fatal | initialization | n/a (our API takes a single area field; exactly-one problem cannot occur) |
| IN.21 | Flat-plate radiation facet reflectance: `albedo`, `albedo_vis`, `albedo_IR`, and `diffuse` must each lie in [0, 1] (`radiation_facet.cc` invalid_setup_error) | fatal | initialization | deferred (same as IN.19 — flat-plate facet config accepts these fields with no range check) |
| IN.22 | Flat-plate facet emitted power must be non-negative; negative emission is non-physical (`flat_plate_radiation_facet.cc` unknown_numerical_error) | fatal | runtime | structural (we compute emission as `eps * sigma * T^4 * A`, all factors non-negative) |
| IN.23 | RadiationThirdBody name must be non-empty and unique within the radiation model (`radiation_third_body.cc` invalid_setup_error; also `radiation_pressure.cc` duplicate-name check) | fatal | initialization | n/a (third bodies identified by entity id, not by name) |
| IN.24 | RadiationThirdBody primary source, inertial-frame pointer, and planet radius must all be set before initialization completes (`radiation_third_body.cc` three invalid_setup_error sites) | fatal | initialization | structural (our API requires the source body, inertial state, and radius at the call site) |
| IN.25 | RadiationThirdBody: name must resolve to a registered planetary or dynamic body (`radiation_third_body.cc` invalid_setup_error) | fatal | initialization | n/a (Entity-based references; registration is entity existence) |
| IN.26 | RadiationThirdBody runtime: vehicle-to-third-body distance squared (`r_mag2`) must be positive; degenerate case puts vehicle in total shadow and exits (`radiation_third_body.cc` invalid_setup_error error-severity) | error | runtime | partial (covered by IN.13 — our shadow.rs returns 0.0 on degenerate distance rather than erroring) |
| IN.27 | RadiationThirdBody: `process_third_body` must not run before initialization (`radiation_third_body.cc` invalid_setup_error) | fatal | runtime | n/a (no initialize/process state machine; shadow compute is a pure function of current state) |
| IN.28 | `set_*_third_body_active` and `set_*_third_body_inactive` warn on no-op (already in target state) (`radiation_pressure.cc` two warn sites) | warn | runtime | n/a (no activate/deactivate state in our shadow body list) |
| IN.29 | RadiationThirdBody activation lookup: name must resolve to an existing third body in the model; otherwise error (`radiation_pressure.cc` invalid_function_call, three sites) | error | runtime | n/a |
| IN.30 | Contact pair bodies must be distinct (`unique_pair` in `contact.cc`) | fatal | initialization | enforced (`register_contact_pair` asserts `body_a != body_b`) |
| IN.31 | Contact forces evaluated at every derivative evaluation (JEOD `check_contact` is a derivative-class job in `contact.sm`) | structural (Trick job scheduling) | runtime | enforced (`Simulation::step_internal` uses `integrate_bodies_contact_coupled` when contact pairs are registered, evaluating every pair at each RK4 stage) |
| IN.32 | IntegrableObject per-step protocol: `snapshot` once at step start, `advance_intermediate` before stages 2–4, `finalize_rk4` after stage 4 — mirrors JEOD's `er7_utils::IntegrableObject` driven by `DynamicsIntegrationGroup` (`trick_source/er7_utils/integration/core/include/integrable_object.hh`; `models/dynamics/dyn_manager/src/dynamics_integration_group.cc`) | structural (trait contract) | runtime | enforced (`crates/jeod_sim/src/integrable.rs`; `FlatPlateState` impl in `interactions.rs`; invoked by `integrate_body_coupled`/`integrate_coupled_sixdof`) |
| IN.33 | ThermalFacetRider initialization: emissivity must be > 0 (else fatal); surface_area must be > 0 (else fatal); both warn-and-clamp at 1e-12 if positive but smaller (`models/interactions/thermal_rider/src/thermal_facet_rider.cc:109-146`) | fatal/warn | initialization | enforced (`crates/jeod_interactions/src/radiation_pressure.rs` `compute_flat_plate_srp` and `compute_flat_plate_srp_thermal_conduction` assert `plate.area > 0.0` and `thermal.emissivity > 0.0` per plate inside the main computation loop, per fail-loudly policy; we fail fast rather than warn-and-clamp the way JEOD's `1e-12` floor does) |
| IN.34 | ThermalFacetRider runtime: `d_temperature` must lie in `(-T_current, 1e6)` per integration step; out-of-range value warns and deactivates the facet (`thermal_rider/src/thermal_facet_rider.cc:226-243`) | warn | runtime | n/a (our `integrate_plate_temperature_euler` / `_rk4` use equilibrium-clamping — when `temp_dot * (T_eq^4 - T_new^4) < 0`, snap to `T_eq` rather than allow the integrator to run away; this is structurally equivalent to JEOD's divergence guard but more conservative because it always produces a physical temperature instead of deactivating the facet) |
| IN.35 | GroundFacet `active` flag must be true. JEOD silently skips inactive `GroundInteraction`s inside `check_contact_ground` (`verif/SIM_ground_contact/models/contact_ground/src/contact_ground.cc:88`); we follow the project's *Fail Loudly* policy (CLAUDE.md) and reject inactive facets at registration time rather than silently dropping their contribution. | fatal | runtime | enforced (`crates/jeod_interactions/src/contact.rs::compute_ground_contact_geometry` asserts `ground_facet.active`; `Simulation::register_ground_contact_pair` also asserts at registration time — diverges from JEOD's silent-skip behaviour by panicking instead) |
| IN.36 | GroundFacet `alt_offset` must be finite — a NaN or ±infinity propagates through the body-frame ground-point comparison and produces undefined contact-detection behaviour | fatal | initialization | enforced (`GroundFacet::new` and `Simulation::register_ground_contact_pair` assert `alt_offset.is_finite()`) |
| IN.37 | `SphericalTerrain.radius` must be strictly positive — a zero or negative radius collapses the ground point to the planet center and yields a NaN normal | fatal | initialization | enforced (`SphericalTerrain::new` asserts `radius.is_finite() && radius > 0.0`) |

## Section DS: Derived States

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| DS.01 | Derived state configuration immutable after initialization (objects created at setup, not toggled at runtime) | structural | structural | structural (`bodies` field is private, no `body_mut()`; `body()` returns `&SimBody`) |

## Section FD: FrameDerivatives

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| FD.01 | `trans_accel = non_grav_accel + grav_accel` | structural | consistency | enforced (`systems.rs` force_collection_system writes FrameDerivativesC) |
| FD.02 | `rot_accel = I^-1 * (tau - omega x I*omega)` | structural | consistency | enforced (`systems.rs` force_collection_system writes FrameDerivativesC) |

## Section IG: Integration

Source: `../jeod/models/utils/integration/` (core + `gauss_jackson/` + `lsode/`). Error identifiers live in `include/integration_messages.hh` and `er7_utils::IntegrationMessages::*`. Our port implements RK4, RKF45 (with adaptive step), ABM4, and a full Gauss-Jackson in `crates/jeod_dynamics/src/` (`integration.rs`, `rkf45.rs`, `abm4.rs`, `gauss_jackson/`). LSODE's stiff-ODE path is not ported; the non-stiff-Adams mode maps to our ABM4 for cross-validation (see `tier3_sim_lsode.rs`).

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| IG.01 | Integration technique must be specified for rotational state (`generalized_second_order_ode_technique.cc:54,82` invalid_request) | fatal | initialization | n/a (integrators are named enum variants, not runtime-selected technique pointers) |
| IG.02 | Integration constructor chosen for a body must support rotational state; constructor-without-rotational-support is fatal (`generalized_second_order_ode_technique.cc:100`) | fatal | initialization | structural (each of our integrator enum variants implements the rotational path) |
| IG.03 | Fallback to Lie-group / Cartesian integration when a constructor does not provide generalized derivative / step integrators (`generalized_second_order_ode_technique.cc:115,132` inform severity) | info | initialization | n/a (no pluggable constructor pattern) |
| IG.04 | Gauss-Jackson `initial_order` must be an even integer in [2, 14] (`gauss_jackson_config.cc:validate_config`) | error | initialization | enforced (`gauss_jackson/config.rs:85-92`) |
| IG.05 | Gauss-Jackson `final_order` must be an even integer in [`initial_order`, 14] (`gauss_jackson_config.cc`) | error | initialization | enforced (`gauss_jackson/config.rs:93-103`) |
| IG.06 | Gauss-Jackson `ndoubling_steps` ≤ 20 (`gauss_jackson_config.cc`) | error | initialization | enforced (`gauss_jackson/config.rs:104-109`) |
| IG.07 | Gauss-Jackson `relative_tolerance` must be finite and in [0, 1] (`gauss_jackson_config.cc`) | error | initialization | enforced (`gauss_jackson/config.rs:110-115`) |
| IG.08 | Gauss-Jackson `absolute_tolerance` must be finite and ≥ 0 (`gauss_jackson_config.cc` — JEOD compares `relative_tolerance` in the message but the variable checked is `absolute_tolerance`) | error | initialization | enforced (`gauss_jackson/config.rs:116-121`) |
| IG.09 | Gauss-Jackson history_length ≤ order throughout priming (`gauss_jackson_state_machine.cc` internal invariant) | structural | consistency | enforced (`gauss_jackson/mod.rs:466` assert) |
| IG.10 | Gauss-Jackson history_length must be odd when reducing order for bootstrap (`gauss_jackson_integration_controls.cc` internal) | structural | consistency | enforced (`gauss_jackson/mod.rs:675`) |
| IG.11 | Gauss-Jackson integrator_constructor::create_integration_controls only accepts GaussJacksonIntegrationControls; failure is fatal (`gauss_jackson_integrator_constructor.cc`) | fatal | initialization | n/a (controls are a concrete struct, not a polymorphic downcast) |
| IG.12 | Gauss-Jackson: state machine must not remain stuck in Reset state after a step (`gauss_jackson_state_machine.cc` equivalent) | fatal | runtime | enforced (`gauss_jackson/mod.rs:319` panic) |
| IG.13 | LSODE: `num_odes` > 0 (`lsode_control_data_interface.cc`) | fatal | initialization | n/a (LSODE not ported; see note above) |
| IG.14 | LSODE: `error_control_indicator` must be a legal enum value (`lsode_control_data_interface.cc`) | fatal | initialization | n/a |
| IG.15 | LSODE: `integration_method` ∈ {1, 2} (`lsode_control_data_interface.cc`) | fatal | initialization | n/a |
| IG.16 | LSODE: `corrector_method` ∈ [1, 5]; method=1 (user-supplied Jacobian) and methods 4–5 (banded Jacobian) explicitly unsupported (`lsode_control_data_interface.cc`, `lsode_first_order_ode_integrator__support.cc`) | fatal | initialization | n/a |
| IG.17 | LSODE: `max_order` ≥ 0 (`lsode_control_data_interface.cc`) | fatal | initialization | n/a |
| IG.18 | LSODE: `max_num_steps` > 0, `max_num_small_step_warnings` ≥ 0 (`lsode_control_data_interface.cc`) | fatal | initialization | n/a |
| IG.19 | LSODE: `max_step_size` ≥ 0, `min_step_size` ≥ 0 (`lsode_control_data_interface.cc`) | fatal | initialization | n/a |
| IG.20 | LSODE: relative and absolute tolerance vectors must be populated and all values ≥ 0 (`lsode_control_data_interface.cc`, both rel and abs variants) | fatal | initialization | n/a |
| IG.21 | LSODE: `initial_step_size` sign must match `cycle_target_time` sign (`lsode_first_order_ode_integrator__manager.cc`) | fatal | initialization | n/a |
| IG.22 | LSODE runtime: step size must not become so small that `t + dt == t` at machine precision; accumulates up to `max_num_small_step_warnings` before suppression (`lsode_first_order_ode_integrator__manager.cc`) | warn | runtime | n/a |
| IG.23 | LSODE runtime: `error_weight` must remain > 0 during stepping (`lsode_first_order_ode_integrator__manager.cc`, two sites) | fatal | runtime | n/a |
| IG.24 | LSODE runtime: `cycle_target_time` must not be too close to `current_time` to start integration (`lsode_first_order_ode_integrator__manager.cc`) | fatal | runtime | n/a |
| IG.25 | LSODE runtime: requested accuracy must be achievable at machine precision (`lsode_first_order_ode_integrator__manager.cc`) | fatal | runtime | n/a |
| IG.26 | LSODE runtime: total steps in one integration cycle must not exceed `max_num_steps` (`lsode_first_order_ode_integrator__manager.cc`) | fatal | runtime | n/a |
| IG.27 | LSODE runtime: error test and corrector convergence must not fail repeatedly (`lsode_first_order_ode_integrator__manager.cc`) | fatal | runtime | n/a |
| IG.28 | LSODE runtime: `cycle_target_time` must lie in the interval `[current_time - prev_good_step, current_time]` (`lsode_first_order_ode_integrator__support.cc`) | fatal | runtime | n/a |
| IG.29 | LSODE runtime: Jacobian computation must not infinite-loop on a repeated singular matrix (`lsode_first_order_ode_integrator__manager.cc`) | fatal | runtime | n/a |
| IG.30 | LSODE: copy constructors deleted on LsodeIntegrationControls, LsodeSecondOrderODEIntegrator, LsodeGeneralizedDerivSecondOrderODEIntegrator, LsodeFirstOrderODEIntegrator (each file has a fatal MessageHandler::fail in the copy ctor) | structural | structural | n/a |
| IG.31 | IntegrationTime: object being added to time-change subscribers must not already be in the list (`jeod_integration_time.cc:73`) | error | initialization | n/a (no subscriber registry — time changes are state, not events) |
| IG.32 | IntegrationTime: object being removed from time-change subscribers must be found in the list (`jeod_integration_time.cc:94`) | warn | runtime | n/a |
| IG.33 | IntegrationGroup::remove_integrable_object: entry must exist (`jeod_integration_group.cc`) | error | runtime | n/a (no runtime integration-group membership; bodies are Bevy entities) |
| IG.34 | Integrator step `dt` must be finite and strictly positive; zero-step silently rotates multi-step history (ABM4) (no JEOD equivalent — JEOD relies on Trick scheduler) | runtime | runtime | enforced (`abm4.rs:172`; `integration.rs` asserts dt non-zero in stepping paths) |
| IG.35 | Integrator state: position and velocity must remain finite across stages (structural JEOD assumption — never asserted in production; verified by test harness) | structural | consistency | n/a (JEOD does not assert; neither do we — unit tests exercise the shape) |
| IG.36 | Gauss-Jackson `BootstrapEdit` accepts a non-converged correction once `correction_iterations >= max_correction_iterations` and proceeds (`gauss_jackson_integration_controls.cc` — JEOD logs via `MessageHandler::error`, non-fatal) | warn | runtime | enforced (`gauss_jackson/state_machine.rs` increments `bootstrap_unconverged_iterations`; `integration.rs` `log::warn!`s on first occurrence) |
| IG.37 | Multi-step integrator history (Gauss-Jackson, ABM4) must be reset on a body's mass / attachment topology change; JEOD calls `DynamicsIntegrationGroup::reset_integrators()` from `dyn_body_attach.cc` (lines 860, 871) and `dyn_body_detach.cc:271-273` whenever the topology mutates, since stale predictor / corrector history reflects pre-attach dynamics | structural | runtime | enforced (`abm4.rs` and `gauss_jackson/mod.rs` carry a `topology_dirty` flag asserted on every step; `jeod_sim::reset_integrators` is called from `bevy_jeod::staging_system`, `jeod_runner::Simulation::{attach,detach,attach_subtree_aligned,detach_subtree}`, and `jeod_runner::Simulation::sync_body_mass_from_tree` — the last is the documented sync site for the lower-level direct-`mass_tree` mutation path) |

## Section QT: Quaternion

Source: `../jeod/models/utils/quaternion/src/`, especially `quat_norm.cc`. Our port: `crates/jeod_math/src/quaternion.rs`.

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| QT.01 | `Quaternion::normalize()` uses the Padé fast path `fact = 2/(1+q²)` when `|q²−1| < NORM_LIMIT`, otherwise `fact = 1/√q²` (`quat_norm.cc:60` approx; same heuristic in our port) | structural | consistency | enforced (`crates/jeod_math/src/quaternion.rs:84-97`) |
| QT.02 | Normalized scalar part is non-negative (canonical hemisphere): a quaternion with scalar < 0 is negated after normalization (`quat_norm.cc:70-72`) | structural | consistency | enforced (`crates/jeod_math/src/quaternion.rs:99-104`) |
| QT.03 | `normalize()` requires a non-zero quaternion (`qmagsq > 0`) — zero quaternion is not normalizable (implicit in JEOD; we assert) | structural | consistency | enforced (`crates/jeod_math/src/quaternion.rs:86`) |
| QT.04 | `normalize_integ` (the integration-safe variant) requires a non-zero quaternion — zero magnitude is unrecoverable. Same invariant as QT.03 but applies inside the typed `BodyAttitude` advance path. | structural | consistency | enforced (`crates/jeod_quantities/src/body_attitude.rs` — `normalize_integ` assert) |
| QT.05 | Body-rate attitude advance uses LEFT-multiply (`q̇ = -½ ω ⊗ q`, integral `q(t+dt) = exp(-½ω·dt) ⊗ q(t)`) per `quat_inline.hh:466`. Operand order is owned by the wrapper, not the caller — `BodyAttitude` exposes no public `multiply` so the wrong order is unrepresentable (issue #252, supersedes the structural mitigation in PR #251 / issue #248). | structural | numeric | enforced (`crates/jeod_quantities/src/body_attitude.rs` — `BodyAttitude::advance_under_body_rate`) |
| QT.06 | Body-to-body attitude composition uses LEFT-multiply of the relation onto the existing attitude (`q_W = T_VW ⊗ q_V`). Owned by `BodyAttitude::compose_with` over a typed `FrameTransform<BodyFrame<V>, BodyFrame<W>>`. | structural | consistency | enforced (`crates/jeod_quantities/src/body_attitude.rs` — `BodyAttitude::compose_with`) |

## Section OE: Orbital Elements

Source: `../jeod/models/utils/orbital_elements/src/orbital_elements.cc`. Our port: `crates/jeod_math/src/orbital_elements.rs`.

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| OE.01 | `mu` (gravitational parameter) must be positive; non-positive or non-finite `mu` fails `from_cartesian` (`orbital_elements.cc:427-436`) | fatal | initialization | enforced (`crates/jeod_math/src/orbital_elements.rs` returns `OrbitalError::InvalidMu`; verified at test line 977-978) |
| OE.02 | Semi-parameter `p = a(1-e²)` must be positive for `to_cartesian`; non-positive `p` fails (`orbital_elements.cc:403-410`) | fatal | runtime | enforced (`orbital_elements.rs:290-292` returns `DegenerateOrbit`) |
| OE.03 | `sin²ν + cos²ν ≈ 1` to tolerance 1e-6 in `to_cartesian` (`orbital_elements.cc:414-424`) | fatal | runtime | enforced (`orbital_elements.rs:301-304`) |
| OE.04 | Eccentricity regime classification: `e < TOLERANCE` ⇒ circular branch; `|e−1| < parabolic-eps` ⇒ parabolic; otherwise elliptic or hyperbolic. Tolerances enforce a branch selection rather than fail (`orbital_elements.cc:560-597`) | structural | consistency | enforced (`orbital_elements.rs:141-142` and surrounding branch selection) |
| OE.05 | Inclination regime classification: `i < TOLERANCE` or `i > π − TOLERANCE` ⇒ equatorial branch (`orbital_elements.cc:218`) | structural | consistency | enforced (`orbital_elements.rs:139-141`) |
| OE.06 | Kepler-equation convergence (mean → eccentric anomaly) is required; non-convergence must fail rather than silently return (`orbital_elements.cc:650-660`) | fatal | runtime | enforced (Newton-Raphson in `orbital_elements.rs`; returns `OrbitalError::KeplerConvergence` after 1000 iterations at 1e-14 tolerance) |
| OE.07 | Initial position and velocity must both be non-zero for `from_cartesian` (either being zero makes `h = r × v` degenerate) | fatal | initialization | enforced (guard rejects either magnitude below 1e-30; verified at `orbital_elements.rs:83-87` and test `invalid_mu`) |

## Section PF: Planet-Fixed

Source: `../jeod/models/utils/planet_fixed/src/planet_fixed_posn.cc`. Our port: `crates/jeod_math/src/geodetic.rs`.

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| PF.01 | Position in PCPF must be far from the planet center: `r_local > r_eq · Small_radius_limit` (JEOD uses 1e-10; `planet_fixed_posn.cc:100-121`) | fatal | runtime | enforced (`geodetic.rs:40-43` and `:92-95`) |
| PF.02 | Input position must contain no NaN/Inf prior to geodetic conversion (`planet_fixed_posn.cc:155-162` checks this implicitly) | fatal | runtime | enforced (`geodetic.rs:82-85`) |
| PF.03 | Polar singularity: at `x_ellipse == 0` (directly over the pole), longitude is not computed — JEOD leaves it unchanged; our port returns 0.0 as convention (`planet_fixed_posn.cc:177-182`) | structural | consistency | enforced (`geodetic.rs:99-105`) |
| PF.04 | Borkowski geodetic iteration must converge to within 1e-12 radians; JEOD silently uses the last iterate on non-convergence, our port asserts (`planet_fixed_posn.cc:263`) | structural | consistency | enforced (`geodetic.rs:156-160`; intentional divergence from JEOD — we assert because iteration is provably convergent for real ellipsoids) |
| PF.05 | Borkowski denominator `d = 2·(cos(y0−w) − c·cos(2·y0))` must stay non-zero; `d==0` would correspond to zero flattening (not a real ellipsoid) — JEOD divides unconditionally, our port asserts (`geodetic.rs:142-145`) | structural | consistency | enforced (`geodetic.rs:142-145`; intentional divergence from JEOD) |

## Section LV: LVLH Frame

Source: `../jeod/models/utils/lvlh_frame/src/lvlh_frame.cc`. Our port: `crates/jeod_math/src/lvlh.rs`.

Our port exposes LVLH as a pure-function frame-conversion utility; JEOD's LVLH is a registered frame in the tree with subject and planet-name lookups. Most setup invariants are therefore `n/a` — the caller supplies the subject and planet references directly.

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| LV.01 | Subject frame must be specified (by name or pointer); neither-set is fatal (`lvlh_frame.cc:85-96`) | fatal | initialization | n/a (direct-argument API) |
| LV.02 | Subject frame must be resolvable in the dynamics manager (`lvlh_frame.cc:100-113`) | fatal | initialization | n/a |
| LV.03 | Planet reference must be specified (by PCI frame or name); neither-set is fatal (`lvlh_frame.cc:123-135`) | fatal | initialization | n/a |
| LV.04 | Named planet must be resolvable in the dynamics manager (`lvlh_frame.cc:139-151`) | fatal | initialization | n/a |
| LV.05 | Zero relative radius at the subject (rmagsq ≈ 0) triggers a singularity: `h_mag/rmagsq` division becomes undefined (`lvlh_frame.cc:266`). JEOD does not guard; failure surfaces downstream | structural | runtime | structural (our port returns NaN if called with zero relative radius — same behavior as JEOD; caller must avoid) |

## Section SM: Surface Model

Source: `../jeod/models/utils/surface_model/src/`, particularly `facet.cc`. Our port: components of `crates/jeod_interactions/` (aerodynamic facet lists, SRP facet surface areas).

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| SM.01 | Facet's `mass_body_name` must be non-empty and resolvable to a registered mass body (`facet.cc:59-74`) | fatal | initialization | n/a (our facets reference the owning Entity by ID; no name-based resolution) |
| SM.02 | Facet articulation calls are gated on `initialize_mass_connection` having run first (`facet.cc:88-101`) | fatal | initialization | structural (our port initializes facet→body references at bundle-spawn time; articulation at runtime is safe) |

## Section BA: Body Action

Source: `../jeod/models/dynamics/body_action/src/`. Our port: `crates/jeod_dynamics/src/body_init.rs`.

Our port expresses body initialization as ECS components/events processed during a startup system; JEOD uses a polymorphic `BodyAction` base class registered with `DynManager`. Registry-hygiene invariants reduce to structural guarantees in the ECS model; physics preconditions transfer directly.

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| BA.01 | Subject body must be a DynBody, not a bare MassBody (`dyn_body_init.cc:70-81`) | fatal | initialization | structural (our body-init components only target entities with DynBody-equivalent components) |
| BA.02 | Subject body must be registered with the dynamics manager before an action fires (`dyn_body_init.cc:85-94`) | fatal | initialization | structural (entities exist in the ECS world; registration is existence) |
| BA.03 | Body-attachment actions require a non-null parent reference (`body_attach.cc:58-71`) | fatal | initialization | enforced (`MassTree::attach` takes a non-null `MassBodyId` by type; bad ids and self-attachment panic via asserts in `crates/jeod_dynamics/src/mass_body.rs`) |
| BA.04 | Body cannot attach to itself, and attachments must not form a cycle in the mass tree (`mass_attach.cc:166-177`) | error | initialization | enforced (covered by MA.08 — cycle detection in `mass_body.rs`) |
| BA.05 | Orbital initializer requires a valid planet with a registered gravity source (`dyn_body_init_orbit.cc:98-111`) | fatal | initialization | enforced (`body_init.rs` `InitialOrbit` requires `mu` and a reference body; startup system panics if missing) |
| BA.06 | Orbit initialization frame must be an ephemeris-type frame (inertial, planet-centered) (`dyn_body_init_orbit.cc:135-145`) | fatal | initialization | structural (our API takes the integration frame directly; non-ephemeris frames cannot be constructed) |
| BA.07 | BA state-init ordering: attitude, then rate, then position, then velocity (`body_action` sequence; also covered by DB.27) | structural | ordering | deferred (still covered by DB.27; concrete event-order enforcement is Phase 5 work) |

## Section MA: Mass (gap fill)

MA.10–MA.21 gap fill. Source: `../jeod/models/dynamics/mass/src/`.

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| MA.22 | Detach-on-drop is safe: destroying a still-attached body must not leave dangling parent pointers (`mass_body.cc:94-108` pattern — `mass_children.remove()` is resilient) | structural | structural | n/a (Rust's ownership model: references don't outlive owners; `MassBodyStore` is an arena of values, so a freed `MassBodyId` cannot produce a dangling pointer) |

## Section DM: DynManager (gap fill)

Gap fill for DM.14+. Source: `../jeod/models/dynamics/dyn_manager/src/dyn_manager.cc`.

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| DM.14 | Gravity manager singleton: at most one can be registered; duplicate registration is an error (`dyn_manager.cc:128-134`) | error | initialization | structural (gravity is a Bevy `Resource` — only one instance can exist per World) |
| DM.15 | Gravity manager must be registered before `initialize_simulation`; late registration is an error (`dyn_manager.cc:137-144`) | error | initialization | structural (resource insertion happens at plugin-build time, before any schedule runs) |
| DM.16 | Gravity manager and `gravity_off` flag are mutually exclusive (`dyn_manager.cc:147-155`) | error | initialization | n/a (we do not expose a `gravity_off` toggle — callers simply omit gravity from the body's configuration) |
| DM.17 | Duplicate body-action registration is rejected (`dyn_manager.cc:179-186`) | error | initialization | n/a (body-init components are inherent to the entity; duplication is an ECS-level concept) |

## Section GV: Gravity (gap fill)

Extensions to the existing GV section. Source: `../jeod/models/environment/gravity/src/`.

| Tag | Invariant | Enforcement | Category | Our Status |
|-----|-----------|-------------|----------|------------|
| GV.19 | Spherical harmonics source: degree and order clamped to `[0, max_degree]` at initialization (`spherical_harmonics_gravity_source.cc:90`) | structural | initialization | enforced (covered by GV.04, GV.05; noted here for source-side clamp) |
| GV.20 | Variational-effect (delta-coefficient) registration must be unique by typeid; duplicate registration fails (`spherical_harmonics_gravity_source.cc:245-250`) | fatal | initialization | deferred (variational effects / solid tides / ocean tides not implemented; Phase 5) |
