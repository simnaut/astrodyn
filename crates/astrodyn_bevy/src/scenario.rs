//! `SimulationBuilder → Bevy App` bridge — the canonical mission entry
//! point for whole-scenario composition.
//!
//! [`SimulationBuilderBevyExt::populate_app`] consumes a fully-composed
//! [`astrodyn::SimulationBuilder`] (sources, bodies, mass tree, ephemeris,
//! atmosphere, polar motion, integrator state) and materializes it into a
//! populated Bevy [`App`] in one call: resources for time, ephemeris,
//! atmosphere, and polar motion; entities for every gravity source and
//! vehicle; a fully-wired mass tree when the scenario registers one;
//! integrator state auto-initialized per body; and the [`AstrodynPlugin`]
//! installed if the caller hadn't already added it. This is the path
//! mission code should reach for when expressing a multi-body scenario.
//!
//! ```ignore
//! use astrodyn::recipes::scenarios;
//! use astrodyn_bevy::SimulationBuilderBevyExt;
//! use bevy::prelude::*;
//!
//! let mut app = App::new();
//! app.add_plugins(MinimalPlugins);
//!
//! let handles = scenarios::iss_leo::iss_leo()
//!     .populate_app::<astrodyn::Earth>(&mut app)
//!     .expect("populate_app");
//! // `handles.source_entities[i]` / `handles.body_entities[i]` are the
//! // Bevy entities for the i-th source / body in the recipe.
//! ```
//!
//! For multi-source scenarios — Earth-central with Moon + Sun
//! perturbations, Sun + planets for SRP / shadow geometry, mass-tree
//! attachments for stage separation — the same one-call shape applies;
//! the recipe layer composes the building blocks and `populate_app`
//! materializes the result. See `examples/multi_body_scenario.rs` for an
//! Apollo trans-lunar (Earth + Moon + Sun) end-to-end run, and
//! `examples/typed_mission.rs` for the complementary single-vehicle
//! flow that uses [`crate::VehicleConfigBevyExt::spawn_bevy`] directly
//! when scenario composition isn't needed.
//!
//! Symmetric to `astrodyn_runner::SimulationBuilderExt::build` (not a
//! dependency of `astrodyn_bevy`, so the link is not resolvable from
//! here), `populate_app` is the Bevy-side terminal of the
//! `SimulationBuilder` shape: every consumer of the same scenario
//! factory — runner, Bevy adapter, Tier 3 cross-validation,
//! `astrodyn_verif_parity::VerificationCaseParityExt` — invokes its
//! own terminal on the same builder Vec, which is what makes the
//! transitivity argument (`runner ↔ JEOD` + `runner ↔ bevy` ⇒
//! `bevy ↔ JEOD`) hold.
//!
//! ## Field-by-field mirror of `Simulation::from_builder`
//!
//! ```text
//! SimulationBuilder field  →  Bevy result
//! ─────────────────────────────────────────────────────────────────────────
//! time                      →  SimulationTimeR(time)
//! dt                        →  Time::<Fixed>::from_seconds(dt)
//! ephemeris                 →  EphemerisR(eph)
//! polar_motion              →  PolarMotionR { xp, yp }
//! atmosphere + planet idx   →  AtmosphereModelR { config, planet_entity }
//! sources[i]                →  Entity with GravitySourceC + …
//! source_ephem_bodies[i]    →  EphemerisBodyC on the source entity
//! sun_source / moon_source  →  SunMarker / MoonMarker on the source entity
//! bodies[i]                 →  cfg.spawn_bevy::<P>(commands, &source_entities)
//! integrator (GJ / ABM4)    →  GaussJacksonStateC / Abm4StateC inserted
//! mass_tree_names[i]        →  pre-allocated MassBodyIdC(id) on the body
//! mass_tree_attachments     →  MassTreeR.attach() + MassChildOf on child
//! ```
//!
//! The resulting `App` steps under `FixedUpdate` and produces the same
//! per-tick state as `astrodyn_runner::Simulation::step_until` for the same
//! scenario. Bit-identity is the contract; see
//! `crates/astrodyn_verif_parity/tests/bevy_parity_*.rs`.
//!
//! ## Single-planet today
//!
//! The bridge is **single-planet** by construction: every body in the
//! scenario integrates in the `<P>`-tagged
//! [`PlanetInertial`](astrodyn::PlanetInertial) frame chosen at the
//! `populate_app::<P>` call site. Mission code (and parity tests)
//! must keep `<P>` consistent across the whole scenario.
//!
//! In practice every shipped Tier 3 scenario is single-body, so the
//! constraint reduces to picking the body's central planet at the
//! call site — `<Earth>` for Earth-centric scenarios (`apollo*`,
//! dyncomp, lvlh, …), `<Moon>` for Moon-centric ones (`earth_moon`
//! Clementine, NESC CC8), `<Sun>` for heliocentric (`mercury`),
//! `<Mars>` for Mars-centric (`mars_orbit`). A future genuine
//! multi-body-multi-planet scenario (two bodies integrating in
//! different planet-inertial frames within one run) would need a
//! per-body dispatch sibling — until such a scenario lands, the
//! single-`<P>` shape is sufficient.

use bevy::prelude::*;
use glam::DVec3;

use astrodyn::{
    Abm4State, FrameTransform, GaussJacksonState, IntegratorType, MassBodyId, MassTree,
    MassTreeAttachment, Planet, PlanetFixed, RootInertial, RotationModel, SimulationBuilder,
    ValidationError, VehicleConfig,
};

use crate::components::{
    Abm4StateC, EphemerisBodyC, GaussJacksonStateC, GravitySourceC, MassBodyIdC, MassChildOf,
    MoonMarker, PlanetFixedRotationC, PlanetOmegaC, RotationModelC, SourceInertialPositionC,
    SourceInertialVelocityC, SunMarker, TidalConfigC, TranslationalStateC,
};
use crate::{
    AstrodynPlugin, AtmosphereModelR, EphemerisR, IntegrationDtR, MassTreeR, PolarMotionR,
    SimulationTimeR, VehicleConfigBevyExt,
};

/// Handles to entities spawned by [`SimulationBuilderBevyExt::populate_app`].
///
/// Indices match the corresponding `SimulationBuilder` `Vec`s: `source_entities[i]`
/// is the entity for the `i`-th gravity source, `body_entities[i]` for the
/// `i`-th vehicle. Callers use these to read state out of the world after
/// stepping (e.g. `world.get::<TranslationalStateC<P>>(handles.body_entities[0])`).
#[derive(Debug, Clone)]
pub struct ScenarioHandles {
    /// Gravity-source entities, indexed parallel to `SimulationBuilder::sources`.
    pub source_entities: Vec<Entity>,
    /// Vehicle entities, indexed parallel to `SimulationBuilder::bodies`.
    pub body_entities: Vec<Entity>,
}

/// Canonical mission entry point: turn a fully-composed
/// [`astrodyn::SimulationBuilder`] into a populated Bevy [`App`] in one
/// call.
///
/// The recipe-driven shape is the recommended use site — mission code
/// composes a scenario via the [`recipes`](astrodyn::recipes) catalog
/// (or by hand), then materializes it in a single line:
///
/// ```ignore
/// use astrodyn::recipes::scenarios;
/// use astrodyn_bevy::SimulationBuilderBevyExt;
///
/// let handles = scenarios::iss_leo::iss_leo()
///     .populate_app::<astrodyn::Earth>(&mut app)
///     .expect("populate_app");
/// ```
///
/// Multi-source scenarios (Earth + Moon + Sun perturbations, mass-tree
/// stage separations, atmosphere + polar motion + ephemeris) compose the
/// same way — the recipe layer assembles the building blocks, the
/// terminal `populate_app` writes the resulting world. The
/// per-vehicle [`crate::VehicleConfigBevyExt::spawn_bevy`] flow remains
/// available for cases where the caller is composing one vehicle at a
/// time without going through a `SimulationBuilder`; see
/// `examples/typed_mission.rs` for that complementary pattern.
///
/// `<P: Planet>` selects the planet whose
/// [`PlanetInertial`](astrodyn::PlanetInertial) frame **every** body
/// in the scenario integrates in. Today's bridge is single-planet by
/// construction; every shipped Tier 3 scenario is single-body, so the
/// constraint reduces to picking the body's central planet at the
/// call site (Earth / Moon / Mars / Sun, …). A future genuine
/// multi-body-multi-planet scenario — two bodies integrating in
/// distinct planet-inertial frames within one run — would need a
/// per-body dispatch sibling on this method; until such a scenario
/// lands the single-`<P>` shape covers every recipe.
///
/// # Returns
///
/// On success, [`ScenarioHandles`] keyed parallel to the builder's
/// `sources` / `bodies` vecs — `source_entities[i]` is the entity for
/// the `i`-th source, `body_entities[i]` for the `i`-th vehicle. Use
/// these to read state out of the world after stepping (e.g.
/// `world.get::<TranslationalStateC<P>>(handles.body_entities[0])`).
/// The `Result` carries the same
/// [`Vec<ValidationError>`](astrodyn::ValidationError) shape
/// `Simulation::from_builder` uses, reserved for a future Bevy-native
/// validator (see the trait method's `# Validation` section for what
/// `populate_app` does and does not check today).
///
/// # Side effects
///
/// Calls `app.add_plugins(AstrodynPlugin)` if it wasn't already added. The
/// plugin is idempotent only when callers haven't pre-installed competing
/// resources; the bridge expects to own time / mass-tree / source spawning
/// for the whole scenario.
pub trait SimulationBuilderBevyExt: Sized {
    /// Materialize this builder into the given Bevy [`App`] under planet `P`.
    ///
    /// This is the canonical "scenario in one call" entry point for the
    /// Bevy adapter. Mirrors every field of [`SimulationBuilder`] into
    /// the `App` world (resources for time / ephemeris / polar motion /
    /// atmosphere, entities for sources and vehicles, mass-tree
    /// pre-allocation + `MassChildOf` edges, integrator-state auto-init)
    /// and returns [`ScenarioHandles`] keyed parallel to the builder's
    /// vectors. The pipeline reads its integrator timestep from
    /// [`IntegrationDtR`], which `populate_app` inserts at `self.dt`;
    /// callers can immediately step the app via
    /// [`crate::AstrodynAppExt::step_fixed_dt`]
    /// (which keeps `Time<Fixed>` and `IntegrationDtR` in sync) or by
    /// running the schedule directly. Direct callers that drive the
    /// schedule manually must also update `IntegrationDtR` if they want
    /// to vary `dt` mid-run — advancing `Time<Fixed>` alone won't change
    /// the physics timestep.
    ///
    /// ```ignore
    /// use astrodyn::recipes::scenarios;
    /// use astrodyn_bevy::SimulationBuilderBevyExt;
    /// use bevy::prelude::*;
    ///
    /// let mut app = App::new();
    /// app.add_plugins(MinimalPlugins);
    /// let handles = scenarios::iss_leo::iss_leo()
    ///     .populate_app::<astrodyn::Earth>(&mut app)
    ///     .expect("populate_app");
    /// assert_eq!(handles.source_entities.len(), 1);
    /// assert_eq!(handles.body_entities.len(), 1);
    /// ```
    ///
    /// # Validation
    ///
    /// `populate_app` does **not** run runner-side `Simulation::from_builder`
    /// validation; the `Result` shape exists so a future Bevy-native
    /// validator can land without an API break. Direct callers that
    /// need validation should build (and discard) a runner `Simulation`
    /// from the same scenario factory first — see the parity trait in
    /// `astrodyn_verif_parity::VerificationCaseParityExt::run_and_assert_parity`
    /// for the canonical pattern.
    fn populate_app<P: Planet>(
        self,
        app: &mut App,
    ) -> Result<ScenarioHandles, Vec<ValidationError>>;
}

impl SimulationBuilderBevyExt for SimulationBuilder {
    fn populate_app<P: Planet>(
        self,
        app: &mut App,
    ) -> Result<ScenarioHandles, Vec<ValidationError>> {
        // No Bevy-native validator yet — see the trait method's
        // `# Validation` doc for the contract. The parity trait pairs
        // each `populate_app` call with a runner-side
        // `Simulation::from_builder` on a fresh builder from the same
        // factory, so validation errors surface there before any
        // bit-identity assertion runs. Direct callers (mission code
        // not going through the parity trait) must validate
        // themselves until a Bevy-native validator lands.

        // Time + dt resources. `AstrodynPlugin::build` calls
        // `init_resource::<SimulationTimeR>()` which constructs a
        // default; overwrite it here with the builder's full
        // `SimulationTime` so `time_scale_factor` and the leap-second
        // table are honored.
        // allowed: `Time::<Fixed>::from_seconds` is Bevy's own constructor
        // for `Time<Fixed>` (a Bevy resource), not the banned
        // `SecondsSince::from_seconds` typed-quantity bypass; the grep
        // pattern catches `from_seconds` indiscriminately. The argument
        // `self.dt` is a plain `f64` integrator timestep, not a typed
        // duration phantom.
        app.insert_resource(Time::<Fixed>::from_seconds(self.dt));
        // `IntegrationDtR` is the bit-exact f64 source of `dt` for the
        // pipeline (`Time<Fixed>::delta_secs_f64()` rounds to integer
        // nanoseconds via `Duration::from_secs_f64`, which breaks
        // `runner ↔ bevy` bit-identity on irrational-in-seconds
        // timesteps like `period / 560`). The runner side reads
        // `SimulationBuilder.dt` directly as f64 through
        // `Simulation::step_internal`; the Bevy side now mirrors that.
        app.insert_resource(IntegrationDtR(self.dt));
        app.insert_resource(SimulationTimeR(self.time));

        // Optional global resources. AstrodynPlugin doesn't insert any
        // of these by default; mission code (and now the bridge)
        // inserts only what the scenario actually configures.
        if let Some(eph) = self.ephemeris {
            app.insert_resource(EphemerisR(eph));
        }
        if let Some((xp, yp)) = self.polar_motion {
            app.insert_resource(PolarMotionR { xp, yp });
        }

        // Add the plugin *after* time + ephemeris + polar-motion are in
        // place: `AstrodynPlugin::build` reads `Time<Fixed>` indirectly
        // and pre-installs `RootFrameEntityR`, which we want to happen
        // before any source-entity spawn so source frame entities can
        // `ChildOf`-link under the existing root.
        if !app.is_plugin_added::<AstrodynPlugin>() {
            app.add_plugins(AstrodynPlugin);
        }

        // `AstrodynPlugin::build` only installs the per-planet system
        // instantiations for `astrodyn::Earth` (frame registration,
        // body-action queue, validator, etc.). Missions integrating in
        // a different planet's inertial frame must additionally call
        // [`register_planet_systems::<P>`](crate::register_planet_systems)
        // — without those instantiations, the FixedUpdate pipeline
        // never picks up the planet's bodies and `populate_app` returns
        // a world that "stays at IC" no matter how many ticks the
        // caller pumps. Do it once here so callers don't have to know
        // about the helper. The `RegisteredPlanetsR` registry is
        // populated by both `AstrodynPlugin::build` (for Earth) and the
        // helper itself, so the contains-check makes this idempotent
        // and safe even when `<P> == astrodyn::Earth` or when
        // populate_app runs twice on the same App.
        if !app
            .world()
            .resource::<crate::RegisteredPlanetsR>()
            .contains::<P>()
        {
            crate::register_planet_systems::<P>(app);
        }

        // ── Sources ──
        let sources_len = self.sources.len();
        let mut source_entities = Vec::with_capacity(sources_len);
        let SimulationBuilder {
            atmosphere,
            atmosphere_planet_source,
            sun_source,
            moon_source,
            sources,
            source_ephem_bodies,
            bodies,
            mass_tree_names,
            mass_tree_attachments,
            ..
        } = self;
        // Validate ephem-body slice length matches sources.
        assert!(
            source_ephem_bodies.len() == sources_len,
            "populate_app: source_ephem_bodies length {} does not match sources length {}",
            source_ephem_bodies.len(),
            sources_len
        );

        for (idx, (name, entry)) in sources.into_iter().enumerate() {
            let ephem = source_ephem_bodies.get(idx).copied().flatten();
            let entity = spawn_source::<P>(app, idx, &name, entry, sun_source, moon_source, ephem);
            source_entities.push(entity);
        }

        // ── Atmosphere ──
        if let Some(config) = atmosphere {
            let planet_idx = atmosphere_planet_source.expect(
                "populate_app: SimulationBuilder.atmosphere is Some but \
                 atmosphere_planet_source is None. Atmosphere computation \
                 requires a planet source whose `PlanetFixedRotationC` the \
                 atmosphere system queries every tick. Call \
                 `SimulationBuilder::atmosphere(config, planet_source)` (not \
                 a direct `sb.atmosphere = Some(_)` field write) and ensure \
                 the source has a `rotation_model` so `populate_app` inserts \
                 `PlanetFixedRotationC` on it. The runner side accepts the \
                 split fields as-is and the in-source-list index validation \
                 catches a stale index, but a `None` planet source is a \
                 misconfiguration the bridge surfaces here.",
            );
            let planet_entity = *source_entities.get(planet_idx).unwrap_or_else(|| {
                panic!(
                    "populate_app: atmosphere_planet_source index {planet_idx} out of \
                     range ({sources_len} sources)"
                )
            });
            app.insert_resource(AtmosphereModelR::new(config, planet_entity));
        }

        // ── Mass tree pre-allocation ──
        // Mirror `Simulation::from_builder`'s
        // `attach_preserving_initial_state` path: pre-build a fresh
        // `MassTree` so each `MassBodyId` is allocated *before* the
        // corresponding entity is spawned. This lets us spawn body
        // entities with `MassBodyIdC(id)` already attached, matching
        // the existing parity tests' explicit pre-allocation pattern.
        let has_tree = mass_tree_names.iter().any(|n| n.is_some());
        // Fail-loudly fence: an attachment without a registered tree
        // would silently drop on the floor under the
        // `if has_tree { … }` branch below. The
        // `SimulationBuilder::attach_bodies` validator already rejects
        // that combination at builder-time, so reaching this assertion
        // means a caller hand-constructed `mass_tree_attachments` with
        // no matching `mass_tree_names`. Mirror the runner's
        // `attach_preserving_initial_state` precondition (both
        // children must be registered) at the bridge boundary.
        assert!(
            has_tree || mass_tree_attachments.is_empty(),
            "populate_app: SimulationBuilder has {} pending mass-tree \
             attachment(s) but no body is registered with a mass-tree name. \
             Call `register_in_mass_tree(idx, name)` on each participating \
             body before `attach_bodies(...)`.",
            mass_tree_attachments.len(),
        );
        let (mass_tree, mass_ids): (Option<MassTree>, Vec<Option<MassBodyId>>) = if has_tree {
            let mut tree = MassTree::new();
            let mut ids: Vec<Option<MassBodyId>> = Vec::with_capacity(bodies.len());
            for (i, name) in mass_tree_names.iter().enumerate() {
                if let Some(name) = name {
                    let mass = bodies[i].mass.unwrap_or_else(|| {
                        panic!(
                            "populate_app: mass-tree-registered body {i} has no mass properties; \
                             SimulationBuilder::register_in_mass_tree should have caught this."
                        )
                    });
                    ids.push(Some(tree.add_body(
                        name.clone(),
                        astrodyn::typed_bridge::mass_typed_to_raw(&mass),
                    )));
                } else {
                    ids.push(None);
                }
            }
            // Apply pending attachments at config-time (the
            // `attach_preserving_initial_state` semantics — no
            // `combine_states_at_attach` writeback). `MassTree::attach`
            // is exactly that: tree mutation + composite-mass resync,
            // no state combine.
            for att in &mass_tree_attachments {
                let MassTreeAttachment {
                    child_idx,
                    parent_idx,
                    offset,
                    t_parent_child,
                } = *att;
                let child_id = ids[child_idx].expect(
                    "populate_app: attachment references a child not registered in the mass tree",
                );
                let parent_id = ids[parent_idx].expect(
                    "populate_app: attachment references a parent not registered in the mass tree",
                );
                tree.attach(child_id, parent_id, offset, t_parent_child);
            }
            (Some(tree), ids)
        } else {
            (None, Vec::new())
        };

        // Need the attachment list available after spawning; clone
        // shape so we can look up parent body indices for `MassChildOf`.
        let attachments_for_mass_child_of = mass_tree_attachments;

        // ── Vehicles ──
        //
        // `VehicleConfig.shadow_body` carries a *source* index — the
        // matching `ShadowBodyC` Bevy component lives on the source
        // entity, not the body. The runner reads
        // `body.shadow_body.source_idx` directly each step; the Bevy
        // pipeline iterates entities with `ShadowBodyC` instead, so the
        // bridge installs the marker on the named source here. We
        // collect the (source_idx, radius) pairs as we walk `bodies`
        // and apply them after spawning so the move into
        // `spawn_vehicle` doesn't need to fork a per-cfg field read.
        let mut shadow_marker_inserts: Vec<(usize, f64)> = Vec::new();
        let mut body_entities = Vec::with_capacity(bodies.len());
        for (i, cfg) in bodies.into_iter().enumerate() {
            if let Some(sb) = cfg.shadow_body {
                shadow_marker_inserts.push((sb.source_idx, sb.radius));
            }
            let integrator = cfg.integrator;
            let entity = spawn_vehicle::<P>(app, cfg, &source_entities);
            // Auto-init integrator state, mirroring
            // `Simulation::validate()`'s GJ / ABM4 auto-init.
            match integrator {
                IntegratorType::GaussJackson(config) => {
                    app.world_mut()
                        .entity_mut(entity)
                        .insert(GaussJacksonStateC(GaussJacksonState::new(config)));
                }
                IntegratorType::Abm4 => {
                    app.world_mut()
                        .entity_mut(entity)
                        .insert(Abm4StateC(Abm4State::new()));
                }
                _ => {}
            }
            // Tag with `MassBodyIdC` if registered in the tree.
            if let Some(Some(id)) = mass_ids.get(i).copied() {
                app.world_mut().entity_mut(entity).insert(MassBodyIdC(id));
            }
            body_entities.push(entity);
        }

        // Per-source `ShadowBodyC` insertions collected above. The
        // marker is idempotent — if multiple bodies name the same source
        // their radii must agree (the runner's
        // `body.shadow_body.radius` is per-body but the Bevy pipeline
        // reads `ShadowBodyC.radius` per source); the assertion catches
        // the mismatch loudly rather than silently overwriting.
        for (source_idx, radius) in shadow_marker_inserts {
            let source_entity = *source_entities.get(source_idx).unwrap_or_else(|| {
                panic!(
                    "populate_app: VehicleConfig.shadow_body.source_idx {source_idx} \
                     out of range ({sources_len} sources)"
                )
            });
            // Read any existing radius first to fail loudly on a mismatch.
            let existing = app
                .world()
                .get::<crate::components::ShadowBodyC>(source_entity)
                .map(|c| c.radius);
            if let Some(prev) = existing {
                assert!(
                    prev.to_bits() == radius.to_bits(),
                    "populate_app: source {source_idx} already has \
                     ShadowBodyC {{ radius: {prev} }} but a later body \
                     specifies radius {radius}; bodies sharing a shadow \
                     body must agree on its radius."
                );
                continue;
            }
            app.world_mut()
                .entity_mut(source_entity)
                .insert(crate::components::ShadowBodyC { radius });
        }

        // ── Mass tree resource + child-edges ──
        if let Some(tree) = mass_tree {
            app.insert_resource(MassTreeR(tree));
            // Per-attachment `MassChildOf` insertions on the child
            // entity. The Bevy adapter uses `MassChildOf` as the
            // ECS-native parent ↔ child edge, parallel to the runner's
            // `MassTree::parent[child_id]` link. Mirror the same
            // edge geometry the tree itself stores.
            for att in attachments_for_mass_child_of {
                let MassTreeAttachment {
                    child_idx,
                    parent_idx,
                    offset,
                    t_parent_child,
                } = att;
                let child_entity = body_entities[child_idx];
                let parent_entity = body_entities[parent_idx];
                app.world_mut()
                    .entity_mut(child_entity)
                    .insert(MassChildOf::with_rotation(
                        parent_entity,
                        offset,
                        t_parent_child,
                    ));
            }
        }

        Ok(ScenarioHandles {
            source_entities,
            body_entities,
        })
    }
}

/// Spawn a single gravity source entity, attaching every `GravitySourceEntry`
/// field that has a Bevy-component analog plus the `Sun`/`Moon` markers when
/// the source's index matches the builder's `sun_source` / `moon_source`.
///
/// `idx` is the source's index in `SimulationBuilder::sources`; the marker
/// comparison reads `sun_source` / `moon_source` directly to decide whether to
/// tag this entity, mirroring `Simulation::sun_source = Some(idx)` on the
/// runner side. `ephem` is the matching slot from
/// `SimulationBuilder::source_ephem_bodies[idx]`, attached as
/// [`EphemerisBodyC`] when `Some` so `ephemeris_update_system` rewrites
/// the source's `SourceInertialPositionC` each tick.
fn spawn_source<P: Planet>(
    app: &mut App,
    idx: usize,
    name: &str,
    entry: astrodyn::GravitySourceEntry,
    sun_source: Option<usize>,
    moon_source: Option<usize>,
    ephem: Option<(astrodyn::EphemerisBody, astrodyn::EphemerisBody)>,
) -> Entity {
    let astrodyn::GravitySourceEntry {
        source,
        position,
        velocity,
        t_inertial_pfix,
        rotation_model,
        delta_c20: _,
        tidal_config,
        planet_omega,
        central: _,
        marker_only,
    } = entry;

    // Marker-only fast-path: spawn the entity with just the marker
    // and translational state — `Name`, `SunMarker` / `MoonMarker`,
    // `TranslationalStateC<P>`. Skips `GravitySourceC`,
    // `SourceInertialPositionC`, and the source's frame-tree entity
    // (no `register_source_frames_system` pickup) so the SRP
    // direction-only-source path matches the hand-rolled
    // `bevy_parity_srp.rs` setup that the recipe family targets.
    // See `astrodyn::GravitySourceEntry::marker_only` for the full
    // contract.
    if marker_only {
        let mut entity_cmds = app.world_mut().spawn((
            Name::new(name.to_string()),
            // `TranslationalStateC<P>` carries the source position
            // for the SRP system's `Query<&TranslationalStateC<P>,
            // With<SunMarker>>` lookup. Mirrors the hand-rolled
            // spawn in the bevy_parity_srp tests.
            TranslationalStateC::<P>::from_untyped(astrodyn::TranslationalState {
                position: position.raw_si(),
                velocity: velocity.raw_si(),
            }),
        ));
        if Some(idx) == sun_source {
            entity_cmds.insert(SunMarker);
        }
        if Some(idx) == moon_source {
            entity_cmds.insert(MoonMarker);
        }
        return entity_cmds.id();
    }

    let mut entity_cmds = app.world_mut().spawn((
        Name::new(name.to_string()),
        GravitySourceC(source),
        SourceInertialPositionC(position),
        // `TranslationalStateC<P>` is what the rest of the Bevy
        // pipeline reads for source kinematic state; populate it from
        // the source's root-inertial position/velocity. The default
        // for the central source is zero — same as the runner's root
        // mapping.
        // allowed: typed↔raw kernel-boundary lift at the gateway
        // gravity-source insertion point (named-method opt-in; see
        // #397).
        TranslationalStateC::<P>::from_untyped(astrodyn::TranslationalState {
            position: position.raw_si(),
            velocity: velocity.raw_si(),
        }),
    ));

    // Source velocity for relativistic / PPN corrections — only
    // attach when non-zero to keep diffs minimal vs the existing
    // hand-rolled parity tests.
    if velocity.raw_si() != DVec3::ZERO {
        entity_cmds.insert(SourceInertialVelocityC(velocity));
    }

    // Rotation model + initial inertial→pfix transform. The runner
    // creates a `pfix` child frame whenever `rotation_model != None`
    // *or* `t_inertial_pfix.is_some()`; mirror the same condition so
    // a hand-set identity transform without a rotation model still
    // gets a `PlanetFixedRotationC<P>` written into the world.
    if rotation_model != RotationModel::None || t_inertial_pfix.is_some() {
        let t = t_inertial_pfix.unwrap_or(glam::DMat3::IDENTITY);
        entity_cmds.insert(PlanetFixedRotationC::<P>(FrameTransform::<
            RootInertial,
            PlanetFixed<P>,
        >::from_matrix(t)));
        // Default rotation model when the source had a `t_inertial_pfix`
        // but no explicit model is `None`; only attach `RotationModelC`
        // when the source actually configures one.
        if rotation_model != RotationModel::None {
            entity_cmds.insert(RotationModelC(rotation_model));
        }
    }
    if planet_omega != 0.0 {
        entity_cmds.insert(PlanetOmegaC(planet_omega));
    }
    if let Some(cfg) = tidal_config {
        entity_cmds.insert(TidalConfigC::from_untyped(&cfg));
    }
    if Some(idx) == sun_source {
        entity_cmds.insert(SunMarker);
    }
    if Some(idx) == moon_source {
        entity_cmds.insert(MoonMarker);
    }
    if let Some((target, observer)) = ephem {
        entity_cmds.insert(EphemerisBodyC { target, observer });
    }

    entity_cmds.id()
}

/// Spawn a single vehicle entity by deferring to
/// [`VehicleConfigBevyExt::spawn_bevy`]. Lives in this module only to keep
/// the `populate_app` body readable; the actual translation logic stays in
/// `lib.rs`.
fn spawn_vehicle<P: Planet>(
    app: &mut App,
    cfg: VehicleConfig,
    source_entities: &[Entity],
) -> Entity {
    let entity = {
        let mut commands = app.world_mut().commands();
        cfg.spawn_bevy::<P>(&mut commands, source_entities)
    };
    // Apply queued component insertions so subsequent post-processing
    // (e.g. `MassBodyIdC` insertion below) lands on the same entity.
    app.world_mut().flush();
    entity
}

#[cfg(test)]
mod tests {
    //! Bridge unit tests: build an Earth point-mass scenario through the
    //! same `SimulationBuilder` on both runtimes and assert
    //! bit-identical post-step state. These tests live in the bridge
    //! module rather than a parity test file so the bridge itself is
    //! exercised by `cargo test -p astrodyn_bevy` — Phase 2's
    //! `VerificationCaseParityExt` then layers on top.
    //!
    //! Runner-side `astrodyn_runner` is a `[dev-dependencies]` of
    //! `astrodyn_bevy`, so the runner is reachable from inside this
    //! test module without bloating the production dep graph.
    use std::time::Duration;

    use astrodyn::{
        GravityControl, GravityControls, GravityGradient, GravityModel, GravitySource,
        GravitySourceEntry, Position, RootInertial, SimulationTime, TranslationalState,
        VehicleConfig, Velocity,
    };
    use astrodyn_runner::SimulationBuilderExt;
    use bevy::prelude::*;
    use glam::DVec3;

    use super::*;
    use crate::TranslationalStateC;

    const MU_EARTH: f64 = 3.986_004_418e14;
    const DT: f64 = 10.0;
    const NUM_STEPS: usize = 50;

    fn iss_trans() -> TranslationalState {
        TranslationalState {
            position: DVec3::new(6_778_137.0, 0.0, 0.0),
            velocity: DVec3::new(0.0, 7668.56, 0.0),
        }
    }

    /// Compose a one-body, one-source point-mass scenario and return its
    /// builder. Same factory drives both runtimes so the scenario
    /// definition lives in exactly one place — the pattern Phase 2's
    /// `VerificationCaseParityExt` makes universal.
    fn point_mass_iss_builder() -> SimulationBuilder {
        let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
        let mut b = SimulationBuilder::new(time, DT);
        let mut earth = GravitySourceEntry::new(
            GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            },
            Position::<RootInertial>::zero(),
            None,
        );
        earth.central = true;
        let earth_idx = b.add_source("Earth", earth);
        b.add_body(VehicleConfig {
            // allowed: typed↔raw kernel boundary (#397)
            trans: astrodyn::typed_bridge::trans_raw_to_root(&iss_trans()),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(
                    earth_idx,
                    GravityGradient::Skip,
                )],
            },
            ..Default::default()
        });
        b
    }

    fn step_bevy(app: &mut App, n: usize) {
        for _ in 0..n {
            app.world_mut()
                .resource_mut::<Time<Fixed>>()
                .advance_by(Duration::from_secs_f64(DT));
            app.world_mut().run_schedule(FixedUpdate);
        }
    }

    fn assert_bits_eq(component: &str, a: f64, b: f64) {
        assert!(
            a.to_bits() == b.to_bits(),
            "{component}: not bit-identical: a={a} ({:#018x}) vs b={b} ({:#018x})",
            a.to_bits(),
            b.to_bits(),
        );
    }

    #[test]
    fn populate_app_point_mass_iss_matches_runner_bit_identical() {
        // Runner — build, validate, step.
        let runner_sim = point_mass_iss_builder()
            .build()
            .expect("runner build must succeed");
        let mut runner_sim = runner_sim;
        runner_sim
            .step_n(NUM_STEPS)
            .expect("runner step_n must succeed");
        // allowed: typed↔raw kernel boundary
        let runner_state = astrodyn::typed_bridge::trans_typed_to_raw(&runner_sim.body(0).trans);

        // Bridge — populate a fresh app from the same builder, step.
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        let handles = point_mass_iss_builder()
            .populate_app::<astrodyn::Earth>(&mut app)
            .expect("populate_app must succeed");
        assert_eq!(handles.source_entities.len(), 1);
        assert_eq!(handles.body_entities.len(), 1);
        step_bevy(&mut app, NUM_STEPS);
        // allowed: typed↔raw kernel boundary
        let bevy_state = astrodyn::typed_bridge::trans_typed_to_raw(
            &app.world()
                .get::<TranslationalStateC<astrodyn::Earth>>(handles.body_entities[0])
                .expect("vehicle entity must carry TranslationalStateC<Earth>")
                .0,
        );

        // Bit-identity per component.
        for i in 0..3 {
            assert_bits_eq(
                &format!("position[{i}]"),
                bevy_state.position[i],
                runner_state.position[i],
            );
            assert_bits_eq(
                &format!("velocity[{i}]"),
                bevy_state.velocity[i],
                runner_state.velocity[i],
            );
        }
    }

    #[test]
    fn populate_app_returns_one_entity_per_source_and_body() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        let mut b = SimulationBuilder::new(
            SimulationTime::at_j2000(astrodyn::default_leap_second_table()),
            DT,
        );
        let mut earth = GravitySourceEntry::new(
            GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            },
            Position::<RootInertial>::zero(),
            None,
        );
        earth.central = true;
        let earth_idx = b.add_source("Earth", earth);
        // Add two third bodies so we exercise the multi-source path.
        let _sun_idx = b.add_source(
            "Sun",
            GravitySourceEntry {
                source: GravitySource {
                    mu: 1.327e20,
                    model: GravityModel::PointMass,
                },
                // allowed: bridge unit test — one-shot synthetic Sun
                // position at scenario-construction time, not a per-step
                // bypass. The call site mints a `Position<RootInertial>`
                // for a `GravitySourceEntry` field that the runner-side
                // `Simulation::add_source` consumes verbatim; using the
                // typed `1.5e11.m_at::<RootInertial>()` lift here would
                // require a `Vec3Ext` import inside the test only and
                // not change the resulting bit pattern.
                position: Position::<RootInertial>::from_raw_si(DVec3::new(1.5e11, 0.0, 0.0)),
                velocity: Velocity::<RootInertial>::zero(),
                t_inertial_pfix: None,
                rotation_model: astrodyn::RotationModel::None,
                delta_c20: 0.0,
                tidal_config: None,
                planet_omega: 0.0,
                central: false,
                marker_only: false,
            },
        );
        b.add_body(VehicleConfig {
            // allowed: typed↔raw kernel boundary (#397)
            trans: astrodyn::typed_bridge::trans_raw_to_root(&iss_trans()),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(
                    earth_idx,
                    GravityGradient::Skip,
                )],
            },
            ..Default::default()
        });
        let handles = b
            .populate_app::<astrodyn::Earth>(&mut app)
            .expect("populate_app");
        assert_eq!(
            handles.source_entities.len(),
            2,
            "two sources spawned, one entity per source"
        );
        assert_eq!(
            handles.body_entities.len(),
            1,
            "one vehicle spawned, one entity"
        );
    }
}
