// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Bevy systems for [`AstrodynSet::Integration`](crate::AstrodynSet::Integration).
//!
//! State integration (RK4 / Gauss-Jackson / ABM4), mass-tree staging
//! (attach / detach), detached-subtree free-flight propagation, and
//! distance-based frame switching.

use astrodyn::{
    Acceleration, AngularAcceleration, BodyFrame, Force, Planet, Position, RootInertial,
    RotationalState, SelfRef, Torque, TranslationalState, Velocity,
};
use bevy::prelude::*;
use glam::DVec3;

use crate::components::*;
use crate::frame_param::{FrameOrigin, RelativeFrameState};
use crate::SimulationTimeR;
use astrodyn::typed_bridge::{
    mass_raw_to_self_ref, mass_typed_to_raw, rot_raw_to_self_ref, rot_typed_to_raw,
    trans_raw_to_planet, trans_typed_to_raw,
};

use super::util::body_integ_origin_in_root_lazy;

/// Evaluate distance-based [`FrameSwitchesC`] entries for each body.
/// On trigger, this system:
///
/// 1. Reparents the body's frame entity under the target source's
///    frame entity via
///    `commands.entity(body_frame).insert(ChildOf(target_frame))`.
/// 2. Rewrites the body's [`TranslationalStateC`] (and the body
///    frame entity's [`FrameTransC`]) in the new integration
///    frame's coordinates, computed via
///    [`crate::frame_param::RelativeFrameState`].
/// 3. Flips [`GravityControlsC`]'s `differential` flags so the new
///    central source becomes non-differential and the prior
///    central source (and any others) becomes differential.
///
/// JEOD reference: `dyn_body_frame_switch.cc:173-182`. The trigger
/// predicates and gravity-control flip mirror
/// `astrodyn_runner::Simulation`'s `evaluate_and_apply_frame_switch` over
/// the arena; the Bevy variant reads/writes the ECS hierarchy
/// directly via [`crate::frame_param::RelativeFrameState`] (which
/// `impl FrameStorage`s and shares the storage-agnostic
/// `compute_relative_state` algorithm in `astrodyn_frames` with the
/// runner's arena).
///
/// Runs in `AstrodynSet::Integration` after [`crate::sync_body_to_frame_system`].
/// Bodies without [`FrameSwitchesC`] entries (or whose entries are
/// all `active = false`) are skipped.
// JEOD_INV: DB.14 — distance-based integration-frame switch reparents
// the body's frame entity under the target source's frame entity and
// rewrites translational state into the new frame's coordinates.
#[allow(clippy::type_complexity)]
pub fn frame_switch_system<P: Planet>(
    mut commands: Commands,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    sources: Query<&FrameEntityC, With<GravitySourceC>>,
    parents: Query<&ChildOf>,
    rel: RelativeFrameState,
    mut bodies: Query<(
        Entity,
        &mut TranslationalStateC<P>,
        &FrameEntityC,
        &mut FrameSwitchesC,
        &mut GravityControlsC,
    )>,
) {
    // Build a set of registered source frame entities once per call so
    // the per-body integ-frame validation below is O(1) rather than
    // O(sources). Without this, the inner check is a linear scan over
    // every source for every body every tick (O(bodies * sources)),
    // which dominates with many bodies and/or many sources even when
    // no switch fires.
    let known_source_frames: std::collections::HashSet<Entity> =
        sources.iter().map(|fe| fe.0).collect();
    for (body_entity, mut trans, body_frame_entity, mut switches, mut gravity_controls) in
        &mut bodies
    {
        if switches.0.is_empty() {
            continue;
        }
        // The body's current integration frame is the parent of its
        // frame entity in the ECS hierarchy.
        let current_integ_frame_entity = parents
            .get(body_frame_entity.0)
            .unwrap_or_else(|err| {
                panic!(
                    "frame_switch_system: body {body_entity:?} frame entity {fe:?} \
                     has no ChildOf parent ({err:?}). The body's frame entity must \
                     be parented under its integration frame entity (set by \
                     register_body_frames_system).",
                    fe = body_frame_entity.0,
                )
            })
            .parent();
        // Validate the current integ frame entity is the root frame
        // entity or a registered source's frame entity. Anything else
        // means the registration / integ-source wiring is corrupt.
        let current_is_known = current_integ_frame_entity == root_frame_entity.0
            || known_source_frames.contains(&current_integ_frame_entity);
        assert!(
            current_is_known,
            "frame_switch_system: body {body_entity:?} frame entity \
             {fe:?} has parent {parent:?} which is neither the root \
             frame entity ({root_e:?}) nor a registered source's \
             frame entity. The integration frame entity must be one \
             of those — register the source via PlanetBundle before \
             spawning the body, or attach the body under the root.",
            fe = body_frame_entity.0,
            parent = current_integ_frame_entity,
            root_e = root_frame_entity.0,
        );

        // Find the first active switch whose predicate triggers.
        let mut trigger_idx = None;
        for (idx, sw) in switches.0.iter().enumerate() {
            if !sw.active {
                continue;
            }
            // Resolve the target source's frame entity. Fail loud if
            // the target isn't a registered gravity source — same
            // contract as `evaluate_and_apply_frame_switch`'s
            // `FrameSwitchTargetMissing` error. The query filter is
            // `With<GravitySourceC>`, so a missing match means the
            // target either isn't a gravity source at all (no
            // `GravitySourceC`) or is one but `FrameEntityC` was never
            // inserted by `register_source_frames_system` — both are
            // user misconfigurations the diagnostic must enumerate.
            let target_frame_entity =
                sources
                    .get(sw.target_source)
                    .map(|fe| fe.0)
                    .unwrap_or_else(|err| {
                        panic!(
                            "frame_switch_system: body {body_entity:?} switch evaluation failed: \
                         target source {target:?} is not a registered gravity source — \
                         it is missing GravitySourceC and/or FrameEntityC. Spawn it via \
                         PlanetBundle (which inserts both) before referencing it from a \
                         FrameSwitchConfig. Underlying error: {err:?}",
                            target = sw.target_source,
                        )
                    });
            // OnApproach: distance from body to target's frame
            // origin. OnDeparture: body's distance from its current
            // integration frame's origin (i.e. body's
            // `TranslationalStateC.position` magnitude, which equals
            // its FrameTransC position in the current integ frame).
            // Mirrors `astrodyn_runner::evaluate_and_apply_frame_switch`.
            let threshold_sq = sw.switch_distance * sw.switch_distance;
            let triggered = match sw.switch_sense {
                astrodyn::SwitchSense::OnApproach => {
                    let pos_in_target = rel.position(target_frame_entity, body_frame_entity.0);
                    pos_in_target.length_squared() < threshold_sq
                }
                astrodyn::SwitchSense::OnDeparture => {
                    trans.0.position.raw_si().length_squared() > threshold_sq
                }
            };
            if triggered {
                trigger_idx = Some(idx);
                break;
            }
        }

        let Some(idx) = trigger_idx else {
            continue;
        };

        let target_source = switches.0[idx].target_source;
        switches.0[idx].active = false;
        // Re-resolve the target frame entity; lookup proven Some above.
        let new_parent_frame_entity = sources.get(target_source).map(|fe| fe.0).expect(
            "frame_switch_system: target source resolved during evaluation \
             but failed during application — caller-side mutation between lookups",
        );

        // Compute the body's full state expressed in the new target
        // frame's coordinates *before* reparenting. The walk uses the
        // body frame entity's pre-switch `ChildOf` parent (the old
        // integ frame) so the math composes through the existing
        // hierarchy — same algorithm `evaluate_and_apply_frame_switch`
        // runs over the arena (`reparent` then read the post-reparent
        // state).
        let new_state = rel.relative_state(new_parent_frame_entity, body_frame_entity.0);

        // Reparent the body's frame entity under the target source's
        // frame entity, and write the new FrameTransC in the same
        // deferred Commands batch so a post-reparent
        // `RelativeFrameState` walk on the next system flush finds
        // the body in the new parent's coordinates. Without the
        // FrameTransC update, the stored value would still reflect
        // the old parent's frame and downstream consumers would
        // observe a discontinuity-equal-to-(new_origin - old_origin)
        // on this tick. Using `Commands::insert` (rather than a
        // `&mut FrameTransC` query) avoids a static query-conflict
        // with `RelativeFrameState`'s read-only `&FrameTransC`
        // query — the body frame entity's `FrameTransC` is rewritten
        // when the Commands buffer flushes after this system.
        commands
            .entity(body_frame_entity.0)
            .insert(ChildOf(new_parent_frame_entity))
            .insert(FrameTransC {
                position: new_state.trans.position,
                velocity: new_state.trans.velocity,
            });

        // Mirror the new state into the body's TranslationalStateC.
        // Re-wrap as the Component's `PlanetInertial<P>` phantom —
        // `new_state.trans` carries planet-inertial coordinates of the
        // *target* source's planet (this is the post-switch frame)
        // which the same `<P>` parameter tags. The system instantiation
        // for `<P>` is responsible for matching the body's planet
        // identity at the call site (see `register_planet_systems`);
        // each instantiation only matches bodies with `TranslationalStateC<P>`.
        // Same boundary lift `evaluate_and_apply_frame_switch` performs.
        // allowed: frame-switch boundary lift, see comment above
        let pos_typed = astrodyn::Position::<astrodyn::PlanetInertial<P>>::from_raw_si(
            new_state.trans.position,
        );
        // allowed: same frame-switch boundary lift
        let vel_typed = astrodyn::Velocity::<astrodyn::PlanetInertial<P>>::from_raw_si(
            new_state.trans.velocity,
        );
        trans.0.position = pos_typed;
        trans.0.velocity = vel_typed;

        // Flip gravity controls: target source becomes
        // non-differential (central body), all others become
        // differential. Identity match by `Entity` — same convention
        // `evaluate_and_apply_frame_switch` uses.
        for ctrl in &mut gravity_controls.0.controls {
            ctrl.differential = ctrl.source_name != target_source;
        }
        // `IntegSourceC` (the config-time intent) is intentionally
        // untouched — the live truth lives in the body frame
        // entity's `ChildOf` parent.
    }
}

/// Advances translational (and optionally rotational) state by one timestep.
///
/// Delegates to [`astrodyn::integrate_body`] for 6-DOF/3-DOF routing and
/// integration stepping. Gravity is recomputed at each intermediate state
/// for proper multi-stage accuracy.
///
/// The integration method is determined by the optional `IntegratorTypeC`
/// component (RK4, RKF45, GaussJackson, Abm4). When absent, RK4 is used.
/// GaussJackson requires `GaussJacksonStateC`; ABM4 requires `Abm4StateC`.
///
/// Per-body integration-frame origins (relative to root) are queried via
/// the [`FrameOrigin`] SystemParam, which walks the ECS frame hierarchy
/// (`Query<&ChildOf>` on the body's frame entity).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn integration_system<P: Planet>(
    frame_origin: FrameOrigin,
    root_frame_entity: Res<crate::RootFrameEntityR>,
    parents: Query<&ChildOf>,
    // The body query filter excludes two disjoint populations:
    //   * Kinematic-chain children — composite-rigid-body integration
    //     only advances the root of every `MassChildOf` chain.
    //     `wrench_aggregation_system` tags every non-root chain member
    //     with `KinematicChildC`. Without this filter, zeroing a
    //     child's `TotalForceC` would not be enough — the per-RK-stage
    //     gravity recompute below would still drift the child's
    //     translational state every step.
    //   * Detached subtrees — advanced ballistically by
    //     `step_detached_system`. Integrating them here would
    //     double-step the same entity per tick, mirroring the runner
    //     split between `Simulation::bodies` and
    //     `Simulation::detached_subtrees`.
    // See `KinematicChildC` and `DetachedSubtreeStateC` for the
    // detailed lifecycles.
    // JEOD_INV: DB.17 — kinematic children skip integration.
    // JEOD_INV: DB.21 — detached subtrees and frame-attached bodies
    //   skip integration. The frame-attach filter mirrors the runner's
    //   `if body.frame_attach.is_some() { continue; }` guard in
    //   `step::integrate.rs`; bodies attached to a non-body reference
    //   frame have their state derived each tick by
    //   `propagate_frame_attached_state_system` (parent frame's
    //   current state composed with the captured offset) and the
    //   integrator must not stomp the kinematic value with a
    //   force-driven update.
    mut bodies: Query<
        (
            Entity,
            &DynamicsConfigC,
            &mut TranslationalStateC<P>,
            Option<&mut RotationalStateC>,
            Option<&MassPropertiesC>,
            &GravityControlsC,
            &mut TotalForceC,
            Option<&IntegratorTypeC>,
            Option<&mut GaussJacksonStateC>,
            Option<&mut Abm4StateC>,
            Option<&mut FlatPlateConfigC>,
            Option<&StructuralTransformC>,
            Option<&mut RadiationForceC>,
            Option<&mut FrameDerivativesC>,
            Option<&FrameEntityC>,
        ),
        (
            Without<KinematicChildC>,
            Without<crate::DetachedSubtreeStateC>,
            Without<crate::components::FrameAttachedC>,
        ),
    >,
    sources: Query<
        (
            &GravitySourceC,
            Option<&PlanetFixedRotationC<P>>,
            &SourceInertialPositionC,
            Option<&SourceInertialVelocityC>,
            Option<&TidalDeltaC20C>,
            Option<&TidalConfigC>,
        ),
        // Static disjointness vs. the `bodies` query's `&mut
        // TranslationalStateC`: no integrated body is also a gravity
        // source. Without this filter Bevy can't prove the queries
        // don't alias and panics with `assert_component_access_compatibility`.
        Without<DynamicsConfigC>,
    >,
    time: Res<Time<Fixed>>,
    sim_time: Res<SimulationTimeR>,
) {
    let dt = time.delta_secs_f64();
    if dt == 0.0 {
        return;
    }
    // Dynamic timestep matches `astrodyn_runner::run_integration`'s
    // `integ_dt = sim_dt * time_scale_factor` so reversed/scaled time
    // produces consistent gravity at RK sub-stages.
    let integ_dt = dt * sim_time.0.time_scale_factor;

    // Helper closure for gravity at an intermediate state — reused by both
    // the standard and coupled dispatch branches. The integrator passes
    // raw `DVec3` per-stage states (the integrator internals are not
    // yet typed); we wrap into `Position<RootInertial>` / `Velocity<RootInertial>`
    // for the typed `*_typed` kernels and unwrap before returning.
    //
    // `integ_origin_pos` / `integ_origin_vel` are the per-body integration
    // frame's translational state (relative to root) at step start. For
    // root-integrated bodies both are zero — the original behavior. For
    // non-root bodies the integ frame may itself be moving, so each
    // RK sub-stage advances the origin linearly by `time_frac * integ_dt`,
    // matching `astrodyn_runner::run_integration`. Source positions are
    // similarly interpolated when the integ frame moves, so the Newtonian
    // gravity field stays consistent across stages. PPN (relativistic)
    // corrections use step-start source state — runner does the same
    // (`step/integrate.rs:199-202`).
    let eval_gravity = |entity: Entity,
                        controls: &GravityControlsC,
                        pos: DVec3,
                        vel: DVec3,
                        integ_origin_pos: DVec3,
                        integ_origin_vel: DVec3,
                        time_frac: f64|
     -> DVec3 {
        // Per-stage interpolation of the integration frame's origin and
        // each source's position, mirroring astrodyn_runner's pattern in
        // `step/integrate.rs:172-184`. `sub_dt` is gated on the integ
        // frame actually moving so root-integrated bodies stay
        // bit-identical to the pre-N3 path.
        let stage_dt = time_frac * integ_dt;
        let stage_origin_pos = integ_origin_pos + integ_origin_vel * stage_dt;
        let sub_dt = if integ_origin_vel != DVec3::ZERO {
            stage_dt
        } else {
            0.0
        };
        // The standard `integrate_body` (and `integrate_body_coupled`
        // for the thermal-SRP path) accept a `gravity_fn` closure
        // that receives raw `DVec3` per-stage state. These lifts are
        // inside `astrodyn` boundary territory, not at the Bevy ECS
        // surface where the typed quantities live.
        let typed_abs_pos = Position::<RootInertial>::from_raw_si(pos + stage_origin_pos); // allowed: integrator-kernel boundary
        let typed_abs_vel = Velocity::<RootInertial>::from_raw_si(vel + integ_origin_vel); // allowed: integrator-kernel boundary
        let typed_origin = Position::<RootInertial>::from_raw_si(stage_origin_pos); // allowed: integrator-kernel boundary

        // Helper: resolve a source's effective velocity from the
        // typed `SourceInertialVelocityC` (which is
        // `Velocity<RootInertial>` — planet-agnostic). Sources that
        // lack this component coast at zero velocity within the step.
        //
        // `SourceInertialVelocityC` is opt-in: `PlanetBundle`,
        // `SunBundle`, and `MoonBundle` do not insert it, and
        // `ephemeris_update_system` only writes through it when it is
        // already present (it does not auto-insert from
        // `EphemerisBodyC`). Callers who want a moving source for
        // per-stage gravity interpolation or relativistic source
        // resolution must attach `SourceInertialVelocityC` explicitly.
        //
        // No `TranslationalStateC<P>` fallback is offered here. The
        // `<P>` instantiation runs gravity-computation in
        // `PlanetInertial<P>` for the body's planet, and a Sun /
        // ephemeris source's `TranslationalStateC<P>` carries that
        // body-side `<P>` tag (per `SunBundle` / `MoonBundle`'s
        // construction-time convention) — so the velocity it stores
        // is "Sun's velocity tagged as the central planet's inertial
        // frame," which has no well-defined source-motion meaning.
        // Treating the source as stationary when no
        // `SourceInertialVelocityC` is present matches
        // `sync_source_to_frame_system`'s precedence: explicit
        // velocity component first, otherwise treat as no source-
        // motion contribution to the per-step kernel.
        let source_vel = |v: Option<&SourceInertialVelocityC>| -> DVec3 {
            v.map(|v| v.0.raw_si()).unwrap_or(DVec3::ZERO)
        };

        let typed_accel = astrodyn::accumulate_gravity_typed(
            typed_abs_pos,
            &controls.0,
            typed_origin,
            |source_entity| match sources.get(source_entity) {
                Ok((s, r, p, v, tidal, tidal_config)) => {
                    let base_pos = p.0.raw_si();
                    let stage_pos = if sub_dt != 0.0 {
                        base_pos + source_vel(v) * sub_dt
                    } else {
                        base_pos
                    };
                    Some(astrodyn::ResolvedSource {
                        source: &s.0,
                        rotation: r.map(|r| r.0.matrix_ref()),
                        position: stage_pos,
                        delta_c20: tidal.map_or(0.0, |t| t.0.value),
                        has_delta_coeffs: tidal_config.is_some(),
                    })
                }
                Err(_) => {
                    panic!(
                        "Entity {entity:?}: GravityControl references source \
                         {source_entity:?} which does not exist or lacks \
                         GravitySourceC + SourceInertialPositionC."
                    );
                }
            },
        );
        let mut accel = typed_accel.grav_accel.raw_si();

        // PPN (relativistic) corrections use step-start source positions
        // and velocities — `astrodyn_runner::run_integration` snapshots both
        // outside the per-stage closure (`step/integrate.rs:199-202`),
        // so per-stage interpolation here would drift from runner.
        let rel = astrodyn::accumulate_relativistic_corrections_typed(
            typed_abs_pos,
            typed_abs_vel,
            &controls.0,
            |source_entity| {
                sources.get(source_entity).ok().map(|(s, _, p, v, _, _)| {
                    // Step-start values for PPN — runner does the
                    // same (snapshots `src_pos`/`src_vel` outside
                    // the per-stage closure).
                    astrodyn::ResolvedRelativisticSource {
                        mu: s.mu,
                        position: p.0.raw_si(),
                        velocity: source_vel(v),
                    }
                })
            },
        );
        accel += rel.raw_si();

        accel
    };

    for (
        entity,
        config,
        mut state,
        mut rot_state,
        mass,
        controls,
        mut total_force,
        integrator,
        mut gj_state,
        mut abm4_state,
        mut flat_config,
        struct_xform,
        mut srp_force,
        mut frame_derivs,
        body_frame_entity,
    ) in &mut bodies
    {
        // Per-body integration-frame origin (relative to root). Computed
        // once per step — the integ frame doesn't move during a single
        // integration step, so the multi-stage RK4 sub-evaluations
        // reuse the same value.
        //
        // The body's integration frame is the parent of its frame
        // entity in the ECS hierarchy (set at registration by
        // `register_body_frames_system`). Bodies registered before
        // the frames-as-entities components landed have no
        // `FrameEntityC`; treat those as root-integrated, matching
        // the pre-migration default.
        let integ_frame_entity = body_frame_entity
            .and_then(|fe| parents.get(fe.0).ok().map(|child_of| child_of.parent()));
        let (integ_origin_pos, integ_origin_vel) = match integ_frame_entity {
            Some(integ_e) if integ_e != root_frame_entity.0 => {
                frame_origin.origin_in(root_frame_entity.0, integ_e)
            }
            _ => (DVec3::ZERO, DVec3::ZERO),
        };
        let integrator_type = integrator.map_or(astrodyn::IntegratorType::Rk4, |c| c.0);
        if matches!(integrator_type, astrodyn::IntegratorType::GaussJackson(..)) {
            assert!(
                gj_state.is_some(),
                "Entity {entity:?}: IntegratorTypeC is GaussJackson but \
                 GaussJacksonStateC component is missing. Create the state \
                 from the same config used in IntegratorTypeC, e.g.: \
                 GaussJacksonStateC(GaussJacksonState::new(config))"
            );
        }
        if matches!(integrator_type, astrodyn::IntegratorType::Abm4) {
            assert!(
                abm4_state.is_some(),
                "Entity {entity:?}: IntegratorTypeC is Abm4 but \
                 Abm4StateC component is missing. Add \
                 Abm4StateC(Abm4State::new()) to the entity."
            );
        }

        // Derivative-class thermal fork: the SRP system cached step-start
        // inputs into `flat_config.stage_inputs`. Recompute SRP force +
        // temperature derivatives per RK4 stage through
        // `integrate_body_coupled`. See `astrodyn_runner::Simulation::step_internal`
        // for the sister implementation.
        let stage_inputs_and_order = flat_config
            .as_ref()
            .and_then(|fc| fc.stage_inputs.map(|si| (si, fc.integration_order)));
        if let Some((srp_inputs, thermal_order)) = stage_inputs_and_order {
            assert!(
                matches!(integrator_type, astrodyn::IntegratorType::Rk4),
                "Entity {entity:?}: derivative-class ThermalIntegrationOrder \
                 requires RK4 integrator; use Scheduled or switch integrator.",
            );
            let t_struct_body = struct_xform.map_or(glam::DMat3::IDENTITY, |s| *s.0.matrix_ref());
            // Drop typed phantoms at the kernel boundary. `total_force`
            // accumulators are typed (`Force<RootInertial>` / `Torque<BodyFrame>`);
            // the integrator API still consumes raw `DVec3`.
            let non_grav_non_srp_force = total_force.force.raw_si();
            let constant_torque = total_force.torque.raw_si();
            let mut final_srp_inertial_force = DVec3::ZERO;
            let mut final_srp_torque = DVec3::ZERO;
            let mut k1_temp_dots: Option<Vec<f64>> = None;
            // Convert typed state to the untyped form the kernel wants.
            // After `integrate_body_coupled` mutates the untyped copies
            // we re-wrap as typed for storage.
            // allowed: typed↔raw kernel boundary
            let mass_copy_untyped = mass.map(|m| mass_typed_to_raw(&m.0));
            let mut state_untyped = trans_typed_to_raw(&state.0);
            let mut rot_state_untyped = rot_state.as_ref().map(|r| rot_typed_to_raw(&r.0));
            let thermal = flat_config
                .as_mut()
                .expect("stage_inputs_and_order => flat_config present");
            astrodyn::integrate_body_coupled(
                config,
                &mut state_untyped,
                rot_state_untyped.as_mut(),
                mass_copy_untyped.as_ref(),
                |stage_trans, stage_rot, stage_thermal, time_frac| {
                    let gravity_accel = eval_gravity(
                        entity,
                        controls,
                        stage_trans.position,
                        stage_trans.velocity,
                        integ_origin_pos,
                        integ_origin_vel,
                        time_frac,
                    );
                    let t_inertial_body = stage_rot.map_or(glam::DMat3::IDENTITY, |r| {
                        r.quaternion.left_quat_to_transformation()
                    });
                    let t_inertial_struct =
                        astrodyn::compute_t_inertial_struct(&t_struct_body, &t_inertial_body);
                    // Per-stage flux recompute from intermediate vehicle
                    // position — matches JEOD's derivative-class
                    // `RadiationSource::calculate_flux`. Sun position is
                    // step-constant (ephemeris is scheduled-class).
                    //
                    // RF.10: `stage_trans.position` is the integrator's
                    // intermediate `DVec3` in the body's *integration*
                    // frame, which equals root inertial only when the
                    // body's frame entity is a direct child of the
                    // root frame entity. For non-root integration we
                    // shift via the per-stage origin before
                    // differencing against `srp_inputs.sun_position`
                    // (which is typed `Position<RootInertial>`). Mirrors
                    // `astrodyn_runner::run_integration`'s coupled SRP path
                    // (`crates/astrodyn_runner/src/simulation/step/integrate.rs:299-305`).
                    use astrodyn::{Position, RootInertial};
                    let stage_dt = time_frac * integ_dt;
                    let stage_origin = if integ_origin_vel != DVec3::ZERO {
                        integ_origin_pos + integ_origin_vel * stage_dt
                    } else {
                        integ_origin_pos
                    };
                    let stage_pos_root: Position<RootInertial> =
                        // allowed: typed-API boundary — `stage_trans.position`
                        // arrives as the integrator's untyped intermediate
                        // DVec3; `stage_pos_root` is the root-inertial value
                        // after the integ-origin shift, ready for the typed
                        // `srp_inputs.sun_position` subtraction.
                        Position::<RootInertial>::from_raw_si(stage_trans.position + stage_origin);
                    let sun_to_vehicle: Position<RootInertial> =
                        stage_pos_root - srp_inputs.sun_position;
                    let sun_to_vehicle = sun_to_vehicle.raw_si();
                    let distance = sun_to_vehicle.length().max(1.0);
                    let stage_flux_inertial_hat = sun_to_vehicle / distance;
                    let stage_flux_mag = astrodyn::solar_flux_at_distance(distance);
                    let flux_struct_hat = t_inertial_struct * stage_flux_inertial_hat;
                    let srp_result = astrodyn::compute_flat_plate_srp_thermal(
                        &stage_thermal.plates,
                        &stage_thermal.t_pow4_cached,
                        flux_struct_hat,
                        stage_flux_mag,
                        // Drop the typed `Position<StructuralFrame<SelfRef>>`
                        // phantom into the kernel's raw-DVec3 contract; the
                        // typed field is the storage-time guard.
                        srp_inputs.center_grav.raw_si(),
                        srp_inputs.illum_factor,
                    );
                    let srp_force_inertial = t_inertial_struct.transpose() * srp_result.force;
                    final_srp_inertial_force = srp_force_inertial;
                    final_srp_torque = srp_result.torque;
                    let temp_dots = match thermal_order {
                        astrodyn::ThermalIntegrationOrder::DerivativeRk4 => srp_result.temp_dots,
                        astrodyn::ThermalIntegrationOrder::DerivativeFirstOrder => {
                            if time_frac == 0.0 {
                                k1_temp_dots = Some(srp_result.temp_dots.clone());
                                srp_result.temp_dots
                            } else {
                                k1_temp_dots
                                    .as_ref()
                                    .expect("stage 1 runs before stages 2-4")
                                    .clone()
                            }
                        }
                        astrodyn::ThermalIntegrationOrder::Scheduled => {
                            unreachable!("Scheduled bodies do not enter the coupled path")
                        }
                    };
                    // `srp_result.torque` is structural-frame per
                    // `FlatPlateSrpResult` docs; `constant_torque` is
                    // body-frame (from `collect_and_resolve_forces`).
                    // Rotate to body frame before summing so the coupled
                    // integrator's rotational dynamics are correct when
                    // `t_struct_body` != IDENTITY.
                    let srp_torque_body = t_struct_body * srp_result.torque;
                    astrodyn::CoupledStageEval {
                        gravity_accel,
                        non_grav_force: non_grav_non_srp_force + srp_force_inertial,
                        torque: constant_torque + srp_torque_body,
                        temp_dots,
                    }
                },
                &mut thermal.0,
                dt,
                sim_time.0.time_scale_factor,
            );

            // Re-wrap kernel-mutated untyped state back into typed
            // components. The frame phantoms are unchanged (the typed
            // storage's `<PlanetInertial<P>>` / `<BodyFrame<SelfRef>>`
            // are the same frames the kernel was operating in — the
            // kernel computes everything in the body's integration
            // frame, which the Component tags as planet-inertial with
            // the system instantiation's `<P>` parameter that matches
            // this entity by query filter).
            // allowed: typed↔raw kernel boundary writeback (integrate_body_coupled signature is untyped).
            state.0 = trans_raw_to_planet::<P>(&state_untyped);
            if let (Some(rs), Some(ru)) = (rot_state.as_mut(), rot_state_untyped) {
                // allowed: same typed↔raw kernel boundary as above.
                rs.0 = rot_raw_to_self_ref(&ru);
            }

            // Write representative `RadiationForceC` from stage 4 so
            // `VehicleOutput`-equivalent observers still see the SRP force.
            if let Some(ref mut srp_force) = srp_force {
                srp_force.force = final_srp_inertial_force;
                srp_force.torque = final_srp_torque;
            }

            // Backfill `TotalForceC` and `FrameDerivativesC` with the
            // final-stage SRP contribution so downstream observers see
            // SRP-inclusive values, matching the Scheduled-mode invariant
            // that `TotalForceC` / `FrameDerivativesC` reflect every
            // applied force / resulting acceleration. In derivative modes
            // this is a "representative stage" (stage 4) snapshot, same
            // as `RadiationForceC` above.
            // allowed: SRP kernel returns DVec3; re-wrap into the typed
            // accumulators (`Force<RootInertial>` / `Torque<BodyFrame<SelfRef>>`).
            total_force.force += Force::<RootInertial>::from_raw_si(final_srp_inertial_force);
            let final_srp_torque_body = t_struct_body * final_srp_torque;
            // allowed: same SRP-kernel boundary.
            total_force.torque += Torque::<BodyFrame<SelfRef>>::from_raw_si(final_srp_torque_body);
            if let (Some(ref mut fd), Some(mass_p)) = (frame_derivs.as_mut(), mass_copy_untyped) {
                // allowed: typed↔untyped acceleration accumulator boundary.
                fd.trans_accel += Acceleration::<RootInertial>::from_raw_si(
                    final_srp_inertial_force * mass_p.inverse_mass,
                );
                // allowed: typed↔untyped angular-acceleration boundary.
                fd.rot_accel += AngularAcceleration::<BodyFrame<SelfRef>>::from_raw_si(
                    mass_p.inverse_inertia * final_srp_torque_body,
                );
            }
            continue;
        }

        // Standard (Scheduled or no-SRP) path. Same typed↔untyped
        // bridging as the coupled path: extract untyped at entry,
        // re-wrap typed at exit.
        // allowed: typed↔raw kernel boundary
        let mut state_untyped = trans_typed_to_raw(&state.0);
        let mut rot_state_untyped = rot_state.as_ref().map(|r| rot_typed_to_raw(&r.0));
        let mass_untyped = mass.map(|m| mass_typed_to_raw(&m.0));
        astrodyn::integrate_body(
            config,
            &mut state_untyped,
            rot_state_untyped.as_mut(),
            mass_untyped.as_ref(),
            |pos, vel, time_frac| {
                eval_gravity(
                    entity,
                    controls,
                    pos,
                    vel,
                    integ_origin_pos,
                    integ_origin_vel,
                    time_frac,
                )
            },
            total_force.force.raw_si(),
            total_force.torque.raw_si(),
            dt,
            sim_time.0.time_scale_factor,
            integrator_type,
            gj_state.as_mut().map(|g| g.0.inner_mut()),
            abm4_state.as_mut().map(|a| a.0.inner_mut()),
        );
        // Re-wrap kernel-mutated state back into typed components;
        // integrate_body signature is untyped, so re-wrapping is the
        // canonical adapter step (analogous to From<Untyped> impls).
        // allowed: typed↔raw kernel boundary; planet-inertial frame matches the body's integration frame.
        state.0 = trans_raw_to_planet::<P>(&state_untyped);
        if let (Some(rs), Some(ru)) = (rot_state.as_mut(), rot_state_untyped) {
            // allowed: typed↔raw kernel boundary
            rs.0 = rot_raw_to_self_ref(&ru);
        }
    }
}

/// Cross-integration-frame attach metadata captured before the mass
/// tree is mutated.
///
/// Lives at module scope (rather than nested inside `staging_system`) so
/// the [`apply_cross_integ_frame_attach`] helper can take it by
/// reference — the structural extraction the cross-integ-frame block
/// needed to fit on a screen.
///
/// `parent_integ_origin_pos` / `..._vel` are the parent's integ-frame
/// origin in root-inertial coordinates. Zero when the parent is
/// integrated in root (the body-frame entity is `ChildOf(root)`); for
/// any body integrating in `PlanetInertial<P>` the shift is the only
/// thing that keeps the per-descendant state rewrite below from silently
/// mixing coordinates across distinct integration frames. RF.10 shift
/// site, mirrors `mass_tree::attach_inner`'s `body_integ_origins`-based
/// shift.
///
/// `new_parent_frame_entity` is the post-attach integ-frame entity for
/// the child + every kinematic descendant of the child in the mass
/// tree. Per JEOD's `dyn_body_integration.cc::set_integ_frame` (lines
/// 64-117) this reparent recurses into `dyn_children` so all
/// descendants follow the child onto the parent's integ frame.
///
/// `reparent_entries` is the per-entity reparent payload: each entry
/// pairs a body-frame entity with its owning body entity plus the
/// body's pre-attach integ-frame origin in root-inertial coordinates,
/// enough to numerically rewrite the body's `TranslationalStateC` (and
/// the body-frame entity's `FrameTransC`) so the stored coordinates
/// remain consistent with the frame-tree's interpretation after the
/// reparent (per `register_body_frames_system`'s docstring: the body's
/// `TranslationalStateC` is interpreted as already in integ-frame
/// coordinates, where "integ frame" is the body-frame entity's current
/// `ChildOf` parent). Without this rewrite, consumers running between
/// `staging_system` and the next `propagate_state_from_root_system`
/// (the entire `Interaction` set: `aero_drag_system`,
/// `gravity_torque_system`, the SRP systems, plus
/// `force_collection_system`) read the body's pre-attach numerical
/// state through the post-attach frame-tree topology and silently mix
/// coordinates across distinct integ frames.
struct CrossIntegFrameAttachWork {
    parent_integ_origin_pos: glam::DVec3,
    parent_integ_origin_vel: glam::DVec3,
    new_parent_frame_entity: Entity,
    reparent_entries: Vec<CrossIntegReparentEntry>,
}

/// Per-entity payload for the cross-integ-frame reparent loop.
///
/// `body_entity` is the body whose typed `TranslationalStateC` is
/// rewritten; `body_frame_entity` is the matching frame-tree node
/// whose `ChildOf` parent + `FrameTransC` are rewritten. The rewrite
/// is a pure translation between two non-rotating integration frames
/// (see [`astrodyn::CrossIntegFrameStateShift`]).
struct CrossIntegReparentEntry {
    body_entity: Entity,
    body_frame_entity: Entity,
    old_integ_origin_pos: glam::DVec3,
    old_integ_origin_vel: glam::DVec3,
}

/// Bevy-glue helper for the cross-integration-frame attach branch:
/// reparents each affected body-frame entity under the new parent
/// frame and rewrites the matching `FrameTransC` /
/// `TranslationalStateC` payloads from old-integ-frame coordinates
/// into new-integ-frame coordinates.
///
/// All physics math is delegated to
/// [`astrodyn::CrossIntegFrameStateShift`] — this function is a thin
/// I/O wrapper that fans the kernel's pure translation across the
/// per-descendant ECS reads/writes, mirroring JEOD's
/// `dyn_body_integration.cc::set_integ_frame` recursion over
/// `core_body`/`composite_body`/`structure` + `dyn_children`.
///
/// **Why both steps run together** (frame reparent + state rewrite):
/// per `register_body_frames_system`'s docstring the body's
/// `TranslationalStateC` is interpreted as already in integ-frame
/// coordinates, where "integ frame" is the body-frame entity's
/// current `ChildOf` parent. After the reparent the body-frame
/// entity's parent has changed, so the stored numerical value must be
/// shifted by `(old_integ_origin - new_integ_origin)` (root-inertial
/// coordinates) to keep this contract. Skipping the rewrite would let
/// consumers running between `staging_system` and the next
/// `propagate_state_from_root_system` (the entire `AstrodynSet::Interaction`
/// set — `aero_drag_system`, `gravity_torque_system`, the SRP systems —
/// plus `force_collection_system`) read pre-attach numerics through
/// post-attach topology and silently mix coordinates across distinct
/// integ frames. `frame_switch_system` does the symmetric pair
/// (reparent + state rewrite) for its own distance-triggered frame
/// transitions; this is the cross-integ-frame attach analogue.
///
/// **Deferred Commands timing**: the reparent itself is issued through
/// deferred Commands so the post-merge frame tree is consistent on the
/// next system flush. The matching `FrameTransC` rewrite goes through
/// `Commands::insert` in the same call so both land on the same flush
/// boundary. The body's `TranslationalStateC` rewrite goes through the
/// existing `&mut TranslationalStateC` borrow on the `bodies` query.
/// After this same-tick rewrite, `propagate_state_from_root_system`
/// (later this tick) re-derives every kinematic child's
/// `TranslationalStateC` / `RotationalStateC` from the parent's
/// freshly-merged composite-body state composed through the
/// `MassChildOf` link, overwriting the staged value. The intermediate
/// rewrite is what keeps the staging → propagate window
/// arithmetic-correct.
///
/// **Why `FrameTransC` is staged immediately + post-integration
/// re-derived**: the `FrameTransC` write here is load-bearing for the
/// staging → integration window: `staging_system` is ordered
/// `.after(AstrodynSet::Environment).before(AstrodynSet::Interaction)`, so
/// within the attach tick every consumer that reads frame state via
/// `RelativeFrameState` *after* staging — the `AstrodynSet::Interaction`
/// set (drag, SRP, gravity-torque), `force_collection_system` /
/// `wrench_aggregation_system` in `AstrodynSet::ForceCollection`, and
/// `integration_system` in `AstrodynSet::Integration` — sees this value.
/// `AstrodynSet::Environment` already ran for this tick and operated on
/// pre-attach `FrameTransC`; the attach physics applies starting at the
/// next Environment pass (tick N+1). After integration,
/// `sync_body_to_frame_system` overwrites `FrameTransC` from the
/// freshly-updated `TranslationalStateC`, so the late-tick value is
/// re-derived. Both writes carry the same physical pose (the
/// staging-time value comes from the already-rewritten pre-integration
/// state; the post-integration value comes from the integrated state),
/// so the apparent "double write" produces a single consistent
/// trajectory. The Commands / immediate-mutation split is dictated by
/// `frame_states` / `FrameOrigin` already holding a shared read borrow
/// on `FrameTransC`; making the write immediate would require a
/// `ParamSet` split that doesn't pay back its complexity. Bevy 0.18's
/// `auto_insert_apply_deferred` (default-on) flushes this `Commands`
/// batch at the `staging_system → AstrodynSet::Interaction` set boundary,
/// given `staging_system.before(AstrodynSet::Interaction)`, so the deferred
/// write is observed by every post-staging consumer above without a
/// manual `ApplyDeferred`.
///
/// **Rotational state intentionally not rewritten**: every legitimate
/// integ-frame entity is non-rotating (root inertial or
/// `PlanetInertial<P>` — both are inertial and co-aligned with root
/// inertial axes by the frame-tree's construction), so the body's
/// attitude expressed `parent → body` is identical in the old and new
/// integ frames. A rotating integ frame would require an
/// attitude/`ang_vel` rewrite analogous to the position / velocity
/// rewrite below; that case is structurally rejected upstream by the
/// cross-integ-frame fence (every legal integ-frame entity is the root
/// or a registered gravity source, none of which are rotating).
///
/// `parent_entity_skip` is the parent body entity whose
/// `TranslationalStateC` was already overwritten with the merged
/// composite by the immediately-preceding `stage_attach_combine`
/// writeback; the entry list is the *child's* subtree and the parent
/// is never in it, but defensively passing the parent here also
/// guards a future change that lifted the parent into the loop from
/// silently double-counting the shift.
///
/// JEOD_INV: DB.14, JEOD_INV: RF.10, JEOD_INV: RF.11 — child
/// frame-tree reparent following the integ-frame switch + matching
/// numerical state rewrite.
#[allow(clippy::type_complexity)]
fn apply_cross_integ_frame_attach<P: Planet>(
    work: &CrossIntegFrameAttachWork,
    commands: &mut Commands,
    bodies: &mut Query<(
        Entity,
        &crate::MassBodyIdC,
        &mut MassPropertiesC,
        Option<&mut TranslationalStateC<P>>,
        Option<&mut RotationalStateC>,
    )>,
    frame_states: &Query<(&FrameTransC, &FrameRotC, &FrameAngVelC)>,
    parent_entity_skip: Entity,
) {
    for entry in &work.reparent_entries {
        let shift = astrodyn::CrossIntegFrameStateShift::between_integ_origins(
            entry.old_integ_origin_pos,
            entry.old_integ_origin_vel,
            work.parent_integ_origin_pos,
            work.parent_integ_origin_vel,
        );

        // Reparent the body-frame entity and rewrite its
        // `FrameTransC` into the new parent frame's coordinates in
        // the same Commands batch so the post-flush frame tree is
        // internally consistent. The stored `position` / `velocity`
        // are parent-frame-relative per `FrameTransC`'s docstring;
        // switching the parent without rewriting the stored value
        // would produce a discontinuity exactly equal to
        // `(old_origin - new_origin)` on any frame-tree walk that
        // goes through this entity.
        let (frame_trans, _, _) = frame_states
            .get(entry.body_frame_entity)
            .unwrap_or_else(|err| {
                // Defensive fail-loud: a body-frame entity without
                // `FrameTransC` cannot exist in production
                // (`register_body_frames_system` always inserts the
                // triplet), but the query type signature still returns a
                // `Result`. Identity-fallback would corrupt the
                // post-reparent state for any consumer that finds the
                // entity. Mirrors `sync_body_to_frame_system`'s
                // unwrap_or_else panic for the same FrameTransC invariant.
                panic!(
                    "staging_system: cross-integ-frame attach: body-frame entity \
                 {fe:?} has no FrameTransC ({err:?}). Every body-frame entity \
                 must be alive with FrameTransC attached (spawned by \
                 register_body_frames_system).",
                    fe = entry.body_frame_entity,
                )
            });
        let (new_pos, new_vel) = shift.apply(frame_trans.position, frame_trans.velocity);
        commands
            .entity(entry.body_frame_entity)
            .insert(ChildOf(work.new_parent_frame_entity))
            .insert(FrameTransC {
                position: new_pos,
                velocity: new_vel,
            });

        // Rewrite the body's `TranslationalStateC` so the typed
        // integ-frame storage holds the new-frame coordinates. The
        // shift is a pure translation between two co-aligned inertial
        // integ frames, so the post-shift value is still in
        // integration-frame coordinates with the `<PlanetInertial<P>>`
        // tag — bit-identical phantom relabel to the original storage
        // type. The parent's body entity is excluded: the parent's
        // `TranslationalStateC` was already overwritten with the
        // merged composite in `parent_integ_origin`-relative
        // coordinates by the immediately-preceding
        // `stage_attach_combine` writeback, and adding the shift here
        // would double-count it. (`reparent_entries` lists the
        // *child's* subtree; the parent's body-frame entity is the
        // reparent *target*, not a payload — the explicit
        // `parent_entity_skip` guard defends against a future change
        // that lifted the parent into the entries list.)
        if entry.body_entity == parent_entity_skip {
            continue;
        }
        if let Ok((_, _, _, Some(mut t), _)) = bodies.get_mut(entry.body_entity) {
            // allowed: cross-integ-frame numerical rewrite boundary
            let old = trans_typed_to_raw(&t.0);
            let (new_pos, new_vel) = shift.apply(old.position, old.velocity);
            // allowed: same typed↔raw re-wrap pattern as the merged-composite writeback.
            // The shift is a pure translation between two inertial integ frames
            // (co-aligned axes, origins differ in root-inertial), so the
            // post-shift value still lives in integration-frame coordinates
            // with the `<PlanetInertial<P>>` tag.
            t.0 = trans_raw_to_planet::<P>(&TranslationalState {
                position: new_pos,
                velocity: new_vel,
            });
        }
    }
}
/// Process mass-tree attach/detach messages and sync composite properties.
///
/// Runs before interactions so that mass changes from staging are
/// reflected in the current step's interaction forces, force collection,
/// and integration.
///
/// On `AttachEvent` this system:
///
/// 1. snapshots both bodies' pre-attach composite-body inertial state
///    (`TranslationalStateC` + `RotationalStateC`) and pre-attach
///    composite mass properties,
/// 2. mutates the [`crate::MassTreeR`] arena (which recomputes composite
///    mass properties for every affected node),
/// 3. runs [`astrodyn::stage_attach_combine`] (the
///    momentum-conservation port of JEOD's `combine_states_at_attach`,
///    `models/dynamics/dyn_body/src/dyn_body_attach.cc`) to derive the
///    merged composite-body inertial state — preserves linear momentum
///    about the integration-frame origin and angular momentum about
///    the new combined CoM. When the parent and child resolve to
///    different integration-frame entities (post root-equivalence
///    fold), each body's pre-attach state is lifted to root inertial
///    via its own `IntegOrigin` before the kernel call so the
///    cross-body composition arithmetic operates on a single inertial
///    frame, and the merged composite is lowered back through the
///    parent's integ origin for the writeback,
/// 4. writes the merged state back into the parent entity's
///    [`crate::TranslationalStateC`] / [`crate::RotationalStateC`],
/// 5. for the cross-integration-frame case, reparents the child's
///    body-frame entity (and every kinematic descendant of the child
///    in the mass tree) under the parent's integ-frame entity via
///    `commands.entity(...).insert(ChildOf(...))`, mirroring JEOD's
///    `dyn_body_attach.cc::attach_establish_links` →
///    `dyn_body_integration.cc::set_integ_frame` recursion. JEOD's
///    `set_integ_frame` itself "does not update state"
///    (`dyn_body_integration.cc:85-86`) — JEOD relies on the
///    immediately-following `propagate_state()` call inside
///    `attach_update_properties` to refill descendants' parent-relative
///    storage from the merged root. Our adapter has no equivalent
///    same-call propagation: the next tick's
///    [`propagate_state_from_root_system`](crate::propagate_state_from_root_system)
///    walk runs many systems later, and the
///    `TranslationalStateC`-is-already-integ-frame-relative contract
///    (`register_body_frames_system`) would otherwise leave every
///    descendant's stored numerics inconsistent with the freshly
///    reparented frame-tree topology for the staging → propagate
///    window. Each reparented descendant's `TranslationalStateC` and
///    body-frame `FrameTransC` are therefore shifted in-place by
///    `(old_integ_origin - new_integ_origin)` (root-inertial
///    coordinates) during this same staging tick — same physical
///    pose, just relabeled into the new integration frame's
///    coordinates. `frame_switch_system` does the symmetric
///    pair (reparent + state rewrite) for its own distance-triggered
///    switches; this is the cross-integ-frame attach analogue,
/// 6. removes [`crate::DetachedSubtreeStateC`] from the child entity if
///    it was previously detached (the captured ballistic state is now
///    consumed by the combine).
///
/// On `DetachEvent` this system:
///
/// 1. captures the about-to-be-detached subtree's instantaneous
///    composite-body inertial state via
///    [`astrodyn::stage_detach_capture`],
/// 2. mutates the arena (which recomputes the former parent's composite
///    mass to reflect the lost subtree),
/// 3. inserts [`crate::DetachedSubtreeStateC`] on the detached entity
///    so [`step_detached_system`] can advance the subtree ballistically
///    each tick.
///
/// Both branches end with the IG.37 mark + reset for any body whose
/// composite mass changed — multi-step integrators (GJ, ABM4) must
/// drop their predictor history on topology change.
///
/// Note: [`crate::MassTreeR`] must be present as a resource for attach/detach
/// messages to have any effect.
///
/// # Example
/// ```
/// use bevy::prelude::*;
/// use astrodyn_bevy::DetachEvent;
///
/// // A user-defined system that emits a DetachEvent for a known booster
/// // entity (e.g. one cached in a Resource).
/// #[derive(Resource)]
/// struct Booster(Entity);
///
/// fn detach_booster(
///     booster: Res<Booster>,
///     mut detach_messages: bevy::ecs::message::MessageWriter<DetachEvent>,
/// ) {
///     detach_messages.write(DetachEvent { child: booster.0 });
/// }
///
/// let mut app = App::new();
/// app.add_message::<DetachEvent>();
/// app.add_systems(Update, detach_booster);
/// ```
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn staging_system<P: Planet>(
    mut commands: Commands,
    tree: Option<ResMut<crate::MassTreeR>>,
    // JEOD_INV: TS.01 — Bevy adapter message-bus storage boundary: the
    // canonical runtime-resolved `AttachEvent<SelfRef, SelfRef>` reader
    // pair with the registration in `AstrodynPlugin::build`.
    mut attach_events: bevy::ecs::message::MessageReader<
        crate::AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>,
    >,
    mut detach_events: bevy::ecs::message::MessageReader<crate::DetachEvent>,
    mut bodies: Query<(
        Entity,
        &crate::MassBodyIdC,
        &mut MassPropertiesC,
        Option<&mut TranslationalStateC<P>>,
        Option<&mut RotationalStateC>,
    )>,
    body_frames: Query<&FrameEntityC>,
    parents: Query<&ChildOf>,
    detached_q: Query<Entity, With<crate::DetachedSubtreeStateC>>,
    // Per-body component presence used by the cross-integ-frame fence
    // to tell apart three distinct "FrameEntityC absent / present"
    // populations:
    //
    //   * **Mass-only attach participant** — entity carries
    //     `MassBodyIdC` + `MassPropertiesC` but lacks at least one of
    //     `DynamicsConfigC` / `TranslationalStateC`. Registration will
    //     never visit it, so `FrameEntityC` will never be inserted.
    //     Legitimate `MassBody`-without-`DynBody` configuration; the
    //     fence has no frame node to protect for it.
    //
    //   * **Registration-race** — entity carries both eligibility
    //     components (`DynamicsConfigC` + `TranslationalStateC`, the
    //     filter for `register_body_frames_system`) but lacks
    //     `FrameEntityC`. `register_body_frames_system` has not yet run
    //     this tick (deferred `Commands` flush ordering). Letting the
    //     attach proceed would silently corrupt the frame tree on the
    //     next register pass. Per Fail Loudly this must panic.
    //
    //   * **Partially-stripped dynamic body** — entity carries
    //     `FrameEntityC` (registration ran) but at least one of
    //     `DynamicsConfigC` / `TranslationalStateC` /
    //     `RotationalStateC` has been removed since. Reading state for
    //     the kernel would silently substitute zero/identity, and the
    //     combine-back-write below conditionally writes the merged
    //     composite only if those components are present — so the
    //     merged state would be silently dropped. Per Fail Loudly the
    //     fence must surface this as well; matches JEOD's
    //     `attach_validate_child` rejecting "Child body has an
    //     incomplete state" (`dyn_body_attach.cc:131-135`).
    eligibility: Query<(
        Has<DynamicsConfigC>,
        Has<TranslationalStateC<P>>,
        Has<RotationalStateC>,
    )>,
    // Frame-state query needed by `is_root_equivalent_entity` so the
    // cross-integ-frame fence below treats Earth.inertial-as-root-
    // equivalent topology (a direct child of root with identity state)
    // as semantically root.
    frame_states: Query<(&FrameTransC, &FrameRotC, &FrameAngVelC)>,
    // Registered source frame entities. Used to verify that a body's
    // resolved live integ-frame entity is a *legal* integ-frame entity
    // (root or a registered source frame), matching the same fence
    // `frame_switch_system` enforces. Without this an attach with both
    // bodies misparented under the same arbitrary frame would otherwise
    // be silently accepted as "same integration frame".
    source_frames: Query<&FrameEntityC, With<GravitySourceC>>,
    mut integrators: Query<(
        &crate::MassBodyIdC,
        Option<&mut GaussJacksonStateC>,
        Option<&mut Abm4StateC>,
    )>,
    // `frame_origin` performs the per-body root-inertial lift required by
    // the cross-integration-frame attach path. The merge kernel composes
    // parent and child state through `omega × r` and
    // `T_inertial_struct.transpose()` shifts — both arithmetic-valid only
    // when both bodies' translational state lives in the same inertial
    // frame. The pre-attach states are lifted to root inertial via each
    // body's `IntegOrigin` (mirrors the runner's `mass_tree::attach_inner`
    // RF.10 shift site), the kernel runs in root coordinates, and the
    // merged result is lowered through the parent's `IntegOrigin` for
    // writeback into `TranslationalStateC`'s integration-frame storage.
    // The lift is identically zero only for root-integrated bodies; for
    // any body integrating in a non-root `PlanetInertial<P>` the lift is
    // non-zero. The same-integ-frame case is a no-op in *physical* terms
    // (parent and child share an `IntegOrigin`, so the post-lift kernel
    // arithmetic matches what the pre-lift integ-frame arithmetic would
    // have produced — the per-body shifts cancel in any inter-body
    // difference the kernel forms), but `frame_origin` is still consulted
    // for both bodies in that case rather than short-circuited.
    frame_origin: FrameOrigin,
    root_frame_entity: Option<Res<crate::RootFrameEntityR>>,
) {
    // No mass tree resource → drain events and return.
    let Some(mut tree) = tree else {
        attach_events.clear();
        detach_events.clear();
        return;
    };

    // The set of mass-tree node ids whose composite mass changes due
    // to the events processed below — i.e. whose multi-step integrator
    // state must be marked topology-dirty (Site A) and later reset
    // (Site B). We accumulate it INLINE with each event-handler branch
    // so the dirty-marking is structurally bound to the topology
    // mutation call site, then mark in one query pass, then reset in
    // a separate observation pass. Splitting Site A and Site B is the
    // structural fix for IG.37 fail-loud (see JEOD_invariants.md): a
    // future code path that adds a new event branch and forgets the
    // reset pass will leave the dirty flag set, so the next
    // `integrate()` panics with the IG.37 diagnostic rather than
    // silently propagating stale predictor history.
    let mut affected_ids: Vec<astrodyn::MassBodyId> = Vec::new();

    // Per-attach work item: captures the pre-attach snapshot needed by
    // `combine_states_at_attach` plus the post-mutation parent entity
    // we'll write the merged composite-body state into. Built before
    // the topology mutation so the snapshot is independent of the
    // tree's post-attach state.
    //
    // `parent_position`/`parent_velocity` and `child_position`/
    // `child_velocity` are stored in **root-inertial** coordinates —
    // each side has been lifted through its own body's `IntegOrigin`
    // at capture time. The combine kernel composes states across
    // bodies (mass-weighted velocity, inertial-frame CoM shift,
    // ω×r over offsets), which is only arithmetic-valid when both
    // sides live in the same inertial frame. Storing the lifted
    // values keeps that invariant explicit; for root-integrated
    // bodies the lift is `IntegOrigin::zero()` and the captured
    // values are bit-identical to the raw `TranslationalStateC`
    // contents. Mirrors the runner's `attach`/`detach` snapshot
    // shape in `astrodyn_runner::Simulation`.
    //
    // `parent_integ_origin_pos`/`parent_integ_origin_vel` are the
    // parent's integ-origin in root-inertial, retained so the
    // writeback below can lower the merged result back into
    // integ-frame storage (`TranslationalStateC` is integ-frame).
    struct AttachWork {
        parent_entity: Entity,
        // The body whose tree is being re-rooted under `parent`. Equals
        // `child_entity` for the simple root-subject case; for
        // chained-attach (subject already has a parent in the mass
        // tree) this is the subject's existing tree-root entity, which
        // is what actually gets reparented under the new parent. The
        // child-side snapshot fields below are read from this entity,
        // not `child_entity`, so the combine kernel sees the subject
        // root's integrator-written composite-body state.
        subject_root_entity: Entity,
        subject_root_id: astrodyn::MassBodyId,
        // True iff this AttachEvent triggers a re-rooting reparent —
        // i.e. `subject_root_entity != child_entity`. Drives the
        // `MassChildOf` insert on the subject root and the
        // post-mutation auto-flag of the rerooted subtree.
        is_chained_attach: bool,
        parent_id: astrodyn::MassBodyId,
        // Pre-attach snapshot for the kernel — lifted to root-inertial.
        parent_position: glam::DVec3,
        parent_velocity: glam::DVec3,
        parent_quaternion: astrodyn::JeodQuat,
        parent_ang_vel_body: glam::DVec3,
        parent_mass: astrodyn::MassProperties,
        orig_parent_cm_struct: glam::DVec3,
        parent_t_inertial_struct: glam::DMat3,
        child_position: glam::DVec3,
        child_velocity: glam::DVec3,
        child_quaternion: astrodyn::JeodQuat,
        child_ang_vel_body: glam::DVec3,
        child_mass: astrodyn::MassProperties,
        // Parent's integ-origin in root-inertial (the displacement
        // from root to the body's integration frame). Used to lower
        // the merged result back into the parent's integ-frame
        // storage at the `TranslationalStateC` writeback. Identity
        // for root-integrated parents; load-bearing for non-root.
        parent_integ_origin_pos: glam::DVec3,
        parent_integ_origin_vel: glam::DVec3,
        // Was the child carrying a `DetachedSubtreeStateC` immediately
        // before this attach? If so the entry is consumed and removed.
        child_was_detached: bool,
        // Was the parent carrying a `DetachedSubtreeStateC` immediately
        // before this attach? If so the parent is still a free-flying
        // tree root post-attach (no integrated ancestor); its tracked
        // ballistic state must be replaced with the merged composite-
        // body state so `step_detached_system` continues advancing the
        // correct value next tick (rather than overwriting the merged
        // state with the stale pre-attach `DetachedSubtreeStateC`).
        parent_was_detached: bool,
        // Cross-integration-frame attach metadata. `Some` when the
        // parent and child resolve to different integ-frame entities
        // (post root-equivalence fold). When set, the merge kernel
        // is run in root-inertial coordinates: the pre-attach states
        // are lifted through `parent_integ_origin_pos/vel` and
        // `child_integ_origin_pos/vel` on input, and the merged
        // composite is lowered back through the parent's integ
        // origin on writeback so the parent's `TranslationalStateC`
        // continues to store integration-frame coordinates. This
        // mirrors `astrodyn_runner::Simulation::attach_inner`'s lift /
        // lower around `combine_states_at_attach`. JEOD source
        // reference: `dyn_body_attach.cc::attach_establish_links` →
        // `dyn_body_integration.cc::set_integ_frame`.
        // [`CrossIntegFrameAttachWork`] is captured before the mass
        // tree is mutated so the integ-origin lifts at the kernel
        // boundary observe the *pre-attach* origins (the post-attach
        // root is the parent's root, so the lower step uses the
        // parent's origin, but the seed-time lifts use the per-body
        // pre-attach origins). The struct lives at module scope so
        // [`apply_cross_integ_frame_attach`] can take it by reference
        // — physics math (the per-descendant translation between two
        // co-aligned inertial integ frames) is delegated to
        // [`astrodyn::CrossIntegFrameStateShift`].
        cross_integ: Option<CrossIntegFrameAttachWork>,
    }

    let mut attach_work: Vec<AttachWork> = Vec::new();
    // Per-detach work: captured pre-detach composite-body state to be
    // attached to the detached entity as `DetachedSubtreeStateC` once
    // the topology mutation is done.
    let mut detach_work: Vec<(Entity, astrodyn::DetachedSubtreeState)> = Vec::new();

    // The set of registered source frame entities is invariant across
    // the entire `staging_system` call — collect it once here rather
    // than once per AttachEvent. Mirrors the optimization in
    // `frame_switch_system` (see lines 737-744) so a tick that drains
    // a batch of attaches doesn't pay the rebuild cost N times.
    let known_source_frames: std::collections::HashSet<Entity> =
        source_frames.iter().map(|fe| fe.0).collect();

    // One-shot mass-body-id → entity map built from a single pass over
    // the `bodies` query. The cross-integ-frame attach branch
    // (descendant subtree walk) and the detach handler both need
    // id-keyed entity lookups; without a shared map each event would
    // re-scan `bodies` once per descendant, giving an `O(subtree_size
    // × body_count)` cost per attach and an `O(body_count)` cost per
    // detach inside the per-event loops. Building the map once amortizes
    // both into a single `O(body_count)` scan plus `O(1)` membership
    // tests, matching the runner's `id_to_entity`-style indexed
    // lookups.
    let id_to_entity: std::collections::HashMap<astrodyn::MassBodyId, Entity> = bodies
        .iter()
        .map(|(e, body_id, _, _, _)| (body_id.0, e))
        .collect();

    for evt in attach_events.read() {
        // Look up child + parent. Fail-loud per CLAUDE.md if either
        // entity is not a mass-tree body.
        let (_, child_body_id, _child_mass_c, _child_trans, _child_rot) =
            bodies.get(evt.child).unwrap_or_else(|_| {
                panic!(
                    "AttachEvent.child = {:?} is not a mass body — entity is missing MassBodyIdC \
                 and/or MassPropertiesC. Spawn the body via the mass-tree API before attaching.",
                    evt.child
                )
            });
        let child_id = child_body_id.0;

        // Resolve the **subject root** — the body whose tree is actually
        // being re-rooted under `parent`. For the simple root-subject
        // case (subject is itself a tree root, no existing parent edge
        // in the mass tree), `subject_root_id == child_id` and the
        // snapshot below reads state from the child entity directly.
        // For the chained-attach case (subject already has a parent in
        // the mass tree) the subject's existing tree root is what
        // actually moves under `parent`; the kernel
        // `MassTree::attach_with_reroot` calls
        // `attach(child_root, parent_id, ...)` with a recomputed offset
        // and rotation so the subject's struct frame still ends up
        // where the caller asked, even though the underlying tree edge
        // runs from `parent` to the subject's existing root. Mirrors
        // JEOD `DynBody::attach_child` (`dyn_body_attach.cc:521`):
        //   `child_root = child.get_root_body_internal();`
        // and the subsequent `child_root->attach_establish_links(*this)`
        // branch when `child_root != &child`.
        //
        // The runner-side dispatch in
        // `astrodyn_runner::Simulation::attach_inner` resolves a separate
        // `subject_root_idx` for the same reason — the
        // momentum-conservation combine kernel needs the subject root's
        // integrator-written composite-body state, not the immediate
        // child's (which is stale or zero for an interior body).
        // JEOD_INV: BA.12 — chained attach re-roots the subject's existing tree under the new parent.
        let subject_root_id = tree.root_of(child_id);
        let is_chained_attach = subject_root_id != child_id;
        let subject_root_entity = if is_chained_attach {
            *id_to_entity.get(&subject_root_id).unwrap_or_else(|| {
                panic!(
                    "AttachEvent.child = {:?} (mass id {:?}): subject root \
                     {subject_root_id:?} for the rerooted subtree has no entity in the bodies \
                     query — every mass-tree node must be spawned with `MassBodyIdC` before \
                     any AttachEvent references it. The chained-attach reroot path needs to \
                     read the subject root's composite-body state for the momentum-conservation \
                     combine and reparent its `MassChildOf` Relations under the new parent; \
                     missing the entity here would silently break either step.",
                    evt.child, child_id,
                )
            })
        } else {
            evt.child
        };

        // Read the subject root's state (== child's state in the
        // root-subject case). The combine kernel reads this as the
        // child-side of the merge; for the reroot case the subject root
        // carries the integrated state of the whole pre-reroot subtree,
        // which is the right composite to merge with the parent-side
        // root (mirrors `astrodyn_runner::Simulation::attach_inner`).
        let (_, subject_root_body_id, subject_root_mass_c, subject_root_trans, subject_root_rot) =
            bodies.get(subject_root_entity).unwrap_or_else(|_| {
                panic!(
                    "AttachEvent.child = {:?}: subject root entity {subject_root_entity:?} \
                     is not a mass body — entity is missing MassBodyIdC and/or \
                     MassPropertiesC. The mass tree's `root_of(child_id)` returned a node \
                     whose backing entity does not satisfy the bodies query.",
                    evt.child,
                )
            });
        // Sanity: the root walker's id and the entity's MassBodyIdC must agree.
        assert_eq!(
            subject_root_body_id.0, subject_root_id,
            "subject root entity {:?} carries MassBodyIdC({:?}) but the mass tree's \
             root_of(child_id={:?}) returned {:?}. The id_to_entity map and the arena \
             tree are out of sync — every entity carrying MassBodyIdC must be reachable \
             via id_to_entity at the same id.",
            subject_root_entity, subject_root_body_id.0, child_id, subject_root_id,
        );
        // allowed: typed↔raw kernel boundary
        let child_mass: astrodyn::MassProperties = mass_typed_to_raw(&subject_root_mass_c.0);
        let (child_position_integ, child_velocity_integ) = subject_root_trans
            .as_ref()
            .map(|t| (t.0.position.raw_si(), t.0.velocity.raw_si()))
            .unwrap_or((glam::DVec3::ZERO, glam::DVec3::ZERO));
        let (child_quaternion, child_ang_vel_body) = subject_root_rot
            .as_ref()
            .map(|r| {
                // allowed: typed↔raw kernel boundary
                let untyped = rot_typed_to_raw(&r.0);
                (untyped.quaternion, untyped.ang_vel_body)
            })
            .unwrap_or((astrodyn::JeodQuat::identity(), glam::DVec3::ZERO));

        let (_, parent_body_id, parent_mass_c, parent_trans, parent_rot) =
            bodies.get(evt.parent).unwrap_or_else(|_| {
                panic!(
                    "AttachEvent.parent = {:?} is not a mass body — entity is missing MassBodyIdC \
                 and/or MassPropertiesC. Spawn the parent via the mass-tree API before attaching.",
                    evt.parent
                )
            });
        let parent_id = parent_body_id.0;
        // allowed: typed↔raw kernel boundary
        let parent_mass: astrodyn::MassProperties = mass_typed_to_raw(&parent_mass_c.0);
        let (parent_position_integ, parent_velocity_integ) = parent_trans
            .as_ref()
            .map(|t| (t.0.position.raw_si(), t.0.velocity.raw_si()))
            .unwrap_or((glam::DVec3::ZERO, glam::DVec3::ZERO));
        let (parent_quaternion, parent_ang_vel_body) = parent_rot
            .as_ref()
            .map(|r| {
                // allowed: typed↔raw kernel boundary
                let untyped = rot_typed_to_raw(&r.0);
                (untyped.quaternion, untyped.ang_vel_body)
            })
            .unwrap_or((astrodyn::JeodQuat::identity(), glam::DVec3::ZERO));

        // Cross-integration-frame attach is not yet supported for
        // bodies that participate in the frame tree. The unsupported
        // piece is the frame-entity reparenting + coordinate rewrite,
        // *not* the multi-step integrator reset (the IG.37 reset for
        // every affected body still runs later in this function via
        // the `affected_ids` walk) and *not* the mass-tree composite
        // recomputation (which is frame-agnostic and runs
        // unconditionally below). JEOD's
        // `dyn_body_attach.cc::attach_establish_links` calls
        // `set_integ_frame(*(dyn_parent->get_integ_frame()))` whenever
        // the child's integ frame differs from the parent's. JEOD's
        // `dyn_body_integration.cc::set_integ_frame` (lines 64-117)
        // reparents the child's `core_body`/`composite_body`/
        // `structure` frames + every registered vehicle point under
        // the new integ frame via low-level
        // `RefFrame::reset_parent()` calls; the in-source comment
        // "NOTE WELL: This uses the low-level reset_parent(). It does
        // not update state." makes explicit that the stored numbers
        // are NOT rewritten. Later propagation reinterprets the
        // existing coordinates against the new parent. Our staging
        // path performs neither the frame-entity `ChildOf` reparent
        // nor the coordinate rewrite, so allowing the merge to
        // proceed silently corrupts every downstream
        // `RelativeFrameState` walk. Per the Fail Loudly rule
        // (CLAUDE.md), surface the misconfiguration at the point of
        // detection rather than producing a wrong trajectory.
        //
        // The live integ-frame for each body is the `ChildOf` parent
        // of its body-frame entity, NOT the body's `IntegSourceC`
        // value. `frame_switch_system` mutates the body-frame
        // entity's `ChildOf` parent on each switch but intentionally
        // leaves `IntegSourceC` (the config-time intent) untouched —
        // comparing `IntegSourceC` would both miss real cross-frame
        // attaches (root-started body that switched to Moon: still
        // `None`) and falsely reject same-frame attaches (a body
        // switched into the parent's frame: stale `IntegSourceC`
        // differs from parent's).
        //
        // The fence has four semantic layers, applied in order so
        // legality is decided on the *original* parent rather than
        // its root-equivalent fold (otherwise an arbitrary entity
        // that happens to be a direct child of root with identity
        // state would silently fold to the root and pass the
        // legality check):
        //
        //   1. **Resolve the live integ-frame entity.** Bodies that
        //      participate in the frame tree carry `FrameEntityC`
        //      (inserted by `register_body_frames_system` for
        //      entities with both `DynamicsConfigC` and
        //      `TranslationalStateC`) and that frame entity must
        //      have a `ChildOf` parent. Mass-only attach
        //      participants (entities carrying only `MassBodyIdC` +
        //      `MassPropertiesC`, matching JEOD's
        //      `MassBody`-without-`DynBody` configuration that
        //      `AttachEvent`'s contract permits) have no
        //      `FrameEntityC` and therefore no frame tree to corrupt
        //      — the fence skips them. If `FrameEntityC` *is*
        //      present but the `ChildOf` is missing the frame tree
        //      itself is corrupt and the attach cannot be safely
        //      processed — panic per Fail Loudly.
        //
        //   1.5. **State-completeness on dynamic participants.** A
        //      body that resolved a frame entity must carry the
        //      full state-component set (`DynamicsConfigC` +
        //      `TranslationalStateC` + `RotationalStateC`). The
        //      kernel snaps any missing input to zero/identity and
        //      the combine-back-write is conditional on the same
        //      components — without them the merged state is
        //      silently dropped. Matches JEOD's
        //      `attach_validate_child` rejecting partial state with
        //      "<role> body has an incomplete state".
        //
        //   2. **Verify the live parent is a legal integ-frame
        //      entity.** Anything that is not the root frame entity
        //      and not a registered gravity source's frame entity
        //      (e.g. both bodies misparented under another body's
        //      frame entity by a buggy mission script) is rejected
        //      here, *before* root-equivalence folding. This matches
        //      `frame_switch_system`'s same-tick check at lines
        //      765-781 so the same misconfig is caught at attach
        //      time rather than only later when a switch evaluates.
        //      Falls back to comparing against `known_source_frames`
        //      when `RootFrameEntityR` is absent (low-level tests
        //      that drive `staging_system` directly without
        //      `AstrodynPlugin`).
        //
        //   3. **Normalize root-equivalent topology for equality.**
        //      In `astrodyn_runner` the central body's inertial frame
        //      *is* the root frame. The Bevy adapter instead
        //      registers every gravity source — including the
        //      central body — one level below a generic root, so
        //      `IntegSourceC(Some(earth))` lands the body's frame
        //      entity under `earth.inertial` (a direct child of root
        //      with identity state). Folding root-equivalent parents
        //      onto root for the equality comparison means an
        //      Earth-centered body and a root-integrated body
        //      ("`IntegSourceC(None)`") count as the same integ
        //      frame. Folding *only* drives the equality check —
        //      legality has already been decided in step 2 against
        //      the un-folded parent.
        //
        // The fail-loud structural and state-completeness checks below
        // (steps 1, 1.5, and 2) run unconditionally — they protect
        // invariants that hold without any reference to root-equivalence
        // semantics, so a low-level test (or a partial app build) that
        // drove `staging_system` directly without `AstrodynPlugin` still
        // sees the same misconfigurations rejected. Only the
        // root-equivalence equality fold (step 3) requires
        // `RootFrameEntityR` and is therefore conditional on its
        // presence; without root the equality fold is skipped, but no
        // production path reaches that branch with `RootFrameEntityR`
        // absent (`AstrodynPlugin::build` always inserts it).
        //
        // Skipped per-event when *both* bodies lack `FrameEntityC`
        // and neither is dynamic — see step 1's narrowed mass-only
        // carve-out. A missing `FrameEntityC` on a body that *would*
        // qualify for `register_body_frames_system` (carries both
        // `DynamicsConfigC` and `TranslationalStateC`) is a
        // registration race, not a mass-only configuration, and is
        // rejected fail-loud below.

        // Step 1: resolve the original `ChildOf` parent of each
        // body's frame entity (no folding yet). Returns `None`
        // only when the entity is intentionally mass-only — its
        // component set fails `register_body_frames_system`'s
        // eligibility filter, so `FrameEntityC` will never be
        // inserted and the entity has no node in the frame tree.
        //
        // An entity that *passes* the eligibility filter
        // (`DynamicsConfigC` + `TranslationalStateC`) but still
        // lacks `FrameEntityC` is a registration race — the body
        // was spawned mid-tick after `register_body_frames_system`
        // already ran, so its frame-tree node does not yet exist
        // even though the rest of the world expects one. Letting
        // the attach proceed would silently corrupt the frame
        // tree on the next register pass; per Fail Loudly we
        // panic with a diagnostic that names the broken
        // assumption (entity has the eligibility components but
        // ran the staging fence before registration).
        let resolve_original_parent = |body: Entity, role: &str| -> Option<Entity> {
            match body_frames.get(body) {
                Ok(frame_handle) => {
                    let child_of = parents.get(frame_handle.0).unwrap_or_else(|err| {
                        panic!(
                            "AttachEvent.{role} = {body:?}: body-frame entity \
                                 {fe:?} has no ChildOf parent. The body-frame entity \
                                 must be parented under its integration-frame entity \
                                 (root frame entity, or a registered source's frame \
                                 entity). `register_body_frames_system` inserts that \
                                 ChildOf when it runs in the AstrodynPlugin schedules \
                                 (Startup, PreUpdate, FixedUpdate); a missing parent \
                                 here means the frame tree is corrupt. Underlying \
                                 query error: {err:?}",
                            fe = frame_handle.0,
                        )
                    });
                    Some(child_of.parent())
                }
                Err(_) => {
                    // No `FrameEntityC`. Distinguish the two
                    // populations: mass-only (carve-out) vs
                    // registration race (fail-loud).
                    let (has_dyn_cfg, has_trans, _has_rot) =
                        eligibility.get(body).unwrap_or((false, false, false));
                    if has_dyn_cfg && has_trans {
                        panic!(
                            "AttachEvent.{role} = {body:?}: entity carries \
                             DynamicsConfigC + TranslationalStateC (the \
                             eligibility filter for register_body_frames_system) \
                             but does not yet carry FrameEntityC. This is a \
                             registration race — the body was spawned mid-tick \
                             after register_body_frames_system already ran in \
                             PreUpdate / FixedUpdate (before \
                             AstrodynSet::EphemerisUpdate), so its frame-tree node \
                             has not been spawned yet by the time staging_system \
                             runs. Spawn the body before the first FixedUpdate \
                             step (e.g. in Startup or PreUpdate ahead of \
                             register_body_frames_system), or defer the \
                             AttachEvent until the next tick so the registration \
                             pass has had a chance to run."
                        );
                    }
                    None
                }
            }
        };

        // Mass-only attach carve-out: both bodies must carry
        // `FrameEntityC` for the fence to apply. If either side
        // has no frame node (legitimate `MassBody`-without-
        // `DynBody` configuration permitted by `AttachEvent`'s
        // contract), the frame tree has no node to corrupt and
        // the equality / legality checks below have nothing to
        // enforce. The mass-tree composite recompute and IG.37
        // integrator reset still run unconditionally outside
        // this branch.
        //
        // One asymmetric case is rejected fail-loud: a dynamic
        // child (with `FrameEntityC`) attaching to a mass-only
        // parent (no `FrameEntityC`). JEOD's
        // `dyn_body_attach.cc::attach_validate_parent` rejects
        // this with "Dynamic attachments can only be made to
        // valid DynBodies" — and our combine-back-write below
        // only writes the merged composite into the parent's
        // `TranslationalStateC` / `RotationalStateC`, which a
        // mass-only parent does not carry. Without this guard
        // the merged state is silently dropped. The dual case
        // (mass-only child on dynamic parent) matches JEOD's
        // legitimate `add_mass_body` path and is allowed.
        let parent_orig = resolve_original_parent(evt.parent, "parent");
        let child_orig = resolve_original_parent(evt.child, "child");
        if parent_orig.is_none() && child_orig.is_some() {
            panic!(
                "AttachEvent: dynamic child {child:?} (carries FrameEntityC) \
                 cannot be attached to mass-only parent {parent:?} (no \
                 FrameEntityC). JEOD's dyn_body_attach.cc::attach_validate_parent \
                 rejects this with \"Dynamic attachments can only be made to \
                 valid DynBodies\" (Modified_data parents need both \
                 DynamicsConfigC and TranslationalStateC). The combine-back-write \
                 in this function only writes the merged composite into the \
                 parent's TranslationalStateC / RotationalStateC, which a \
                 mass-only parent does not carry — the merged state would be \
                 silently lost. Either promote the parent to a dynamic body \
                 (add DynamicsConfigC + TranslationalStateC + RotationalStateC) \
                 before the attach, or attach the parent to its own dynamic \
                 ancestor first so the composite has a free-flying root.",
                child = evt.child,
                parent = evt.parent,
            );
        }

        // Step 1.5: state-completeness for any body that *did*
        // resolve a frame entity. The kernel reads
        // `parent_position` / `parent_velocity` /
        // `parent_quaternion` / `parent_ang_vel_body` (and the
        // child analogs) from `TranslationalStateC` /
        // `RotationalStateC`, falling back to zero / identity when
        // those components are absent — and the combine-back-write
        // below only writes the merged composite back when the
        // same components are present. A body that carries
        // `FrameEntityC` (registration ran) but has had
        // `DynamicsConfigC` / `TranslationalStateC` removed since
        // is therefore in a miscomputing-attach state: missing
        // inputs silently snap to zero, and any merged result is
        // silently dropped. JEOD's `attach_validate_child`
        // (`dyn_body_attach.cc:121-180`) rejects partial state
        // with "Child body has an incomplete state" / "Root body
        // has an incomplete state"; we surface the same
        // misconfiguration here at the staging fence so the bug is
        // caught at the event boundary rather than silently
        // corrupting downstream state.
        //
        // `RotationalStateC` is required only when the attach
        // partner also has it: the bevy adapter supports a 3-DOF
        // configuration (`DynamicsConfigC` + `TranslationalStateC`
        // without `RotationalStateC`) — `register_body_frames_system`'s
        // filter mirrors this — and an attach between two 3-DOF
        // bodies merges translational state only, leaving rotation
        // identity on both sides consistently. The dangerous case
        // is *asymmetric* rotation: one body 6-DOF, the other 3-DOF,
        // where the 3-DOF side's rotation snaps to identity and the
        // merged attitude / angular momentum is silently wrong. We
        // reject the asymmetric case below.
        let parent_has_state =
            parent_orig.map(|_| eligibility.get(evt.parent).unwrap_or((false, false, false)));
        let child_has_state =
            child_orig.map(|_| eligibility.get(evt.child).unwrap_or((false, false, false)));
        let rotational_asymmetry = match (parent_has_state, child_has_state) {
            (Some((_, _, parent_rot)), Some((_, _, child_rot))) => parent_rot != child_rot,
            _ => false,
        };
        for (entity, orig, role) in [
            (evt.parent, parent_orig, "parent"),
            (evt.child, child_orig, "child"),
        ] {
            if orig.is_none() {
                continue;
            }
            let (has_dyn_cfg, has_trans, has_rot) =
                eligibility.get(entity).unwrap_or((false, false, false));
            let mut missing: Vec<&'static str> = Vec::new();
            if !has_dyn_cfg {
                missing.push("DynamicsConfigC");
            }
            if !has_trans {
                missing.push("TranslationalStateC");
            }
            if rotational_asymmetry && !has_rot {
                missing.push("RotationalStateC");
            }
            if !missing.is_empty() {
                let missing = missing.join(", ");
                panic!(
                    "AttachEvent.{role} = {entity:?}: dynamic body carries \
                     FrameEntityC (registration ran) but is missing required \
                     state component(s): {missing}. The stage_attach_combine \
                     kernel reads TranslationalStateC / RotationalStateC for \
                     pre-attach pose + velocity, and the merged composite is \
                     written back only into those same components — without \
                     them the kernel silently substitutes zero / identity for \
                     the missing input and the merged result is silently \
                     dropped. JEOD's dyn_body_attach.cc::attach_validate_child \
                     rejects this same case with \"Child body has an \
                     incomplete state\" / \"Root body has an incomplete state\". \
                     Re-insert the missing component(s) on the entity before \
                     firing the AttachEvent, or remove the body from the \
                     mass-tree before stripping its state. (Note: \
                     RotationalStateC is required only when the attach \
                     partner carries it — pure 3-DOF attach between two \
                     bodies that both lack RotationalStateC is allowed.)"
                );
            }
        }

        // Cross-integration-frame attach metadata, computed below
        // when both bodies resolved frame entities and the post-fold
        // parents differ. Stays `None` for the same-integ-frame case
        // (the common one) so the writeback loop bypasses the lift /
        // lower / reparent code paths bit-identically.
        let mut cross_integ: Option<CrossIntegFrameAttachWork> = None;

        if let (Some(parent_orig), Some(child_orig)) = (parent_orig, child_orig) {
            // Step 2: legality is decided against the *original*
            // parent — never the root-equivalent fold. An arbitrary
            // entity that happens to satisfy root-equivalence (direct
            // child of root with identity state) but is not itself a
            // registered source frame must still be rejected, because
            // `frame_switch_system` will reject the same parent on the
            // very next tick. Match its legality check at lines
            // 765-781.
            //
            // The legality predicate uses `known_source_frames`
            // (built without `RootFrameEntityR`) directly, plus an
            // optional equality with the root entity when the
            // resource is present — so when `RootFrameEntityR` is
            // absent the check still rejects illegal parents (those
            // not under any registered source) instead of silently
            // bypassing.
            let root_e_opt = root_frame_entity.as_ref().map(|r| r.0);
            for (entity, integ_frame, role) in [
                (evt.parent, parent_orig, "parent"),
                (evt.child, child_orig, "child"),
            ] {
                let is_root = root_e_opt == Some(integ_frame);
                let is_legal = is_root || known_source_frames.contains(&integ_frame);
                assert!(
                    is_legal,
                    "AttachEvent.{role} = {entity:?}: live integration-frame \
                     entity {integ_frame:?} (the ChildOf parent of the body's \
                     frame entity) is neither the root frame entity \
                     ({root_e_opt:?}) nor a registered gravity source's frame \
                     entity. The body-frame entity must be parented under one \
                     of those — register the source via PlanetBundle (which \
                     inserts GravitySourceC and FrameEntityC) before spawning \
                     the body, or attach the body under the root frame entity."
                );
            }

            // Step 3: fold root-equivalent topology *only* for the
            // equality comparison below. Both `parent_orig` and
            // `child_orig` are now known to be legal integ frames, so
            // any fold to `root_e` happens on a registered source
            // (typically the central body's `*.inertial` frame).
            //
            // The fold (and the equality check it drives) requires
            // the root entity to be known. When `RootFrameEntityR`
            // is absent (low-level tests / partial app builds), we
            // skip just this final equality — the structural
            // fail-loud checks above have already run unconditionally,
            // and production paths always set the resource via
            // `AstrodynPlugin::build`.
            if let Some(root_e) = root_e_opt {
                let fold_root_equivalent = |parent: Entity| -> Entity {
                    if crate::validation::is_root_equivalent_entity(
                        parent,
                        root_e,
                        &parents,
                        &frame_states,
                    ) {
                        root_e
                    } else {
                        parent
                    }
                };
                let parent_frame = fold_root_equivalent(parent_orig);
                let child_frame = fold_root_equivalent(child_orig);

                if parent_frame != child_frame {
                    // Cross-integration-frame attach. Mirrors JEOD's
                    // `dyn_body_attach.cc::attach_establish_links`
                    // calling `set_integ_frame(*(dyn_parent->get_integ_frame()))`
                    // when the child's integ frame differs from the
                    // parent's. The merged body will integrate in the
                    // parent's frame post-attach — the runner's
                    // `mass_tree::attach_inner` keeps the same
                    // post-attach invariant (the merged composite is
                    // written back into the integrated tree root).
                    //
                    // Compute each body's pre-attach integ-frame
                    // origin in root-inertial coordinates. The lift
                    // is identically zero for any body whose folded
                    // integ frame is root (`parent_frame == root_e` /
                    // `child_frame == root_e`), so for the asymmetric
                    // case "parent in root + child in
                    // PlanetInertial<P>" only the child carries a
                    // non-zero shift; for the symmetric case "parent
                    // in PlanetInertial<P> + child in PlanetInertial<Q>"
                    // both lifts are non-zero and distinct. Note that
                    // `parent_frame`/`child_frame` are the *folded*
                    // values (root-equivalent topology mapped onto
                    // `root_e`); the unfolded `parent_orig` /
                    // `child_orig` may name a registered source frame
                    // that is itself root-equivalent.
                    let resolve_integ_origin = |frame: Entity| -> (glam::DVec3, glam::DVec3) {
                        if frame == root_e {
                            (glam::DVec3::ZERO, glam::DVec3::ZERO)
                        } else {
                            let (p, v) = frame_origin.origin_in_root(root_e, frame);
                            (p.raw_si(), v.raw_si())
                        }
                    };
                    let (parent_integ_origin_pos, parent_integ_origin_vel) =
                        resolve_integ_origin(parent_frame);

                    // Walk the child's mass-tree subtree (inclusive)
                    // and resolve a body-frame entity for each
                    // descendant. Mirrors JEOD's `set_integ_frame`
                    // recursing into `dyn_children`. Mass-only
                    // descendants (no `FrameEntityC`) are skipped —
                    // they have no frame-tree node to reparent.
                    //
                    // The walk uses the *pre-attach* mass tree: the
                    // child has not been linked to the parent yet, so
                    // walking from `child_id` collects the original
                    // child subtree (i.e. every body that was a
                    // descendant of the child before the attach
                    // mutation).
                    //
                    // Per descendant we also capture the body's old
                    // integ-frame origin in root-inertial coordinates
                    // (the body-frame entity's current `ChildOf`
                    // parent's frame state). The numerical rewrite at
                    // the reparent site uses this to shift the body's
                    // stored `TranslationalStateC` from old-frame to
                    // new-frame coordinates without arithmetically
                    // mixing the two; without it, consumers in the
                    // staging → propagate window read pre-attach
                    // numerics through post-attach topology.
                    let mut reparent_entries: Vec<CrossIntegReparentEntry> = Vec::new();
                    let mut subtree: Vec<astrodyn::MassBodyId> = vec![child_id];
                    let mut idx = 0;
                    while idx < subtree.len() {
                        let id = subtree[idx];
                        for child in tree.children(id) {
                            subtree.push(*child);
                        }
                        idx += 1;
                    }
                    for id in subtree {
                        // O(1) id → entity lookup via the shared map
                        // built once at the top of `staging_system`,
                        // mirroring the runner's `id_to_entity`. A
                        // linear `bodies.iter()` scan per id would be
                        // O(subtree_size × body_count) per attach.
                        if let Some(&entity) = id_to_entity.get(&id) {
                            // A descendant can legitimately lack a
                            // body-frame entity *only* when it is a
                            // pure mass-only node — i.e. it has no
                            // `DynamicsConfigC` / `TranslationalStateC`
                            // and therefore no integ-frame
                            // interpretation to keep in sync with the
                            // post-attach topology. A descendant that
                            // *does* carry those components but is
                            // missing `FrameEntityC` is the same
                            // registration-race state we already
                            // fail-loud on for the attach participants
                            // (lines above, mirroring
                            // `attach_validate_child`). Skipping it
                            // would silently leave that body's stored
                            // `TranslationalStateC` interpreted under
                            // the pre-attach integ frame for every
                            // staging→propagate consumer in the same
                            // tick, so surface the misconfiguration
                            // here rather than letting the stale
                            // numerics propagate.
                            let fe = body_frames.get(entity).ok();
                            let body_frame_entity = match fe {
                                Some(fe) => fe.0,
                                None => {
                                    let (has_dyn_cfg, has_trans, _has_rot) =
                                        eligibility.get(entity).unwrap_or((false, false, false));
                                    assert!(
                                        !(has_dyn_cfg && has_trans),
                                        "staging_system: cross-integ-frame attach: \
                                         descendant body {entity:?} carries DynamicsConfigC \
                                         and TranslationalStateC (dynamic body) but has no \
                                         FrameEntityC — registration race vs \
                                         register_body_frames_system. The cross-integ-frame \
                                         reparent would otherwise leave this descendant's \
                                         stored TranslationalStateC interpreted under the \
                                         pre-attach integ frame while every other body in \
                                         the subtree gets shifted into the new frame, \
                                         silently mixing coordinates across distinct \
                                         integration frames. Spawn the body with a \
                                         registered IntegSourceC (or under the root frame) \
                                         and ensure register_body_frames_system has run \
                                         before firing the AttachEvent."
                                    );
                                    continue;
                                }
                            };
                            // Resolve this descendant's pre-attach
                            // integ-frame origin from its
                            // body-frame entity's current `ChildOf`
                            // parent (the live integ-frame source
                            // of truth, same fold rule used above
                            // for the child/parent equality check).
                            let descendant_integ_frame = parents
                                .get(body_frame_entity)
                                .unwrap_or_else(|err| {
                                    panic!(
                                        "staging_system: cross-integ-frame attach: \
                                         descendant body {entity:?} body-frame entity \
                                         {body_frame_entity:?} has no ChildOf parent \
                                         ({err:?}). Every body-frame entity must be \
                                         parented under its integration frame entity \
                                         (set by register_body_frames_system)."
                                    )
                                })
                                .parent();
                            let descendant_integ_frame_folded =
                                if crate::validation::is_root_equivalent_entity(
                                    descendant_integ_frame,
                                    root_e,
                                    &parents,
                                    &frame_states,
                                ) {
                                    root_e
                                } else {
                                    descendant_integ_frame
                                };
                            let (old_pos, old_vel) = if descendant_integ_frame_folded == root_e {
                                (glam::DVec3::ZERO, glam::DVec3::ZERO)
                            } else {
                                let (p, v) =
                                    frame_origin.origin_in_root(root_e, descendant_integ_frame);
                                (p.raw_si(), v.raw_si())
                            };
                            reparent_entries.push(CrossIntegReparentEntry {
                                body_entity: entity,
                                body_frame_entity,
                                old_integ_origin_pos: old_pos,
                                old_integ_origin_vel: old_vel,
                            });
                        }
                    }

                    // Resolve the new parent frame entity: in the
                    // root-equivalent case (parent's integ-frame
                    // entity folded onto `root_e` for the equality
                    // check above), the actual reparent target must
                    // be the *unfolded* parent frame entity — the
                    // child's body-frame entity becomes
                    // `ChildOf(parent_orig)`, mirroring exactly what
                    // `register_body_frames_system` would have done
                    // for a body spawned with the parent's
                    // `IntegSourceC`. Reparenting onto the folded
                    // root entity directly would bypass the central
                    // body's frame entity and silently break any
                    // consumer that walks the body's `ChildOf` chain
                    // expecting a registered source.
                    let new_parent_frame_entity = parent_orig;

                    cross_integ = Some(CrossIntegFrameAttachWork {
                        parent_integ_origin_pos,
                        parent_integ_origin_vel,
                        new_parent_frame_entity,
                        reparent_entries,
                    });
                }
            }
        }

        // Lift each body's translational state from its own
        // integration frame to root-inertial before feeding the
        // combine kernel. `TranslationalStateC` is stored in the
        // body's `IntegrationFrame` (planet-relative for a non-root
        // integ source), but `combine_states_at_attach` does
        // cross-body composition (mass-weighted velocity, inertial
        // CoM shift, ω×r) which is only arithmetic-valid when both
        // sides live in the same inertial frame. Add each body's
        // `IntegOrigin` (its integ-frame origin in root-inertial) to
        // get root-inertial coordinates. For root-integrated bodies
        // the origin is identically zero so the lift is a numerical
        // no-op; for two bodies that integrate in distinct
        // `PlanetInertial<P>` frames (or one in root + one in a
        // planet) the lift is the only thing that prevents the
        // kernel from silently mixing coordinates across distinct
        // origins. Mirrors the runner's seed-time lift in
        // `astrodyn_runner::Simulation::attach`.
        //
        // JEOD_INV: RF.10 — `body.trans` is typed
        // `TranslationalStateTyped<IntegrationFrame>`; the only
        // safe transition to `RootInertial` is the integ-origin
        // shift, and the combine kernel is a root-inertial-shift
        // consumer.
        let parent_body_frame_capture = body_frames.get(evt.parent).ok();
        let (parent_integ_origin_pos_typed, parent_integ_origin_vel_typed) =
            body_integ_origin_in_root_lazy(
                parent_body_frame_capture,
                &parents,
                root_frame_entity.as_deref().map(|r| r.0),
                &frame_origin,
            );
        let parent_integ_origin_pos = parent_integ_origin_pos_typed.raw_si();
        let parent_integ_origin_vel = parent_integ_origin_vel_typed.raw_si();
        // Read the subject root's integ-frame origin (== child's origin
        // in the root-subject case) — the subject root carries the
        // integrated state of the whole pre-reroot subtree, so its own
        // integ-frame origin is the right shift to lift its trans state
        // to root-inertial for the combine kernel.
        let child_body_frame_capture = body_frames.get(subject_root_entity).ok();
        let (child_integ_origin_pos_typed, child_integ_origin_vel_typed) =
            body_integ_origin_in_root_lazy(
                child_body_frame_capture,
                &parents,
                root_frame_entity.as_deref().map(|r| r.0),
                &frame_origin,
            );
        let child_integ_origin_pos = child_integ_origin_pos_typed.raw_si();
        let child_integ_origin_vel = child_integ_origin_vel_typed.raw_si();
        let parent_position = parent_position_integ + parent_integ_origin_pos;
        let parent_velocity = parent_velocity_integ + parent_integ_origin_vel;
        let child_position = child_position_integ + child_integ_origin_pos;
        let child_velocity = child_velocity_integ + child_integ_origin_vel;

        // T_inertial_to_struct = T_struct_to_body^T · T_inertial_to_body
        // Per JEOD `dyn_body_collect.cc:219-221` and
        // `astrodyn_dynamics::compute_t_inertial_struct` — the kernel needs
        // this to rotate the structure-frame CoM-shift vector
        // (`combined.position - orig_parent_cm_struct`) into the
        // inertial frame for the parent's post-attach position.
        let parent_t_struct_to_body = parent_mass.t_parent_this;
        let parent_t_inertial_to_body = parent_quaternion.left_quat_to_transformation();
        let parent_t_inertial_struct = astrodyn::compute_t_inertial_struct(
            &parent_t_struct_to_body,
            &parent_t_inertial_to_body,
        );

        // The bodies whose composite mass changes are the subject root
        // (== child for the simple case, the subject's existing tree
        // root for the chained-attach case) plus every ancestor of the
        // new parent in the pre-attach tree
        // (`MassTree::recompute_composites` walks the entire forest
        // post-order, so any ancestor of the new parent is touched).
        // For the chained-attach reroot path the subject's *whole*
        // pre-reroot subtree (`subject_root` + every descendant) ends
        // up under `parent`, and `recompute_composites` walks both the
        // new combined tree and the old tree's former root chain — the
        // conservative set is therefore subject_root + descendants +
        // parent's ancestor chain. Including the descendants matters
        // because, post-reroot, those bodies are kinematic children of
        // the merged tree's root and any GJ/ABM4 history they carried
        // (they were interior nodes in the subject tree and may have
        // been integrated standalone before the original attach) is
        // now stale topology-wise. Mirrors the runner-side dispatch in
        // `astrodyn_runner::Simulation::attach_inner`.
        // JEOD_INV: IG.37 — multi-step integrator history must be reset on topology change
        affected_ids.push(subject_root_id);
        affected_ids.extend(tree.ancestors_inclusive(parent_id));
        if is_chained_attach {
            // Reject cross-integ-frame attach while chained: the
            // cross-integ-frame branch above resolves a single child
            // entity (and its descendants) for the frame-entity
            // reparent walk, but the actual rerooted subtree is the
            // subject root + its descendants — a different entity set.
            // Plumbing the cross-integ-frame numerical rewrite through
            // the rerooted subtree is its own surgery (chained reroot
            // changes which bodies need the lift/lower), and JEOD's
            // `dyn_body_attach.cc::attach_child` reroot path does not
            // exercise cross-integ-frame attaches in any of the
            // verification sims we currently port. Surface the
            // unsupported combination loudly here rather than letting
            // the lift/lower walk silently leave coordinates in the
            // wrong frame.
            assert!(
                cross_integ.is_none(),
                "AttachEvent: chained-attach (re-rooting) combined with cross-integration-frame \
                 attach is not yet supported. The subject root {subject_root_entity:?} (mass id \
                 {subject_root_id:?}) is being reparented under {:?} but the parent and child \
                 also live in different integration frames. Either align the two bodies' \
                 integration frames before the chained attach (e.g. via a frame switch), or \
                 detach the subject from its current parent before the cross-integ-frame attach \
                 so the simple root-subject path runs.",
                evt.parent,
            );
            affected_ids.extend(tree.subtree_ids(subject_root_id));
        }

        // `child_was_detached` / `parent_was_detached` track whether
        // the two integrated tree roots about to be merged were
        // free-flying ballistic subtrees pre-attach (carrying
        // `DetachedSubtreeStateC`). For the simple root-subject case
        // the child IS the subject root, so reading from
        // `subject_root_entity` is bit-identical to reading from
        // `evt.child`; for the chained-attach case the subject's
        // existing tree root is the integrated free-flier (the
        // immediate child is interior to its own subtree and never
        // carries `DetachedSubtreeStateC` by the same DB.21 contract).
        let child_was_detached = detached_q.contains(subject_root_entity);
        let parent_was_detached = detached_q.contains(evt.parent);

        attach_work.push(AttachWork {
            parent_entity: evt.parent,
            subject_root_entity,
            subject_root_id,
            is_chained_attach,
            parent_id,
            parent_position,
            parent_velocity,
            parent_quaternion,
            parent_ang_vel_body,
            parent_mass,
            orig_parent_cm_struct: parent_mass.position,
            parent_t_inertial_struct,
            child_position,
            child_velocity,
            child_quaternion,
            child_ang_vel_body,
            child_mass,
            parent_integ_origin_pos,
            parent_integ_origin_vel,
            child_was_detached,
            parent_was_detached,
            cross_integ,
        });

        // `tree.attach_with_reroot` takes raw structural-frame DVec3;
        // drop the typed phantom at this kernel boundary. The typed
        // `AttachEvent.offset` field guards the structural-frame
        // contract at the writer site. The reroot-aware kernel handles
        // both the simple root-subject case (bit-identical to plain
        // `attach`) and the chained-attach case (recomputes geometry
        // and reparents the subject's existing tree root under
        // `parent_id`). Mirrors JEOD `dyn_body_attach.cc:521-567`'s
        // `attach_child` path.
        // JEOD_INV: BA.12 — Bevy adapter dispatches every attach through
        // the reroot-aware kernel so chained-attach scenarios pick the
        // JEOD `dyn_body_attach.cc:521-567` path automatically.
        let _attached_root = tree.attach_with_reroot(
            child_id,
            parent_id,
            evt.offset.raw_si(),
            evt.t_parent_child.matrix(),
        );
    }

    // Per-detach post-mutation work: tree_root entity whose
    // `TranslationalStateC` / `RotationalStateC` (and possibly
    // `DetachedSubtreeStateC`) must be shifted by the inertial-frame
    // composite-CoM delta after the topology change, since the parent's
    // composite-CoM moves within its own struct frame when the subtree
    // leaves. Mirrors the runner's `detach_subtree` parent-side update.
    struct ParentShift {
        tree_root_entity: Entity,
        parent_pre_position: glam::DVec3,
        parent_pre_velocity: glam::DVec3,
        parent_pre_quat: astrodyn::JeodQuat,
        parent_pre_ang_vel_body: glam::DVec3,
        parent_pre_composite_props: astrodyn::MassProperties,
        parent_was_detached: bool,
    }
    let mut parent_shifts: Vec<(astrodyn::MassBodyId, ParentShift)> = Vec::new();

    // The mass-body-id → entity map built at the top of this system
    // (above the attach loop) is reused here. Mirrors the runner's
    // `detach_subtree` which indexes `self.bodies` by id directly.

    for evt in detach_events.read() {
        let (_, child_body_id, _, _, _) = bodies.get(evt.child).unwrap_or_else(|_| {
            panic!(
                "DetachEvent.child = {:?} is not a mass body — entity is missing MassBodyIdC \
                 and/or MassPropertiesC.",
                evt.child
            )
        });
        let child_id = child_body_id.0;

        // Walk up to the current tree root. The runner's
        // `detach_subtree` does this same walk; the parent's composite-
        // body inertial state lives at the root (only the integrated /
        // free-flying root carries the merged composite — attached
        // children's `TranslationalStateC` is stale post-attach, since
        // post-attach state is propagated down from the root by
        // `propagate_state_from_root_system` rather than re-merged at
        // each child).
        let mut tree_root_id = child_id;
        while let Some(p) = tree.parent(tree_root_id) {
            tree_root_id = p;
        }
        if tree_root_id == child_id {
            // Detaching a body that has no parent in the mass tree is
            // a misconfiguration: the rigid-body subtree is already
            // free-flying with respect to every other tree, so there
            // is no parent composite to derive child state from.
            panic!(
                "DetachEvent.child = {:?} (mass id {:?}) has no parent in the mass tree — \
                 detaching a tree root is a no-op in JEOD and indicates a stale event \
                 (e.g. firing DetachEvent twice without a re-AttachEvent in between).",
                evt.child, child_id,
            );
        }

        let tree_root_entity = *id_to_entity.get(&tree_root_id).unwrap_or_else(|| {
            panic!(
                "DetachEvent.child = {:?}: tree root {:?} has no entity in the bodies query — \
                 every mass-tree node must be spawned with `MassBodyIdC` before any \
                 attach/detach event references it.",
                evt.child, tree_root_id,
            )
        });

        // Pre-mutation snapshot of the parent's composite-body inertial
        // state (read from the root entity, which is the only place
        // post-attach that carries the merged composite — attached
        // children's `TranslationalStateC` is stale, populated by
        // root-down propagation rather than re-merged in place).
        // Keeping these as raw f64 fields (not borrowing the query)
        // avoids holding a borrow across the `bodies.iter()` /
        // `bodies.get_mut` calls below.
        //
        // `parent_pre_composite_props` is read from the legacy
        // `MassTreeR` arena rather than the entity's
        // `MassPropertiesC` because the ECS-tree fast path in
        // `composite_mass_system` reverts `MassPropertiesC` to its
        // `CoreMassPropertiesC` cache for any entity that has no
        // `MassChildOf` edge, and the arena attach/detach path
        // exercised here never adds those edges. Without this
        // arena-read, by the time the detach handler runs (in the
        // same tick, after `composite_mass_system`), reading the
        // entity's `MassPropertiesC` component would yield the
        // just-reverted *core* mass instead of the live post-attach
        // composite — and the CoM-shift formula below would key off
        // `composite_properties.position == core.position` (typically
        // zero), corrupting the parent's post-detach inertial position.
        // The arena tree is the same source of truth the runner reads
        // in `Simulation::detach_subtree`, so this also keeps the two
        // adapters bit-identical for the parent-side post-detach
        // CoM-shift. Mirrors `astrodyn_runner::Simulation::detach_subtree`'s
        // `tree.get(tree_root_id).composite_properties` access.
        //
        // JEOD_INV: MA.23 — composite-property reads at detach must
        // see the live (pre-detach) composite, not a downstream
        // cache; the `MassTree` arena is the canonical store.
        let (
            parent_pre_position,
            parent_pre_velocity,
            parent_pre_quat,
            parent_pre_ang_vel_body,
            parent_pre_composite_props,
        ) = {
            let (_, _, _, parent_trans, parent_rot) = bodies
                .get(tree_root_entity)
                .expect("id_to_entity points at a valid mass body");
            let position = parent_trans
                .as_ref()
                .map(|t| t.0.position.raw_si())
                .unwrap_or(glam::DVec3::ZERO);
            let velocity = parent_trans
                .as_ref()
                .map(|t| t.0.velocity.raw_si())
                .unwrap_or(glam::DVec3::ZERO);
            let (q, w) = parent_rot
                .as_ref()
                .map(|r| {
                    // allowed: typed↔raw kernel boundary
                    let u = rot_typed_to_raw(&r.0);
                    (u.quaternion, u.ang_vel_body)
                })
                .unwrap_or((astrodyn::JeodQuat::identity(), glam::DVec3::ZERO));
            let composite = tree.get(tree_root_id).composite_properties;
            (position, velocity, q, w, composite)
        };

        // Walk root → subtree applying `propagate_forward` at each
        // level using the mass-tree's `composite_wrt_pstr` offsets.
        // This is the JEOD-faithful derivation of the subtree's
        // instantaneous composite-body inertial state at the detach
        // instant — i.e. the rigid-body composition of the parent's
        // composite-body state plus the subtree's offset within the
        // composite. Runner does the same in `detach_subtree`.
        let mut chain: Vec<astrodyn::MassBodyId> = Vec::new();
        let mut walker = child_id;
        while walker != tree_root_id {
            chain.push(walker);
            walker = tree
                .parent(walker)
                .expect("chain walk hit a parentless intermediate before reaching tree root");
        }
        chain.reverse();

        // Lift the parent's `TranslationalStateC` from its integration
        // frame to root-inertial before walking the rigid-body offset
        // chain. The storage convention pins `TranslationalStateC` to
        // the body's integration frame, so for a parent integrated in a
        // non-root `PlanetInertial<P>` the raw position/velocity are
        // planet-relative; running `propagate_forward` on planet-relative
        // coords would seed the walk in integ-frame and produce a
        // subtree state that lives in the same integ-frame, while the
        // captured `DetachedSubtreeState` is typed `Position/Velocity<
        // RootInertial>` and propagated as such by `step_ballistic`.
        // The runner mirrors this exact lift in
        // `crates/astrodyn_runner/src/simulation/mass_tree.rs:583-585`
        // (root-pre-state) and feeds the inertial seed to the same
        // `derive_subtree_composite_state` walk. Identity for root-
        // integrated parents (`integ_origin == zero`); load-bearing for
        // non-root.
        // JEOD_INV: RF.10 — root-inertial-shift consumer: the kernel
        // walks rigid-body composition in root-inertial coordinates.
        let parent_body_frame = body_frames.get(tree_root_entity).ok();
        let (parent_integ_origin_pos, parent_integ_origin_vel) = body_integ_origin_in_root_lazy(
            parent_body_frame,
            &parents,
            root_frame_entity.as_deref().map(|r| r.0),
            &frame_origin,
        );
        let parent_pre_position_inertial = parent_pre_position + parent_integ_origin_pos.raw_si();
        let parent_pre_velocity_inertial = parent_pre_velocity + parent_integ_origin_vel.raw_si();
        let parent_composite_state = astrodyn::RefFrameState {
            trans: astrodyn::RefFrameTrans {
                position: parent_pre_position_inertial,
                velocity: parent_pre_velocity_inertial,
            },
            rot: astrodyn::RefFrameRot {
                q_parent_this: parent_pre_quat,
                t_parent_this: parent_pre_quat.left_quat_to_transformation(),
                ang_vel_this: parent_pre_ang_vel_body,
            },
        };
        let mut current_state = parent_composite_state;
        let mut current_node_id = tree_root_id;
        for next_id in &chain {
            let next_node = tree.get(*next_id);
            let current_node = tree.get(current_node_id);
            // Body-aware step (matches runner's detach walk):
            //   offset_in_current_body = T_current_struct_to_body
            //                          · (next.composite_wrt_pstr.position
            //                             − current.composite_properties.position)
            //   T_current_body_to_next_body = T_next_struct_to_body
            //                               · next.structure_point.t_parent_this
            //                               · T_current_body_to_struct
            let t_current_struct_to_body = current_node.composite_properties.t_parent_this;
            let t_next_struct_to_body = next_node.composite_properties.t_parent_this;
            let offset_struct =
                next_node.composite_wrt_pstr.position - current_node.composite_properties.position;
            let offset_in_current_body = t_current_struct_to_body * offset_struct;
            let t_current_body_to_next_body = t_next_struct_to_body
                * next_node.structure_point.t_parent_this
                * t_current_struct_to_body.transpose();
            let rel = astrodyn::MassPointState {
                position: offset_in_current_body,
                t_parent_this: t_current_body_to_next_body,
            };
            current_state = astrodyn::propagate_forward(&current_state, &rel);
            current_node_id = *next_id;
        }
        let subtree_state = current_state;

        let captured = astrodyn::stage_detach_capture(
            subtree_state.trans.position,
            subtree_state.trans.velocity,
            subtree_state.rot.q_parent_this,
            subtree_state.rot.ang_vel_this,
        );
        detach_work.push((evt.child, captured));

        // Stash the parent-side post-mutation update for later (after
        // tree.detach + composite mass sync). The CoM-shift uses the
        // pre/post composite properties — the post is read after
        // mutation so we record only the pre-state here.
        let parent_was_detached_root = detached_q.contains(tree_root_entity);
        parent_shifts.push((
            tree_root_id,
            ParentShift {
                tree_root_entity,
                parent_pre_position,
                parent_pre_velocity,
                parent_pre_quat,
                parent_pre_ang_vel_body,
                parent_pre_composite_props,
                parent_was_detached: parent_was_detached_root,
            },
        ));

        // Bodies whose composite changes: the (about-to-be-detached)
        // child plus the former parent's full ancestor chain. Capture
        // BEFORE mutating the tree.
        affected_ids.push(child_id);
        affected_ids.extend(tree.ancestors_inclusive(tree_root_id));
        tree.detach(child_id);
    }

    if affected_ids.is_empty() && attach_work.is_empty() && detach_work.is_empty() {
        return;
    }
    affected_ids.sort_unstable();
    affected_ids.dedup();

    // Sync composite mass properties for all affected nodes.
    //
    // These writes go through `bypass_change_detection` because the value being
    // written is the *composite* (post-Steiner) mass, not a core-mass
    // edit by mission code. The `composite_mass_system` ECS path uses
    // `Changed<MassPropertiesC>` to detect mid-sim core edits (fuel
    // burn, propellant offload) and refresh its hidden
    // [`crate::mass_tree::CoreMassPropertiesC`] cache. If the legacy
    // arena `staging_system` write tripped that filter, the next tick
    // the ECS path would seed `CoreMassPropertiesC` from a *composite*
    // value — corrupting the core cache so every subsequent
    // recomposition would Steiner-shift the already-composed mass on
    // top of itself. Bypassing change detection here keeps the two
    // composition paths (legacy arena via `MassBodyIdC`/`AttachEvent`
    // and ECS-native via `MassChildOf`) safe to coexist on the same
    // entity during the migration window. The `MassPropertiesC` value
    // is still updated; only the change-detection signal is silenced.
    for (_, body_id, mut mass, _, _) in &mut bodies {
        if affected_ids.binary_search(&body_id.0).is_ok() {
            // allowed: typed↔raw kernel-boundary lift on mass-tree
            // composite writeback (named-method opt-in; see #397).
            *mass.bypass_change_detection() = MassPropertiesC::from(mass_raw_to_self_ref(
                &tree.get(body_id.0).composite_properties,
            ));
        }
    }

    // Run the JEOD momentum-conservation combine for every staged
    // attach. This must happen *after* the composite-mass sync above
    // so the merged mass we feed the kernel matches the parent's
    // post-attach `MassPropertiesC` (which is what subsequent
    // gravity / force-collection / integration reads in the same tick).
    //
    // JEOD_INV: DB.13 — state propagation across attached subtrees: only the
    // root carries the integrated composite-body state; child sub-trees ride
    // it via the MassChildOf / mass-tree composition.
    // JEOD_INV: DB.14 — integration-frame switch on attach: when the
    // child's pre-attach integ frame differs from the parent's, the
    // child's body-frame entity (and every kinematic descendant) is
    // reparented under the parent's integ-frame entity here, mirroring
    // JEOD's `dyn_body_attach.cc::attach_establish_links` calling
    // `set_integ_frame(*(dyn_parent->get_integ_frame()))` and the
    // recursive `dyn_body_integration.cc::set_integ_frame` walk over
    // `core_body`/`composite_body`/`structure` + `dyn_children`. The
    // integrator-state reset (JEOD's `reset_integrators()`) is handled
    // independently below via the `affected_ids` IG.37 walk.
    // JEOD_INV: DB.21 — only unattached bodies integrate: after attach the
    // detached-subtree-state is removed from the child so it stops drifting
    // ballistically; the integrated body's state is the merged composite.
    // JEOD_INV: RF.10 — cross-integration-frame attach is a kernel-input
    // shift site. The combine kernel does cross-body composition
    // (`omega × r`, `T_inertial_struct.transpose()`) which is only
    // arithmetic-valid when both bodies' state lives in the same
    // inertial frame. Lift each body to root inertial via its
    // pre-attach integ origin on input, then lower the merged
    // composite back through the parent's integ origin on writeback so
    // the parent's `TranslationalStateC` continues to hold
    // integration-frame coordinates. Mirrors
    // `astrodyn_runner::Simulation::attach_inner`. For the same-integ-frame
    // case both lifts are identically zero so the kernel call and
    // writeback collapse to the previous bit-identical behaviour.
    for work in &attach_work {
        let combined_mass = tree.get(work.parent_id).composite_properties;

        // The kernel runs in root-inertial coordinates: `work.parent_position`
        // and `work.child_position` were already lifted from each body's
        // pre-attach integration frame through its own `IntegOrigin` at the
        // construction site above (`parent_position_integ +
        // parent_integ_origin_pos`). For root-integrated bodies the lift is
        // identically zero and the kernel input collapses bit-identically to
        // the integ-frame value. Mirrors `astrodyn_runner::Simulation::attach`'s
        // seed-time lift through `body_integ_origins`.
        let merged = astrodyn::stage_attach_combine(astrodyn::StageAttachInputs {
            parent_position: work.parent_position,
            parent_velocity: work.parent_velocity,
            parent_quaternion: work.parent_quaternion,
            parent_ang_vel_body: work.parent_ang_vel_body,
            parent_mass: work.parent_mass,
            orig_parent_cm_struct: work.orig_parent_cm_struct,
            parent_t_inertial_struct: work.parent_t_inertial_struct,
            child_position: work.child_position,
            child_velocity: work.child_velocity,
            child_quaternion: work.child_quaternion,
            child_ang_vel_body: work.child_ang_vel_body,
            child_mass: work.child_mass,
            combined_mass,
        });

        // Lower the merged composite through the parent's integ
        // origin so the writeback into `TranslationalStateC` lands in
        // the parent's integration-frame coordinates. The parent's
        // integ frame is the new integ frame for the merged body — in
        // the runner this corresponds to writing the merged composite
        // back into the integrated tree root's `body.trans` (the
        // parent IS the tree root post-attach). For the
        // same-integ-frame case `parent_integ_origin_pos/vel` are
        // zero and the subtraction is bit-identically a no-op.
        let merged_position = merged.position - work.parent_integ_origin_pos;
        let merged_velocity = merged.velocity - work.parent_integ_origin_vel;

        if let Ok((_, _, _, mut trans, mut rot)) = bodies.get_mut(work.parent_entity) {
            if let Some(ref mut t) = trans {
                // Kernel returned the merged composite in root-inertial
                // (the captured snapshots were lifted before the call).
                // `TranslationalStateC` is integ-frame storage, so
                // lower back through the parent's `IntegOrigin` —
                // identity for root-integrated parents, load-bearing
                // for non-root. Symmetric partner of the seed-time
                // lift above; mirrors the runner's writeback in
                // `astrodyn_runner::Simulation::attach`.
                //
                // JEOD_INV: RF.10 — `body.trans` is typed
                // `TranslationalStateTyped<IntegrationFrame>`; the only
                // safe transition from `RootInertial` is the
                // integ-origin shift.
                // allowed: stage_attach_combine kernel boundary — the
                // kernel returns untyped DVec3 by design.
                t.0 = trans_raw_to_planet::<P>(&TranslationalState {
                    position: merged_position,
                    velocity: merged_velocity,
                });
            }
            if let Some(ref mut r) = rot {
                // allowed: stage_attach_combine kernel boundary — the
                // output quaternion is the parent's pre-attach unit-norm
                // quaternion, so the BodyAttitude witness is satisfied.
                r.0 = rot_raw_to_self_ref(&RotationalState {
                    quaternion: merged.quaternion,
                    ang_vel_body: merged.ang_vel_body,
                });
            }
        }

        // Cross-integration-frame attach: reparent the child's
        // body-frame entity (and every kinematic descendant of the
        // child in the mass tree) under the parent's integ-frame
        // entity, AND numerically rewrite each affected body's
        // `TranslationalStateC` / `FrameTransC` so the staged values
        // stay consistent with the frame-tree's post-reparent
        // interpretation. The full rationale (Bevy-vs-JEOD timing,
        // deferred-Commands flush boundary, why rotational state
        // doesn't need a rewrite) lives on
        // [`apply_cross_integ_frame_attach`]. Bypassed bit-identically
        // for same-integ-frame attaches (the common case), where
        // `cross_integ` is `None`.
        if let Some(ci) = work.cross_integ.as_ref() {
            apply_cross_integ_frame_attach::<P>(
                ci,
                &mut commands,
                &mut bodies,
                &frame_states,
                work.parent_entity,
            );
        }

        if work.child_was_detached {
            // Re-attach consumes the captured ballistic state — the
            // free-flying integrated tree root is no longer ballistic.
            // For the simple root-subject case `subject_root_entity ==
            // child_entity`; for the chained-attach case the subject's
            // existing tree root is the integrated free-flier carrying
            // `DetachedSubtreeStateC`, which is the entity we tracked
            // in `child_was_detached` above.
            commands
                .entity(work.subject_root_entity)
                .remove::<crate::DetachedSubtreeStateC>();
        }

        if work.is_chained_attach {
            // Fail-loud on a 3-DOF body anywhere in the rerooted
            // subtree. The kinematic walk
            // (`propagate_state_from_root_system`) derives both
            // `TranslationalStateC` and `RotationalStateC` from the
            // integrating root, so any attached (non-root) body must
            // be 6-DOF. A 3-DOF body in the rerooted subtree would no
            // longer be the integrated tree root *and* could not be
            // derived by the kinematic walk (the walk reads the body's
            // existing `RotationalStateC` to compose attitude); its
            // stored `TranslationalStateC` would silently go stale
            // post-reroot. Surface this at the attach site that
            // introduced the configuration rather than letting a
            // confusing "non-kinematic ancestor" diagnostic fire on
            // the first post-reroot step. Mirrors the runner's
            // `attach_inner` reroot-path 3-DOF check
            // (`crates/astrodyn_runner/src/simulation/mass_tree.rs`).
            // JEOD `dyn_body_attach.cc::attach_validate_child` rejects
            // partial state with the same intent; we apply the
            // analogous rejection per Bevy adapter semantics.
            for id in tree.subtree_ids(work.subject_root_id) {
                if let Some(&entity) = id_to_entity.get(&id) {
                    let (_has_dyn_cfg, _has_trans, has_rot) =
                        eligibility.get(entity).unwrap_or((false, false, false));
                    // Mass-only entities (no DynamicsConfigC /
                    // TranslationalStateC, no RotationalStateC) are
                    // legitimate JEOD `MassBody`-without-`DynBody`
                    // configuration and don't participate in the
                    // kinematic walk — they have no `TranslationalStateC`
                    // to go stale. The 3-DOF concern is specifically
                    // about a dynamic body (carries DynamicsConfigC +
                    // TranslationalStateC) that lacks RotationalStateC.
                    let body_frame = body_frames.get(entity).ok();
                    if body_frame.is_some() {
                        assert!(
                            has_rot,
                            "AttachEvent: chained-attach reroot: dynamic body {entity:?} \
                             (mass id {id:?}) is in the rerooted subtree under \
                             subject_root {subject:?} (mass id {subject_id:?}) but has no \
                             RotationalStateC. Kinematic propagation derives both \
                             TranslationalStateC and RotationalStateC from the integrating \
                             root, so any attached (non-root) body must be 6-DOF — a 3-DOF \
                             body's state would go stale post-attach. Make this body 6-DOF \
                             by inserting RotationalStateC before the chained attach, or \
                             restructure the topology so this body never becomes a non-root \
                             member of the merged tree.",
                            subject = work.subject_root_entity,
                            subject_id = work.subject_root_id,
                        );
                    }
                }
            }

            // Reparent the rerooted subtree's `MassChildOf` Relations
            // so the ECS-native composition path
            // (`composite_mass_system`, `wrench_aggregation_system`,
            // `propagate_state_from_root_system`) sees the same shape
            // the arena tree just landed via
            // `tree.attach_with_reroot(...)`.
            //
            // The walk is over `tree.subtree_ids(subject_root)` and
            // touches:
            //
            //   * **The subject root.** Pre-reroot it was a tree root
            //     with no `MassChildOf`; post-reroot it must carry
            //     `MassChildOf(parent=new_parent)` with the
            //     JEOD-recomputed (offset, t_parent_child) pair. The
            //     recomputed pair is what `attach_with_reroot` stamped
            //     into the arena's `structure_point[subject_root]`,
            //     so we read it back from the tree to keep the ECS
            //     side bit-identical to the arena.
            //
            //   * **Every descendant.** Pre-reroot a descendant may
            //     have no `MassChildOf` (the simple-attach path
            //     doesn't auto-insert one — mission code retains
            //     explicit control). Post-reroot, the kinematic walk
            //     and the wrench aggregation must derive the
            //     descendant's state from the new tree root, which
            //     requires the entire chain of `MassChildOf` edges to
            //     be intact. We synthesize each descendant's
            //     `MassChildOf` from the arena's
            //     `parent[descendant]` + `structure_point[descendant]`
            //     so the ECS path observes exactly the same edge the
            //     arena tree already has.
            //
            // This is the auto-promote-to-kinematic semantics on the
            // Bevy side: `wrench_aggregation_system` walks every
            // `MassChildOf` chain after staging and inserts
            // `KinematicChildC` on every non-root entity it visits.
            // Mission code does NOT need to manually mark the
            // rerooted subtree's bodies as kinematic — the marker
            // emerges from the reparented chain on the next pass
            // through the schedule. Matches the runner-side
            // `Simulation::attach_inner`'s explicit
            // `body.kinematic_only = true` auto-flag for the rerooted
            // subtree, just expressed through the ECS relation rather
            // than a per-body bool.
            // JEOD_INV: BA.12 — chained attach re-roots the subject's existing tree under the new parent.
            for id in tree.subtree_ids(work.subject_root_id) {
                let entity = match id_to_entity.get(&id) {
                    Some(&e) => e,
                    None => continue,
                };
                let arena_parent_id = tree.parent(id).expect(
                    "subtree_ids(subject_root) is the post-reroot subtree; the \
                             subject root now has `parent_entity` and every descendant has \
                             its in-subtree arena parent — both branches must yield Some(_)",
                );
                let arena_parent_entity = match id_to_entity.get(&arena_parent_id) {
                    Some(&e) => e,
                    None => continue,
                };
                // Read the offset / rotation from the arena's
                // `structure_point` so the ECS-side `MassChildOf`
                // matches the arena tree edge bit-identically.
                // Post-reroot, `tree.get(subject_root).structure_point`
                // is the JEOD-recomputed pair the kernel just stamped
                // (matches the (`rerooted_child_offset`,
                // `rerooted_child_t_parent_child`) we computed
                // pre-mutation from the same formulas), and every
                // descendant's `structure_point` is unchanged from its
                // pre-reroot value (the reroot only changes the
                // parent of `subject_root`, not its descendants'
                // edges).
                let sp = tree.get(id).structure_point;
                commands
                    .entity(entity)
                    .insert(crate::MassChildOf::with_rotation(
                        arena_parent_entity,
                        sp.position,
                        sp.t_parent_this,
                    ));
            }
        }

        if work.parent_was_detached {
            // The merged composite is still a free-flying tree root
            // (the parent had no integrated ancestor to graft onto).
            // Replace the parent's stale pre-attach `DetachedSubtreeStateC`
            // with the merged composite-body inertial state so
            // `step_detached_system` continues advancing the right
            // value next tick rather than overwriting the merged state
            // with the captured pre-attach snapshot.
            //
            // `merged` is already in root-inertial — both parent and
            // child snapshots were lifted through their own
            // `IntegOrigin` before feeding the kernel, so the kernel
            // produced the merged composite in root-inertial too.
            // `DetachedSubtreeState.composite_*` is typed
            // `Position/Velocity<RootInertial>` by witness, so this
            // is a direct relabel with no further shift. The runner
            // mirrors this contract by tracking `composite_state` in
            // root-inertial inside its detached-subtree map.
            //
            // JEOD_INV: DB.21 — detached subtrees keep advancing
            // ballistically post-attach; the merged composite simply
            // becomes the new "free-flying root" state.
            // JEOD_INV: RF.10 — root-inertial-shift consumer: the
            // typed `DetachedSubtreeState.composite_*` is
            // `Position/Velocity<RootInertial>`.
            use astrodyn::Vec3Ext as _;
            let updated = astrodyn::DetachedSubtreeState {
                composite_position: merged.position.m_at::<astrodyn::RootInertial>(),
                composite_velocity: merged.velocity.m_per_s_at::<astrodyn::RootInertial>(),
                composite_attitude: astrodyn::DetachedSubtreeState::attitude_from_raw_jeod_quat(
                    merged.quaternion,
                ),
                composite_ang_vel_body: merged.ang_vel_body,
            };
            commands
                .entity(work.parent_entity)
                .insert(crate::DetachedSubtreeStateC(updated));
        }
    }

    // Apply detach captures: insert `DetachedSubtreeStateC` on each
    // detached child so `step_detached_system` advances it ballistically
    // each tick.
    for (entity, captured) in detach_work {
        commands
            .entity(entity)
            .insert(crate::DetachedSubtreeStateC(captured));
    }

    // Parent-side post-detach composite-CoM shift: when a subtree is
    // removed from a tree, the parent's composite-CoM moves within its
    // own struct frame. The parent's rigid-body structure point hasn't
    // moved in inertial space, but the composite-body inertial state
    // (which is what `TranslationalStateC` stores after the
    // composite-body refactor) must shift by the corresponding
    // kinematic offset to track the new (smaller) composite. Mirrors
    // `astrodyn_runner::Simulation::detach_subtree`'s integrated-body /
    // detached-parent branches; both produce the same inertial
    // CoM-delta formula.
    //
    // JEOD_INV: DB.13 — composite-body propagation on topology change.
    for (tree_root_id, shift) in parent_shifts {
        let parent_post_composite_props = tree.get(tree_root_id).composite_properties;
        let cm_delta_struct =
            parent_post_composite_props.position - shift.parent_pre_composite_props.position;
        // composite_properties.t_parent_this is struct→body. Compose
        // with the body's inertial-to-body to map struct → inertial.
        let t_struct_to_body = shift.parent_pre_composite_props.t_parent_this;
        let cm_delta_body = t_struct_to_body * cm_delta_struct;
        let t_inertial_to_body = shift.parent_pre_quat.left_quat_to_transformation();
        let cm_delta_inertial = t_inertial_to_body.transpose() * cm_delta_body;
        // Velocity offset from rigid-body rotation: ω × Δr in body
        // frame, then rotated to inertial.
        let omega_body = shift.parent_pre_ang_vel_body;
        let dvel_inertial = t_inertial_to_body.transpose() * omega_body.cross(cm_delta_body);

        let new_position = shift.parent_pre_position + cm_delta_inertial;
        let new_velocity = shift.parent_pre_velocity + dvel_inertial;

        if let Ok((_, _, _, Some(mut t), _)) = bodies.get_mut(shift.tree_root_entity) {
            // allowed: detach-handler kernel boundary; CoM-shift is a
            // pure kinematic update — same `PlanetInertial<P>` frame
            // as the pre-detach value.
            t.0 = trans_raw_to_planet::<P>(&TranslationalState {
                position: new_position,
                velocity: new_velocity,
            });
        }

        if shift.parent_was_detached {
            // The parent is itself a detached free-flying root — keep
            // its `DetachedSubtreeStateC` in lock-step with the shifted
            // `TranslationalStateC` so the next `step_detached_system`
            // tick advances from the post-detach composite state.
            // Quaternion / ang_vel are unchanged because the parent's
            // body axes don't rotate just because mass left the tree
            // (composite_properties.t_parent_this == core_properties
            // .t_parent_this throughout — see mass tree recompute).
            //
            // `new_position` / `new_velocity` are computed by adding a
            // CoM-delta to `shift.parent_pre_position/velocity`, which
            // were captured directly from the parent's
            // `TranslationalStateC` (integration-frame coords). The
            // delta itself is frame-invariant (kinematic offset of a
            // CoM within rigid-body inertial space). To stamp the
            // `RootInertial` phantom for the typed
            // `DetachedSubtreeState`, lift through the parent's
            // `IntegOrigin`. Identity for root-integrated parents
            // (origin = zero); load-bearing for non-root.
            //
            // JEOD_INV: RF.10 — root-inertial-shift consumer: the
            // typed `DetachedSubtreeState.composite_*` is
            // `Position/Velocity<RootInertial>`.
            let parent_body_frame_shift = body_frames.get(shift.tree_root_entity).ok();
            let (parent_integ_origin_pos_shift, parent_integ_origin_vel_shift) =
                body_integ_origin_in_root_lazy(
                    parent_body_frame_shift,
                    &parents,
                    root_frame_entity.as_deref().map(|r| r.0),
                    &frame_origin,
                );
            use astrodyn::Vec3Ext as _;
            let updated = astrodyn::DetachedSubtreeState {
                composite_position: (new_position + parent_integ_origin_pos_shift.raw_si())
                    .m_at::<astrodyn::RootInertial>(),
                composite_velocity: (new_velocity + parent_integ_origin_vel_shift.raw_si())
                    .m_per_s_at::<astrodyn::RootInertial>(),
                composite_attitude: astrodyn::DetachedSubtreeState::attitude_from_raw_jeod_quat(
                    shift.parent_pre_quat,
                ),
                composite_ang_vel_body: shift.parent_pre_ang_vel_body,
            };
            commands
                .entity(shift.tree_root_entity)
                .insert(crate::DetachedSubtreeStateC(updated));
        }
    }

    // Site A: mark every affected body's integrators dirty.
    // JEOD_INV: IG.37 — kept strictly before Site B so a regression
    // that drops Site B leaves the dirty flag set and panics on next
    // integrate.
    for (body_id, mut gj_opt, mut abm_opt) in &mut integrators {
        if affected_ids.binary_search(&body_id.0).is_ok() {
            if let Some(ref mut gj) = gj_opt {
                gj.0.mark_topology_dirty();
            }
            if let Some(ref mut abm) = abm_opt {
                abm.0.mark_topology_dirty();
            }
        }
    }

    // Site B: reset integrator history. Mirrors JEOD's
    // `dyn_body_attach.cc::reset_integrators()` (lines 860, 871) and
    // `dyn_body_detach.cc:271-273`.
    // JEOD_INV: IG.37 — multi-step integrator history must be reset on topology change
    for (body_id, mut gj_opt, mut abm_opt) in &mut integrators {
        if affected_ids.binary_search(&body_id.0).is_ok() {
            astrodyn::reset_integrators(
                gj_opt.as_mut().map(|c| c.0.inner_mut()),
                abm_opt.as_mut().map(|c| c.0.inner_mut()),
            );
        }
    }
}

/// Advance every entity carrying [`crate::DetachedSubtreeStateC`] by
/// the schedule's fixed `dt` under ballistic dynamics — no force, no
/// torque. Position drifts at `composite_velocity`; attitude rotates
/// at `composite_ang_vel_body` via JEOD's left-multiply convention
/// (`q̇ = -½(ω ⊗ q)`, owned by [`astrodyn::BodyAttitude`]).
///
/// Also synchronizes the entity's [`crate::TranslationalStateC`] /
/// [`crate::RotationalStateC`] with the advanced subtree state each
/// tick so downstream consumers (gravity-source position lookups,
/// derived-state systems, mission code) see the body's current
/// inertial state without having to special-case detached vs
/// integrated bodies. Mirrors
/// `astrodyn_runner::Simulation::step_detached_subtrees`.
///
/// The ballistic timestep is `dt * time_scale_factor` (matching
/// `integration_system`'s `integ_dt` and the runner's
/// `step_detached_subtrees(dt * time.time_scale_factor)`); under
/// reversed or scaled time the detached subtree advances at the same
/// rate as integrated bodies, so the two stay phase-locked.
///
/// JEOD_INV: DB.21 — only unattached bodies integrate; detached subtrees
/// drift ballistically here while the integrator targets the integrated
/// body.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn step_detached_system<P: Planet>(
    time: Res<Time<Fixed>>,
    sim_time: Res<SimulationTimeR>,
    mut detached: Query<(
        Entity,
        &mut crate::DetachedSubtreeStateC,
        Option<&mut TranslationalStateC<P>>,
        Option<&mut RotationalStateC>,
    )>,
    body_frames: Query<&FrameEntityC>,
    parents: Query<&ChildOf>,
    frame_origin: FrameOrigin,
    root_frame_entity: Option<Res<crate::RootFrameEntityR>>,
) {
    let dt = time.delta().as_secs_f64();
    if dt == 0.0 {
        return;
    }
    let integ_dt = dt * sim_time.0.time_scale_factor;
    for (entity, mut state, trans, rot) in &mut detached {
        state.0.step_ballistic(integ_dt);
        if let Some(mut t) = trans {
            // Lower the typed `Position/Velocity<RootInertial>` back
            // through the body's `IntegOrigin` to match
            // `TranslationalStateC`'s integration-frame storage
            // convention. For a root-integrated body the origin is
            // zero and the subtraction is bit-identical to a no-op;
            // for a body integrated in `PlanetInertial<P>` (set up
            // at config time via `IntegSourceC`) it is the only
            // thing that prevents stamping a root-inertial coord into
            // an integration-frame slot. Symmetric partner of the
            // staging-system lift; mirrors the runner's writeback in
            // `crates/astrodyn_runner/src/simulation/mass_tree.rs:681-688`.
            // JEOD_INV: RF.10 — root-inertial-shift consumer:
            // step-time writeback lowers from root-inertial to integ
            // frame.
            let body_frame = body_frames.get(entity).ok();
            let (integ_origin_pos, integ_origin_vel) = body_integ_origin_in_root_lazy(
                body_frame,
                &parents,
                root_frame_entity.as_deref().map(|r| r.0),
                &frame_origin,
            );
            let position = state.0.composite_position.raw_si() - integ_origin_pos.raw_si();
            let velocity = state.0.composite_velocity.raw_si() - integ_origin_vel.raw_si();
            // allowed: DetachedSubtreeState kernel boundary — ballistic-
            // step result is returned as raw DVec3 fields by design.
            t.0 = trans_raw_to_planet::<P>(&TranslationalState { position, velocity });
        }
        if let Some(mut r) = rot {
            // allowed: DetachedSubtreeState kernel boundary. The advanced
            // `composite_attitude` is a `BodyAttitude<SelfRef>` whose
            // `to_jeod_quat` returns the underlying scalar-first
            // left-transformation quaternion. The wrapper guarantees
            // unit-norm post-step (`BodyAttitude::advance_under_body_rate`),
            // so the witness is satisfied.
            r.0 = rot_raw_to_self_ref(&RotationalState {
                quaternion: state.0.composite_attitude.to_jeod_quat(),
                ang_vel_body: state.0.composite_ang_vel_body,
            });
        }
    }
}
