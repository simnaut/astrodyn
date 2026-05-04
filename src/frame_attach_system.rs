//! Bevy systems for frame-attached body integration.
//!
//! Bevy adapter for the runner's
//! `Simulation::attach_to_frame` API (the runner-side equivalent in
//! `jeod_runner::Simulation`). The two consumers (Bevy and runner) drive the same
//! `jeod_dynamics::derive_frame_attached_state` kernel against the
//! same JEOD precedent (`DynBody::attach_to_frame`,
//! `models/dynamics/dyn_body/src/dyn_body_attach.cc:271-379` and the
//! `frame_attach.isAttached()` integration branch at
//! `dyn_body_integration.cc:309-333`).
//!
//! ### Two systems, mirroring the runner's two call sites
//!
//! - [`frame_attach_system`] — processes [`FrameAttachEvent`] /
//!   [`FrameDetachEvent`] messages: inserts / removes the
//!   [`FrameAttachedC`] component on the body entity and resets
//!   multi-step integrator history (Gauss–Jackson, ABM4) so the
//!   topology change doesn't carry stale predictor cache forward.
//!   Mirrors `Simulation::attach_to_frame` /
//!   `Simulation::detach_from_frame`.
//! - [`propagate_frame_attached_state_system`] — runs each tick:
//!   for every entity carrying [`FrameAttachedC`], reads the parent
//!   frame entity's current state via [`RelativeFrameState`], composes
//!   with the captured offset, and overwrites the body's
//!   [`TranslationalStateC`] / [`RotationalStateC`]. Mirrors the
//!   runner's `propagate_frame_attached_state` per-step pass.
//!
//! ### Mutual exclusion with mass-tree attachment
//!
//! Frame attachment and mass-tree attachment are mutually exclusive
//! per JEOD's
//! [DynBody::attach_to_frame](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/dynamics/dyn_body/src/dyn_body_attach.cc#L271):
//! `attach_to_frame` writes the attachment on the integrated tree
//! root, never on a child body. The Bevy adapter's
//! [`frame_attach_system`] enforces this by panicking if a
//! [`FrameAttachEvent`] targets an entity that already carries
//! [`MassChildOf`].
//!
//! ### Schedule placement
//!
//! - [`frame_attach_system`] is wired in
//!   [`JeodSet::ForceCollection`](crate::JeodSet::ForceCollection)
//!   alongside `staging_system` so attach/detach events take effect
//!   for the current tick's force collection and integration.
//! - [`propagate_frame_attached_state_system`] runs in
//!   [`JeodSet::ForceCollection`](crate::JeodSet::ForceCollection)
//!   *after* `frame_attach_system` (so freshly-attached bodies pick up
//!   the frame composition the same tick they were attached),
//!   *before*
//!   [`propagate_state_from_root_system`](crate::propagate_state_from_root_system)
//!   (so a frame-attached body that is also a mass-tree root has its
//!   freshly-derived state available when the kinematic walk derives
//!   its children — otherwise the subtree would lag the root by one
//!   tick), and *before*
//!   [`integration_system`](crate::systems::integration_system) (so
//!   the integrator sees the frame-derived state when deciding to
//!   skip via the `FrameAttachedC` filter).

use std::collections::HashSet;

use bevy::ecs::message::MessageReader;
use bevy::ecs::system::ParamSet;
use bevy::prelude::*;

use jeod_sim::MassPointState;

use crate::components::{
    Abm4StateC, FrameAngVelC, FrameAttachEvent, FrameAttachedC, FrameDetachEvent, FrameEntityC,
    FrameRotC, FrameTransC, GaussJacksonStateC, MassChildOf, RotationalStateC, TranslationalStateC,
};
use crate::frame_param::RelativeFrameState;
use crate::RootFrameEntityR;

/// Process [`FrameAttachEvent`] / [`FrameDetachEvent`] messages by
/// inserting / removing [`FrameAttachedC`] and resetting multi-step
/// integrator history on the body entity.
///
/// Bevy adapter for `Simulation::attach_to_frame` /
/// `Simulation::detach_from_frame`. Schedule placement: alongside
/// `staging_system` in [`JeodSet::ForceCollection`](crate::JeodSet::ForceCollection),
/// so attach/detach events take effect on the same tick they're
/// dispatched.
///
/// # Panics
///
/// Panics with a "Fail Loudly" diagnostic when:
/// - A [`FrameAttachEvent`]'s `parent_frame` is not a frame entity —
///   defined here as carrying the full
///   [`FrameTransC`] / [`FrameRotC`] / [`FrameAngVelC`] triplet that
///   `RelativeFrameState` walks during the per-tick propagation pass.
///   A non-frame entity (e.g. a body, a source, or an arbitrary
///   placeholder) would silently misbehave: the relative-state walk
///   would observe undefined translation / rotation / angular
///   velocity contributions at that segment of the chain. Detecting
///   the mismatch at attach time turns a silent garbage-state
///   trajectory into a loud configuration error pointing at the
///   miswired entity.
/// - A [`FrameAttachEvent`] targets an entity that already carries
///   [`FrameAttachedC`]: a silent overwrite would lose the original
///   parent-frame relationship and leave the captured offset
///   desynchronized from the body's actual position.
/// - Two [`FrameAttachEvent`]s in the same tick target the same
///   entity: only the first event's `commands.insert` will land before
///   the apply boundary, so a `Query<&FrameAttachedC>` check alone
///   cannot observe the in-flight insert. The second event would
///   silently overwrite the first's offset / `t_parent_body` /
///   `parent_frame`, masking paired-event bugs in mission code. A
///   per-call `HashSet` of bodies that have already had an insert
///   queued this tick rejects the duplicate before the queue grows.
/// - A [`FrameAttachEvent`] targets an entity that has a
///   [`MassChildOf`] parent: JEOD's `attach_to_frame` writes the
///   attachment on the root body, never on a child body; mixing
///   mass-tree attach with frame-tree attach would let
///   `propagate_frame_attached_state_system` overwrite the parent's
///   chosen child state with a contradicting parent-frame composition.
/// - A [`FrameDetachEvent`] targets an entity that does not currently
///   carry [`FrameAttachedC`]: silently no-op'ing would mask
///   paired-event bugs in mission code.
/// - Two [`FrameDetachEvent`]s in the same tick target the same
///   entity: same in-flight `commands.remove` blind spot as the
///   double-attach case above. Tracked through the same per-call
///   `HashSet` so the second detach panics rather than silently
///   no-op'ing.
// JEOD_INV: DB.21 — only unattached bodies integrate (frame-attach gate)
// JEOD_INV: IG.37 — multi-step integrator history reset on topology change
#[allow(clippy::type_complexity)]
pub fn frame_attach_system(
    mut commands: Commands,
    mut attach_events: MessageReader<FrameAttachEvent>,
    mut detach_events: MessageReader<FrameDetachEvent>,
    already_frame_attached: Query<Entity, With<FrameAttachedC>>,
    has_mass_parent: Query<&MassChildOf>,
    // Frame-tree triplet check for `evt.parent_frame`. The triplet
    // (`FrameTransC` + `FrameRotC` + `FrameAngVelC`) is what
    // `RelativeFrameState` reads during the per-tick propagation pass;
    // an entity missing any of the three would produce undefined
    // contributions at that segment of the chain. The `Has<…>` access
    // pattern keeps the query disjoint from the writeback paths and
    // works whether the parent frame happens to also carry other
    // components (e.g., a body frame entity that lives under the root
    // frame and would pull in `FrameEntityC` / `BodyFrameMarker`).
    parent_frame_components: Query<(
        bevy::ecs::query::Has<FrameTransC>,
        bevy::ecs::query::Has<FrameRotC>,
        bevy::ecs::query::Has<FrameAngVelC>,
    )>,
    mut integrators: Query<(Option<&mut GaussJacksonStateC>, Option<&mut Abm4StateC>)>,
) {
    // Bevy's `Commands` queue is not flushed until the next system
    // boundary, so two events in the same `MessageReader` batch both
    // see the pre-tick component snapshot via `already_frame_attached`
    // / `has_mass_parent`. Tracking bodies whose insert/remove has
    // already been queued in this call closes the window — without it
    // the second event silently overwrites the first's
    // `FrameAttachedC` (or no-ops the first detach), violating the
    // fail-loud contract.
    let mut attached_this_tick: HashSet<Entity> = HashSet::new();
    let mut detached_this_tick: HashSet<Entity> = HashSet::new();

    for evt in attach_events.read() {
        // Reject a second attach event for an entity that already had
        // one queued earlier in this same tick. The
        // `already_frame_attached` query reflects the pre-tick
        // component snapshot only; in-flight `commands.insert` calls
        // are invisible to it until the next apply boundary.
        assert!(
            !attached_this_tick.contains(&evt.body),
            "FrameAttachEvent: body {:?} already had a FrameAttachEvent processed \
             earlier in this tick. Two simultaneous attach events on the same \
             body would silently overwrite the first event's offset and \
             `t_parent_body` (the queued `commands.insert` is invisible to the \
             component query until the next apply boundary). Coalesce duplicate \
             events in mission code, or send a FrameDetachEvent on the \
             intervening tick before re-attaching.",
            evt.body
        );
        assert!(
            already_frame_attached.get(evt.body).is_err(),
            "FrameAttachEvent: body {:?} is already frame-attached. Send a \
             FrameDetachEvent before re-attaching to a different parent frame; \
             silent overwrite would lose the original frame-tree relationship \
             and leave the captured offset desynchronized from the body's \
             actual position.",
            evt.body
        );
        // Mass-tree attachment is mutually exclusive with frame
        // attachment (see module docs + JEOD_INV: DB.21).
        assert!(
            has_mass_parent.get(evt.body).is_err(),
            "FrameAttachEvent: body {:?} is a mass-tree child (carries \
             MassChildOf). Send a DetachEvent first — frame attachment and \
             mass-tree attachment are mutually exclusive (JEOD's \
             `attach_to_frame` writes the attachment on the integrated tree \
             root, not on a child body). Mixing both would let the \
             frame-attached propagation overwrite the parent-derived state \
             every tick.",
            evt.body
        );

        // Validate `evt.parent_frame` carries the full frame-tree
        // triplet read by `RelativeFrameState` on every tick. A
        // non-frame entity would produce undefined translation /
        // rotation / angular velocity contributions during the
        // per-tick propagation walk, silently corrupting the
        // attached body's derived state. Surface the misconfiguration
        // here so the diagnostic names the offending entity and the
        // missing components instead of letting the downstream walk
        // produce garbage. `Query::get` returns `Err` for despawned
        // entities; treat that as "not a frame entity" with the same
        // message so the caller learns whichever invariant they
        // broke. The resulting tuple of `Has<_>` flags is the per-
        // event view of which of the three components are present.
        let (has_trans, has_rot, has_angvel) = parent_frame_components
            .get(evt.parent_frame)
            .unwrap_or((false, false, false));
        assert!(
            has_trans && has_rot && has_angvel,
            "FrameAttachEvent: parent_frame {:?} is not a frame entity \
             (missing{}{}{}). Frame-tree consumers walk every parent_frame \
             via `RelativeFrameState`, which requires the full \
             FrameTransC / FrameRotC / FrameAngVelC triplet on each node. \
             Spawn the parent via `PlanetBundle` (for planet-inertial / \
             planet-fixed frames) or by inserting the triplet directly \
             (e.g., for a custom joint frame), and pass that frame's \
             entity id — not a body, source, or arbitrary placeholder \
             entity. Body {:?}.",
            evt.parent_frame,
            if has_trans { "" } else { " FrameTransC" },
            if has_rot { "" } else { " FrameRotC" },
            if has_angvel { "" } else { " FrameAngVelC" },
            evt.body,
        );

        commands.entity(evt.body).insert(FrameAttachedC {
            parent_frame: evt.parent_frame,
            offset: evt.offset,
            t_parent_body: evt.t_parent_body,
        });
        attached_this_tick.insert(evt.body);

        // Reset multi-step integrator history.
        if let Ok((gj, abm4)) = integrators.get_mut(evt.body) {
            if let Some(mut state) = gj {
                state.0.reset();
            }
            if let Some(mut state) = abm4 {
                state.0.reset();
            }
        }
    }

    for evt in detach_events.read() {
        // Same in-flight blind spot as the attach loop: a queued
        // `commands.remove` won't show up in `already_frame_attached`
        // until the next apply boundary, so a second detach event on
        // the same body would silently pass the component check.
        assert!(
            !detached_this_tick.contains(&evt.body),
            "FrameDetachEvent: body {:?} already had a FrameDetachEvent processed \
             earlier in this tick. Two simultaneous detach events would silently \
             no-op the second one (the queued `commands.remove` is invisible to \
             the component query until the next apply boundary). Coalesce \
             duplicate events in mission code.",
            evt.body
        );
        assert!(
            already_frame_attached.get(evt.body).is_ok(),
            "FrameDetachEvent: body {:?} is not currently frame-attached. \
             Send a FrameAttachEvent first, or remove the duplicate detach \
             to avoid masking caller bugs.",
            evt.body
        );

        commands.entity(evt.body).remove::<FrameAttachedC>();
        detached_this_tick.insert(evt.body);

        if let Ok((gj, abm4)) = integrators.get_mut(evt.body) {
            if let Some(mut state) = gj {
                state.0.reset();
            }
            if let Some(mut state) = abm4 {
                state.0.reset();
            }
        }
    }
}

/// Per-tick pass that derives every [`FrameAttachedC`] body's
/// [`TranslationalStateC`] / [`RotationalStateC`] from the parent
/// frame entity's current state composed with the captured offset.
///
/// Bevy adapter for `Simulation::propagate_frame_attached_state`. Runs
/// in [`JeodSet::ForceCollection`](crate::JeodSet::ForceCollection)
/// after `frame_attach_system` (so events processed this tick take
/// effect immediately) and before `integration_system` (so the
/// integrator sees the frame-derived state when deciding to skip via
/// the [`FrameAttachedC`] filter applied in
/// [`integration_system`](crate::systems::integration_system)).
///
/// Fast-paths to a no-op when no entity carries [`FrameAttachedC`].
// JEOD_INV: DB.13 — propagate_state delegates to parent frame
// JEOD_INV: DB.14 — integration-frame switch on attach: the attached body's
//   state is owned by the parent frame, not the integrator
// JEOD_INV: RF.10 — frame-attach is a shift site: the parent frame state is
//   computed in root-inertial coords via `RelativeFrameState(root, parent)`,
//   the kernel composes with the offset in those coords, and the body
//   integration-frame is the root frame in the realistic Bevy config
//   (ISS-LEO etc.) so the writeback is a no-op shift. Cross-source
//   integration frames remain on the runner-side TODO list.
#[allow(clippy::type_complexity)]
pub fn propagate_frame_attached_state_system(
    attached: Query<(Entity, &FrameAttachedC)>,
    // Body entities only — exclude frame entities (which carry
    // `FrameTransC`) so `state_q` is statically disjoint from the
    // frame-state query in the `ParamSet` below.
    mut state_q: Query<
        (
            &mut TranslationalStateC,
            Option<&mut RotationalStateC>,
            Option<&FrameEntityC>,
        ),
        Without<crate::components::FrameTransC>,
    >,
    // ParamSet that gates the read-only `RelativeFrameState`
    // SystemParam (slot 0) against the per-step write of the body's
    // own frame entity (slot 1). Both touch frame-state components on
    // frame entities; serializing them through a `ParamSet` is the
    // standard Bevy idiom for "borrow checker says no" between an
    // unfiltered read and a marker-gated write of the same components
    // on overlapping entity sets, mirroring the same pattern in
    // `propagate_state_from_root_system` for `RotationalStateC` /
    // `TranslationalStateC`.
    mut frame_qs: ParamSet<(
        RelativeFrameState,
        Query<
            (
                &'static mut crate::components::FrameTransC,
                Option<&'static mut crate::components::FrameRotC>,
                Option<&'static mut crate::components::FrameAngVelC>,
            ),
            Without<TranslationalStateC>,
        >,
    )>,
    root_frame: Res<RootFrameEntityR>,
) {
    if attached.is_empty() {
        return;
    }

    let root = root_frame.0;

    // Pre-compute parent-frame states for every attached body so we
    // can release the `RelativeFrameState` borrow before taking the
    // mutable frame-state query in the writeback loop. Mirrors the
    // pre-compute pattern in `propagate_state_from_root_system`
    // (build the per-node state map under the read query, then
    // release and write back through the write query).
    struct AttachWork {
        body_entity: Entity,
        derived: jeod_sim::RefFrameState,
    }
    let mut work: Vec<AttachWork> = Vec::with_capacity(attached.iter().len());
    {
        let rel = frame_qs.p0();
        for (body_entity, attach) in attached.iter() {
            // Read the parent reference frame's state in root-inertial
            // coordinates. `RelativeFrameState::relative_state` walks the
            // ECS hierarchy via `ChildOf` and composes per-segment
            // transforms — the same algorithm the runner's
            // `FrameTree::compute_relative_state` uses, single-sourced
            // through the storage-agnostic helper.
            let parent_state = rel.relative_state(root, attach.parent_frame);

            let derived = jeod_sim::derive_frame_attached_state(jeod_sim::FrameAttachInputs {
                parent_frame: parent_state,
                attach_offset: MassPointState {
                    position: attach.offset,
                    t_parent_this: attach.t_parent_body,
                },
            });
            work.push(AttachWork {
                body_entity,
                derived,
            });
        }
    }

    // Writeback pass. Re-acquire each side's mutable view; the
    // pre-computed `derived` value is already in root-inertial
    // coordinates so we don't need any further frame walks here.
    let mut frame_writeback_q = frame_qs.p1();
    for AttachWork {
        body_entity,
        derived,
    } in &work
    {
        let Ok((mut trans, rot_opt, frame_opt)) = state_q.get_mut(*body_entity) else {
            // Entity may have been despawned between event processing
            // and propagation; skip silently rather than panicking.
            // The marker will be reaped by Bevy's despawn cleanup.
            continue;
        };
        // The kernel produces composite-body inertial state in
        // root-inertial coords (the parent frame state was read in
        // root-inertial via `RelativeFrameState`). Bevy stores body
        // state as `TranslationalStateC`, which is tagged with the
        // wildcard `<PlanetInertial<SelfPlanet>>` phantom — the
        // type-system convention for Bevy bodies that integrate in
        // their own planet's inertial frame. In the realistic Bevy
        // mission config (ISS-LEO etc.) the root inertial frame is
        // numerically coincident with the central body's
        // `PlanetInertial<Earth>` frame, so writing the kernel's
        // root-inertial output through the bypass constructor (a
        // zero-cost phantom relabel) preserves the SI coordinates.
        // RF.10: cross-source integration frames (body in
        // `PlanetInertial<P>` for non-central P) would need an
        // `IntegOrigin`-style shift here, analogous to the runner's
        // per-body lift in `propagate_frame_attached_state`. The
        // runner already enforces the shift; the Bevy adapter
        // currently only wires the central-body case (every existing
        // recipe), so the relabel is correct for the in-tree
        // configurations.
        // Kernel-boundary writeback: pack the root-inertial state into
        // the wildcard-tagged Bevy storage (`PlanetInertial<SelfPlanet>`).
        // The wildcard relabel preserves SI coordinates exactly because
        // every existing Bevy recipe integrates in the central body's
        // planet-inertial frame, which is bit-coincident with root
        // inertial.
        let derived_trans = jeod_sim::TranslationalState {
            position: derived.trans.position,
            velocity: derived.trans.velocity,
        };
        // allowed: kernel boundary — the kernel returns root-inertial values; Bevy storage uses the wildcard PlanetInertial<SelfPlanet> tag, which is bit-coincident in the central-body recipes.
        trans.0 = jeod_sim::TranslationalStateTyped::<jeod_sim::PlanetInertial<jeod_sim::SelfPlanet>>::from_untyped_unchecked(&derived_trans);

        if let Some(mut rot) = rot_opt {
            let derived_rot = jeod_sim::RotationalState {
                quaternion: derived.rot.q_parent_this,
                ang_vel_body: derived.rot.ang_vel_this,
            };
            // allowed: kernel boundary — the kernel produces a composed quaternion + body-frame angular velocity; re-tagging into the SelfRef-marked RotationalStateTyped is the same zero-cost phantom relabel `staging_system` uses on the mass-tree side.
            rot.0 = jeod_sim::RotationalStateTyped::from_untyped_unchecked(&derived_rot);
        }

        // Sync the body's frame entity (if it has one) so frame-tree
        // consumers see the same value as `TranslationalStateC`. The
        // body frame is `ChildOf(root)` in the realistic config, so
        // its `FrameTransC` is the body's root-inertial position
        // (relative to root = root-inertial absolute).
        if let Some(frame_entity) = frame_opt.map(|f| f.0) {
            if let Ok((mut frame_trans, frame_rot, frame_angvel)) =
                frame_writeback_q.get_mut(frame_entity)
            {
                frame_trans.position = derived.trans.position;
                frame_trans.velocity = derived.trans.velocity;
                if let Some(mut rot) = frame_rot {
                    rot.q_parent_this = derived.rot.q_parent_this;
                    rot.t_parent_this = derived.rot.t_parent_this;
                }
                if let Some(mut av) = frame_angvel {
                    av.0 = derived.rot.ang_vel_this;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{
        DynamicsConfigC, ExternalForceC, ExternalTorqueC, FrameDerivativesC, KinematicChildC,
        MassChildOf, MassPropertiesC, RotationalStateC, TotalForceC, TranslationalStateC,
    };
    use crate::JeodPlugin;
    use bevy::prelude::FixedUpdate;
    use bevy::time::{Fixed, Time};
    use glam::DVec3;
    use jeod_sim::{MassProperties, RotationalState, TranslationalState};
    use std::time::Duration;

    fn step_bevy(app: &mut App, n: usize, dt: f64) {
        for _ in 0..n {
            app.world_mut()
                .resource_mut::<Time<Fixed>>()
                .advance_by(Duration::from_secs_f64(dt));
            app.world_mut().run_schedule(FixedUpdate);
        }
    }

    /// Smoke test: spawning a body, dispatching a `FrameAttachEvent`,
    /// and stepping `FixedUpdate` once must result in the body
    /// carrying `FrameAttachedC` after the event is processed.
    #[test]
    fn attach_event_inserts_marker() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JeodPlugin));

        let body = app.world_mut().spawn_empty().id();
        let parent_frame = **app.world().resource::<RootFrameEntityR>();
        app.world_mut()
            .resource_mut::<bevy::ecs::message::Messages<FrameAttachEvent>>()
            .write(FrameAttachEvent {
                body,
                parent_frame,
                offset: DVec3::ZERO,
                t_parent_body: glam::DMat3::IDENTITY,
            });

        step_bevy(&mut app, 1, 0.1);

        assert!(
            app.world().entity(body).contains::<FrameAttachedC>(),
            "FrameAttachedC should be present after FrameAttachEvent processed"
        );
    }

    /// `FrameDetachEvent` removes the marker.
    #[test]
    fn detach_event_removes_marker() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JeodPlugin));

        let body = app.world_mut().spawn_empty().id();
        let parent_frame = **app.world().resource::<RootFrameEntityR>();
        app.world_mut()
            .resource_mut::<bevy::ecs::message::Messages<FrameAttachEvent>>()
            .write(FrameAttachEvent {
                body,
                parent_frame,
                offset: DVec3::ZERO,
                t_parent_body: glam::DMat3::IDENTITY,
            });

        step_bevy(&mut app, 1, 0.1);
        assert!(app.world().entity(body).contains::<FrameAttachedC>());

        app.world_mut()
            .resource_mut::<bevy::ecs::message::Messages<FrameDetachEvent>>()
            .write(FrameDetachEvent { body });

        step_bevy(&mut app, 1, 0.1);

        assert!(
            !app.world().entity(body).contains::<FrameAttachedC>(),
            "FrameAttachedC should have been removed after FrameDetachEvent"
        );
    }

    /// Two `FrameAttachEvent`s targeting the same body in the same
    /// tick must panic with the duplicate-attach diagnostic.
    ///
    /// Without per-call deduplication, only the first event's
    /// `commands.insert` lands before the apply boundary, so the
    /// component-only check (`already_frame_attached.get(...).is_err()`)
    /// passes for both events. The second event would then silently
    /// overwrite the first event's `parent_frame` / `offset` /
    /// `t_parent_body`, masking paired-event bugs in mission code.
    /// The fail-loud rule requires a panic instead.
    #[test]
    #[should_panic(expected = "already had a FrameAttachEvent processed earlier in this tick")]
    fn duplicate_attach_event_in_same_tick_panics() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JeodPlugin));

        let body = app.world_mut().spawn_empty().id();
        let parent_frame = **app.world().resource::<RootFrameEntityR>();
        let mut messages = app
            .world_mut()
            .resource_mut::<bevy::ecs::message::Messages<FrameAttachEvent>>();
        messages.write(FrameAttachEvent {
            body,
            parent_frame,
            offset: DVec3::new(1.0, 0.0, 0.0),
            t_parent_body: glam::DMat3::IDENTITY,
        });
        messages.write(FrameAttachEvent {
            body,
            parent_frame,
            offset: DVec3::new(2.0, 0.0, 0.0),
            t_parent_body: glam::DMat3::IDENTITY,
        });

        step_bevy(&mut app, 1, 0.1);
    }

    /// Two `FrameDetachEvent`s targeting the same body in the same
    /// tick must panic with the duplicate-detach diagnostic.
    ///
    /// Same in-flight `commands.remove` blind spot as the duplicate-
    /// attach case: the queued removal isn't visible to
    /// `already_frame_attached` until the next apply boundary.
    #[test]
    #[should_panic(expected = "already had a FrameDetachEvent processed earlier in this tick")]
    fn duplicate_detach_event_in_same_tick_panics() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JeodPlugin));

        let body = app.world_mut().spawn_empty().id();
        let parent_frame = **app.world().resource::<RootFrameEntityR>();
        app.world_mut()
            .resource_mut::<bevy::ecs::message::Messages<FrameAttachEvent>>()
            .write(FrameAttachEvent {
                body,
                parent_frame,
                offset: DVec3::ZERO,
                t_parent_body: glam::DMat3::IDENTITY,
            });
        step_bevy(&mut app, 1, 0.1);
        assert!(app.world().entity(body).contains::<FrameAttachedC>());

        let mut messages = app
            .world_mut()
            .resource_mut::<bevy::ecs::message::Messages<FrameDetachEvent>>();
        messages.write(FrameDetachEvent { body });
        messages.write(FrameDetachEvent { body });

        step_bevy(&mut app, 1, 0.1);
    }

    /// `FrameAttachEvent::parent_frame` must be an actual frame entity
    /// — i.e. carry the full `FrameTransC` / `FrameRotC` /
    /// `FrameAngVelC` triplet that the per-tick propagation pass
    /// reads. A bare `spawn_empty()` entity carries none of those, so
    /// passing it as `parent_frame` must panic at attach time rather
    /// than silently misbehaving later when `RelativeFrameState`
    /// walks an undefined node.
    #[test]
    #[should_panic(expected = "is not a frame entity")]
    fn attach_event_with_non_frame_parent_panics() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JeodPlugin));

        let body = app.world_mut().spawn_empty().id();
        // Bare entity — no FrameTransC / FrameRotC / FrameAngVelC.
        // This stands in for a caller that mistakenly passed a body
        // entity, a source entity, or an arbitrary placeholder.
        let bogus_parent = app.world_mut().spawn_empty().id();

        app.world_mut()
            .resource_mut::<bevy::ecs::message::Messages<FrameAttachEvent>>()
            .write(FrameAttachEvent {
                body,
                parent_frame: bogus_parent,
                offset: DVec3::ZERO,
                t_parent_body: glam::DMat3::IDENTITY,
            });

        step_bevy(&mut app, 1, 0.1);
    }

    /// Schedule-order regression for issue #309 thread 1.
    ///
    /// A frame-attached body that is also a mass-tree root with a
    /// kinematic child must propagate to its parent reference frame
    /// *before* the mass-tree kinematic walk derives its child. The
    /// schedule wires
    /// `propagate_frame_attached_state_system.before(propagate_state_from_root_system)`
    /// — without that ordering the kinematic walk would derive the
    /// child from the root's pre-frame-attach state, leaving the
    /// subtree one tick stale.
    ///
    /// Setup: spawn a frame-attached parent at a non-zero offset from
    /// the (stationary) root frame, attach a kinematic child to it via
    /// `MassChildOf` with a known link offset, run one tick, and
    /// verify the child's `TranslationalStateC` matches the analytical
    /// "frame-attach derived parent + link" composition rather than
    /// the "default-init parent + link" composition the bad order
    /// would produce.
    #[test]
    fn frame_attached_parent_propagates_before_kinematic_child() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, JeodPlugin));

        // Use the root frame (which sits at the origin and has zero
        // velocity) as the parent reference frame. The captured offset
        // becomes the body's root-inertial position and is non-zero so
        // the test distinguishes "frame-attach derived state" from a
        // default-initialized `TranslationalStateC` (which is also
        // zero — the bad order would not change the child either way
        // unless the parent's state visibly differs from the default).
        let parent_frame = **app.world().resource::<RootFrameEntityR>();
        let parent_offset = DVec3::new(1234.5, -678.9, 42.0);

        let parent_body = app
            .world_mut()
            .spawn((
                Name::new("frame_attached_root"),
                MassPropertiesC::from(MassProperties::new(10.0)),
                RotationalStateC::from_untyped(RotationalState::default()),
                TranslationalStateC::from_untyped(TranslationalState::default()),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id();

        // Kinematic child: identity link rotation, fixed structural
        // offset in the parent's frame. The expected child position is
        // `parent_offset + composite-CoM-routed link_offset`.
        //
        // Pre-insert `KinematicChildC` so the kinematic walk's
        // marker-gated writeback hits the child on tick 1 — without
        // it the marker is only installed by `wrench_aggregation_system`,
        // which runs *after* the propagation pass we're trying to
        // observe (the schedule chain is propagation → wrench, so the
        // child's first marker-gated write only lands on tick 2). The
        // schedule-order regression we're guarding against is "frame-
        // attach propagation runs before mass-tree kinematic
        // propagation", and we need the kinematic write to land on the
        // same tick as the frame-attach write to make the difference
        // observable. Pre-inserting the marker is exactly what the
        // wrench system would do on tick 2 in steady state.
        let child_link_offset = DVec3::new(0.0, 100.0, 0.0);
        let child_body = app
            .world_mut()
            .spawn((
                Name::new("kinematic_child"),
                MassPropertiesC::from(MassProperties::new(5.0)),
                MassChildOf::with_rotation(parent_body, child_link_offset, glam::DMat3::IDENTITY),
                KinematicChildC,
                RotationalStateC::from_untyped(RotationalState::default()),
                TranslationalStateC::from_untyped(TranslationalState::default()),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id();

        // Send the FrameAttachEvent and run the schedule.
        app.world_mut()
            .resource_mut::<bevy::ecs::message::Messages<FrameAttachEvent>>()
            .write(FrameAttachEvent {
                body: parent_body,
                parent_frame,
                offset: parent_offset,
                t_parent_body: glam::DMat3::IDENTITY,
            });

        // Run a single tick. With the correct ordering
        // (`propagate_frame_attached_state_system.before(propagate_state_from_root_system)`),
        // the parent's state is updated before the kinematic walk
        // reads it, so the child's derived position reflects the new
        // attach offset on the same tick. With the inverted ordering,
        // the kinematic walk reads the parent's pre-attach default
        // state (zero) and writes the child at `link_offset_routed`
        // alone — missing `parent_offset`. The two orderings only
        // diverge on the first tick of an attach (or the first tick
        // after the parent state changes); steady-state convergence
        // would mask the regression after tick 2, so the assertion
        // must fire on tick 1.
        step_bevy(&mut app, 1, 0.1);

        // The frame-attached parent's state must reflect the captured
        // offset (root frame is at origin, so root-inertial position =
        // offset).
        let parent_state = app
            .world()
            .get::<TranslationalStateC>(parent_body)
            .expect("parent body should still have TranslationalStateC");
        let parent_pos = parent_state.0.position.raw_si();
        assert!(
            (parent_pos - parent_offset).length() < 1e-9,
            "frame-attached parent must end the tick at its captured offset \
             ({parent_offset:?}); got {parent_pos:?}",
        );

        // The kinematic child must inherit the *frame-attach-derived*
        // parent state (parent_offset) plus the kernel's structural
        // routing through the parent's composite CoM. The kernel
        // computes:
        //   r_inertial_child = r_inertial_parent + T_inertial_pstr · pcm_to_ccm
        //   pcm_to_ccm = link_offset + child_composite_CoM_in_cstr
        //              - parent_composite_CoM_in_pstr
        // Both bodies have identity struct→body and a composite CoM
        // collapsed to a single weighted-sum point along the link axis.
        // With parent mass 10 at origin and child mass 5 at link_offset,
        // parent's composite CoM in its own struct = (5/15) * link_offset.
        // Child's composite CoM in its own struct = origin (no further
        // children).
        let child_state = app
            .world()
            .get::<TranslationalStateC>(child_body)
            .expect("kinematic child should still have TranslationalStateC");
        let child_pos = child_state.0.position.raw_si();
        let parent_mass = 10.0;
        let child_mass = 5.0;
        let parent_composite_cm = child_link_offset * (child_mass / (parent_mass + child_mass));
        let pcm_to_ccm = child_link_offset - parent_composite_cm;
        // Parent has identity attitude (default) so T_inertial_pstr is
        // identity — `pcm_to_ccm` is already in inertial frame.
        let expected_child_pos = parent_offset + pcm_to_ccm;
        assert!(
            (child_pos - expected_child_pos).length() < 1e-9,
            "kinematic child of a frame-attached root must derive its state from \
             the freshly-propagated parent. Expected {expected_child_pos:?} \
             (= parent_offset + composite-CoM-routed link), got {child_pos:?}. \
             If the schedule order regressed (kinematic walk before frame-attach \
             propagation), the child would read the default-zero parent state and \
             end up at the link contribution alone ({pcm_to_ccm:?}).",
        );
    }
}
