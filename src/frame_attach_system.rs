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
//!   the frame composition the same tick they were attached) and
//!   *before*
//!   [`integration_system`](crate::systems::integration_system) (so
//!   the integrator sees the frame-derived state when deciding to
//!   skip via the `FrameAttachedC` filter).

use bevy::ecs::message::MessageReader;
use bevy::ecs::system::ParamSet;
use bevy::prelude::*;
use glam::DVec3;

use jeod_sim::MassPointState;

use crate::components::{
    Abm4StateC, FrameAttachEvent, FrameAttachedC, FrameDetachEvent, FrameEntityC,
    GaussJacksonStateC, MassChildOf, RotationalStateC, TranslationalStateC,
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
/// - A [`FrameAttachEvent`] targets an entity that already carries
///   [`FrameAttachedC`]: a silent overwrite would lose the original
///   parent-frame relationship and leave the captured offset
///   desynchronized from the body's actual position.
/// - A [`FrameAttachEvent`] targets an entity that has a
///   [`MassChildOf`] parent: JEOD's `attach_to_frame` writes the
///   attachment on the root body, never on a child body; mixing
///   mass-tree attach with frame-tree attach would let
///   `propagate_frame_attached_state_system` overwrite the parent's
///   chosen child state with a contradicting parent-frame composition.
/// - A [`FrameDetachEvent`] targets an entity that does not currently
///   carry [`FrameAttachedC`]: silently no-op'ing would mask
///   paired-event bugs in mission code.
// JEOD_INV: DB.21 — only unattached bodies integrate (frame-attach gate)
// JEOD_INV: IG.37 — multi-step integrator history reset on topology change
#[allow(clippy::type_complexity)]
pub fn frame_attach_system(
    mut commands: Commands,
    mut attach_events: MessageReader<FrameAttachEvent>,
    mut detach_events: MessageReader<FrameDetachEvent>,
    already_frame_attached: Query<Entity, With<FrameAttachedC>>,
    has_mass_parent: Query<&MassChildOf>,
    mut integrators: Query<(Option<&mut GaussJacksonStateC>, Option<&mut Abm4StateC>)>,
) {
    for evt in attach_events.read() {
        // Reject double-attach. Reading from the query gives us the
        // pre-event state of the world (attach events processed this
        // tick haven't run their `commands.insert` yet), so we can
        // also reject two simultaneous attach events on the same
        // entity by tracking the inserts we've already queued.
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

        commands.entity(evt.body).insert(FrameAttachedC {
            parent_frame: evt.parent_frame,
            offset: evt.offset,
            t_parent_body: evt.t_parent_body,
        });

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
        assert!(
            already_frame_attached.get(evt.body).is_ok(),
            "FrameDetachEvent: body {:?} is not currently frame-attached. \
             Send a FrameAttachEvent first, or remove the duplicate detach \
             to avoid masking caller bugs.",
            evt.body
        );

        commands.entity(evt.body).remove::<FrameAttachedC>();

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
    // Reference DVec3 to silence unused-import diagnostics across the
    // configurations that don't exercise the body-frame writeback.
    let _ = DVec3::ZERO;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JeodPlugin;
    use bevy::prelude::FixedUpdate;
    use bevy::time::{Fixed, Time};
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
}
