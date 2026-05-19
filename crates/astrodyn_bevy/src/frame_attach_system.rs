//! Bevy systems for frame-attached body integration.
//!
//! Bevy adapter for the runner's
//! `Simulation::attach_to_frame` API (the runner-side equivalent in
//! `astrodyn_runner::Simulation`). The two consumers (Bevy and runner) drive the same
//! `astrodyn_dynamics::derive_frame_attached_state` kernel against the
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
//! - [`frame_attach_system`] is pinned between
//!   [`AstrodynSet::EphemerisUpdate`](crate::AstrodynSet::EphemerisUpdate) and
//!   [`AstrodynSet::Environment`](crate::AstrodynSet::Environment) so
//!   attach/detach events take effect on the same tick they were
//!   dispatched and the propagation pass below sees the freshly-
//!   processed events. Mirrors the runner's pre-stage-3a event
//!   application.
//! - [`propagate_frame_attached_state_system`] runs *after*
//!   `frame_attach_system` (so freshly-attached bodies pick up the
//!   frame composition the same tick they were attached) and
//!   *before* [`AstrodynSet::Environment`](crate::AstrodynSet::Environment)
//!   so gravity / atmosphere read the parent-frame-derived body
//!   state rather than a one-tick-stale composition. Pre-Environment
//!   placement also keeps [`AstrodynSet::Interaction`](crate::AstrodynSet::Interaction)
//!   consumers (drag, SRP, gravity-torque) and the pre-integration
//!   kinematic walk
//!   ([`propagate_state_from_root_system`](crate::propagate_state_from_root_system))
//!   reading the freshly-derived root state — a frame-attached
//!   body that is also a mass-tree root would otherwise hand its
//!   kinematic descendants a stale pre-frame-attach root state.
//!   Mirrors stage 3a of the runner's `Simulation::step_internal` in
//!   `crates/astrodyn_runner/src/simulation/step/mod.rs`.
//! - [`propagate_frame_attached_state_post_integration_system`] runs
//!   in [`AstrodynSet::Integration`](crate::AstrodynSet::Integration) *after*
//!   `sync_body_to_frame_system` and `frame_switch_system` (so the
//!   frame-tree state the kernel reads reflects the just-finished
//!   intra-step updates: ephemeris advance, planet-fixed rotation
//!   update, frame-switch reparent), and *before*
//!   [`AstrodynSet::DerivedState`](crate::AstrodynSet::DerivedState) (so
//!   `orbital_elements_system` / `geodetic_system` / `lvlh_system` /
//!   `solar_beta_system` / `earth_lighting_system` observe the
//!   parent reference frame's *current* state rather than a
//!   one-tick-stale composition). Mirrors stage 8c of the runner's
//!   `Simulation::step_internal` in
//!   `crates/astrodyn_runner/src/simulation/step/mod.rs`.

use std::collections::HashSet;

use bevy::ecs::message::MessageReader;
use bevy::ecs::system::ParamSet;
use bevy::prelude::*;

use astrodyn::{MassPointState, Planet};

use crate::components::{
    Abm4StateC, FrameAngVelC, FrameAttachEvent, FrameAttachedC, FrameDetachEvent, FrameEntityC,
    FrameRotC, FrameTransC, GaussJacksonStateC, MassChildOf, MassPropertiesC, RotationalStateC,
    TranslationalStateC,
};
use crate::frame_param::RelativeFrameState;
use crate::RootFrameEntityR;
use glam::DVec3;

/// Process [`FrameAttachEvent`] / [`FrameDetachEvent`] messages by
/// inserting / removing [`FrameAttachedC`] and resetting multi-step
/// integrator history on the body entity.
///
/// Bevy adapter for `Simulation::attach_to_frame` /
/// `Simulation::detach_from_frame`. Schedule placement: pinned
/// between [`AstrodynSet::EphemerisUpdate`](crate::AstrodynSet::EphemerisUpdate)
/// and [`AstrodynSet::Environment`](crate::AstrodynSet::Environment) so
/// attach/detach events take effect on the same tick they're
/// dispatched and the per-tick propagation pass downstream sees the
/// freshly-processed events.
///
/// # Panics
///
/// Panics with a "Fail Loudly" diagnostic when:
/// - A [`FrameAttachEvent`]'s `body` is not a body entity — defined
///   here as carrying [`TranslationalStateC`]. Attaching the
///   [`FrameAttachedC`] marker to an entity that lacks the body-state
///   storage would silently mis-route the per-tick propagation:
///   `propagate_frame_attached_state_system` would not find a
///   `TranslationalStateC` to overwrite, the body would never observe
///   the parent-frame composition, and any consumer that *does* read
///   the marker (the integration gate, derived-state systems) would
///   skip the entity. Catching the misconfiguration at attach time
///   surfaces it as a loud "wrong target" error pointing at the body
///   id and the missing component instead of a quiet trajectory
///   divergence dozens of ticks downstream. The same check rejects a
///   frame entity (which carries [`FrameTransC`]) being passed as the
///   attachment target — a frame entity is not a body, and the
///   propagation pass's `Without<FrameTransC>` filter would silently
///   skip it.
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
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn frame_attach_system<P: Planet>(
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
    // Body-shape check for `evt.body`. The propagation pass owns a
    // `Query<&mut TranslationalStateC<P>, Without<FrameTransC>>`, so a
    // valid attach target must (a) carry `TranslationalStateC<P>` so
    // the writeback lands and (b) not carry `FrameTransC` so the
    // propagation query's filter doesn't silently drop the entity.
    // Detect either mismatch here and reject at event time. The
    // `Has<_>` access pattern keeps the query disjoint from the
    // mutable writebacks in the same system.
    body_components: Query<(
        bevy::ecs::query::Has<TranslationalStateC<P>>,
        bevy::ecs::query::Has<FrameTransC>,
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
        // Validate `evt.body` is a body entity — i.e. carries
        // `TranslationalStateC` and is not itself a frame entity
        // (`FrameTransC`). The propagation pass owns
        // `Query<&mut TranslationalStateC, Without<FrameTransC>>`, so
        // an entity that lacks the storage (or that the filter
        // excludes) would never receive the parent-frame composition
        // — `propagate_frame_attached_state_system` would silently
        // skip it while the integration gate happily kept the
        // (now-marker-bearing) entity out of integration. Surface
        // both mismatches here so the diagnostic names the offending
        // entity and the structural reason instead of letting a
        // misconfigured target produce a quiet, never-updated body.
        // `Query::get` returns `Err` for despawned entities; treat
        // that the same as "missing TranslationalStateC" so the
        // caller learns whichever invariant they broke.
        let (body_has_trans, body_has_frame_trans) =
            body_components.get(evt.body).unwrap_or((false, false));
        assert!(
            body_has_trans && !body_has_frame_trans,
            "FrameAttachEvent: body {:?} is not a valid body entity \
             (TranslationalStateC present: {}, FrameTransC present: {}). \
             A frame-attach target must carry `TranslationalStateC` so the \
             per-tick propagation can overwrite its state, and must NOT carry \
             `FrameTransC` (the propagation query is filtered \
             `Without<FrameTransC>` to keep body and frame state disjoint). \
             Pass the entity id of a body spawned via the typestate \
             `VehicleBuilder` (or a manually-spawned body that includes \
             `TranslationalStateC` / `RotationalStateC`) — not a frame entity, \
             a source entity, or an arbitrary placeholder. Parent frame {:?}.",
            evt.body,
            body_has_trans,
            body_has_frame_trans,
            evt.parent_frame,
        );

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
/// after `frame_attach_system` (so events processed this tick take
/// effect immediately) and before
/// [`AstrodynSet::Environment`](crate::AstrodynSet::Environment) so gravity
/// and atmosphere read the freshly-derived parent-frame composition
/// rather than a one-tick-stale body state. The pre-Environment slot
/// also fronts every downstream consumer that reads body state
/// in-tick: [`AstrodynSet::Interaction`](crate::AstrodynSet::Interaction)
/// (drag / SRP / gravity-torque), the pre-integration kinematic
/// walk ([`propagate_state_from_root_system`](crate::propagate_state_from_root_system)),
/// and [`integration_system`](crate::systems::integration_system)
/// (which then skips frame-attached bodies via the [`FrameAttachedC`]
/// filter).
///
/// Fast-paths to a no-op when no entity carries [`FrameAttachedC`].
// JEOD_INV: DB.13 — propagate_state delegates to parent frame
// JEOD_INV: DB.14 — integration-frame switch on attach: the attached body's
//   state is owned by the parent frame, not the integrator
// JEOD_INV: RF.10 — frame-attach is a shift site: the parent frame state is
//   computed in root-inertial coords via `RelativeFrameState(root, parent)`,
//   the kernel composes with the offset in those coords, and the result is
//   lowered through the body's `IntegOrigin` before writing back to
//   `TranslationalStateC` (whose storage convention is the body's
//   integration frame). Bit-identical no-op for root-integrated bodies
//   (origin = zero); load-bearing for any body whose `IntegSourceC` is a
//   non-root planet. Mirrors the runner's writeback in
//   `crates/astrodyn_runner/src/simulation/frame_attach.rs:335-339`.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn propagate_frame_attached_state_system<P: Planet>(
    attached: Query<(Entity, &FrameAttachedC)>,
    // Body entities only — exclude frame entities (which carry
    // `FrameTransC`) so `state_q` is statically disjoint from the
    // frame-state query in the `ParamSet` below.
    mut state_q: Query<
        (
            &mut TranslationalStateC<P>,
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
            Without<TranslationalStateC<P>>,
        >,
    )>,
    // Body→frame-entity lookup and the frame-tree parent walk are
    // needed to identify the body's integration frame and lower the
    // kernel's root-inertial output through its `IntegOrigin`. Both are
    // read-only and disjoint from the writable frame query gated by
    // the `ParamSet` above.
    body_frames: Query<&FrameEntityC>,
    parents: Query<&ChildOf>,
    // Per-body mass properties — needed for the kernel's
    // structure → composite CoM correction
    // (JEOD `dyn_body_propagate_state.cc:571`'s
    // `compute_derived_state_forward(structure,
    // mass.composite_properties, composite_body)`). The captured
    // `attach_offset` is the parent → struct pose; without the body's
    // CoM offset the composite-body writeback would land
    // `|composite_properties.position|` away from JEOD's value
    // whenever the CoM is offset from the structural origin. Bodies
    // without a `MassPropertiesC` get a zero CoM offset (matches
    // JEOD's default-constructed `composite_properties`).
    body_mass: Query<&MassPropertiesC>,
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
    //
    // Each `AttachWork` also caches the body's `IntegOrigin` —
    // `(position, velocity)` of the body's integration frame in
    // root-inertial coordinates — so the writeback can lower the
    // kernel's root-inertial output back into the integration-frame
    // storage convention without re-acquiring the read-side frame
    // query.
    struct AttachWork {
        body_entity: Entity,
        derived: astrodyn::RefFrameState,
        integ_origin_pos: DVec3,
        integ_origin_vel: DVec3,
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

            // Body's structure → composite-CoM offset for the
            // kernel's struct → composite step (`dyn_body_propagate_state.cc:571`'s
            // `compute_derived_state_forward(structure,
            // mass.composite_properties, composite_body)`). Bodies
            // without a `MassPropertiesC` get a zero offset (matches
            // JEOD's default-constructed `composite_properties`); for
            // bodies that do carry one, `to_untyped()` exposes the
            // CoM position and struct → body rotation in the same
            // shape JEOD's `MassPoint` uses.
            let composite_offset = body_mass
                .get(body_entity)
                .ok()
                .map(|mp| {
                    // allowed: typed↔raw kernel boundary
                    let untyped = astrodyn::typed_bridge::mass_typed_to_raw(&mp.0);
                    MassPointState {
                        position: untyped.position,
                        t_parent_this: untyped.t_parent_this,
                    }
                })
                .unwrap_or_default();
            // JEOD_INV: DB.31 — frame-attach derives composite from
            // captured struct offset.
            let derived = astrodyn::derive_frame_attached_state(astrodyn::FrameAttachInputs {
                parent_frame: parent_state,
                attach_offset: MassPointState {
                    position: attach.offset,
                    t_parent_this: attach.t_parent_body,
                },
                composite_offset,
            });

            // Body's integration-frame origin in root-inertial.
            // Resolved via the body's `FrameEntityC` parent (the body
            // frame is `ChildOf` its integ frame); identity zero when
            // the body has no `FrameEntityC` (legacy minimal config) or
            // the integ frame is the root frame itself.
            let integ_frame_entity = body_frames
                .get(body_entity)
                .ok()
                .and_then(|fe| parents.get(fe.0).ok().map(|child_of| child_of.parent()));
            let (integ_origin_pos, integ_origin_vel) = match integ_frame_entity {
                Some(integ_e) if integ_e != root => {
                    let trans = rel.relative_state(root, integ_e).trans;
                    (trans.position, trans.velocity)
                }
                _ => (DVec3::ZERO, DVec3::ZERO),
            };

            work.push(AttachWork {
                body_entity,
                derived,
                integ_origin_pos,
                integ_origin_vel,
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
        integ_origin_pos,
        integ_origin_vel,
    } in &work
    {
        // The body-shape validation in `frame_attach_system` rejects
        // any attach target that lacks `TranslationalStateC` or that
        // happens to carry `FrameTransC` (the propagation query's
        // `Without<FrameTransC>` filter). By the time this writeback
        // runs the marker has been observed — the entity was a valid
        // body when the marker was inserted, and there is no in-tree
        // path that mutates the marker except the matched detach
        // event. A `get_mut` failure here therefore means an upstream
        // invariant has broken: silently skipping it would leave the
        // body's state stuck at whatever value preceded the corrupt
        // tick while the marker still gates it out of integration —
        // exactly the silent-garbage-state failure mode the fail-loud
        // rule forbids. Panic with a diagnostic that names the body
        // and points at the most likely caller bugs (manual marker
        // insertion, despawn-without-detach) rather than continuing.
        let (mut trans, rot_opt, frame_opt) = state_q.get_mut(*body_entity).unwrap_or_else(|err| {
            panic!(
                "propagate_frame_attached_state_system: body {body_entity:?} \
                 carries FrameAttachedC but the body-state query returned \
                 {err:?}. The attach-time validation in `frame_attach_system` \
                 rejects bodies that lack TranslationalStateC, so this means \
                 either the marker was inserted manually (forbidden — use \
                 FrameAttachEvent / FrameDetachEvent) or the body was \
                 despawned without a paired FrameDetachEvent. Send a \
                 FrameDetachEvent before despawning a frame-attached body."
            )
        });
        // The kernel produces composite-body inertial state in
        // root-inertial coords (the parent frame state was read in
        // root-inertial via `RelativeFrameState`). `TranslationalStateC`
        // storage is pinned to the body's integration frame, so the
        // root-inertial output must be lowered through the body's
        // `IntegOrigin` before stamping the wildcard
        // `PlanetInertial<SelfPlanet>` phantom. For root-integrated
        // bodies the origin is zero and the subtraction is a numerical
        // no-op; for a body with `IntegSourceC(Some(planet))` it is
        // the only thing that keeps the typed phantom and the actual
        // coordinates consistent. Mirrors the runner's writeback in
        // `crates/astrodyn_runner/src/simulation/frame_attach.rs:335-339`
        // (`from_inertial(trans_root, integ_origin)`) — same arithmetic
        // shape, ECS-backed origin lookup.
        // JEOD_INV: RF.10 — root-inertial-shift consumer: the kernel
        // returns root-inertial composite-body state; the writeback
        // lowers through the body's `IntegOrigin` to match
        // `TranslationalStateC`'s integration-frame storage.
        let derived_trans = astrodyn::TranslationalState {
            position: derived.trans.position - *integ_origin_pos,
            velocity: derived.trans.velocity - *integ_origin_vel,
        };
        // allowed: typed↔raw kernel boundary (#397)
        trans.0 = astrodyn::typed_bridge::trans_raw_to_planet::<P>(&derived_trans);

        if let Some(mut rot) = rot_opt {
            let derived_rot = astrodyn::RotationalState {
                quaternion: derived.rot.q_parent_this,
                ang_vel_body: derived.rot.ang_vel_this,
            };
            // allowed: typed↔raw kernel boundary (#397)
            rot.0 = astrodyn::typed_bridge::rot_raw_to_self_ref(&derived_rot);
        }

        // Sync the body's frame entity (if it has one) so frame-tree
        // consumers see the same value as `TranslationalStateC`.
        // `FrameTransC` is parent-relative in the parent frame's
        // coordinates (relative to the body frame's `ChildOf` parent —
        // see `components::FrameTransC` doc and `sync_body_to_frame_system`).
        // The body frame entity is `ChildOf(integ_frame_entity)`
        // (set by `register_body_frames_system`; frame-attach does not
        // reparent the frame node), so the parent-relative value is
        // the kernel's root-inertial output lowered through the
        // `IntegOrigin` — i.e. the same `derived_trans` already written
        // into `TranslationalStateC` above. Mirrors the runner's
        // `node.state.trans = bodies[idx].trans` line in
        // `crates/astrodyn_runner/src/simulation/frame_attach.rs:369-370`.
        //
        // Fail loudly when `FrameEntityC` points at an entity that
        // isn't a frame: skipping silently would leave the frame tree
        // stale for the rest of the tick (consumers reading through
        // `RelativeFrameState` see a snapshot pre-dating this
        // attach-step propagation) and surface as a panic in some
        // downstream system instead of pointing at the actual root
        // cause. The two failure modes the diagnostic names — a
        // despawned frame entity and a misconfigured `FrameEntityC`
        // pointing at a non-frame entity — are exactly the
        // misconfigurations the fail-loud rule requires the system to
        // panic on at the point of detection.
        // JEOD_INV: RF.10 — root-inertial-shift consumer: `FrameTransC`
        // is parent-relative against the body frame's `ChildOf`
        // (= integ frame), so the writeback uses the lowered value.
        if let Some(frame_entity) = frame_opt.map(|f| f.0) {
            let (mut frame_trans, frame_rot, frame_angvel) = frame_writeback_q
                .get_mut(frame_entity)
                .unwrap_or_else(|err| {
                    panic!(
                        "propagate_frame_attached_state_system: body {body_entity:?} \
                         carries FrameEntityC({frame_entity:?}) but the frame-state \
                         writeback query returned {err:?}. The frame entity must \
                         carry FrameTransC (and the optional FrameRotC / \
                         FrameAngVelC) — either the entity has been despawned \
                         without clearing FrameEntityC on the body, or \
                         FrameEntityC was set to an entity that was never \
                         spawned as a frame. Spawn frame entities through \
                         `register_*_frames_system` (which inserts the full \
                         FrameTrans/Rot/AngVel triple) and clear FrameEntityC \
                         before despawning the frame entity."
                    )
                });
            frame_trans.position = derived_trans.position;
            frame_trans.velocity = derived_trans.velocity;
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

/// Post-integration twin of [`propagate_frame_attached_state_system`].
///
/// Re-runs the same parent-frame → body composition after
/// `integration_system` + `sync_body_to_frame_system` + `frame_switch_system`
/// have landed, so any consumer reading body state in
/// [`AstrodynSet::DerivedState`](crate::AstrodynSet::DerivedState) sees a
/// frame-attached body whose state reflects the just-finished frame-tree
/// updates and the parent reference frame's current state. Mirrors
/// stage 8c of the runner's `Simulation::step_internal` (see
/// `crates/astrodyn_runner/src/simulation/step/mod.rs`), which fires
/// `propagate_frame_attached_state` both before and after integration.
///
/// Without this pass the parent frame's intra-step changes (ephemeris
/// advance, planet-fixed rotation update, frame-switch reparent) only
/// propagate into the attached body on the *next* tick — derived states
/// (`orbital_elements_system`, `geodetic_system`, …) would observe a
/// one-tick-stale body state.
///
/// Distinct fn from the pre-integration sibling so Bevy's
/// `SystemTypeSet` treats the two registrations as independent system
/// instances; the body delegates to the same logic to keep the two
/// passes byte-for-byte equivalent.
// JEOD_INV: DB.13 — propagate_state delegates to parent frame
// JEOD_INV: DB.21 — frame-attached bodies are not integrated; the post-integration
//   sweep refreshes their state from the parent frame after the frame tree's
//   own intra-step updates (ephemeris / pfix rotation / frame switch) so
//   derived-state consumers don't observe a one-tick-stale composition.
// JEOD_INV: RF.10 — same shift-site reasoning as the pre-integration sibling.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn propagate_frame_attached_state_post_integration_system<P: Planet>(
    attached: Query<(Entity, &FrameAttachedC)>,
    state_q: Query<
        (
            &mut TranslationalStateC<P>,
            Option<&mut RotationalStateC>,
            Option<&FrameEntityC>,
        ),
        Without<crate::components::FrameTransC>,
    >,
    frame_qs: ParamSet<(
        RelativeFrameState,
        Query<
            (
                &'static mut crate::components::FrameTransC,
                Option<&'static mut crate::components::FrameRotC>,
                Option<&'static mut crate::components::FrameAngVelC>,
            ),
            Without<TranslationalStateC<P>>,
        >,
    )>,
    body_frames: Query<&FrameEntityC>,
    parents: Query<&ChildOf>,
    body_mass: Query<&MassPropertiesC>,
    root_frame: Res<RootFrameEntityR>,
) {
    propagate_frame_attached_state_system::<P>(
        attached,
        state_q,
        frame_qs,
        body_frames,
        parents,
        body_mass,
        root_frame,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{
        DynamicsConfigC, ExternalForceC, ExternalTorqueC, FrameDerivativesC, KinematicChildC,
        MassChildOf, MassPropertiesC, RotationalStateC, TotalForceC, TranslationalStateC,
    };
    use crate::test_utils::create_astrodyn_test_app;
    use crate::IntegrationDtR;
    use astrodyn::{MassProperties, RotationalState, TranslationalState};
    use bevy::prelude::FixedUpdate;
    use bevy::time::{Fixed, Time};
    use glam::DVec3;
    use std::time::Duration;

    fn step_bevy(app: &mut App, n: usize, dt: f64) {
        // Keep `IntegrationDtR` in sync with the caller's `dt` so the
        // pipeline systems see the exact f64 the test passed in
        // (mirrors `AstrodynAppExt::step_fixed_dt`).
        app.insert_resource(IntegrationDtR(dt));
        for _ in 0..n {
            app.world_mut()
                .resource_mut::<Time<Fixed>>()
                .advance_by(Duration::from_secs_f64(dt));
            app.world_mut().run_schedule(FixedUpdate);
        }
    }

    // Minimal body bundle used by the smoke tests below. The
    // attach-time validation now requires `TranslationalStateC`
    // (and rejects `FrameTransC` on the body), which is exactly
    // the body-shape contract production code already produces
    // through the typestate `VehicleBuilder`. Keeping the bundle
    // local to tests avoids dragging the full vehicle composer
    // into a smoke test.
    fn spawn_test_body(app: &mut App) -> Entity {
        app.world_mut()
            .spawn((
                MassPropertiesC::from(astrodyn::typed_bridge::mass_raw_to_self_ref(
                    &(MassProperties::new(1.0)),
                )),
                RotationalStateC::from(astrodyn::typed_bridge::rot_raw_to_self_ref(
                    &(RotationalState::default()),
                )),
                TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState::default()),
                TotalForceC::default(),
                FrameDerivativesC::default(),
                DynamicsConfigC::default(),
                ExternalForceC::default(),
                ExternalTorqueC::default(),
            ))
            .id()
    }

    /// Smoke test: spawning a body, dispatching a `FrameAttachEvent`,
    /// and stepping `FixedUpdate` once must result in the body
    /// carrying `FrameAttachedC` after the event is processed.
    #[test]
    fn attach_event_inserts_marker() {
        let mut app = create_astrodyn_test_app();

        let body = spawn_test_body(&mut app);
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
        let mut app = create_astrodyn_test_app();

        let body = spawn_test_body(&mut app);
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
        let mut app = create_astrodyn_test_app();

        let body = spawn_test_body(&mut app);
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
        let mut app = create_astrodyn_test_app();

        let body = spawn_test_body(&mut app);
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
        let mut app = create_astrodyn_test_app();

        let body = spawn_test_body(&mut app);
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

    /// `FrameAttachEvent::body` must be an actual body entity — i.e.
    /// carry `TranslationalStateC` and *not* carry `FrameTransC`. A
    /// bare `spawn_empty()` entity carries neither component, so
    /// passing it as `body` must panic at attach time. Without the
    /// validation the marker would land on an entity that the
    /// per-tick propagation query (`Without<FrameTransC>` filtered
    /// over `&mut TranslationalStateC`) cannot match — the marker
    /// would gate the entity out of integration while the propagation
    /// pass silently skipped it, leaving the body's state stuck and
    /// every downstream consumer reading the stale value.
    #[test]
    #[should_panic(expected = "is not a valid body entity")]
    fn attach_event_with_non_body_target_panics() {
        let mut app = create_astrodyn_test_app();

        // Bare entity — no TranslationalStateC. Stands in for a
        // caller that mistakenly passed a frame entity, a source
        // entity, or an arbitrary placeholder.
        let bogus_body = app.world_mut().spawn_empty().id();
        let parent_frame = **app.world().resource::<RootFrameEntityR>();

        app.world_mut()
            .resource_mut::<bevy::ecs::message::Messages<FrameAttachEvent>>()
            .write(FrameAttachEvent {
                body: bogus_body,
                parent_frame,
                offset: DVec3::ZERO,
                t_parent_body: glam::DMat3::IDENTITY,
            });

        step_bevy(&mut app, 1, 0.1);
    }

    /// `FrameAttachEvent::body` must not be a frame entity (i.e. must
    /// not carry `FrameTransC`). Frame entities are the *parents* of
    /// frame attachment, not the targets — the propagation pass owns
    /// `Query<&mut TranslationalStateC, Without<FrameTransC>>`, so
    /// the filter would silently skip a frame-entity target even if
    /// it happened to also carry `TranslationalStateC` somehow. The
    /// fail-loud rule requires a panic at attach time instead.
    #[test]
    #[should_panic(expected = "is not a valid body entity")]
    fn attach_event_with_frame_entity_target_panics() {
        let mut app = create_astrodyn_test_app();

        // Use the root frame as the (invalid) body target. The root
        // frame is a frame entity (carries the FrameTransC / FrameRotC
        // / FrameAngVelC triplet) — passing it as `body` is exactly
        // the misconfiguration the validation must catch.
        let bogus_body = **app.world().resource::<RootFrameEntityR>();
        let parent_frame = bogus_body;

        app.world_mut()
            .resource_mut::<bevy::ecs::message::Messages<FrameAttachEvent>>()
            .write(FrameAttachEvent {
                body: bogus_body,
                parent_frame,
                offset: DVec3::ZERO,
                t_parent_body: glam::DMat3::IDENTITY,
            });

        step_bevy(&mut app, 1, 0.1);
    }

    /// Schedule-order regression: frame-attach must precede the
    /// mass-tree kinematic walk.
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
        let mut app = create_astrodyn_test_app();

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
                MassPropertiesC::from(astrodyn::typed_bridge::mass_raw_to_self_ref(
                    &(MassProperties::new(10.0)),
                )),
                RotationalStateC::from(astrodyn::typed_bridge::rot_raw_to_self_ref(
                    &(RotationalState::default()),
                )),
                TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState::default()),
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
                MassPropertiesC::from(astrodyn::typed_bridge::mass_raw_to_self_ref(
                    &(MassProperties::new(5.0)),
                )),
                MassChildOf::with_rotation(parent_body, child_link_offset, glam::DMat3::IDENTITY),
                KinematicChildC,
                RotationalStateC::from(astrodyn::typed_bridge::rot_raw_to_self_ref(
                    &(RotationalState::default()),
                )),
                TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState::default()),
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

        // The frame-attached parent's state is the composite-body
        // inertial position derived from the captured offset (treated
        // as the parent's structure pose in the parent reference
        // frame, mirroring JEOD `frame_attach.initialize_attachment`)
        // composed with the parent's structure → composite CoM offset
        // (mirroring JEOD `propagate_state_from_structure`'s
        // `compute_derived_state_forward(structure,
        // mass.composite_properties, composite_body)` step). With
        // parent mass 10 at struct origin and child mass 5 at
        // `link_offset`, the parent's composite CoM in struct frame is
        // `(5/15) * link_offset`; identity attitude makes the rotation
        // a no-op, so `parent_pos = parent_offset + parent_composite_cm`.
        let parent_state = app
            .world()
            .get::<TranslationalStateC<astrodyn::Earth>>(parent_body)
            .expect("parent body should still have TranslationalStateC");
        let parent_pos = parent_state.0.position.raw_si();
        let parent_mass = 10.0;
        let child_mass = 5.0;
        let parent_composite_cm = child_link_offset * (child_mass / (parent_mass + child_mass));
        let expected_parent_pos = parent_offset + parent_composite_cm;
        assert!(
            (parent_pos - expected_parent_pos).length() < 1e-9,
            "frame-attached parent must end the tick at its struct → composite \
             derived state. Expected {expected_parent_pos:?} (= captured offset \
             {parent_offset:?} + composite CoM {parent_composite_cm:?}); \
             got {parent_pos:?}",
        );

        // The kinematic child must inherit the *frame-attach-derived*
        // parent state plus the kernel's structural routing through
        // the parent's composite CoM. With the captured offset now
        // treated as the parent's struct pose, the child's
        // composite-body position simplifies to
        // `parent_offset + child_link_offset + child_composite_CoM`
        // (= `parent.struct + struct → child.struct → child.composite`).
        // The intermediate composite-CoM contributions cancel
        // algebraically: any shift the parent's composite CoM adds to
        // the parent's writeback is undone when the kinematic walk
        // routes back through the same offset to the child's struct.
        // The child has no children and identity orientation, so its
        // composite CoM is at its struct origin and the expected
        // composite is `parent_offset + child_link_offset`.
        let child_state = app
            .world()
            .get::<TranslationalStateC<astrodyn::Earth>>(child_body)
            .expect("kinematic child should still have TranslationalStateC");
        let child_pos = child_state.0.position.raw_si();
        let expected_child_pos = parent_offset + child_link_offset;
        assert!(
            (child_pos - expected_child_pos).length() < 1e-9,
            "kinematic child of a frame-attached root must derive its state from \
             the freshly-propagated parent. Expected {expected_child_pos:?} \
             (= parent_offset + child_link_offset), got {child_pos:?}. \
             If the schedule order regressed (kinematic walk before frame-attach \
             propagation), the child would read the default-zero parent state \
             and miss the parent_offset contribution.",
        );
    }
}
