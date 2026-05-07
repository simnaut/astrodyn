//! Glue: ECS-frame-entity cleanup on owner despawn.
//!
//! Frame entities are owned by the source / body / pfix entity that
//! references them via [`FrameEntityC`] / [`PfixFrameEntityC`] /
//! [`RetiredPfixFrameEntityC`]. When the owner despawns, the referenced
//! frame entity must despawn alongside it; these observers wire that.

use bevy::prelude::*;

#[allow(unused_imports)] // intra-doc-link resolution
use super::frame_registration::{
    register_body_frames_system, register_pfix_frames_system, register_source_frames_system,
};
use crate::components::*;

/// On entity despawn, despawn the orphan pfix *frame entity* stashed
/// in [`RetiredPfixFrameEntityC`] (left over from a
/// `RotationModel::None` toggle that wasn't followed by a re-toggle
/// before despawn). The orphan ECS entity has no other owner — it
/// was kept alive specifically so the next `None → rotating` retoggle
/// could reuse it — so without this observer it would leak when the
/// owning source despawns.
///
/// `try_despawn` (not `despawn`) because the retired pfix entity's
/// `ChildOf` parent is the source frame entity, which is despawned
/// recursively by [`on_frame_entity_despawn`] when the source
/// despawns; the retired pfix may already be gone by the time this
/// observer's command flushes.
pub fn on_retired_pfix_frame_entity_despawn(
    trigger: On<Despawn, RetiredPfixFrameEntityC>,
    sources: Query<&RetiredPfixFrameEntityC>,
    mut commands: Commands,
) {
    if let Ok(retired) = sources.get(trigger.entity) {
        commands.entity(retired.0).try_despawn();
    }
}

/// On entity despawn, despawn the *frame entity* the source / body
/// entity carries in [`FrameEntityC`]. Without this observer,
/// despawning a source or body would leave its dual-write frame
/// entity (and the pfix child it parents, when present) alive
/// indefinitely under the root frame entity, growing the entity
/// count over time and potentially shadowing future re-spawns of
/// the same `Name`.
///
/// Fires for *any* entity that carries [`FrameEntityC`], i.e. both
/// source entities (registered by [`register_source_frames_system`])
/// and body entities (registered by [`register_body_frames_system`]).
/// The cleanup logic is identical for the two cases — the despawning
/// entity hands us its frame-entity handle and we tear down the
/// referenced frame entity — so the observer is named for the
/// component it watches, not for either of the owner kinds (a
/// previous name `on_source_frame_entity_despawn` misled readers
/// into thinking the observer only handled sources).
///
/// `try_despawn` (not `despawn`) because Bevy's `ChildOf` /
/// `Children` relationship already triggers recursive despawn on the
/// frame entity's children — the pfix child of a source frame, the
/// body frame entity if a body shares the integration frame entity
/// — so a sibling observer ([`on_source_pfix_frame_entity_despawn`])
/// firing on the same entity-despawn event may find its target
/// already queued for despawn. `try_despawn` silently no-ops in that
/// case.
///
/// Pairs with [`on_source_pfix_frame_entity_despawn`] (covers the
/// source's pfix child) and
/// [`on_retired_pfix_frame_entity_despawn`] (covers a stashed
/// orphan from a `RotationModel::None` toggle that wasn't followed
/// by a re-toggle before despawn) to provide complete cleanup for
/// the spawn sites in [`register_source_frames_system`],
/// [`register_pfix_frames_system`], and
/// [`register_body_frames_system`].
pub fn on_frame_entity_despawn(
    trigger: On<Despawn, FrameEntityC>,
    owners: Query<&FrameEntityC>,
    mut commands: Commands,
) {
    if let Ok(frame) = owners.get(trigger.entity) {
        commands.entity(frame.0).try_despawn();
    }
}

/// On entity despawn, despawn the pfix *frame entity* the source
/// entity carries in [`PfixFrameEntityC`]. Pair to
/// [`on_frame_entity_despawn`] for the source's pfix child.
///
/// Independent of [`on_frame_entity_despawn`] so the
/// per-component `Despawn` order doesn't matter: in the common case
/// the pfix entity is `ChildOf(source_frame_entity)` and gets
/// despawned recursively when its parent does, but this observer is
/// the safety net for any future configuration where the pfix entity
/// is parented elsewhere (or for entities that hold
/// `PfixFrameEntityC` without `FrameEntityC`). `try_despawn` silently
/// no-ops when the recursive despawn has already claimed the entity.
pub fn on_source_pfix_frame_entity_despawn(
    trigger: On<Despawn, PfixFrameEntityC>,
    owners: Query<&PfixFrameEntityC>,
    mut commands: Commands,
) {
    if let Ok(frame) = owners.get(trigger.entity) {
        commands.entity(frame.0).try_despawn();
    }
}
