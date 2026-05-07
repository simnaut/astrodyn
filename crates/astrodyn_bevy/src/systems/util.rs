//! Shared helpers for `src/systems/`.
//!
//! Hosts the integ-origin lookup that gravity, integration, derived
//! states, and several interaction systems all need to read the body's
//! current integration frame's origin (and its time derivative) in root
//! coordinates. Stage modules access these via `super::util::...`.

use astrodyn::{Position, RootInertial, Velocity};
use bevy::prelude::*;

use crate::components::*;
use crate::frame_param::FrameOrigin;

/// Compute the typed root-inertial origin offset of `body_frame`'s
/// integration frame — the RF.10 shift that lifts a body's
/// `PlanetInertial<P>` state into absolute `RootInertial`
/// coordinates. Returns `(zero, zero)` when:
///
/// - the body has no [`FrameEntityC`] (legacy entities registered
///   before the frames-as-entities components landed are treated as
///   root-integrated), or
/// - the body's frame entity's parent is the root frame.
///
/// In both cases the integ-origin shift is identically zero, so
/// relabeling the body state to `RootInertial` is a no-op
/// numerically. For non-root-integrated bodies the shift is the
/// translational state of the integration frame relative to root,
/// supplied by the [`FrameOrigin`] SystemParam.
///
/// Mirrors the `body_integ_origins` helper that
/// `astrodyn_runner::Simulation::step_internal` builds before each shift
/// site (gravity, integration, derived states); same algorithm,
/// ECS-backed storage.
pub(crate) fn body_integ_origin_in_root(
    body_frame: Option<&FrameEntityC>,
    parents: &Query<&ChildOf>,
    root_frame_entity: Entity,
    frame_origin: &FrameOrigin,
) -> (Position<RootInertial>, Velocity<RootInertial>) {
    let integ_frame_entity =
        body_frame.and_then(|fe| parents.get(fe.0).ok().map(|child_of| child_of.parent()));
    match integ_frame_entity {
        Some(integ_e) if integ_e != root_frame_entity => {
            frame_origin.origin_in_root(root_frame_entity, integ_e)
        }
        _ => (
            Position::<RootInertial>::zero(),
            Velocity::<RootInertial>::zero(),
        ),
    }
}

/// Lazy fail-loud variant of [`body_integ_origin_in_root`] for systems
/// that take `Option<Res<RootFrameEntityR>>`: a body with a
/// `FrameEntityC` whose parent is *not* the root needs the
/// integ-origin shift, and the shift cannot be computed without the
/// root entity. Panicking here surfaces the misconfiguration at the
/// exact site where wrong physics would otherwise propagate silently
/// (per the *Fail Loudly* rule in CLAUDE.md): a non-root-integrated
/// body's `TranslationalStateC` is planet-relative, and treating
/// every integ-origin as zero would feed planet-relative coordinates
/// into a kernel that composes in root-inertial — silently producing
/// merged states off by the integration-frame's full translational
/// state (~3.8e8 m / 1 km/s for lunar bodies).
///
/// Pure root-integrated worlds (the common minimal-test shape: no
/// `AstrodynPlugin`, so no `FrameEntityC` on bodies) keep working — the
/// `body_frame.is_none()` branch returns zero without consulting the
/// root entity. Tests that exercise non-root-integrated bodies must
/// register `AstrodynPlugin` (which inserts `RootFrameEntityR` and the
/// frame-tree infrastructure) or supply an equivalent mock resource.
pub(crate) fn body_integ_origin_in_root_lazy(
    body_frame: Option<&FrameEntityC>,
    parents: &Query<&ChildOf>,
    root_frame_entity: Option<Entity>,
    frame_origin: &FrameOrigin,
) -> (Position<RootInertial>, Velocity<RootInertial>) {
    // Resolve the body's integ-frame entity (parent of its
    // `FrameEntityC` in the frame-tree). Two legitimate paths return
    // a zero origin without consulting the frame tree:
    //
    //   * `body_frame.is_none()` — the body has no `FrameEntityC` at
    //     all (minimal-test shape with no `AstrodynPlugin`); root-
    //     integrated by convention.
    //
    // A body that *does* carry `FrameEntityC` but whose frame entity
    // has no `ChildOf` parent is malformed: every frame entity must
    // be parented in the frame tree (under the root frame entity for
    // root-integrated bodies, or under a planet's inertial frame
    // entity for planet-integrated bodies). Treating that corruption
    // as "root-integrated" would silently feed planet-relative coords
    // into a kernel that composes in root-inertial — exactly the
    // failure mode the rest of the staging path rejects loudly.
    let Some(fe) = body_frame else {
        return (
            Position::<RootInertial>::zero(),
            Velocity::<RootInertial>::zero(),
        );
    };
    let integ_e = parents
        .get(fe.0)
        .map(|child_of| child_of.parent())
        .unwrap_or_else(|err| {
            panic!(
                "malformed frame tree: body's FrameEntityC ({:?}) has no ChildOf parent \
                 ({err:?}). Every body frame entity must be parented under either the root \
                 frame entity (root-integrated) or a planet's inertial frame entity \
                 (planet-integrated). Detached or freshly reparented bodies must restore \
                 the ChildOf edge before the next staging/step; treating this as \
                 root-integrated would feed planet-relative coordinates into a \
                 root-inertial kernel and silently corrupt the merged composite by the \
                 missing integ-frame's full root-inertial state. Likely cause: an attach \
                 or detach handler dropped the frame-tree reparent step.",
                fe.0,
            )
        });
    // The body has a registered frame entity. Without the root entity
    // we cannot tell whether `integ_e == root` (root-integrated, safe
    // zero shift) or `integ_e != root` (non-root, load-bearing shift).
    // Demand the resource and panic with a fix-it diagnostic if it is
    // absent — silently returning zero in the latter case would
    // corrupt the merged composite by the integration-frame's full
    // root-inertial state.
    let root = root_frame_entity.unwrap_or_else(|| {
        panic!(
            "RootFrameEntityR resource not present, but a body carries FrameEntityC \
             ({:?}) whose integ-frame parent is {integ_e:?} — the integ-origin shift \
             cannot be computed without the root frame entity. AstrodynPlugin must be \
             loaded for systems that lift integration-frame coordinates to \
             root-inertial (staging_system, step_detached_system). If your test \
             intentionally omits AstrodynPlugin, also omit FrameEntityC from the body \
             (root-integrated bodies skip this path entirely).",
            fe.0,
        )
    });
    if integ_e == root {
        (
            Position::<RootInertial>::zero(),
            Velocity::<RootInertial>::zero(),
        )
    } else {
        frame_origin.origin_in_root(root, integ_e)
    }
}
