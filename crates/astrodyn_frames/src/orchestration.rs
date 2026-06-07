//! Frame-tree orchestration helpers.
//!
//! Functions every consumer of `astrodyn` (the standalone runner harness
//! and ECS adapters) needs to update planet-fixed rotations and resolve
//! frame origins against a [`FrameTree`].

use crate::{FrameId, FrameTree, RefFrameStateTyped};
use astrodyn_math::JeodQuat;
use astrodyn_quantities::aliases::{Position, Velocity};
use astrodyn_quantities::frame::Frame;
use glam::{DMat3, DVec3};

/// Sync a planet-fixed frame node's rotation state from a computed matrix.
///
/// Sets `t_parent_this`, derives `q_parent_this` from it, and sets
/// `ang_vel_this = [0, 0, planet_omega]` matching JEOD's `planet_rnp.cc`.
pub fn sync_pfix_rotation(
    frame_tree: &mut FrameTree,
    pfix_id: FrameId,
    rotation: DMat3,
    planet_omega: f64,
) {
    let node = frame_tree.get_mut(pfix_id);
    node.state.rot.t_parent_this = rotation;
    node.state.rot.q_parent_this = JeodQuat::left_quat_from_transformation(&rotation);
    // JEOD sets ang_vel_this = [0, 0, planet_omega] in planet_rnp.cc.
    // This is used by compute_relative_state velocity composition.
    node.state.rot.ang_vel_this = DVec3::new(0.0, 0.0, planet_omega);
}

/// Get the position and velocity of a frame relative to the root inertial
/// frame. Mirrors `astrodyn_runner::Simulation::frame_origin` so consumers that
/// don't own a `Simulation` (the Bevy adapter) can compute frame origins
/// using only the frame tree.
pub fn frame_origin(
    frame_tree: &FrameTree,
    root_frame_id: FrameId,
    frame_id: FrameId,
) -> (DVec3, DVec3) {
    if frame_id == root_frame_id {
        return (DVec3::ZERO, DVec3::ZERO);
    }
    let state = frame_tree.compute_relative_state(root_frame_id, frame_id);
    (state.trans.position, state.trans.velocity)
}

/// Typed sibling of [`frame_origin`] returning the frame's position and
/// velocity already wrapped as `Position<F>` / `Velocity<F>`. The frame
/// marker `F` is the **root frame's** marker (the result expresses the
/// query frame's origin in root-relative coordinates).
///
/// **Checked lookup**: the root node's stamped identity is verified
/// against `F` — a wrong marker fails loudly instead of silently
/// mislabeling the result's coordinate basis.
///
/// # Panics
/// - the root's stored identity does not match `F`.
///
/// Phase C5 of issue #71: lets consumers compute the integration-frame
/// origin without re-lifting raw `DVec3` through `from_raw_si` at the
/// hot-path boundary.
pub fn frame_origin_typed<F: Frame>(
    frame_tree: &FrameTree,
    root_frame_id: FrameId,
    frame_id: FrameId,
) -> (Position<F>, Velocity<F>) {
    // JEOD_INV: RF.02 — the result is expressed in the root's frame, so
    // the requested marker must name the root node's stamped identity.
    let root_uid = frame_tree.get(root_frame_id).uid();
    assert!(
        root_uid.is::<F>(),
        "frame_origin_typed: root frame {root_frame_id} has stored identity `{root_uid}`, but the requested marker is `{}` — the result is expressed in the root's frame, so F must name the root's identity.",
        core::any::type_name::<F>()
    );
    let (pos_raw, vel_raw) = frame_origin(frame_tree, root_frame_id, frame_id);
    (
        Position::<F>::from_raw_si(pos_raw),
        Velocity::<F>::from_raw_si(vel_raw),
    )
}

/// Typed sibling of [`FrameTree::compute_relative_state`] returning a
/// [`RefFrameStateTyped<From, To>`].
///
/// **Checked lookup**: both endpoints' stamped identities are verified
/// against the requested markers — a wrong marker fails loudly instead of
/// silently mislabeling physics.
///
/// # Panics
/// - `From` or `To` is non-mintable (a storage-boundary wildcard or
///   `IntegrationFrame`) — such a marker can never name a stored identity;
/// - either endpoint's stored identity does not match its marker.
///
/// Phase C5 of issue #71.
// JEOD_INV: RF.02 — checked typed recovery at the orchestration boundary:
// both endpoints' stored identities must equal the requested markers.
pub fn compute_relative_state_typed<From: Frame, To: Frame>(
    frame_tree: &FrameTree,
    from: FrameId,
    to: FrameId,
) -> RefFrameStateTyped<From, To> {
    use astrodyn_quantities::frame_descriptor::MintPolicy;
    let check = |id: FrameId, marker: &str, name: &str, mint: MintPolicy, is_match: bool| {
        assert!(
            matches!(mint, MintPolicy::Stable),
            "compute_relative_state_typed: {marker} marker `{name}` is not mintable (a runtime-resolved wildcard or IntegrationFrame) and can never name a stored identity. Resolve the concrete frame type and request that.",
        );
        assert!(
            is_match,
            "compute_relative_state_typed: frame {id} has stored identity `{}`, but the requested {marker} marker is `{name}` — identity mismatch.",
            frame_tree.get(id).uid(),
        );
    };
    let from_name = core::any::type_name::<From>();
    let to_name = core::any::type_name::<To>();
    check(
        from,
        "From",
        from_name,
        From::DESCRIPTOR.mint,
        frame_tree.get(from).uid().is::<From>(),
    );
    check(
        to,
        "To",
        to_name,
        To::DESCRIPTOR.mint,
        frame_tree.get(to).uid().is::<To>(),
    );
    let untyped = frame_tree.compute_relative_state(from, to);
    RefFrameStateTyped::<From, To>::from_untyped_unchecked(&untyped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RefFrameRot, RefFrameState, RefFrameTrans};
    use astrodyn_quantities::frame::{Ecef, RootInertial};
    use astrodyn_quantities::frame_descriptor::FrameUid;

    #[test]
    fn frame_origin_typed_matches_untyped() {
        let mut tree = FrameTree::new();
        let root = tree.add_root_typed::<RootInertial>("root".into());
        let child = tree.add_child_uid(
            root,
            FrameUid::of::<Ecef>(),
            "child".into(),
            RefFrameState {
                trans: RefFrameTrans {
                    position: DVec3::new(1.0e7, 2.0e7, 3.0e7),
                    velocity: DVec3::new(100.0, 200.0, 300.0),
                },
                rot: RefFrameRot::default(),
            },
            None,
        );

        let (untyped_pos, untyped_vel) = frame_origin(&tree, root, child);
        let (typed_pos, typed_vel) = frame_origin_typed::<RootInertial>(&tree, root, child);

        assert_eq!(typed_pos.raw_si(), untyped_pos);
        assert_eq!(typed_vel.raw_si(), untyped_vel);
    }

    #[test]
    fn frame_origin_typed_root_is_zero() {
        let mut tree = FrameTree::new();
        let root = tree.add_root_typed::<RootInertial>("root".into());
        let (typed_pos, typed_vel) = frame_origin_typed::<RootInertial>(&tree, root, root);
        assert_eq!(typed_pos.raw_si(), DVec3::ZERO);
        assert_eq!(typed_vel.raw_si(), DVec3::ZERO);
    }

    #[test]
    fn compute_relative_state_typed_matches_untyped() {
        let mut tree = FrameTree::new();
        let root = tree.add_root_typed::<RootInertial>("root".into());
        let child = tree.add_child_uid(
            root,
            FrameUid::of::<Ecef>(),
            "ecef".into(),
            RefFrameState {
                trans: RefFrameTrans {
                    position: DVec3::new(1.0, 2.0, 3.0),
                    velocity: DVec3::new(0.1, 0.2, 0.3),
                },
                rot: RefFrameRot::default(),
            },
            None,
        );

        let untyped = tree.compute_relative_state(root, child);
        let typed = compute_relative_state_typed::<RootInertial, Ecef>(&tree, root, child);

        assert_eq!(typed.trans.position.raw_si(), untyped.trans.position);
        assert_eq!(typed.trans.velocity.raw_si(), untyped.trans.velocity);
        assert_eq!(typed.rot.t_parent_this(), untyped.rot.t_parent_this);
    }
}
