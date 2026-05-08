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
/// query frame's origin in root-relative coordinates) — caller asserts
/// the root is `F`. For the Bevy adapter and `astrodyn_runner` today, the
/// root is by construction inertial, so callers pass `F = Inertial`.
///
/// Phase C5 of issue #71: lets consumers compute the integration-frame
/// origin without re-lifting raw `DVec3` through `from_raw_si` at the
/// hot-path boundary.
pub fn frame_origin_typed<F: Frame>(
    frame_tree: &FrameTree,
    root_frame_id: FrameId,
    frame_id: FrameId,
) -> (Position<F>, Velocity<F>) {
    let (pos_raw, vel_raw) = frame_origin(frame_tree, root_frame_id, frame_id);
    (
        Position::<F>::from_raw_si(pos_raw),
        Velocity::<F>::from_raw_si(vel_raw),
    )
}

/// Typed sibling of [`FrameTree::compute_relative_state`] returning a
/// [`RefFrameStateTyped<From, To>`]. The caller asserts the supplied
/// frame IDs correspond to frames whose markers are `From` and `To` —
/// the arena is heterogeneous and stores untyped `RefFrameState` per
/// node, so this is a `from_untyped_unchecked` boundary lift.
///
/// Phase C5 of issue #71.
pub fn compute_relative_state_typed<From: Frame, To: Frame>(
    frame_tree: &FrameTree,
    from: FrameId,
    to: FrameId,
) -> RefFrameStateTyped<From, To> {
    let untyped = frame_tree.compute_relative_state(from, to);
    RefFrameStateTyped::<From, To>::from_untyped_unchecked(&untyped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RefFrameKind, RefFrameRot, RefFrameState, RefFrameTrans};
    use astrodyn_quantities::frame::{Ecef, RootInertial};

    #[test]
    fn frame_origin_typed_matches_untyped() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let child = tree.add_child(
            root,
            "child".into(),
            RefFrameKind::Inertial,
            RefFrameState {
                trans: RefFrameTrans {
                    position: DVec3::new(1.0e7, 2.0e7, 3.0e7),
                    velocity: DVec3::new(100.0, 200.0, 300.0),
                },
                rot: RefFrameRot::default(),
            },
        );

        let (untyped_pos, untyped_vel) = frame_origin(&tree, root, child);
        let (typed_pos, typed_vel) = frame_origin_typed::<RootInertial>(&tree, root, child);

        assert_eq!(typed_pos.raw_si(), untyped_pos);
        assert_eq!(typed_vel.raw_si(), untyped_vel);
    }

    #[test]
    fn frame_origin_typed_root_is_zero() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let (typed_pos, typed_vel) = frame_origin_typed::<RootInertial>(&tree, root, root);
        assert_eq!(typed_pos.raw_si(), DVec3::ZERO);
        assert_eq!(typed_vel.raw_si(), DVec3::ZERO);
    }

    #[test]
    fn compute_relative_state_typed_matches_untyped() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let child = tree.add_child(
            root,
            "ecef".into(),
            RefFrameKind::PlanetFixed,
            RefFrameState {
                trans: RefFrameTrans {
                    position: DVec3::new(1.0, 2.0, 3.0),
                    velocity: DVec3::new(0.1, 0.2, 0.3),
                },
                rot: RefFrameRot::default(),
            },
        );

        let untyped = tree.compute_relative_state(root, child);
        let typed = compute_relative_state_typed::<RootInertial, Ecef>(&tree, root, child);

        assert_eq!(typed.trans.position.raw_si(), untyped.trans.position);
        assert_eq!(typed.trans.velocity.raw_si(), untyped.trans.velocity);
        assert_eq!(typed.rot.t_parent_this(), untyped.rot.t_parent_this);
    }
}
