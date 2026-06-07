//! Frame-tree ⇄ frame-document bridge (issue #663; feature `frame-doc`).
//!
//! [`export_tree`] walks a stamped [`FrameTree`] into a
//! [`FrameDocument`] snapshot;
//! [`load_document`] rebuilds a standalone `FrameTree` from one. Spec
//! trace: RFS-601 (lossless round-trip of identity, topology,
//! classification, origin, epochs), RFS-602 (the document crate carries no
//! physics — this module is the only place tree and document meet),
//! RFS-603 (per-record epoch + origin).
//!
//! ## Composition
//!
//! `load_document` returns a **standalone** tree. Landing it inside an
//! existing tree reuses the PR-2 (#661) namespace mechanics unchanged:
//! `FrameTree::import_subtree(&loaded, ns)` with a namespace ≥ 2 re-stamps
//! the foreign identities, and an explicit `graft` is the only topology
//! bridge — the loader never auto-connects trees, so un-grafted
//! cross-namespace queries keep failing loudly.
//!
//! ## Unstamped nodes
//!
//! [`export_tree`] **panics** on an unstamped node. Production trees are
//! fully stamped after issue #662; an unstamped node at export time is a
//! construction-path bug, and inventing an identity for it on the wire
//! would be silent misattribution.

use astrodyn_frame_doc::{
    CanonicalRotation, DocError, DocHeader, FrameDocument, FrameRecord, FrameUid, Origin,
    TransRecord,
};
use astrodyn_frames::{
    FrameId, FrameTree, FrameTreeError, RefFrameRot, RefFrameState, RefFrameTrans,
};
use astrodyn_quantities::quat::JeodQuat;
use astrodyn_quantities::time_scale::{SecondsSince, TDB};
use glam::{DMat3, DVec3};

/// Errors from [`load_document`]. The document-level shape errors come
/// from [`DocError`]; the structural errors mirror the
/// [`FrameTreeError`] vocabulary (`UnresolvedParent` / `Cycle`), which
/// this load path makes reachable for the first time — the tree's own
/// `validate_forest()` runs as the final belt.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// The document failed shape/conventions validation (checked before
    /// any state is interpreted).
    #[error(transparent)]
    Doc(#[from] DocError),
    /// A record declares a parent uid for which the document carries no
    /// record — the document equivalent of
    /// [`FrameTreeError::UnresolvedParent`].
    #[error(
        "record {record} (`{name}`) declares parent `{parent_uid}`, but the \
         document has no record for that identity"
    )]
    UnresolvedParent {
        /// Record position in the document.
        record: usize,
        /// The record's diagnostic name.
        name: String,
        /// The declared-but-absent parent identity.
        parent_uid: FrameUid,
    },
    /// Records whose parent chains never reach a root — the document
    /// equivalent of [`FrameTreeError::Cycle`].
    #[error("records {names:?} form a parent cycle (no chain reaches a root)")]
    Cycle {
        /// Diagnostic names of the unplaceable records.
        names: Vec<String>,
    },
    /// A root record's identity class cannot root a tree
    /// ([`FrameTreeError::NonInertialRoot`] at the wire).
    #[error(
        "root record {record} (`{name}`) has identity `{uid}` whose class \
         cannot root a tree (only inertial-flavor classes may)"
    )]
    NonInertialRoot {
        /// Record position in the document.
        record: usize,
        /// The record's diagnostic name.
        name: String,
        /// The offending identity.
        uid: FrameUid,
    },
    /// The rebuilt tree failed the frame tree's own integrity belt.
    #[error(transparent)]
    Tree(#[from] FrameTreeError),
}

/// Serialize a stamped frame tree into a snapshot document.
///
/// `origin_of` supplies each node's [`Origin`] — host knowledge the tree
/// does not carry (which nodes project an authoritative body store, which
/// are model evaluations, which are caller-injected).
///
/// Rotation canonicity is class-driven: `PlanetFixed`-class nodes are
/// written by `sync_pfix_rotation`, which is **matrix-canonical** (the
/// RNP/IAU matrix is stored verbatim and the quaternion derived), so their
/// records carry [`CanonicalRotation::Matrix`]; every other node is
/// quaternion-canonical (JEOD_INV RF.04) and carries
/// [`CanonicalRotation::Quat`].
///
/// # Panics
/// - any node is unstamped (`uid() == None`) — see the module docs;
/// - the resulting document fails validation (non-finite state is the
///   practical case: broken upstream physics must not be laundered into
///   "data").
pub fn export_tree(
    tree: &FrameTree,
    header: DocHeader,
    origin_of: impl Fn(FrameId) -> Origin,
) -> FrameDocument {
    let mut uids: Vec<FrameUid> = Vec::with_capacity(tree.len());
    let mut records = Vec::with_capacity(tree.len());
    // Node id == position in `uids`: the tree is walked in arena order and
    // every node is recorded exactly once, so interning is the identity
    // map from FrameId. Parent references still go through the uid table
    // (never the arena index) on the wire.
    for id in 0..tree.len() {
        let node = tree.get(id);
        let uid = node.uid().unwrap_or_else(|| {
            panic!(
                "export_tree: frame {id} (`{}`) is unstamped — production trees are \
                 fully stamped (issue #662); refusing to serialize a node without an \
                 identity. Stamp it via the typed constructors or add_child_uid.",
                node.name
            )
        });
        uids.push(uid.clone());
        let parent = tree.parent(id).map(|pid| {
            u32::try_from(pid)
                .expect("frame tree node count exceeds u32 — unsupported document size")
        });
        let rotation = if uid.class == astrodyn_frame_doc::FrameClass::PlanetFixed {
            CanonicalRotation::Matrix(node.state.rot.t_parent_this.to_cols_array_2d())
        } else {
            CanonicalRotation::Quat(node.state.rot.q_parent_this.data)
        };
        records.push(FrameRecord {
            name: node.name.clone(),
            uid_index: u32::try_from(id)
                .expect("frame tree node count exceeds u32 — unsupported document size"),
            parent,
            epoch: node.epoch.map(SecondsSince::<TDB>::as_seconds),
            trans: TransRecord {
                position: node.state.trans.position.to_array(),
                velocity: node.state.trans.velocity.to_array(),
            },
            rotation,
            ang_vel_this: node.state.rot.ang_vel_this.to_array(),
            origin: origin_of(id),
        });
    }
    let doc = FrameDocument {
        header,
        uids,
        records,
    };
    doc.validate()
        .unwrap_or_else(|err| panic!("export_tree: produced an invalid document: {err}"));
    doc
}

/// Rebuild a record's `RefFrameState`, re-deriving the non-canonical
/// rotation representation from the canonical one (RF.04: both forms must
/// agree; re-deriving with the same conversion the producer used keeps
/// reload → continue bit-identical for both canonicity regimes).
pub fn record_state(rec: &FrameRecord) -> RefFrameState {
    let (q_parent_this, t_parent_this) = match &rec.rotation {
        CanonicalRotation::Quat(q) => {
            let q = JeodQuat::from_array(*q);
            (q, q.left_quat_to_transformation())
        }
        CanonicalRotation::Matrix(cols) => {
            let t = DMat3::from_cols_array_2d(cols);
            (JeodQuat::left_quat_from_transformation(&t), t)
        }
    };
    RefFrameState {
        trans: RefFrameTrans {
            position: DVec3::from_array(rec.trans.position),
            velocity: DVec3::from_array(rec.trans.velocity),
        },
        rot: RefFrameRot {
            q_parent_this,
            t_parent_this,
            ang_vel_this: DVec3::from_array(rec.ang_vel_this),
        },
    }
}

/// A record's epoch lifted back into the tree's typed epoch form, for
/// hosts applying records to an existing tree via
/// [`FrameTree::set_epoch`].
pub fn record_epoch(rec: &FrameRecord) -> Option<SecondsSince<TDB>> {
    rec.epoch.map(SecondsSince::<TDB>::from_seconds)
}

/// Rebuild a standalone [`FrameTree`] from a snapshot document.
///
/// Validates the document (header conventions **before** any state is
/// interpreted), then places records root-first. The wire's structural
/// failure modes map onto the [`FrameTreeError`] vocabulary that was
/// structurally unreachable before this load path existed:
/// a declared parent with no record → [`LoadError::UnresolvedParent`];
/// records whose parent chains never reach a root → [`LoadError::Cycle`].
/// The rebuilt tree's own `validate_forest()` runs as the final belt.
// JEOD_INV: RF.02 — the loader cross-checks every record's declared parent
// identity while rebuilding topology; the tree-level belt (validate_forest)
// runs on the result. This is the bulk-load path the Cycle/UnresolvedParent
// variants were reserved for (issue #663).
pub fn load_document(doc: &FrameDocument) -> Result<FrameTree, LoadError> {
    doc.validate()?;

    // Map uid-table index → record position (validate() guarantees each
    // uid_index appears on at most one record).
    let mut record_of_uid: Vec<Option<usize>> = vec![None; doc.uids.len()];
    for (pos, rec) in doc.records.iter().enumerate() {
        record_of_uid[rec.uid_index as usize] = Some(pos);
    }

    // A declared parent uid with no record is unresolvable up front.
    for (pos, rec) in doc.records.iter().enumerate() {
        if let Some(p) = rec.parent {
            if record_of_uid[p as usize].is_none() {
                return Err(LoadError::UnresolvedParent {
                    record: pos,
                    name: rec.name.clone(),
                    parent_uid: doc.uids[p as usize].clone(),
                });
            }
        }
    }

    let mut tree = FrameTree::new();
    let mut placed: Vec<Option<FrameId>> = vec![None; doc.records.len()];
    let mut frontier: Vec<usize> = Vec::new();

    // Roots first.
    for (pos, rec) in doc.records.iter().enumerate() {
        if rec.parent.is_none() {
            let uid = doc.uids[rec.uid_index as usize].clone();
            if !uid.class.may_be_root_or_integ() {
                return Err(LoadError::NonInertialRoot {
                    record: pos,
                    name: rec.name.clone(),
                    uid,
                });
            }
            let fid = tree.add_root_uid(uid, rec.name.clone());
            tree.get_mut(fid).state = record_state(rec);
            tree.set_epoch(fid, rec.epoch.map(SecondsSince::<TDB>::from_seconds));
            placed[pos] = Some(fid);
            frontier.push(pos);
        }
    }

    // Breadth-first placement: a record is placeable once its parent is.
    while let Some(parent_pos) = frontier.pop() {
        let parent_uid_index = doc.records[parent_pos].uid_index;
        let parent_fid = placed[parent_pos].expect("frontier entries are placed");
        for (pos, rec) in doc.records.iter().enumerate() {
            if placed[pos].is_none() && rec.parent == Some(parent_uid_index) {
                let uid = doc.uids[rec.uid_index as usize].clone();
                let fid = tree.add_child_uid(
                    parent_fid,
                    uid,
                    rec.name.clone(),
                    record_state(rec),
                    rec.epoch.map(SecondsSince::<TDB>::from_seconds),
                );
                placed[pos] = Some(fid);
                frontier.push(pos);
            }
        }
    }

    // Anything unplaced has a parent chain that never reaches a root: the
    // wire form of a cycle.
    let unplaced: Vec<String> = placed
        .iter()
        .zip(&doc.records)
        .filter(|(p, _)| p.is_none())
        .map(|(_, r)| r.name.clone())
        .collect();
    if !unplaced.is_empty() {
        return Err(LoadError::Cycle { names: unplaced });
    }

    // Final belt: the tree's own integrity validation (forest form — a
    // multi-root document is legitimate pre-graft state).
    tree.validate_forest()?;
    Ok(tree)
}

#[cfg(test)]
mod tests {
    use super::*;
    use astrodyn_frame_doc::{Conventions, SCHEMA_VERSION};
    use astrodyn_quantities::frame::{Earth, PlanetFixed, PlanetInertial, RootInertial};

    fn header() -> DocHeader {
        DocHeader {
            schema_version: SCHEMA_VERSION,
            conventions: Conventions::current(),
            simtime: 100.0,
            tai_tjt_at_epoch: 213.818,
        }
    }

    /// A stamped three-node tree exercising both canonicity regimes: a
    /// matrix-canonical pfix node (written via the sync_pfix_rotation
    /// convention) and a quaternion-canonical inertial child.
    fn stamped_tree() -> FrameTree {
        let mut tree = FrameTree::new();
        let root = tree.add_root_typed::<RootInertial>("root".into());
        tree.set_epoch(root, Some(SecondsSince::from_seconds(100.0)));
        let earth = tree.add_child_uid(
            root,
            FrameUid::of::<PlanetInertial<Earth>>(),
            "Earth.inertial".into(),
            RefFrameState {
                trans: RefFrameTrans {
                    position: DVec3::new(1.0e9, -2.0e9, 3.0e9),
                    velocity: DVec3::new(10.0, -20.0, 30.0),
                },
                rot: RefFrameRot::default(),
            },
            Some(SecondsSince::from_seconds(100.0)),
        );
        // Matrix-canonical pfix node: store a non-trivial rotation matrix
        // verbatim and derive q from it, exactly as sync_pfix_rotation does.
        let rotation = JeodQuat::left_quat_from_eigen_rotation(0.7, DVec3::new(0.1, 0.2, 0.97))
            .left_quat_to_transformation();
        tree.add_child_uid(
            earth,
            FrameUid::of::<PlanetFixed<Earth>>(),
            "Earth.pfix".into(),
            RefFrameState {
                trans: RefFrameTrans::default(),
                rot: RefFrameRot {
                    q_parent_this: JeodQuat::left_quat_from_transformation(&rotation),
                    t_parent_this: rotation,
                    ang_vel_this: DVec3::new(0.0, 0.0, 7.292_115_1e-5),
                },
            },
            Some(SecondsSince::from_seconds(100.0)),
        );
        tree
    }

    fn export(tree: &FrameTree) -> FrameDocument {
        export_tree(tree, header(), |_| Origin::Injected)
    }

    #[test]
    fn export_load_round_trips_identity_topology_state() {
        let tree = stamped_tree();
        let doc = export(&tree);
        // Pfix node carries the matrix (its canonical form), others the quat.
        assert!(matches!(
            doc.records[2].rotation,
            CanonicalRotation::Matrix(_)
        ));
        assert!(matches!(
            doc.records[0].rotation,
            CanonicalRotation::Quat(_)
        ));

        let loaded = load_document(&doc).expect("load");
        assert_eq!(loaded.len(), tree.len());
        for id in 0..tree.len() {
            let (a, b) = (tree.get(id), loaded.get(id));
            assert_eq!(a.uid(), b.uid(), "identity must round-trip");
            assert_eq!(
                tree.parent(id),
                loaded.parent(id),
                "topology must round-trip"
            );
            assert_eq!(a.name, b.name);
            assert_eq!(
                a.epoch.map(|e| e.as_seconds().to_bits()),
                b.epoch.map(|e| e.as_seconds().to_bits()),
                "epoch must round-trip bit-exactly"
            );
            assert_eq!(
                a.state.trans.position.to_array().map(f64::to_bits),
                b.state.trans.position.to_array().map(f64::to_bits)
            );
            assert_eq!(
                a.state.trans.velocity.to_array().map(f64::to_bits),
                b.state.trans.velocity.to_array().map(f64::to_bits)
            );
            // BOTH rotation representations must agree bit-for-bit: the
            // canonical one travelled, the other was re-derived with the
            // same conversion the producer used (RF.04).
            assert_eq!(
                a.state.rot.q_parent_this.data.map(f64::to_bits),
                b.state.rot.q_parent_this.data.map(f64::to_bits),
                "quaternion (node {id})"
            );
            assert_eq!(
                a.state.rot.t_parent_this.to_cols_array().map(f64::to_bits),
                b.state.rot.t_parent_this.to_cols_array().map(f64::to_bits),
                "matrix (node {id})"
            );
            assert_eq!(
                a.state.rot.ang_vel_this.to_array().map(f64::to_bits),
                b.state.rot.ang_vel_this.to_array().map(f64::to_bits)
            );
        }
    }

    #[test]
    fn export_load_round_trips_through_json() {
        let tree = stamped_tree();
        let json = export(&tree).to_json_string();
        let doc = FrameDocument::from_json_str(&json).expect("parse");
        let loaded = load_document(&doc).expect("load");
        for id in 0..tree.len() {
            assert_eq!(
                tree.get(id)
                    .state
                    .rot
                    .t_parent_this
                    .to_cols_array()
                    .map(f64::to_bits),
                loaded
                    .get(id)
                    .state
                    .rot
                    .t_parent_this
                    .to_cols_array()
                    .map(f64::to_bits),
                "JSON round trip drifted (node {id})"
            );
        }
    }

    #[test]
    #[should_panic(expected = "is unstamped")]
    fn export_unstamped_node_panics() {
        let mut tree = FrameTree::new();
        let _ = tree.add_root(
            "legacy-root".into(),
            astrodyn_frames::RefFrameKind::Inertial,
        );
        let _ = export(&tree);
    }

    #[test]
    fn load_rejects_dangling_parent() {
        let mut doc = export(&stamped_tree());
        // Drop the Earth.inertial record: pfix's declared parent now has
        // no record (its uid stays in the table).
        doc.records.remove(1);
        match load_document(&doc).err() {
            Some(LoadError::UnresolvedParent { parent_uid, .. }) => {
                assert_eq!(parent_uid, FrameUid::of::<PlanetInertial<Earth>>());
            }
            other => panic!("expected UnresolvedParent, got {other:?}"),
        }
    }

    #[test]
    fn load_rejects_parent_cycle() {
        let mut doc = export(&stamped_tree());
        // Earth.inertial's parent becomes Earth.pfix while pfix stays
        // parented to Earth.inertial — a two-node cycle detached from root.
        doc.records[1].parent = Some(doc.records[2].uid_index);
        match load_document(&doc).err() {
            Some(LoadError::Cycle { names }) => {
                assert_eq!(
                    names,
                    vec!["Earth.inertial".to_string(), "Earth.pfix".into()]
                );
            }
            other => panic!("expected Cycle, got {other:?}"),
        }
    }

    #[test]
    fn load_rejects_non_inertial_root() {
        let mut doc = export(&stamped_tree());
        // Detach the pfix record from its parent: a PlanetFixed-class root.
        doc.records[2].parent = None;
        assert!(matches!(
            load_document(&doc).err(),
            Some(LoadError::NonInertialRoot { record: 2, .. })
        ));
    }

    #[test]
    fn load_accepts_multi_root_forest() {
        // Two stamped roots, no graft — the legitimate post-import,
        // pre-graft state; the loader validates with validate_forest.
        let mut tree = FrameTree::new();
        let r = tree.add_root_typed::<RootInertial>("root".into());
        tree.set_epoch(r, Some(SecondsSince::from_seconds(1.0)));
        let r2 = tree.add_root_uid(
            FrameUid::of::<PlanetInertial<Earth>>(),
            "imported-root".into(),
        );
        tree.set_epoch(r2, Some(SecondsSince::from_seconds(1.0)));
        let doc = export(&tree);
        let loaded = load_document(&doc).expect("forest loads");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.parent(0), None);
        assert_eq!(loaded.parent(1), None);
    }
}
