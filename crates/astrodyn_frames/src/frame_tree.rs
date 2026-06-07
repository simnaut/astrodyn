//! Arena-based frame tree: a faithful port of JEOD's RefFrame hierarchy.
//!
//! JEOD models reference frames as a tree. Each node stores its state
//! (position, velocity, orientation, angular velocity) relative to its
//! parent. Relative states between arbitrary frames are computed by
//! walking to the common ancestor and composing/negating states.
//!
//! This module is pure Rust with zero Bevy dependency.

use std::collections::{HashMap, HashSet};

use crate::ref_frame_state::{RefFrameKind, RefFrameState, RefFrameStateTyped};
use astrodyn_quantities::frame::Frame;
use astrodyn_quantities::frame_descriptor::{FrameClass, FrameUid, MintPolicy, Namespace};
use astrodyn_quantities::quat::{LeftTransform, NormalizedQuat, ScalarFirst};
use astrodyn_quantities::time_scale::{SecondsSince, TDB};
use glam::DVec3;

/// Handle into the [`FrameTree`] arena.
pub type FrameId = usize;

/// A node in the frame tree.
#[derive(Debug, Clone)]
pub struct FrameNode {
    /// Human-readable name (e.g., "Earth.inertial", "ISS.composite_body").
    pub name: String,
    /// Kind of reference frame.
    pub kind: RefFrameKind,
    /// State relative to parent. Identity for root frames.
    pub state: RefFrameState,
    /// Runtime frame identity. Private: mutating it directly would desync
    /// the tree's identity index (`find`/`resolve`) — read via
    /// [`FrameNode::uid`]; stamping happens only through the typed
    /// constructors, [`FrameTree::add_child_uid`], or
    /// [`FrameTree::import_subtree`].
    uid: Option<FrameUid>,
    /// Frame epoch (TDB seconds): the time-validity of `state`. `None`
    /// until stamped (per-step stamping lands with issue #662).
    pub epoch: Option<SecondsSince<TDB>>,
}

impl FrameNode {
    /// Runtime frame identity: `None` for nodes created via the untyped
    /// [`FrameTree::add_root`] / [`FrameTree::add_child`] path (unstamped —
    /// no identity is synthesized); `Some` once stamped via a typed
    /// constructor, [`FrameTree::add_child_uid`], or
    /// [`FrameTree::import_subtree`]. Read-only: the field is private so a
    /// caller cannot desync the tree's identity index through
    /// [`FrameTree::get_mut`]. The untyped path is a migration scaffold:
    /// issue #662 stamps all production sites and issue #664 makes the
    /// identity required.
    pub fn uid(&self) -> Option<&FrameUid> {
        self.uid.as_ref()
    }
}

/// Integrity violations reported by [`FrameTree::validate`] /
/// [`FrameTree::validate_forest`].
///
/// Identity-dependent checks skip unstamped (`uid: None`) nodes — they have
/// no identity to contradict. `Cycle` and `UnresolvedParent` are structurally
/// unreachable through this module's construction API; the issue #663
/// frame-document loader (`astrodyn::frame_doc_io`) is the bulk-load path
/// they exist for — its load errors mirror this vocabulary and it runs
/// [`FrameTree::validate_forest`] as the post-build belt.
#[derive(Debug, thiserror::Error)]
pub enum FrameTreeError {
    /// A parent walk from this frame exceeded the node count — the
    /// parent links contain a cycle.
    #[error("frame tree contains a cycle reachable from frame {0}")]
    Cycle(FrameId),
    /// The same stamped identity appears on more than one node.
    #[error("duplicate frame identity `{0}` appears on more than one node")]
    DuplicateUid(FrameUid),
    /// A frame names a parent id that is not a valid node.
    #[error("frame {0} names parent {1}, which is not a valid node")]
    UnresolvedParent(FrameId, FrameId),
    /// More than one root in a tree validated with the strict
    /// [`FrameTree::validate`] (use [`FrameTree::validate_forest`] for the
    /// legitimate post-import, pre-graft state).
    #[error(
        "tree has multiple roots {0:?}; expected a single root \
         (use validate_forest to allow a forest)"
    )]
    MultiRoot(Vec<FrameId>),
    /// A stamped root's class is not root-eligible
    /// ([`FrameClass::may_be_root_or_integ`]).
    #[error(
        "stamped root frame {id} has class {class:?}, which cannot root a tree \
         (only inertial-flavor classes may be a root or integration frame)"
    )]
    NonInertialRoot {
        /// The offending root frame.
        id: FrameId,
        /// Its stamped class.
        class: FrameClass,
    },
    /// A stamped node whose class structurally guarantees zero angular
    /// velocity carries a non-zero `ang_vel_this`.
    #[error(
        "stamped frame {id} has non-rotating class {class:?} but a non-zero \
         angular velocity — classification contradicts state"
    )]
    ClassStateContradiction {
        /// The offending frame.
        id: FrameId,
        /// Its stamped class.
        class: FrameClass,
    },
    /// A node's rotation quaternion has drifted from unit norm beyond
    /// `NormalizedQuat::DEFAULT_TOLERANCE` (a missed renormalization
    /// upstream, or corrupt loaded data).
    #[error("frame {id} quaternion norm {norm} drifted beyond unit tolerance")]
    UnitNormDrift {
        /// The offending frame.
        id: FrameId,
        /// The drifted norm.
        norm: f64,
    },
}

/// Arena-based frame tree. Portable (no ECS dependency).
///
/// Frames are stored in a flat `Vec`; parent/child relationships are tracked
/// with parallel vectors of `Option<FrameId>` and `Vec<FrameId>`. Stamped
/// frame identities are indexed for [`FrameTree::find`] /
/// [`FrameTree::resolve`].
pub struct FrameTree {
    nodes: Vec<FrameNode>,
    parent: Vec<Option<FrameId>>,
    children: Vec<Vec<FrameId>>,
    /// Identity → arena-id index, maintained by the stamped constructors
    /// and [`FrameTree::import_subtree`]. Unstamped nodes are absent.
    uid_index: HashMap<FrameUid, FrameId>,
}

impl FrameTree {
    /// Create an empty tree.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            parent: Vec::new(),
            children: Vec::new(),
            uid_index: HashMap::new(),
        }
    }

    // -- construction -------------------------------------------------------

    /// Add a root frame (no parent). State is identity. The node is
    /// **unstamped** (`uid: None`) — prefer [`Self::add_root_typed`] for a
    /// frame with a runtime identity. This untyped path is removed once all
    /// construction sites are stamped (issue #664).
    pub fn add_root(&mut self, name: String, kind: RefFrameKind) -> FrameId {
        let id = self.nodes.len();
        self.nodes.push(FrameNode {
            name,
            kind,
            state: RefFrameState::default(),
            uid: None,
            epoch: None,
        });
        self.parent.push(None);
        self.children.push(Vec::new());
        id
    }

    /// Add a stamped root frame whose identity is `FrameUid::of::<F>()`.
    /// State is identity.
    ///
    /// # Panics
    /// - `F`'s class is not root-eligible
    ///   ([`FrameClass::may_be_root_or_integ`]): integrating in (or rooting
    ///   on) a rotating frame is silently wrong physics, rejected at the
    ///   point of introduction.
    /// - `F` is non-mintable (a storage-boundary wildcard or
    ///   `IntegrationFrame`): `FrameUid::of` panics with the mint-policy
    ///   diagnostic.
    /// - the minted identity is already registered in this tree.
    pub fn add_root_typed<F: Frame>(&mut self, name: String) -> FrameId {
        let uid = FrameUid::of::<F>();
        assert!(
            uid.class.may_be_root_or_integ(),
            "add_root_typed: frame `{}` has class {:?}, which cannot root a tree \
             (only RootInertial / PlanetInertial / BarycenterInertial classes may \
             be a root or integration frame). Add it as a child via add_child_typed.",
            F::NAME,
            uid.class
        );
        let id = self.nodes.len();
        self.nodes.push(FrameNode {
            name,
            kind: RefFrameKind::Inertial,
            state: RefFrameState::default(),
            uid: Some(uid.clone()),
            epoch: None,
        });
        self.parent.push(None);
        self.children.push(Vec::new());
        self.register_uid(uid, id);
        id
    }

    /// Add a stamped root frame whose identity the caller supplies as a
    /// value — the root-level sibling of [`Self::add_child_uid`], for
    /// hosts (and the issue #663 document loader) that resolve the root's
    /// identity at runtime rather than from a compile-time marker. State
    /// is identity.
    ///
    /// # Panics
    /// - `uid.class` is not root-eligible
    ///   ([`FrameClass::may_be_root_or_integ`]): integrating in (or rooting
    ///   on) a rotating frame is silently wrong physics, rejected at the
    ///   point of introduction (matching [`Self::add_root_typed`] and
    ///   `validate()`'s `NonInertialRoot`).
    /// - the identity is already registered in this tree.
    pub fn add_root_uid(&mut self, uid: FrameUid, name: String) -> FrameId {
        assert!(
            uid.class.may_be_root_or_integ(),
            "add_root_uid: identity `{uid}` has class {:?}, which cannot root a \
             tree (only RootInertial / PlanetInertial / BarycenterInertial \
             classes may be a root or integration frame). Add it as a child \
             via add_child_uid.",
            uid.class
        );
        let id = self.nodes.len();
        self.nodes.push(FrameNode {
            name,
            kind: Self::kind_for_class(uid.class),
            state: RefFrameState::default(),
            uid: Some(uid.clone()),
            epoch: None,
        });
        self.parent.push(None);
        self.children.push(Vec::new());
        self.register_uid(uid, id);
        id
    }

    /// Add a child frame with the given state relative to its parent. The
    /// node is **unstamped** (`uid: None`) — prefer [`Self::add_child_typed`]
    /// or [`Self::add_child_uid`] for a frame with a runtime identity. This
    /// untyped path is removed once all construction sites are stamped
    /// (issue #664).
    ///
    /// # Panics
    /// Panics if `parent_id` does not refer to an existing frame.
    pub fn add_child(
        &mut self,
        parent_id: FrameId,
        name: String,
        kind: RefFrameKind,
        state: RefFrameState,
    ) -> FrameId {
        assert!(
            parent_id < self.nodes.len(),
            "parent_id {parent_id} out of range (have {} frames)",
            self.nodes.len()
        );
        let id = self.nodes.len();
        self.nodes.push(FrameNode {
            name,
            kind,
            state,
            uid: None,
            epoch: None,
        });
        self.parent.push(Some(parent_id));
        self.children.push(Vec::new());
        self.children[parent_id].push(id);
        id
    }

    /// Register a stamped identity in the lookup index, rejecting
    /// duplicates at the point of introduction.
    fn register_uid(&mut self, uid: FrameUid, id: FrameId) {
        if let Some(&existing) = self.uid_index.get(&uid) {
            panic!(
                "FrameTree: duplicate frame identity `{uid}` — already registered at \
                 frame {existing}, cannot register it again at frame {id}. Each \
                 FrameUid maps to exactly one node. If these are genuinely distinct \
                 frames, mint distinct identities (different tag or role), or import \
                 the foreign tree under a non-LOCAL namespace via import_subtree."
            );
        }
        self.uid_index.insert(uid, id);
    }

    /// Map a stamped class onto the legacy [`RefFrameKind`] discriminant.
    /// The field is removed in issue #664; nothing branches on it — this
    /// keeps the stopgap value honest in the meantime.
    fn kind_for_class(class: FrameClass) -> RefFrameKind {
        if matches!(class, FrameClass::Body) {
            RefFrameKind::Body
        } else if class.is_rotating() {
            RefFrameKind::PlanetFixed
        } else {
            RefFrameKind::Inertial
        }
    }

    /// Typed sibling of [`Self::add_child`]: stamps the child with
    /// `FrameUid::of::<C>()` (no hand-supplied kind — the runtime identity
    /// is derived from the compile-time marker) and checks the typed
    /// state's parent marker `P` against the parent node's stamped
    /// identity. The arena's storage stays untyped — the typed state is
    /// converted via [`RefFrameStateTyped::to_untyped`] at the boundary;
    /// numeric values are preserved exactly.
    ///
    /// # Panics
    /// - `parent_id` out of range.
    /// - the parent node is unstamped (created via the untyped path).
    /// - the parent's stamped identity is not `FrameUid::of::<P>()`.
    /// - `C` is non-mintable (`FrameUid::of` panics).
    /// - the child identity is already registered in this tree.
    pub fn add_child_typed<P: Frame, C: Frame>(
        &mut self,
        parent_id: FrameId,
        name: String,
        state: RefFrameStateTyped<P, C>,
        epoch: Option<SecondsSince<TDB>>,
    ) -> FrameId {
        assert!(
            parent_id < self.nodes.len(),
            "add_child_typed: parent_id {parent_id} out of range (have {} frames)",
            self.nodes.len()
        );
        assert!(
            matches!(P::DESCRIPTOR.mint, MintPolicy::Stable),
            "add_child_typed: parent marker `{}` is not mintable (a runtime-resolved \
             wildcard or IntegrationFrame) and can never name a stamped parent \
             identity. Resolve the concrete parent frame type and use that instead.",
            core::any::type_name::<P>()
        );
        // JEOD_INV: RF.02 — the typed state's parent marker must name the
        // parent node's stamped identity (valid predecessor).
        let parent_uid = self.nodes[parent_id].uid.as_ref().unwrap_or_else(|| {
            panic!(
                "add_child_typed: parent frame {parent_id} (`{}`) is unstamped — it \
                 was created via the untyped add_root/add_child path and has no \
                 identity to check `{}` against. Stamp the parent via \
                 add_root_typed / add_child_typed / add_child_uid, or use add_child \
                 for an unstamped child.",
                self.nodes[parent_id].name,
                core::any::type_name::<P>()
            )
        });
        assert!(
            parent_uid.is::<P>(),
            "add_child_typed: parent frame {parent_id} has identity `{parent_uid}`, \
             which is not `{}` — the typed state's parent marker P must match the \
             frame it is parented to.",
            core::any::type_name::<P>()
        );
        let child_uid = FrameUid::of::<C>();
        self.add_child_uid(parent_id, child_uid, name, state.to_untyped(), epoch)
    }

    /// Add a child stamped with a caller-supplied identity — the flexible
    /// primitive for dynamically-resolved identities (a host that knows the
    /// planet only at runtime) and producer-defined runtime frames.
    ///
    /// No namespace restriction: `LOCAL` identities minted via
    /// `FrameUid::of` are accepted (the LOCAL reservation is enforced at
    /// uid construction by `FrameUid::external`, not here).
    ///
    /// # Panics
    /// - `parent_id` out of range.
    /// - the identity is already registered in this tree.
    pub fn add_child_uid(
        &mut self,
        parent_id: FrameId,
        uid: FrameUid,
        name: String,
        state: RefFrameState,
        epoch: Option<SecondsSince<TDB>>,
    ) -> FrameId {
        assert!(
            parent_id < self.nodes.len(),
            "add_child_uid: parent_id {parent_id} out of range (have {} frames)",
            self.nodes.len()
        );
        let id = self.nodes.len();
        self.nodes.push(FrameNode {
            name,
            kind: Self::kind_for_class(uid.class),
            state,
            uid: Some(uid.clone()),
            epoch,
        });
        self.parent.push(Some(parent_id));
        self.children.push(Vec::new());
        self.children[parent_id].push(id);
        self.register_uid(uid, id);
        id
    }

    /// Set (or clear) a node's frame epoch. Groundwork for per-step
    /// stamping (issue #662); not called on the hot path here.
    pub fn set_epoch(&mut self, id: FrameId, epoch: Option<SecondsSince<TDB>>) {
        assert!(
            id < self.nodes.len(),
            "set_epoch: frame id {id} out of range (have {} frames)",
            self.nodes.len()
        );
        self.nodes[id].epoch = epoch;
    }

    /// Speculative identity lookup: the arena id of `uid`, or `None` —
    /// an observable miss the caller can surface or recover from.
    pub fn find(&self, uid: &FrameUid) -> Option<FrameId> {
        self.uid_index.get(uid).copied()
    }

    /// Load-bearing identity lookup: the arena id of `uid`.
    ///
    /// # Panics
    /// Panics naming the identity if it is not registered in this tree —
    /// a missing load-bearing frame is a misconfiguration, never silently
    /// substituted.
    pub fn resolve(&self, uid: &FrameUid) -> FrameId {
        self.uid_index.get(uid).copied().unwrap_or_else(|| {
            panic!(
                "FrameTree::resolve: no frame with identity `{uid}` in this tree \
                 ({} frames, {} stamped). Stamp the frame via a typed constructor, \
                 add_child_uid, or import_subtree before resolving it.",
                self.nodes.len(),
                self.uid_index.len()
            )
        })
    }

    /// Read the state at `id` as a typed [`RefFrameStateTyped<P, C>`],
    /// **checked against the stored identities**: the node's stamped uid
    /// must be `FrameUid::of::<C>()` and its parent's stamped uid must be
    /// `FrameUid::of::<P>()`. Replaces the former caller-asserts boundary —
    /// a wrong marker now fails loudly instead of silently mislabeling
    /// physics. The wrapped quaternion is additionally checked against
    /// `NormalizedQuat::DEFAULT_TOLERANCE` by the underlying lift.
    ///
    /// # Panics
    /// - `P` or `C` is non-mintable (a storage-boundary wildcard or
    ///   `IntegrationFrame`) — such a marker can never name a stored
    ///   identity; request the concrete frame type.
    /// - the node (or its parent) is unstamped.
    /// - the node is a root (it has no predecessor state to read).
    /// - the stored child or parent identity does not match the markers.
    // JEOD_INV: RF.02 — checked typed recovery: stored child identity must
    // equal of::<C>() and the parent node's identity must equal of::<P>()
    // (valid predecessor); unstamped or root nodes are rejected loudly.
    pub fn get_state_typed<P: Frame, C: Frame>(&self, id: FrameId) -> RefFrameStateTyped<P, C> {
        assert!(
            matches!(C::DESCRIPTOR.mint, MintPolicy::Stable),
            "get_state_typed: child marker `{}` is not mintable (a runtime-resolved \
             wildcard or IntegrationFrame) and can never name a stored identity. \
             Resolve the concrete frame type and request that instead.",
            core::any::type_name::<C>()
        );
        assert!(
            matches!(P::DESCRIPTOR.mint, MintPolicy::Stable),
            "get_state_typed: parent marker `{}` is not mintable (a runtime-resolved \
             wildcard or IntegrationFrame) and can never name a stored identity. \
             Resolve the concrete frame type and request that instead.",
            core::any::type_name::<P>()
        );
        let node_uid = self.nodes[id].uid.as_ref().unwrap_or_else(|| {
            panic!(
                "get_state_typed: frame {id} (`{}`) is unstamped — it was created \
                 via the untyped path; stamp its identity via add_child_typed / \
                 add_child_uid before reading it as a typed state.",
                self.nodes[id].name
            )
        });
        assert!(
            node_uid.is::<C>(),
            "get_state_typed: frame {id} has stored identity `{node_uid}`, but the \
             requested child marker is `{}` — identity mismatch.",
            core::any::type_name::<C>()
        );
        let parent_id = self.parent[id].unwrap_or_else(|| {
            panic!(
                "get_state_typed: frame {id} (`{node_uid}`) is a root (no parent), \
                 but a parent marker `{}` was requested — a root has no predecessor \
                 state to read.",
                core::any::type_name::<P>()
            )
        });
        let parent_uid = self.nodes[parent_id].uid.as_ref().unwrap_or_else(|| {
            panic!(
                "get_state_typed: parent of frame {id} (frame {parent_id}, `{}`) is \
                 unstamped; cannot verify it is `{}`.",
                self.nodes[parent_id].name,
                core::any::type_name::<P>()
            )
        });
        assert!(
            parent_uid.is::<P>(),
            "get_state_typed: parent of frame {id} has identity `{parent_uid}`, but \
             the requested parent marker is `{}` — predecessor identity mismatch.",
            core::any::type_name::<P>()
        );
        RefFrameStateTyped::<P, C>::from_untyped_unchecked(&self.nodes[id].state)
    }

    // -- accessors ----------------------------------------------------------

    /// Borrow a frame node by id.
    pub fn get(&self, id: FrameId) -> &FrameNode {
        &self.nodes[id]
    }

    /// Mutably borrow a frame node by id.
    pub fn get_mut(&mut self, id: FrameId) -> &mut FrameNode {
        &mut self.nodes[id]
    }

    /// Parent of the given frame, or `None` for a root.
    pub fn parent(&self, id: FrameId) -> Option<FrameId> {
        self.parent[id]
    }

    /// Direct children of the given frame.
    pub fn children(&self, id: FrameId) -> &[FrameId] {
        &self.children[id]
    }

    /// Number of frames in the tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    // -- tree traversal -----------------------------------------------------

    /// Depth of `id` in the tree (root has depth 0).
    pub fn depth(&self, id: FrameId) -> usize {
        let mut d = 0usize;
        let mut current = id;
        while let Some(p) = self.parent[current] {
            d += 1;
            current = p;
        }
        d
    }

    /// Find the common ancestor of two frames in O(depth).
    ///
    /// Computes both depths, brings the deeper frame up to the other's depth,
    /// then walks both up in lockstep until they meet.
    ///
    /// Returns `None` if the frames do not share a common root (disconnected
    /// subtrees).
    pub fn common_ancestor(&self, a: FrameId, b: FrameId) -> Option<FrameId> {
        let mut da = self.depth(a);
        let mut db = self.depth(b);
        let mut ca = a;
        let mut cb = b;

        // Equalize depths.
        while da > db {
            ca = self.parent[ca]?;
            da -= 1;
        }
        while db > da {
            cb = self.parent[cb]?;
            db -= 1;
        }

        // Walk up in lockstep.
        while ca != cb {
            ca = self.parent[ca]?;
            cb = self.parent[cb]?;
        }
        Some(ca)
    }

    /// Identity-or-name rendering of a frame for diagnostics.
    fn display_id(&self, id: FrameId) -> String {
        match &self.nodes[id].uid {
            Some(uid) => format!("`{uid}` (frame {id})"),
            None => format!("`{}` (frame {id}, unstamped)", self.nodes[id].name),
        }
    }

    /// Walk to the root of `id`'s tree.
    fn root_of(&self, mut id: FrameId) -> FrameId {
        while let Some(p) = self.parent[id] {
            id = p;
        }
        id
    }

    /// Find the common ancestor of two frames.
    ///
    /// Convenience wrapper around `common_ancestor` that panics if the frames
    /// do not share a common root, naming both endpoints and both roots —
    /// an undeclared cross-source relationship is never silently answered.
    /// Prefer `common_ancestor` in new code.
    pub fn find_common_ancestor(&self, a: FrameId, b: FrameId) -> FrameId {
        self.common_ancestor(a, b).unwrap_or_else(|| {
            let (ra, rb) = (self.root_of(a), self.root_of(b));
            let ns = |id: FrameId| self.nodes[id].uid.as_ref().map(|u| u.namespace);
            let suggest = if ns(ra) != ns(rb) {
                " The two frames live under roots in different namespaces; if they \
                 belong in one tree, declare the relationship explicitly by \
                 attaching one root under the other with FrameTree::graft before \
                 querying relative state."
            } else {
                ""
            };
            panic!(
                "find_common_ancestor: frames {} and {} do not share a common \
                 ancestor (roots {} and {}).{suggest}",
                self.display_id(a),
                self.display_id(b),
                self.display_id(ra),
                self.display_id(rb)
            )
        })
    }

    // -- integrity ------------------------------------------------------------

    /// Validate the tree's structural and identity integrity, requiring a
    /// **single root** (the production invariant). Use
    /// [`Self::validate_forest`] for the legitimate multi-root state
    /// between [`Self::import_subtree`] and [`Self::graft`].
    ///
    /// Identity checks skip unstamped nodes; see [`FrameTreeError`] for the
    /// rejected conditions.
    pub fn validate(&self) -> Result<(), FrameTreeError> {
        let roots = self.validate_common()?;
        if roots.len() > 1 {
            return Err(FrameTreeError::MultiRoot(roots));
        }
        Ok(())
    }

    /// [`Self::validate`] minus the single-root requirement — a forest of
    /// disconnected roots (e.g. freshly imported, not yet grafted) passes.
    pub fn validate_forest(&self) -> Result<(), FrameTreeError> {
        self.validate_common().map(|_| ())
    }

    /// Shared validation pass; returns the root set on success.
    fn validate_common(&self) -> Result<Vec<FrameId>, FrameTreeError> {
        let len = self.nodes.len();
        let mut roots = Vec::new();
        let mut seen_uids: HashSet<&FrameUid> = HashSet::new();
        for id in 0..len {
            // Parent link resolution (belt: structurally impossible via this
            // module's construction API; reachable via future load paths).
            if let Some(p) = self.parent[id] {
                if p >= len {
                    return Err(FrameTreeError::UnresolvedParent(id, p));
                }
            } else {
                roots.push(id);
            }
            // Cycle detection: a parent walk longer than the node count
            // cannot terminate (belt, as above).
            let mut current = id;
            let mut hops = 0usize;
            while let Some(p) = self.parent[current] {
                if p >= len {
                    return Err(FrameTreeError::UnresolvedParent(current, p));
                }
                current = p;
                hops += 1;
                if hops > len {
                    return Err(FrameTreeError::Cycle(id));
                }
            }
            let node = &self.nodes[id];
            // Identity checks (stamped nodes only — unstamped nodes have no
            // identity to contradict).
            if let Some(uid) = &node.uid {
                if !seen_uids.insert(uid) {
                    return Err(FrameTreeError::DuplicateUid(uid.clone()));
                }
                if self.parent[id].is_none() && !uid.class.may_be_root_or_integ() {
                    return Err(FrameTreeError::NonInertialRoot {
                        id,
                        class: uid.class,
                    });
                }
                if !uid.class.is_rotating()
                    && !matches!(uid.class, FrameClass::Body | FrameClass::External)
                    && node.state.rot.ang_vel_this != DVec3::ZERO
                {
                    return Err(FrameTreeError::ClassStateContradiction {
                        id,
                        class: uid.class,
                    });
                }
            }
            // Unit-norm drift (all nodes — state integrity is
            // identity-independent). Tolerance check, not bit-equality; the
            // explicit NaN arm exists because NaN compares false to every
            // threshold and would otherwise pass silently.
            let norm = node.state.rot.q_parent_this.norm();
            if norm.is_nan()
                || (norm - 1.0).abs()
                    > NormalizedQuat::<ScalarFirst, LeftTransform>::DEFAULT_TOLERANCE
            {
                return Err(FrameTreeError::UnitNormDrift { id, norm });
            }
        }
        Ok(roots)
    }

    // -- multi-source composition ----------------------------------------------

    /// Deep-copy `other`'s entire forest into this tree, re-stamping every
    /// stamped identity into namespace `ns` (unstamped nodes stay
    /// unstamped). Returns `(old_id, new_id)` pairs in `other`'s insertion
    /// order so callers holding foreign ids can locate the imported nodes
    /// (e.g. to [`Self::graft`] an imported root).
    ///
    /// # Panics
    /// - `ns == Namespace::LOCAL`: LOCAL is reserved for type-derived
    ///   identities — a foreign tree must not be able to impersonate them.
    /// - a re-stamped identity collides with one already registered here
    ///   (e.g. two imports into the same namespace).
    pub fn import_subtree(&mut self, other: &FrameTree, ns: Namespace) -> Vec<(FrameId, FrameId)> {
        assert!(
            ns != Namespace::LOCAL,
            "import_subtree: Namespace::LOCAL is reserved for type-derived \
             identities (FrameUid::of). Import foreign trees into a \
             host-allocated non-LOCAL namespace."
        );
        let base = self.nodes.len();
        let mut map = Vec::with_capacity(other.nodes.len());
        for (old, node) in other.nodes.iter().enumerate() {
            let new = base + old;
            let new_uid = node.uid.clone().map(|u| u.with_namespace(ns));
            self.nodes.push(FrameNode {
                name: node.name.clone(),
                kind: node.kind,
                state: node.state,
                uid: new_uid.clone(),
                epoch: node.epoch,
            });
            self.parent.push(other.parent[old].map(|p| base + p));
            self.children
                .push(other.children[old].iter().map(|&c| base + c).collect());
            if let Some(u) = new_uid {
                self.register_uid(u, new);
            }
            map.push((old, new));
        }
        map
    }

    /// Attach root `root_id` under `new_parent` with the caller-supplied
    /// relative state — the **explicit declaration** of a cross-source
    /// relationship. Unlike [`Self::reparent`] (which recomputes state from
    /// an existing same-tree relationship), a freshly imported root has no
    /// prior relationship to this tree: the supplied `state` *is* the
    /// host's declared physical claim about where the foreign root sits.
    ///
    /// # Panics
    /// - `root_id` is not a root (already has a parent).
    /// - `new_parent` is a descendant of `root_id` (would create a cycle).
    pub fn graft(&mut self, root_id: FrameId, new_parent: FrameId, state: RefFrameState) {
        assert!(
            self.parent[root_id].is_none(),
            "graft: frame {} is not a root (it has parent {:?}); only roots may be \
             grafted — use reparent for frames already related to this tree.",
            root_id,
            self.parent[root_id]
        );
        assert!(
            !self.is_descendant_of(new_parent, root_id),
            "graft: new_parent {new_parent} is a descendant of {root_id} — grafting \
             would create a cycle."
        );
        self.parent[root_id] = Some(new_parent);
        self.children[new_parent].push(root_id);
        self.nodes[root_id].state = state;
    }

    /// Path from `descendant` up to `ancestor` (inclusive of both endpoints).
    ///
    /// Returns `None` if `ancestor` is not actually an ancestor of `descendant`.
    pub fn path_to_ancestor(&self, descendant: FrameId, ancestor: FrameId) -> Option<Vec<FrameId>> {
        let mut path = Vec::new();
        let mut current = descendant;
        loop {
            path.push(current);
            if current == ancestor {
                return Some(path);
            }
            current = self.parent[current]?;
        }
    }

    /// Compute the relative state between two frames, returning `None` if
    /// they don't share a common ancestor.
    ///
    /// Option-returning variant of `compute_relative_state`. Useful when the
    /// frame relationship isn't statically known.
    pub fn try_compute_relative_state(&self, from: FrameId, to: FrameId) -> Option<RefFrameState> {
        let ancestor = self.common_ancestor(from, to)?;
        let state_from = self.compose_to_ancestor(from, ancestor);
        let state_to = self.compose_to_ancestor(to, ancestor);
        let from_negated = RefFrameState::negate(&state_from);
        Some(from_negated.incr_right(&state_to))
    }

    /// Compute the relative state between two frames.
    ///
    /// Returns the state of `to` relative to `from` (i.e., if you are
    /// "standing in" the `from` frame, this tells you where `to` is).
    ///
    /// Port of JEOD `ref_frame_compute_relative_state.cc`. The algorithm:
    /// 1. Find common ancestor of `from` and `to`.
    /// 2. Compose states from `from` up to the ancestor.
    /// 3. Compose states from `to` up to the ancestor.
    /// 4. Result = negate(from_composed) composed with to_composed.
    ///
    /// This gives the state of `to` as seen from `from`.
    // JEOD_INV: RF.01 — same-tree requirement: both FrameIds are arena indices into this
    // single `FrameTree`, so `compute_relative_state` cannot be called across trees.
    // JEOD_INV: RF.02 — predecessor validity: `find_common_ancestor` and `parent()` both
    // bounds-check the arena; an invalid `FrameId` panics immediately.
    pub fn compute_relative_state(&self, from: FrameId, to: FrameId) -> RefFrameState {
        let ancestor = self.find_common_ancestor(from, to);

        // Compose state from `from` to ancestor.
        let state_from = self.compose_to_ancestor(from, ancestor);

        // Compose state from `to` to ancestor.
        let state_to = self.compose_to_ancestor(to, ancestor);

        // state_from is the state of `from` relative to ancestor (ancestor -> from).
        // state_to is the state of `to` relative to ancestor (ancestor -> to).
        // We want state of `to` relative to `from` (from -> to).
        // from -> to = negate(ancestor -> from) composed with (ancestor -> to)
        //           = (from -> ancestor) composed with (ancestor -> to)
        let from_negated = RefFrameState::negate(&state_from);
        from_negated.incr_right(&state_to)
    }

    // -- lookup --------------------------------------------------------------

    /// Find a frame by name. Returns the first match, or `None`.
    pub fn find_by_name(&self, name: &str) -> Option<FrameId> {
        self.nodes.iter().position(|n| n.name == name)
    }

    // -- tree mutation -------------------------------------------------------

    /// Test whether `id` is a descendant of `ancestor` (or equal to it).
    pub fn is_descendant_of(&self, id: FrameId, ancestor: FrameId) -> bool {
        if id == ancestor {
            return true;
        }
        let mut current = id;
        while let Some(p) = self.parent[current] {
            if p == ancestor {
                return true;
            }
            current = p;
        }
        false
    }

    /// Move a frame to a new parent, preserving its absolute state.
    ///
    /// Port of JEOD's `RefFrame::transplant_node()`: the frame's position in
    /// the tree changes, but its state relative to the root is preserved by
    /// recomputing the relative state with respect to the new parent.
    ///
    /// # Panics
    /// - `new_parent` is a descendant of `id` (would create a cycle).
    /// - `id` is a root frame with no parent.
    /// - `id` and `new_parent` do not share a common root.
    pub fn reparent(&mut self, id: FrameId, new_parent: FrameId) {
        // Check root first — root has no parent to detach from.
        assert!(
            self.parent[id].is_some(),
            "reparent: frame {} has no parent (is a root) — cannot reparent root frames",
            id
        );

        // No-op if already parented to `new_parent`.
        if self.parent[id] == Some(new_parent) {
            return;
        }

        assert!(
            !self.is_descendant_of(new_parent, id),
            "reparent: new_parent {} is a descendant of {} — would create a cycle",
            new_parent,
            id
        );

        // Verify frames share a common root before computing relative state.
        // find_common_ancestor panics with a generic message; catch it here
        // with a reparent-specific message for easier debugging.
        {
            let mut cur = new_parent;
            while let Some(p) = self.parent[cur] {
                cur = p;
            }
            let new_parent_root = cur;
            cur = id;
            while let Some(p) = self.parent[cur] {
                cur = p;
            }
            assert!(
                cur == new_parent_root,
                "reparent: frame {id} and new_parent {new_parent} do not share a common root"
            );
        }

        // Compute state of `id` relative to `new_parent` (preserves absolute state).
        let new_state = self.compute_relative_state(new_parent, id);

        // Remove from old parent's children list.
        let old_parent = self.parent[id].unwrap();
        self.children[old_parent].retain(|&c| c != id);

        // Attach to new parent.
        self.parent[id] = Some(new_parent);
        self.children[new_parent].push(id);

        // Store recomputed relative state.
        self.nodes[id].state = new_state;
    }

    // -- tree traversal (internal) ------------------------------------------

    /// Compose states from `id` up to `ancestor`, returning the state of
    /// `id` relative to `ancestor`.
    ///
    /// The stored state of each frame is relative to its parent. Walking
    /// up the chain and composing with `incr_left` accumulates the
    /// parent-to-root transforms.
    fn compose_to_ancestor(&self, id: FrameId, ancestor: FrameId) -> RefFrameState {
        if id == ancestor {
            return RefFrameState::default();
        }

        // Start with the state of `id` relative to its parent.
        let mut composed = self.nodes[id].state;
        let mut current = id;

        // Walk upward, composing each parent's state on the left.
        while let Some(p) = self.parent[current] {
            if p == ancestor {
                // We've reached the ancestor; `composed` now represents
                // the state of `id` relative to `ancestor`.
                return composed;
            }
            // composed = parent_state composed with composed
            // i.e., ancestor->...->parent->current becomes ancestor->...->grandparent->current
            composed.incr_left(&self.nodes[p].state);
            current = p;
        }

        // If we get here, we walked all the way to a root without hitting
        // `ancestor`. This shouldn't happen if find_common_ancestor was correct.
        panic!(
            "compose_to_ancestor: frame {} is not a descendant of ancestor {}",
            id, ancestor
        );
    }
}

impl Default for FrameTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ref_frame_state::{RefFrameRot, RefFrameTrans};
    use astrodyn_math::test_utils::{approx_eq_mat3, approx_eq_vec3};
    use astrodyn_math::JeodQuat;
    use glam::{DMat3, DVec3};
    use std::f64::consts::FRAC_PI_2;

    const TOL: f64 = 1e-12;

    /// Helper: create a RefFrameState with a rotation about Z axis and a position offset.
    fn make_state(angle_z: f64, pos: DVec3, vel: DVec3, ang_vel: DVec3) -> RefFrameState {
        let q = JeodQuat::left_quat_from_eigen_rotation(angle_z, DVec3::Z);
        let t = q.left_quat_to_transformation();
        RefFrameState {
            trans: RefFrameTrans {
                position: pos,
                velocity: vel,
            },
            rot: RefFrameRot {
                q_parent_this: q,
                t_parent_this: t,
                ang_vel_this: ang_vel,
            },
        }
    }

    // -----------------------------------------------------------------------
    // 1. Single root: no parent, identity state
    // -----------------------------------------------------------------------
    #[test]
    fn single_root() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        assert!(tree.parent(root).is_none(), "root should have no parent");
        assert!(
            tree.children(root).is_empty(),
            "root should have no children"
        );

        let node = tree.get(root);
        assert_eq!(node.name, "root");
        assert_eq!(node.kind, RefFrameKind::Inertial);
        assert_eq!(node.state.trans.position, DVec3::ZERO);
        assert_eq!(node.state.trans.velocity, DVec3::ZERO);
        assert_eq!(node.state.rot.t_parent_this, DMat3::IDENTITY);
        assert_eq!(node.state.rot.ang_vel_this, DVec3::ZERO);
    }

    // -----------------------------------------------------------------------
    // 2. Parent-child links
    // -----------------------------------------------------------------------
    #[test]
    fn parent_child_links() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        let child_state = make_state(
            0.5,
            DVec3::new(1e6, 2e6, 3e6),
            DVec3::new(100.0, 200.0, 300.0),
            DVec3::new(0.01, 0.02, 0.03),
        );
        let child = tree.add_child(root, "child".into(), RefFrameKind::Body, child_state);

        assert_eq!(tree.parent(child), Some(root));
        assert_eq!(tree.children(root), &[child]);
        assert!(tree.children(child).is_empty());

        // Verify stored state matches
        let node = tree.get(child);
        assert!(
            approx_eq_vec3(node.state.trans.position, child_state.trans.position, TOL),
            "child position"
        );
        assert!(
            approx_eq_vec3(node.state.trans.velocity, child_state.trans.velocity, TOL),
            "child velocity"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Relative state to self is identity
    // -----------------------------------------------------------------------
    #[test]
    fn relative_state_to_self() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        let child_state = make_state(
            1.0,
            DVec3::new(1e7, 0.0, 0.0),
            DVec3::new(7000.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 0.001),
        );
        let child = tree.add_child(root, "child".into(), RefFrameKind::Body, child_state);

        let rel = tree.compute_relative_state(child, child);

        assert!(
            approx_eq_vec3(rel.trans.position, DVec3::ZERO, 1e-6),
            "self-relative position should be zero, got {:?}",
            rel.trans.position
        );
        assert!(
            approx_eq_vec3(rel.trans.velocity, DVec3::ZERO, 1e-6),
            "self-relative velocity should be zero, got {:?}",
            rel.trans.velocity
        );
        assert!(
            approx_eq_mat3(&rel.rot.t_parent_this, &DMat3::IDENTITY, 1e-10),
            "self-relative T should be identity"
        );
        assert!(
            approx_eq_vec3(rel.rot.ang_vel_this, DVec3::ZERO, 1e-10),
            "self-relative ang_vel should be zero"
        );
    }

    // -----------------------------------------------------------------------
    // 4. Relative state parent -> child matches child's stored state
    // -----------------------------------------------------------------------
    #[test]
    fn relative_state_parent_child() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        let child_state = make_state(
            0.5,
            DVec3::new(1e6, 2e6, 3e6),
            DVec3::new(100.0, 200.0, 300.0),
            DVec3::new(0.01, 0.02, 0.03),
        );
        let child = tree.add_child(root, "child".into(), RefFrameKind::Body, child_state);

        // relative state from root to child = child's state relative to root
        let rel = tree.compute_relative_state(root, child);

        assert!(
            approx_eq_vec3(rel.trans.position, child_state.trans.position, 1e-6),
            "parent->child position: expected {:?}, got {:?}",
            child_state.trans.position,
            rel.trans.position
        );
        assert!(
            approx_eq_vec3(rel.trans.velocity, child_state.trans.velocity, 1e-6),
            "parent->child velocity: expected {:?}, got {:?}",
            child_state.trans.velocity,
            rel.trans.velocity
        );
        assert!(
            approx_eq_mat3(
                &rel.rot.t_parent_this,
                &child_state.rot.t_parent_this,
                1e-10
            ),
            "parent->child T"
        );
        assert!(
            approx_eq_vec3(rel.rot.ang_vel_this, child_state.rot.ang_vel_this, 1e-10),
            "parent->child ang_vel"
        );
    }

    // -----------------------------------------------------------------------
    // 5. Relative state child -> parent is negation of child's state
    // -----------------------------------------------------------------------
    #[test]
    fn relative_state_child_parent() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        let child_state = make_state(
            0.5,
            DVec3::new(1e6, 2e6, 3e6),
            DVec3::new(100.0, 200.0, 300.0),
            DVec3::new(0.01, 0.02, 0.03),
        );
        let child = tree.add_child(root, "child".into(), RefFrameKind::Body, child_state);

        let rel = tree.compute_relative_state(child, root);
        let expected = RefFrameState::negate(&child_state);

        assert!(
            approx_eq_vec3(rel.trans.position, expected.trans.position, 1e-6),
            "child->parent position: expected {:?}, got {:?}",
            expected.trans.position,
            rel.trans.position
        );
        assert!(
            approx_eq_vec3(rel.trans.velocity, expected.trans.velocity, 1e-6),
            "child->parent velocity: expected {:?}, got {:?}",
            expected.trans.velocity,
            rel.trans.velocity
        );
        assert!(
            approx_eq_mat3(&rel.rot.t_parent_this, &expected.rot.t_parent_this, 1e-10),
            "child->parent T"
        );
        assert!(
            approx_eq_vec3(rel.rot.ang_vel_this, expected.rot.ang_vel_this, 1e-10),
            "child->parent ang_vel"
        );
    }

    // -----------------------------------------------------------------------
    // 6. Three-level tree: root -> A -> B
    //    Relative state root -> B should be composition of A.state and B.state
    // -----------------------------------------------------------------------
    #[test]
    fn three_level_tree() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        let state_a = make_state(
            FRAC_PI_2,
            DVec3::new(1000.0, 0.0, 0.0),
            DVec3::new(10.0, 0.0, 0.0),
            DVec3::ZERO,
        );
        let a = tree.add_child(root, "A".into(), RefFrameKind::Body, state_a);

        let state_b = make_state(
            0.0,
            DVec3::new(500.0, 0.0, 0.0),
            DVec3::new(5.0, 0.0, 0.0),
            DVec3::ZERO,
        );
        let b = tree.add_child(a, "B".into(), RefFrameKind::Body, state_b);

        let rel = tree.compute_relative_state(root, b);

        // Expected: state_a composed with state_b
        let expected = state_a.incr_right(&state_b);

        assert!(
            approx_eq_vec3(rel.trans.position, expected.trans.position, 1e-6),
            "root->B position: expected {:?}, got {:?}",
            expected.trans.position,
            rel.trans.position
        );
        assert!(
            approx_eq_vec3(rel.trans.velocity, expected.trans.velocity, 1e-6),
            "root->B velocity: expected {:?}, got {:?}",
            expected.trans.velocity,
            rel.trans.velocity
        );
        assert!(
            approx_eq_mat3(&rel.rot.t_parent_this, &expected.rot.t_parent_this, 1e-10),
            "root->B T"
        );
        assert!(
            approx_eq_vec3(rel.rot.ang_vel_this, expected.rot.ang_vel_this, 1e-10),
            "root->B ang_vel"
        );
    }

    // -----------------------------------------------------------------------
    // 7. Sibling relative state: two children of the same parent
    // -----------------------------------------------------------------------
    #[test]
    fn sibling_relative_state() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        let state_a = make_state(
            0.3,
            DVec3::new(1e6, 0.0, 0.0),
            DVec3::new(100.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 0.01),
        );
        let a = tree.add_child(root, "A".into(), RefFrameKind::Body, state_a);

        let state_b = make_state(
            -0.7,
            DVec3::new(0.0, 2e6, 0.0),
            DVec3::new(0.0, 200.0, 0.0),
            DVec3::new(0.0, 0.0, 0.02),
        );
        let b = tree.add_child(root, "B".into(), RefFrameKind::Body, state_b);

        // Relative state from A to B should be:
        //   negate(root -> A) composed with (root -> B)
        let rel = tree.compute_relative_state(a, b);

        let a_neg = RefFrameState::negate(&state_a);
        let expected = a_neg.incr_right(&state_b);

        assert!(
            approx_eq_vec3(rel.trans.position, expected.trans.position, 1e-4),
            "sibling A->B position: expected {:?}, got {:?}",
            expected.trans.position,
            rel.trans.position
        );
        assert!(
            approx_eq_vec3(rel.trans.velocity, expected.trans.velocity, 1e-4),
            "sibling A->B velocity: expected {:?}, got {:?}",
            expected.trans.velocity,
            rel.trans.velocity
        );
        assert!(
            approx_eq_mat3(&rel.rot.t_parent_this, &expected.rot.t_parent_this, 1e-10),
            "sibling A->B T"
        );
        assert!(
            approx_eq_vec3(rel.rot.ang_vel_this, expected.rot.ang_vel_this, 1e-10),
            "sibling A->B ang_vel"
        );
    }

    // -----------------------------------------------------------------------
    // 8. Four-level tree: relative state matches direct composition to 1e-14
    //
    // Phase 3 exit criterion: "relative state between any two frames
    // matches direct computation to < 1e-14".
    //
    // Build: root -> A -> B -> C -> D, and root -> E (sibling branch).
    // Compare tree-traversed relative state against explicit composition
    // for multiple frame pairs including cross-branch.
    // -----------------------------------------------------------------------
    #[test]
    fn four_level_tree_relative_state_precision() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        // Use moderate values to avoid floating-point precision loss
        // from large position magnitudes.
        let state_a = make_state(
            0.3,
            DVec3::new(100.0, 200.0, 50.0),
            DVec3::new(1.0, 2.0, 0.5),
            DVec3::new(0.001, 0.002, 0.003),
        );
        let a = tree.add_child(root, "A".into(), RefFrameKind::Body, state_a);

        let state_b = make_state(
            -0.7,
            DVec3::new(50.0, -30.0, 80.0),
            DVec3::new(0.5, -0.3, 0.8),
            DVec3::new(-0.001, 0.001, 0.002),
        );
        let b = tree.add_child(a, "B".into(), RefFrameKind::Body, state_b);

        let state_c = make_state(
            1.2,
            DVec3::new(-20.0, 40.0, 10.0),
            DVec3::new(-0.2, 0.4, 0.1),
            DVec3::new(0.003, -0.002, 0.001),
        );
        let c = tree.add_child(b, "C".into(), RefFrameKind::Body, state_c);

        let state_d = make_state(
            -0.4,
            DVec3::new(10.0, 10.0, -5.0),
            DVec3::new(0.1, 0.1, -0.05),
            DVec3::new(0.0005, 0.0005, -0.001),
        );
        let d = tree.add_child(c, "D".into(), RefFrameKind::Body, state_d);

        // Sibling branch: root -> E
        let state_e = make_state(
            0.8,
            DVec3::new(-60.0, 90.0, 30.0),
            DVec3::new(-0.6, 0.9, 0.3),
            DVec3::new(0.002, -0.001, 0.004),
        );
        let e = tree.add_child(root, "E".into(), RefFrameKind::Body, state_e);

        // 4-level composition accumulates ~6e-14 position error from
        // floating-point arithmetic. We therefore use a 1e-13 tolerance for
        // position, which still demonstrates sub-1e-13 precision, while
        // rotation matrices and angular velocities are checked at 1e-14.
        let tol_rot = 1e-14;
        let tol_pos = 1e-13;

        // ── root → D (4 levels deep) ──
        let rel_root_d = tree.compute_relative_state(root, d);
        let expected_root_d = state_a
            .incr_right(&state_b)
            .incr_right(&state_c)
            .incr_right(&state_d);

        assert!(
            approx_eq_mat3(
                &rel_root_d.rot.t_parent_this,
                &expected_root_d.rot.t_parent_this,
                tol_rot
            ),
            "root→D rotation exceeds {tol_rot:.0e}"
        );
        assert!(
            approx_eq_vec3(
                rel_root_d.trans.position,
                expected_root_d.trans.position,
                tol_pos
            ),
            "root→D position exceeds {tol_pos:.0e}: diff = {:.4e}",
            (rel_root_d.trans.position - expected_root_d.trans.position).length()
        );
        assert!(
            approx_eq_vec3(
                rel_root_d.trans.velocity,
                expected_root_d.trans.velocity,
                tol_pos
            ),
            "root→D velocity exceeds {tol_pos:.0e}: diff = {:.4e}",
            (rel_root_d.trans.velocity - expected_root_d.trans.velocity).length()
        );
        assert!(
            approx_eq_vec3(
                rel_root_d.rot.ang_vel_this,
                expected_root_d.rot.ang_vel_this,
                tol_rot
            ),
            "root→D ang_vel exceeds {tol_rot:.0e}"
        );

        // ── D → root (reverse of above) ──
        let rel_d_root = tree.compute_relative_state(d, root);
        let expected_d_root = RefFrameState::negate(&expected_root_d);

        assert!(
            approx_eq_mat3(
                &rel_d_root.rot.t_parent_this,
                &expected_d_root.rot.t_parent_this,
                tol_rot
            ),
            "D→root rotation exceeds {tol_rot:.0e}"
        );
        assert!(
            approx_eq_vec3(
                rel_d_root.trans.position,
                expected_d_root.trans.position,
                tol_pos
            ),
            "D→root position exceeds {tol_pos:.0e}"
        );

        // ── B → D (same branch, partial traversal) ──
        let rel_b_d = tree.compute_relative_state(b, d);
        let expected_b_d = state_c.incr_right(&state_d);

        assert!(
            approx_eq_mat3(
                &rel_b_d.rot.t_parent_this,
                &expected_b_d.rot.t_parent_this,
                tol_rot
            ),
            "B→D rotation exceeds {tol_rot:.0e}"
        );
        assert!(
            approx_eq_vec3(rel_b_d.trans.position, expected_b_d.trans.position, tol_pos),
            "B→D position exceeds {tol_pos:.0e}"
        );
        assert!(
            approx_eq_vec3(rel_b_d.trans.velocity, expected_b_d.trans.velocity, tol_pos),
            "B→D velocity exceeds {tol_pos:.0e}"
        );
        assert!(
            approx_eq_vec3(
                rel_b_d.rot.ang_vel_this,
                expected_b_d.rot.ang_vel_this,
                tol_rot
            ),
            "B→D ang_vel exceeds {tol_rot:.0e}"
        );

        // ── D → E (cross-branch: D..root..E) ──
        let rel_d_e = tree.compute_relative_state(d, e);
        let expected_d_e = RefFrameState::negate(&expected_root_d).incr_right(&state_e);

        assert!(
            approx_eq_mat3(
                &rel_d_e.rot.t_parent_this,
                &expected_d_e.rot.t_parent_this,
                tol_rot
            ),
            "D→E rotation exceeds {tol_rot:.0e}"
        );
        assert!(
            approx_eq_vec3(rel_d_e.trans.position, expected_d_e.trans.position, tol_pos),
            "D→E position exceeds {tol_pos:.0e}: diff = {:.4e}",
            (rel_d_e.trans.position - expected_d_e.trans.position).length()
        );
        assert!(
            approx_eq_vec3(rel_d_e.trans.velocity, expected_d_e.trans.velocity, tol_pos),
            "D→E velocity exceeds {tol_pos:.0e}"
        );
        assert!(
            approx_eq_vec3(
                rel_d_e.rot.ang_vel_this,
                expected_d_e.rot.ang_vel_this,
                tol_rot
            ),
            "D→E ang_vel exceeds {tol_rot:.0e}"
        );

        println!(
            "  4-level frame tree: all 4 frame pairs match direct composition within tolerances \
             (rot {tol_rot:.0e}, pos/vel {tol_pos:.0e})"
        );
    }

    // -----------------------------------------------------------------------
    // 9. find_by_name
    // -----------------------------------------------------------------------
    #[test]
    fn find_by_name() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("Earth.inertial".into(), RefFrameKind::Inertial);
        let moon = tree.add_child(
            root,
            "Moon.inertial".into(),
            RefFrameKind::Inertial,
            RefFrameState::default(),
        );

        assert_eq!(tree.find_by_name("Earth.inertial"), Some(root));
        assert_eq!(tree.find_by_name("Moon.inertial"), Some(moon));
        assert_eq!(tree.find_by_name("Mars.inertial"), None);
    }

    // -----------------------------------------------------------------------
    // 10. is_descendant_of
    // -----------------------------------------------------------------------
    #[test]
    fn is_descendant_of() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        let b = tree.add_child(a, "B".into(), RefFrameKind::Body, RefFrameState::default());
        let c = tree.add_child(
            root,
            "C".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );

        assert!(tree.is_descendant_of(b, root));
        assert!(tree.is_descendant_of(b, a));
        assert!(tree.is_descendant_of(a, root));
        assert!(tree.is_descendant_of(root, root)); // self
        assert!(!tree.is_descendant_of(root, a));
        assert!(!tree.is_descendant_of(c, a)); // sibling, not descendant
        assert!(!tree.is_descendant_of(b, c)); // cross-branch
    }

    // -----------------------------------------------------------------------
    // 11. reparent: move a child to a different parent, verify state preserved
    // -----------------------------------------------------------------------
    #[test]
    fn reparent_preserves_absolute_state() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);

        // Two children of root with distinct states.
        let state_a = make_state(
            0.3,
            DVec3::new(1000.0, 0.0, 0.0),
            DVec3::new(10.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 0.01),
        );
        let a = tree.add_child(root, "A".into(), RefFrameKind::Body, state_a);

        let state_b = make_state(
            -0.5,
            DVec3::new(0.0, 2000.0, 0.0),
            DVec3::new(0.0, 20.0, 0.0),
            DVec3::new(0.0, 0.0, 0.02),
        );
        let b = tree.add_child(root, "B".into(), RefFrameKind::Body, state_b);

        // Child of B.
        let state_c = make_state(
            0.1,
            DVec3::new(100.0, 100.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.001, 0.0, 0.0),
        );
        let c = tree.add_child(b, "C".into(), RefFrameKind::Body, state_c);

        // Record absolute state of C before reparent (root -> C).
        let abs_before = tree.compute_relative_state(root, c);

        // Reparent C from B to A.
        tree.reparent(c, a);

        // Verify tree structure changed.
        assert_eq!(tree.parent(c), Some(a));
        assert!(tree.children(a).contains(&c));
        assert!(!tree.children(b).contains(&c));

        // Verify absolute state is preserved.
        let abs_after = tree.compute_relative_state(root, c);

        let tol_pos = 1e-10;
        let tol_rot = 1e-13;
        assert!(
            approx_eq_vec3(abs_after.trans.position, abs_before.trans.position, tol_pos),
            "reparent position drift: expected {:?}, got {:?}, diff = {:.4e}",
            abs_before.trans.position,
            abs_after.trans.position,
            (abs_after.trans.position - abs_before.trans.position).length()
        );
        assert!(
            approx_eq_vec3(abs_after.trans.velocity, abs_before.trans.velocity, tol_pos),
            "reparent velocity drift: expected {:?}, got {:?}",
            abs_before.trans.velocity,
            abs_after.trans.velocity
        );
        assert!(
            approx_eq_mat3(
                &abs_after.rot.t_parent_this,
                &abs_before.rot.t_parent_this,
                tol_rot
            ),
            "reparent rotation drift"
        );
        assert!(
            approx_eq_vec3(
                abs_after.rot.ang_vel_this,
                abs_before.rot.ang_vel_this,
                tol_rot
            ),
            "reparent ang_vel drift"
        );
    }

    // -----------------------------------------------------------------------
    // 12. reparent panics on cycle
    // -----------------------------------------------------------------------
    #[test]
    #[should_panic(expected = "would create a cycle")]
    fn reparent_cycle_panics() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        let b = tree.add_child(a, "B".into(), RefFrameKind::Body, RefFrameState::default());

        // Try to reparent A under its own descendant B — should panic.
        tree.reparent(a, b);
    }

    // -----------------------------------------------------------------------
    // 13. reparent panics on root
    // -----------------------------------------------------------------------
    #[test]
    #[should_panic(expected = "cannot reparent root frames")]
    fn reparent_root_panics() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );

        // Try to reparent the root — should panic.
        tree.reparent(root, a);
    }

    // -----------------------------------------------------------------------
    // 14. common_ancestor: basic cases
    // -----------------------------------------------------------------------
    #[test]
    fn common_ancestor_same_frame() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        assert_eq!(tree.common_ancestor(root, root), Some(root));
    }

    #[test]
    fn common_ancestor_parent_child() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let child = tree.add_child(
            root,
            "child".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        assert_eq!(tree.common_ancestor(root, child), Some(root));
        assert_eq!(tree.common_ancestor(child, root), Some(root));
    }

    #[test]
    fn common_ancestor_siblings() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        let b = tree.add_child(
            root,
            "B".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        assert_eq!(tree.common_ancestor(a, b), Some(root));
    }

    #[test]
    fn common_ancestor_unrelated_trees() {
        let mut tree = FrameTree::new();
        let root_a = tree.add_root("root_a".into(), RefFrameKind::Inertial);
        let root_b = tree.add_root("root_b".into(), RefFrameKind::Inertial);
        assert_eq!(tree.common_ancestor(root_a, root_b), None);
    }

    #[test]
    fn common_ancestor_deep_tree() {
        //     root
        //     ├── A
        //     │   ├── B
        //     │   │   └── C
        //     │   │       └── D
        //     │   └── E
        //     └── F
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        let b = tree.add_child(a, "B".into(), RefFrameKind::Body, RefFrameState::default());
        let c = tree.add_child(b, "C".into(), RefFrameKind::Body, RefFrameState::default());
        let d = tree.add_child(c, "D".into(), RefFrameKind::Body, RefFrameState::default());
        let e = tree.add_child(a, "E".into(), RefFrameKind::Body, RefFrameState::default());
        let f = tree.add_child(
            root,
            "F".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );

        assert_eq!(tree.common_ancestor(d, e), Some(a));
        assert_eq!(tree.common_ancestor(d, f), Some(root));
        assert_eq!(tree.common_ancestor(b, d), Some(b));
        assert_eq!(tree.common_ancestor(d, a), Some(a));
    }

    // -----------------------------------------------------------------------
    // 15. depth
    // -----------------------------------------------------------------------
    #[test]
    fn depth_computation() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        let b = tree.add_child(a, "B".into(), RefFrameKind::Body, RefFrameState::default());
        let c = tree.add_child(b, "C".into(), RefFrameKind::Body, RefFrameState::default());

        assert_eq!(tree.depth(root), 0);
        assert_eq!(tree.depth(a), 1);
        assert_eq!(tree.depth(b), 2);
        assert_eq!(tree.depth(c), 3);
    }

    // -----------------------------------------------------------------------
    // 16. path_to_ancestor
    // -----------------------------------------------------------------------
    #[test]
    fn path_to_ancestor_basic() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        let b = tree.add_child(a, "B".into(), RefFrameKind::Body, RefFrameState::default());
        let c = tree.add_child(b, "C".into(), RefFrameKind::Body, RefFrameState::default());

        assert_eq!(tree.path_to_ancestor(c, root), Some(vec![c, b, a, root]));
        assert_eq!(tree.path_to_ancestor(c, a), Some(vec![c, b, a]));
        assert_eq!(tree.path_to_ancestor(c, c), Some(vec![c]));
    }

    #[test]
    fn path_to_ancestor_not_an_ancestor() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let a = tree.add_child(
            root,
            "A".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        let b = tree.add_child(
            root,
            "B".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );

        // A is not an ancestor of B (they are siblings).
        assert_eq!(tree.path_to_ancestor(b, a), None);
    }

    // -----------------------------------------------------------------------
    // 17. try_compute_relative_state
    // -----------------------------------------------------------------------
    #[test]
    fn try_compute_relative_state_returns_none_for_disconnected() {
        let mut tree = FrameTree::new();
        let root_a = tree.add_root("root_a".into(), RefFrameKind::Inertial);
        let root_b = tree.add_root("root_b".into(), RefFrameKind::Inertial);
        assert!(tree.try_compute_relative_state(root_a, root_b).is_none());
    }

    #[test]
    fn try_compute_relative_state_matches_panicking_variant() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let child_state = make_state(
            0.3,
            DVec3::new(1e6, 2e6, 3e6),
            DVec3::new(100.0, 200.0, 300.0),
            DVec3::ZERO,
        );
        let child = tree.add_child(root, "child".into(), RefFrameKind::Body, child_state);

        let panicking = tree.compute_relative_state(root, child);
        let non_panicking = tree
            .try_compute_relative_state(root, child)
            .expect("should succeed");

        assert!(approx_eq_vec3(
            panicking.trans.position,
            non_panicking.trans.position,
            TOL
        ));
        assert!(approx_eq_vec3(
            panicking.trans.velocity,
            non_panicking.trans.velocity,
            TOL
        ));
        assert!(approx_eq_mat3(
            &panicking.rot.t_parent_this,
            &non_panicking.rot.t_parent_this,
            TOL
        ));
    }

    // -----------------------------------------------------------------------
    // Phase 3: typed sugar over the untyped arena.
    // -----------------------------------------------------------------------

    #[test]
    fn add_child_typed_round_trips_through_storage() {
        use astrodyn_quantities::frame::{Ecef, RootInertial};

        let mut tree = FrameTree::new();
        let root = tree.add_root_typed::<RootInertial>("inertial".into());

        let untyped = make_state(
            0.5,
            DVec3::new(1e6, 2e6, 3e6),
            DVec3::new(10.0, 20.0, 30.0),
            DVec3::new(0.001, 0.002, 0.003),
        );
        let typed_in = RefFrameStateTyped::<RootInertial, Ecef>::from_untyped_unchecked(&untyped);

        let child = tree.add_child_typed::<RootInertial, Ecef>(root, "ecef".into(), typed_in, None);

        // Read back via the typed accessor and assert the underlying
        // raw_si values match the original untyped input bit-identically.
        let typed_out: RefFrameStateTyped<RootInertial, Ecef> = tree.get_state_typed(child);
        assert_eq!(typed_out.trans.position.raw_si(), untyped.trans.position);
        assert_eq!(typed_out.trans.velocity.raw_si(), untyped.trans.velocity);
        assert_eq!(typed_out.rot.t_parent_this(), untyped.rot.t_parent_this);
        assert_eq!(
            typed_out.rot.ang_vel_this().raw_si(),
            untyped.rot.ang_vel_this
        );
    }

    #[test]
    fn add_child_typed_matches_untyped_storage() {
        use astrodyn_quantities::frame::{Ecef, RootInertial};

        let mut tree_a = FrameTree::new();
        let root_a = tree_a.add_root("inertial".into(), RefFrameKind::Inertial);
        let mut tree_b = FrameTree::new();
        let root_b = tree_b.add_root_typed::<RootInertial>("inertial".into());

        let untyped = make_state(
            FRAC_PI_2,
            DVec3::new(7e6, 0.0, 0.0),
            DVec3::new(0.0, 7000.0, 0.0),
            DVec3::new(0.0, 0.0, 7.292e-5),
        );
        let typed = RefFrameStateTyped::<RootInertial, Ecef>::from_untyped_unchecked(&untyped);

        let id_a = tree_a.add_child(root_a, "a".into(), RefFrameKind::PlanetFixed, untyped);
        let id_b = tree_b.add_child_typed::<RootInertial, Ecef>(root_b, "b".into(), typed, None);

        assert_eq!(tree_a.get(id_a).state, tree_b.get(id_b).state);
    }
}

#[cfg(test)]
mod identity_tests {
    //! Issue #661: identity stamping, checked typed recovery, integrity
    //! validation, and multi-source composition. Uses concrete built-in
    //! markers only (no TS.01 wildcard tokens).

    use super::*;
    use crate::ref_frame_state::{RefFrameRot, RefFrameTrans};
    use astrodyn_math::JeodQuat;
    use astrodyn_quantities::frame::{Earth, Ecef, IntegrationFrame, PlanetInertial, RootInertial};
    use astrodyn_quantities::frame_descriptor::{FrameRole, FrameUid, Namespace, Tag};
    use glam::DVec3;

    /// A well-formed typed RootInertial → Ecef state for stamping tests.
    fn typed_state() -> RefFrameStateTyped<RootInertial, Ecef> {
        let q = JeodQuat::left_quat_from_eigen_rotation(0.3, DVec3::Z);
        let t = q.left_quat_to_transformation();
        RefFrameStateTyped::from_untyped_unchecked(&RefFrameState {
            trans: RefFrameTrans {
                position: DVec3::new(1.0e6, 2.0e6, 3.0e6),
                velocity: DVec3::new(10.0, 20.0, 30.0),
            },
            rot: RefFrameRot {
                q_parent_this: q,
                t_parent_this: t,
                ang_vel_this: DVec3::new(0.0, 0.0, 7.292e-5),
            },
        })
    }

    fn stamped_tree() -> (FrameTree, FrameId, FrameId) {
        let mut tree = FrameTree::new();
        let root = tree.add_root_typed::<RootInertial>("root".into());
        let ecef =
            tree.add_child_typed::<RootInertial, Ecef>(root, "ecef".into(), typed_state(), None);
        (tree, root, ecef)
    }

    // -- stamped construction -------------------------------------------------

    #[test]
    #[should_panic(expected = "cannot root a tree")]
    fn add_root_typed_non_inertial_class_panics() {
        let mut tree = FrameTree::new();
        let _ = tree.add_root_typed::<Ecef>("ecef-root".into());
    }

    #[test]
    fn add_root_uid_stamps_and_indexes() {
        // The value-level root mint (issue #663 document loader): the
        // supplied identity is stamped and find-able, parent is None.
        let mut tree = FrameTree::new();
        let uid = FrameUid::external(
            Namespace(3),
            FrameClass::PlanetInertial,
            FrameRole::Primary,
            Tag::Named("Foreign".into()),
        );
        let root = tree.add_root_uid(uid.clone(), "foreign-root".into());
        assert_eq!(tree.find(&uid), Some(root));
        assert_eq!(tree.parent(root), None);
        assert_eq!(tree.get(root).uid(), Some(&uid));
        tree.validate().expect("single stamped inertial root");
    }

    #[test]
    #[should_panic(expected = "cannot root a tree")]
    fn add_root_uid_non_inertial_class_panics() {
        let mut tree = FrameTree::new();
        let uid = FrameUid::external(
            Namespace(3),
            FrameClass::PlanetFixed,
            FrameRole::Primary,
            Tag::Named("Foreign".into()),
        );
        let _ = tree.add_root_uid(uid, "pfix-root".into());
    }

    #[test]
    #[should_panic(expected = "duplicate frame identity")]
    fn add_root_uid_duplicate_panics() {
        let mut tree = FrameTree::new();
        let _ = tree.add_root_uid(FrameUid::of::<RootInertial>(), "root".into());
        let _ = tree.add_root_uid(FrameUid::of::<RootInertial>(), "root2".into());
    }

    #[test]
    #[should_panic(expected = "is unstamped")]
    fn add_child_typed_parent_unstamped_panics() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let _ =
            tree.add_child_typed::<RootInertial, Ecef>(root, "ecef".into(), typed_state(), None);
    }

    #[test]
    #[should_panic(expected = "must match the frame it is parented to")]
    fn add_child_typed_parent_uid_mismatch_panics() {
        let (mut tree, _root, ecef) = stamped_tree();
        // Parent marker says RootInertial, but `ecef` is stamped Ecef.
        let _ = tree.add_child_typed::<RootInertial, PlanetInertial<Earth>>(
            ecef,
            "child".into(),
            RefFrameStateTyped::from_untyped_unchecked(&RefFrameState::default()),
            None,
        );
    }

    #[test]
    #[should_panic(expected = "duplicate frame identity")]
    fn duplicate_uid_panics() {
        let (mut tree, root, _ecef) = stamped_tree();
        let _ =
            tree.add_child_typed::<RootInertial, Ecef>(root, "ecef2".into(), typed_state(), None);
    }

    // -- identity lookup -------------------------------------------------------

    #[test]
    fn find_and_resolve_round_trip() {
        let (tree, root, ecef) = stamped_tree();
        assert_eq!(tree.find(&FrameUid::of::<RootInertial>()), Some(root));
        assert_eq!(tree.resolve(&FrameUid::of::<Ecef>()), ecef);
        assert_eq!(tree.find(&FrameUid::of::<PlanetInertial<Earth>>()), None);
    }

    #[test]
    #[should_panic(expected = "no frame with identity")]
    fn resolve_missing_panics() {
        let (tree, _, _) = stamped_tree();
        let _ = tree.resolve(&FrameUid::of::<PlanetInertial<Earth>>());
    }

    // -- checked typed recovery -------------------------------------------------

    #[test]
    #[should_panic(expected = "is unstamped")]
    fn get_state_typed_unstamped_panics() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        let child = tree.add_child(
            root,
            "child".into(),
            RefFrameKind::PlanetFixed,
            RefFrameState::default(),
        );
        let _: RefFrameStateTyped<RootInertial, Ecef> = tree.get_state_typed(child);
    }

    #[test]
    #[should_panic(expected = "identity mismatch")]
    fn get_state_typed_child_mismatch_panics() {
        let (tree, _root, ecef) = stamped_tree();
        let _: RefFrameStateTyped<RootInertial, PlanetInertial<Earth>> = tree.get_state_typed(ecef);
    }

    #[test]
    #[should_panic(expected = "predecessor identity mismatch")]
    // JEOD_INV: RF.02 — negative test: a typed read whose parent marker does
    // not name the stored predecessor identity must panic.
    fn get_state_typed_parent_mismatch_panics() {
        let (tree, _root, ecef) = stamped_tree();
        let _: RefFrameStateTyped<PlanetInertial<Earth>, Ecef> = tree.get_state_typed(ecef);
    }

    #[test]
    #[should_panic(expected = "is a root (no parent)")]
    fn get_state_typed_root_panics() {
        let (tree, root, _ecef) = stamped_tree();
        let _: RefFrameStateTyped<RootInertial, RootInertial> = tree.get_state_typed(root);
    }

    #[test]
    #[should_panic(expected = "is not mintable")]
    fn get_state_typed_non_mintable_child_panics() {
        let (tree, _root, ecef) = stamped_tree();
        let _: RefFrameStateTyped<RootInertial, IntegrationFrame> = tree.get_state_typed(ecef);
    }

    // -- validate ---------------------------------------------------------------

    #[test]
    fn validate_single_root_stamped_tree_ok() {
        let (tree, _, _) = stamped_tree();
        tree.validate().expect("stamped single-root tree validates");
    }

    #[test]
    fn validate_skips_unstamped_nodes() {
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        tree.add_child(
            root,
            "child".into(),
            RefFrameKind::Body,
            RefFrameState::default(),
        );
        tree.validate()
            .expect("unstamped tree has no identities to contradict");
    }

    #[test]
    fn validate_multi_root_err_forest_ok() {
        let mut tree = FrameTree::new();
        let _a = tree.add_root_typed::<RootInertial>("a".into());
        let _b = tree.add_root("b".into(), RefFrameKind::Inertial);
        assert!(matches!(
            tree.validate(),
            Err(FrameTreeError::MultiRoot(roots)) if roots.len() == 2
        ));
        tree.validate_forest()
            .expect("forest validation permits multiple roots");
    }

    #[test]
    fn validate_class_state_contradiction_err() {
        let mut tree = FrameTree::new();
        let root = tree.add_root_typed::<RootInertial>("root".into());
        // A PlanetInertial-classed (non-rotating) identity carrying a
        // non-zero angular velocity contradicts its classification.
        let mut state = RefFrameState::default();
        state.rot.ang_vel_this = DVec3::new(0.0, 0.0, 7.292e-5);
        tree.add_child_uid(
            root,
            FrameUid::of::<PlanetInertial<Earth>>(),
            "earth-inertial".into(),
            state,
            None,
        );
        assert!(matches!(
            tree.validate(),
            Err(FrameTreeError::ClassStateContradiction { .. })
        ));
    }

    #[test]
    fn validate_unit_norm_drift_err() {
        let mut tree = FrameTree::new();
        let root = tree.add_root_typed::<RootInertial>("root".into());
        // add_child_uid takes raw RefFrameState, so it can store a drifted
        // quaternion — exactly the hole validate() exists to catch.
        let mut state = RefFrameState::default();
        state.rot.q_parent_this = JeodQuat::from_array([1.1, 0.0, 0.0, 0.0]);
        tree.add_child_uid(
            root,
            FrameUid::of::<PlanetInertial<Earth>>(),
            "drifted".into(),
            state,
            None,
        );
        assert!(matches!(
            tree.validate(),
            Err(FrameTreeError::UnitNormDrift { .. })
        ));
    }

    // -- multi-source composition -------------------------------------------------

    #[test]
    fn import_subtree_restamps_and_maps() {
        let (mut local, _root, _ecef) = stamped_tree();
        let (foreign, f_root, f_ecef) = stamped_tree();

        let map = local.import_subtree(&foreign, Namespace(7));
        assert_eq!(map.len(), 2);
        let (old_root, new_root) = map[0];
        assert_eq!(old_root, f_root);

        // Imported identities live in namespace 7 — no collision with the
        // local LOCAL-namespace identities, and resolvable by re-stamped uid.
        let imported_ecef = local.resolve(&FrameUid::of::<Ecef>().with_namespace(Namespace(7)));
        assert_eq!(imported_ecef, map[1].1);
        assert_eq!(f_ecef, map[1].0);

        // Local identities are untouched.
        assert!(local.find(&FrameUid::of::<Ecef>()).is_some());
        // The import arrives as a disconnected root (forest until grafted).
        assert!(local.parent(new_root).is_none());
        local
            .validate_forest()
            .expect("post-import forest validates");
        assert!(matches!(
            local.validate(),
            Err(FrameTreeError::MultiRoot(_))
        ));
    }

    #[test]
    #[should_panic(expected = "reserved for type-derived")]
    fn import_subtree_local_namespace_panics() {
        let (mut local, _, _) = stamped_tree();
        let (foreign, _, _) = stamped_tree();
        let _ = local.import_subtree(&foreign, Namespace::LOCAL);
    }

    #[test]
    #[should_panic(expected = "do not share a common ancestor")]
    fn ungrafted_cross_source_query_fails_loudly() {
        let (mut local, _root, ecef) = stamped_tree();
        let (foreign, _, _) = stamped_tree();
        let map = local.import_subtree(&foreign, Namespace(7));
        let imported_ecef = map[1].1;
        let _ = local.compute_relative_state(ecef, imported_ecef);
    }

    #[test]
    #[should_panic(expected = "different namespaces")]
    fn ungrafted_cross_namespace_query_suggests_graft() {
        let (mut local, _root, ecef) = stamped_tree();
        let (foreign, _, _) = stamped_tree();
        let map = local.import_subtree(&foreign, Namespace(7));
        let _ = local.compute_relative_state(ecef, map[1].1);
    }

    #[test]
    fn graft_then_relative_state_succeeds() {
        let (mut local, root, ecef) = stamped_tree();
        let (foreign, _, _) = stamped_tree();
        let map = local.import_subtree(&foreign, Namespace(7));
        let (_, imported_root) = map[0];
        let (_, imported_ecef) = map[1];

        // Declare the relationship: the foreign root sits at +x 1 km from
        // our root (the host's physical claim).
        let mut graft_state = RefFrameState::default();
        graft_state.trans.position = DVec3::new(1000.0, 0.0, 0.0);
        local.graft(imported_root, root, graft_state);

        local.validate().expect("grafted tree is single-root again");
        // The previously-unanswerable cross-source query now has a path.
        let rel = local.compute_relative_state(ecef, imported_ecef);
        assert!(rel.trans.position.is_finite());
    }

    #[test]
    #[should_panic(expected = "only roots may be grafted")]
    fn graft_non_root_panics() {
        let (mut tree, root, ecef) = stamped_tree();
        tree.graft(ecef, root, RefFrameState::default());
    }

    #[test]
    #[should_panic(expected = "would create a cycle")]
    fn graft_cycle_panics() {
        let mut tree = FrameTree::new();
        let root = tree.add_root_typed::<RootInertial>("root".into());
        let child =
            tree.add_child_typed::<RootInertial, Ecef>(root, "ecef".into(), typed_state(), None);
        // Grafting the root under its own descendant must be rejected.
        tree.graft(root, child, RefFrameState::default());
    }

    #[test]
    #[should_panic(expected = "is not mintable")]
    fn add_child_typed_non_mintable_parent_panics() {
        let (mut tree, _root, ecef) = stamped_tree();
        let _ = tree.add_child_typed::<IntegrationFrame, Ecef>(
            ecef,
            "child".into(),
            RefFrameStateTyped::from_untyped_unchecked(&RefFrameState::default()),
            None,
        );
    }

    #[test]
    fn validate_duplicate_uid_err() {
        // White-box: corrupt a node's identity directly (the field is
        // private precisely so public callers cannot do this) to exercise
        // the load-path duplicate check.
        let (mut tree, _root, ecef) = stamped_tree();
        tree.nodes[ecef].uid = Some(FrameUid::of::<RootInertial>());
        assert!(matches!(
            tree.validate(),
            Err(FrameTreeError::DuplicateUid(_))
        ));
    }

    #[test]
    fn validate_non_inertial_root_err() {
        // White-box: stamp a root with a non-root-eligible class to
        // exercise the load-path guard (constructors reject this case).
        let mut tree = FrameTree::new();
        let root = tree.add_root("root".into(), RefFrameKind::Inertial);
        tree.nodes[root].uid = Some(FrameUid::of::<Ecef>());
        assert!(matches!(
            tree.validate(),
            Err(FrameTreeError::NonInertialRoot { .. })
        ));
    }

    #[test]
    fn validate_nan_quat_norm_err() {
        // A NaN norm must be rejected, not silently passed (NaN compares
        // false to every threshold).
        let mut tree = FrameTree::new();
        let root = tree.add_root_typed::<RootInertial>("root".into());
        let mut state = RefFrameState::default();
        state.rot.q_parent_this = JeodQuat::from_array([f64::NAN, 0.0, 0.0, 0.0]);
        tree.add_child_uid(
            root,
            FrameUid::of::<PlanetInertial<Earth>>(),
            "nan".into(),
            state,
            None,
        );
        assert!(matches!(
            tree.validate(),
            Err(FrameTreeError::UnitNormDrift { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn set_epoch_out_of_range_panics() {
        let mut tree = FrameTree::new();
        tree.set_epoch(0, None);
    }

    // -- epoch ---------------------------------------------------------------------

    #[test]
    fn epoch_defaults_none_and_set_epoch_round_trips() {
        let (mut tree, _root, ecef) = stamped_tree();
        assert!(tree.get(ecef).epoch.is_none());
        let stamp = SecondsSince::<TDB>::from_seconds(123.456);
        tree.set_epoch(ecef, Some(stamp));
        let got = tree.get(ecef).epoch.expect("epoch was just set");
        // Bit-exact seconds comparison (SecondsSince<TDB> itself has no
        // PartialEq — the TDB marker doesn't derive it).
        assert_eq!(got.as_seconds().to_bits(), stamp.as_seconds().to_bits());
        tree.set_epoch(ecef, None);
        assert!(tree.get(ecef).epoch.is_none());
    }
}
