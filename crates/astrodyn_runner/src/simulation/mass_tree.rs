// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! Mass-tree topology and detached-subtree machinery for [`super::Simulation`].
//!
//! Carries the bigger attach/detach methods that previously lived in
//! `lib.rs` (~600 lines): `add_body_to_tree`, `attach`, `detach`,
//! `detach_subtree` (~200 lines), `attach_subtree_aligned` (~240 lines,
//! ports JEOD's `DynBody::attach_child` momentum-conservation
//! algorithm), `step_detached_subtrees`, and the
//! `subtree_composite_inertial` chain-walk accessor.

use glam::{DMat3, DVec3};

use astrodyn::typed_bridge::{
    mass_raw_to_self_ref, mass_typed_to_raw, rot_raw_to_self_ref, trans_raw_to_typed,
};
use astrodyn::{
    combine_states_at_attach, AttachCombineInputs, CrossIntegFrameStateShift, DetachedSubtreeState,
    IntegrationFrame, MassBodyId, MassPointState, RefFrameRot, RefFrameState, RefFrameTrans,
    RotationalState, TranslationalState,
};

use super::Simulation;

/// Pre-attach state captured by `attach_inner` when the combine kernel
/// is going to run. Bundled into a struct so the gating between the
/// runtime path (`attach`) and the configuration-time path
/// (`attach_preserving_initial_state`) lives in a single
/// `Option<AttachSnapshot>` instead of nine individually-conditioned
/// locals — the configuration-time path skips the snapshot, so the
/// `IntegOrigin` lookups that would silently no-op cost nothing.
struct AttachSnapshot {
    root_integ_origin_pos: glam::DVec3,
    root_integ_origin_vel: glam::DVec3,
    root_pre_state: RefFrameState,
    root_pre_composite_props: astrodyn::MassProperties,
    child_pre_state: RefFrameState,
    child_pre_composite_props: astrodyn::MassProperties,
    root_has_rot: bool,
}

impl Simulation {
    /// Register a body in the simulation's mass tree.
    ///
    /// Creates (or reuses) a `MassTree` and adds the body's mass as a node.
    /// Returns the `MassBodyId` for use with [`attach`](Self::attach) and
    /// [`detach`](Self::detach). The body's `mass` field must be `Some`.
    ///
    /// # Panics
    /// Panics if the body has no mass properties.
    pub fn add_body_to_tree(
        &mut self,
        body_idx: usize,
        name: impl Into<String>,
    ) -> astrodyn::MassBodyId {
        let mass = self.bodies[body_idx]
            .mass
            .expect("add_body_to_tree requires mass properties");
        let tree = self.mass_tree.get_or_insert_with(astrodyn::MassTree::new);
        let id = tree.add_body(name.into(), mass_typed_to_raw(&mass));
        self.bodies[body_idx].mass_body_id = Some(id);
        id
    }

    /// Attach a child body to a parent body in the mass tree.
    ///
    /// Both bodies must have been registered via [`add_body_to_tree`](Self::add_body_to_tree).
    /// After attachment, every ancestor's composite mass properties are
    /// updated automatically (via `MassTree::recompute_composites`).
    ///
    /// If both bodies have an integrated state (translational, and
    /// optionally rotational), the parent's `body.trans` / `body.rot`
    /// are then overwritten with JEOD's momentum-conservation merge of
    /// the two pre-attach composite-body states (port of
    /// `combine_states_at_attach` in `dyn_body_attach.cc`):
    ///
    /// - Linear momentum about the integration-frame origin is
    ///   conserved (post-attach velocity = mass-weighted average of the
    ///   pre-attach pair).
    /// - The composite-body inertial position is shifted by the
    ///   inertial-frame CoM-delta so it tracks the new combined CoM in
    ///   the parent's struct frame.
    /// - Angular momentum about the new combined CoM is conserved (the
    ///   solve `ω_t = I_t⁻¹ · L_total` runs only when `body.rot` is
    ///   `Some` on both bodies). The merged body inherits the parent's
    ///   attitude per JEOD's "colinear body axes" comment in
    ///   `dyn_body_attach.cc`.
    ///
    /// The integrators of every body whose composite changed (the child
    /// plus the parent's full ancestor chain to the root) are then
    /// reset to match JEOD's `dyn_body_attach.cc::reset_integrators()`
    /// semantics.
    ///
    /// # Panics
    /// Panics if either body is not in the tree, or if the child already has a parent.
    pub fn attach(
        &mut self,
        child_idx: usize,
        parent_idx: usize,
        offset: DVec3,
        t_parent_child: DMat3,
    ) {
        // Public runtime entry point: a runtime in-flight attach must
        // conserve momentum, so the JEOD combine kernel runs and the
        // merged composite-body state is written back to the integrated
        // tree root. See `attach_preserving_initial_state` for the
        // configuration-time sibling that skips the writeback.
        self.attach_inner(child_idx, parent_idx, offset, t_parent_child, true);
    }

    /// Configuration-time sibling of [`attach`](Self::attach) that
    /// performs the tree mutation, composite-mass resync, and
    /// integrator-history bookkeeping **without** running JEOD's
    /// `combine_states_at_attach` momentum-conservation kernel — so the
    /// caller's pre-set `body.trans` / `body.rot` on the child and on
    /// the integrated tree root are preserved verbatim.
    ///
    /// This exists for the [`Simulation::from_builder`] path
    /// (`crates/astrodyn_runner/src/builder.rs`), which materialises a
    /// [`SimulationBuilder`](astrodyn::SimulationBuilder)'s
    /// `attach_bodies` declarations at construction time. Builder
    /// callers have already specified the post-attach state they want
    /// (parent and child `VehicleConfig::trans` / `rot` are part of the
    /// build spec); running the runtime combine over those would treat
    /// the user-supplied initial conditions as a pre-attach pair to
    /// merge, silently turning the build-time topology declaration into
    /// an in-flight impulse merge. That would change the meaning of
    /// `SimulationBuilder` for any consumer that registers bodies with
    /// independent initial states (orbital element + LVLH child, etc.)
    /// — a breaking API change disguised as a momentum-conservation
    /// fix. Configuration-time attaches must instead preserve the
    /// spec'd states; the runtime [`attach`](Self::attach) is the one
    /// that needs momentum conservation.
    ///
    /// The integrator-history reset still runs (topology has changed,
    /// so `gj_state` / `abm4_state` must be reset per JEOD_INV: IG.37).
    /// This method is `pub(crate)` because the only legitimate caller
    /// is `from_builder`; mission-code attaches must go through the
    /// public [`attach`](Self::attach) so momentum is conserved.
    ///
    /// # Panics
    /// Panics if either body is not in the tree, or if the child
    /// already has a parent.
    pub(crate) fn attach_preserving_initial_state(
        &mut self,
        child_idx: usize,
        parent_idx: usize,
        offset: DVec3,
        t_parent_child: DMat3,
    ) {
        self.attach_inner(child_idx, parent_idx, offset, t_parent_child, false);
    }

    /// Shared implementation backing [`attach`](Self::attach) and
    /// [`attach_preserving_initial_state`](Self::attach_preserving_initial_state).
    ///
    /// `combine_writeback = true` runs JEOD's `combine_states_at_attach`
    /// kernel and overwrites the integrated tree root's `body.trans` /
    /// `body.rot` with the merged composite-body state (the runtime
    /// attach contract). `combine_writeback = false` skips both the
    /// snapshot and the writeback, leaving the caller-supplied initial
    /// states in place (the configuration-time builder contract). All
    /// other steps — tree mutation, composite-mass resync on every
    /// affected body, integrator-history dirty-mark + reset — run in
    /// both modes because they reflect topology changes that hold
    /// regardless of whether the kinematic state is being merged.
    fn attach_inner(
        &mut self,
        child_idx: usize,
        parent_idx: usize,
        offset: DVec3,
        t_parent_child: DMat3,
        combine_writeback: bool,
    ) {
        let child_id = self.bodies[child_idx]
            .mass_body_id
            .expect("attach: child body not in mass tree");
        let parent_id = self.bodies[parent_idx]
            .mass_body_id
            .expect("attach: parent body not in mass tree");

        // ── Resolve the *subject root*: the body whose tree is actually
        //    being re-rooted under `parent`. For the root-subject case
        //    (subject == its own tree root, no existing parent edge in
        //    the mass tree), `subject_root_id == child_id` and
        //    `subject_root_idx == child_idx`; for the chained-attach /
        //    re-rooting case (subject is itself a non-root body in some
        //    other tree, e.g. JEOD's `RUN_complex_attach_detach` where
        //    veh1.attach_to_3 fires while veh1 is already attached to
        //    veh2), the subject root is veh1's existing tree root —
        //    veh2 in that example — and that's the body whose
        //    integrator-written composite-body state seeds the
        //    momentum-conservation combine.
        //
        //    Mirrors JEOD `DynBody::attach_child` (line 521 of
        //    `dyn_body_attach.cc`):
        //      `child_root = child.get_root_body_internal();`
        //    and the subsequent `child_root->attach_establish_links(*this)`
        //    branch when `child_root != &child`.
        let (subject_root_id, subject_root_idx) = {
            let tree_ro = self.mass_tree.as_ref().expect("attach: no mass tree");
            let root = tree_ro.root_of(child_id);
            if root == child_id {
                (child_id, child_idx)
            } else {
                let idx = self
                    .bodies
                    .iter()
                    .position(|b| b.mass_body_id == Some(root))
                    .unwrap_or_else(|| {
                        panic!(
                            "attach: subject root {root:?} for re-rooting child {child_id:?} \
                             has no Simulation body backing it — `Simulation::attach` requires \
                             the subject's tree root to be a registered SimBody so its \
                             integrator-written composite-body state seeds the combine."
                        )
                    });
                (root, idx)
            }
        };

        // ── Resolve the integrated tree root that owns the parent's
        //    pre-attach composite-body state. Per JEOD_INV: DB.17 only
        //    the root carries integrator-written `body.trans` /
        //    `body.rot`; every interior SimBody on the chain has its
        //    state derived each tick from the root via
        //    `propagate_kinematic_state`. Reading the immediate
        //    parent's `body.trans` directly would silently consume one-
        //    tick-stale derived state when `parent_idx` is interior —
        //    or, for an interior body that was never flagged
        //    `kinematic_only`, frozen state from `add_body` time —
        //    feeding the combine kernel a wrong `parent_composite` and
        //    breaking momentum conservation across the whole tree.
        //    Walking to the root and using its authoritative state
        //    (composed forward through the tree to the parent's node
        //    when needed) gives the same answer regardless of whether
        //    parent IS the root or sits arbitrarily deep underneath
        //    it; the root-as-parent case collapses to the previous
        //    code path bit-identically (`root_idx == parent_idx`,
        //    `derive_subtree_composite_state` is the trivial walk).
        let (root_id, root_idx) = {
            let tree_ro = self.mass_tree.as_ref().expect("attach: no mass tree");
            let walker = tree_ro.root_of(parent_id);
            let idx = self
                .bodies
                .iter()
                .position(|b| b.mass_body_id == Some(walker))
                .unwrap_or_else(|| {
                    panic!(
                        "attach: integrated tree root {walker:?} for parent {parent_id:?} \
                         has no Simulation body backing it — `Simulation::attach` requires \
                         the parent's tree root to be a registered SimBody so its \
                         integrator-written composite-body state seeds the combine. \
                         For tree-only-rooted subtrees use `attach_subtree_aligned` instead."
                    )
                });
            (walker, idx)
        };

        // ── Site A: mark every body whose composite mass is about to
        //    change as topology-dirty. JEOD_INV: IG.37 — this is bound
        //    to the topology mutation itself; if we ever do this in a
        //    new method but forget the matching reset (Site B below),
        //    the dirty flag remains set and `integrate()` panics on
        //    the next step with the IG.37 diagnostic.
        //
        //    For the plain root-subject attach, the bodies whose
        //    composite changes are the subject (== child) plus the
        //    parent's full ancestor chain (since `MassTree::attach`
        //    walks `recompute_composites` from leaves to every root, so
        //    any ancestor of the new parent is touched).
        //
        //    For the re-rooting case the subject root *and every body
        //    in its existing subtree* now sit under `parent`, and
        //    `recompute_composites` walks both the new combined tree
        //    and the old tree's former root chain. The conservative
        //    set is therefore: subject_root + its descendants +
        //    parent's ancestor chain. Including the descendants matters
        //    because, post-reroot, those bodies are kinematic children
        //    of the merged tree's root and any GJ/ABM4 history they
        //    accumulated as pre-reroot integrated bodies (they were
        //    interior nodes in the subject tree and may have been
        //    integrated standalone before the original attach) is now
        //    stale topology-wise.
        //
        // The set is sorted + deduped so the helpers below (and the
        // mass-sync pass) can use `binary_search` for O(log n)
        // membership instead of a linear `Vec::contains` scan,
        // mirroring the Bevy path's affected-id discipline.
        let mut affected_ids: Vec<astrodyn::MassBodyId> = vec![child_id];
        {
            let tree_ro = self.mass_tree.as_ref().expect("attach: no mass tree");
            affected_ids.extend(tree_ro.ancestors_inclusive(parent_id));
            // Subject's whole subtree (rooted at subject_root_id,
            // including subject_root_id itself). For the root-subject
            // case this just adds `subject_root_id == child_id` again;
            // the dedup below collapses it. For the reroot case it adds
            // every body that hung off the subject's old root chain.
            affected_ids.extend(tree_ro.subtree_ids(subject_root_id));
        }
        affected_ids.sort_unstable();
        affected_ids.dedup();
        Self::mark_body_integrators_dirty_by_id(&mut self.bodies, &affected_ids);

        // ── Snapshot pre-attach state required by the combine kernel:
        //    the integrated tree root's composite-body inertial state +
        //    composite mass (the whole pre-attach tree's authoritative
        //    state), the new child's composite-body state + mass, and
        //    whether the root carries a rotational state. The kernel
        //    composes the new whole-tree composite from these inputs;
        //    `combine_states_at_attach` is symmetric in "parent" and
        //    "child" sides at the algorithmic level, so passing the
        //    pre-attach root in place of the immediate parent is the
        //    rigorous formulation when the parent is interior. For the
        //    parent-IS-root case (`root_idx == parent_idx`) this is
        //    bit-identical to the previous direct read once the
        //    integ-frame lift below collapses to the no-op path.
        //
        //    JEOD_INV: RF.10 — `body.trans` is typed
        //    `TranslationalStateTyped<IntegrationFrame>`; the combine
        //    kernel does cross-body composition (mass-weighted velocity,
        //    inertial-frame CoM shift, ω×r over offsets) which is only
        //    arithmetic-valid when both bodies' translational state
        //    lives in the same inertial frame. Lift each body's state
        //    to root-inertial at the kernel boundary via its
        //    `IntegOrigin`. For root-integrated bodies the shift is a
        //    bit-identical `IntegOrigin::zero()` no-op; for any body
        //    integrating in a non-root planet (the cross-frame chain
        //    case `parent in root + child in PlanetInertial<P>`) the
        //    lift is the only thing that keeps the kernel from
        //    silently mixing coordinates across distinct integration
        //    frames. Mirrors the kinematic walk's seed-time lift in
        //    `step::kinematic`.
        //
        // The snapshot is gated on `combine_writeback`: the
        // configuration-time path (`attach_preserving_initial_state`)
        // skips both the snapshot and the writeback so the caller's
        // pre-set `body.trans` / `body.rot` survive the topology
        // change verbatim. The integ-frame lift, kernel call, and
        // writeback below all live behind the same gate so the entire
        // momentum-merge stays within one branch.
        // The body whose integrator-written composite-body state seeds
        // the *child side* of the combine kernel is the **subject
        // root** — not necessarily `child_idx` itself. For the
        // root-subject case `subject_root_idx == child_idx` so this is
        // bit-identical to the previous code; for the re-rooting case
        // the subject root carries the integrated state of the whole
        // pre-reroot subtree, which is the right composite to merge
        // with the parent-side root.
        let snapshot = if combine_writeback {
            let root_integ_origin_pos = {
                let (p, _v) = self.frame_origin(self.bodies[root_idx].integ_frame_id);
                p
            };
            let subject_root_integ_origin_pos = {
                let (p, _v) = self.frame_origin(self.bodies[subject_root_idx].integ_frame_id);
                p
            };
            let root_integ_origin_vel = {
                let (_p, v) = self.frame_origin(self.bodies[root_idx].integ_frame_id);
                v
            };
            let subject_root_integ_origin_vel = {
                let (_p, v) = self.frame_origin(self.bodies[subject_root_idx].integ_frame_id);
                v
            };
            // Lift each side's pre-attach state from its own integration
            // frame to root-inertial via the
            // [`CrossIntegFrameStateShift`] kernel. `old = body's integ
            // origin`, `new = ZERO` makes `apply` add the integ origin to
            // the stored `(position, velocity)` — the lift partner of
            // the writeback shift below. Same kernel as the Bevy adapter
            // uses per-descendant in `apply_cross_integ_frame_attach`;
            // here it runs once per side because the runner stores at
            // most one integ-frame-typed state per body. For
            // root-integrated bodies the kernel collapses to
            // `between_integ_origins(ZERO, ZERO, ZERO, ZERO)` and
            // `apply` is a bit-identical no-op.
            let root_lift = CrossIntegFrameStateShift::between_integ_origins(
                root_integ_origin_pos,
                root_integ_origin_vel,
                glam::DVec3::ZERO,
                glam::DVec3::ZERO,
            );
            let subject_root_lift = CrossIntegFrameStateShift::between_integ_origins(
                subject_root_integ_origin_pos,
                subject_root_integ_origin_vel,
                glam::DVec3::ZERO,
                glam::DVec3::ZERO,
            );
            let mut root_pre_state = Self::body_composite_state_or_default(&self.bodies[root_idx]);
            (root_pre_state.trans.position, root_pre_state.trans.velocity) =
                root_lift.apply(root_pre_state.trans.position, root_pre_state.trans.velocity);
            let root_pre_composite_props = mass_typed_to_raw(
                self.bodies[root_idx]
                    .mass
                    .as_ref()
                    .expect("attach: tree root has no mass properties"),
            );
            let mut child_pre_state =
                Self::body_composite_state_or_default(&self.bodies[subject_root_idx]);
            (
                child_pre_state.trans.position,
                child_pre_state.trans.velocity,
            ) = subject_root_lift.apply(
                child_pre_state.trans.position,
                child_pre_state.trans.velocity,
            );
            let child_pre_composite_props = mass_typed_to_raw(
                self.bodies[subject_root_idx]
                    .mass
                    .as_ref()
                    .expect("attach: subject root has no mass properties"),
            );
            let root_has_rot = self.bodies[root_idx].rot.is_some();
            Some(AttachSnapshot {
                root_integ_origin_pos,
                root_integ_origin_vel,
                root_pre_state,
                root_pre_composite_props,
                child_pre_state,
                child_pre_composite_props,
                root_has_rot,
            })
        } else {
            None
        };

        // ── Mutate the tree itself. The plain `MassTree::attach`
        //    panics if the child already has a parent; for re-rooting
        //    chained-attach scenarios the subject root has no parent
        //    (we just walked up to find it) but the *subject* might,
        //    and JEOD's `attach_child` reroot path is exactly the
        //    pattern `attach_with_reroot` ports — recompute the
        //    geometry so the subject ends up where the caller asked,
        //    even though the underlying tree edge runs from `parent`
        //    to the subject's existing root.
        let tree = self.mass_tree.as_mut().expect("attach: no mass tree");
        // JEOD_INV: BA.12 — runner dispatches every public attach through the
        // reroot-aware kernel so chained-attach scenarios pick the JEOD
        // `dyn_body_attach.cc:521-567` path automatically.
        let _attached_root = tree.attach_with_reroot(child_id, parent_id, offset, t_parent_child);
        // Sync every affected body's composite mass from the tree.
        // `affected_ids` is sorted + deduped above; binary_search keeps
        // this O(n_bodies · log n_affected) instead of O(n²).
        for body in self.bodies.iter_mut() {
            if let Some(id) = body.mass_body_id {
                if affected_ids.binary_search(&id).is_ok() {
                    // allowed: typed↔raw kernel-boundary lift from
                    // mass-tree raw composite (see #397).
                    let raw = tree.get(id).composite_properties;
                    body.mass = Some(mass_raw_to_self_ref(&raw));
                }
            }
        }

        // ── Run the JEOD momentum-conservation combine and write the
        //    merged composite-body state back into the integrated tree
        //    root. The kernel always runs (matching the Bevy adapter's
        //    `staging_system`); for 3-DOF bodies the missing rotational
        //    state is treated as identity attitude + zero ang_vel,
        //    which yields the linear-momentum-only result the kernel
        //    produces in that degenerate case.
        // JEOD_INV: DB.13 — composite-body propagation on topology change.
        // JEOD_INV: DB.14 — integration-frame switch on attach: the merged
        // body integrates in the root's frame; the root's composite_body
        // state is the new integration target. Interior kinematic-only
        // bodies along the chain are rederived from the freshly-written
        // root state by `propagate_kinematic_state` on the next step
        // (root → leaves walk), so they don't need an explicit
        // writeback here.
        if let Some(snap) = snapshot {
            // Read the post-mutation root composite mass — this is the
            // *combined* mass of the whole merged tree, which the kernel
            // uses to solve `ω_t = I_t⁻¹ · L_total` and to shift the
            // inertial position by the CoM-delta. Reading at the root
            // (not at `parent_id`) is the change that matches the
            // parent-as-root semantics and keeps the kernel's
            // invariants when the attaching parent is an interior node.
            let combined_root_composite_props = self
                .mass_tree
                .as_ref()
                .expect("attach: mass tree dropped between mutate and read")
                .get(root_id)
                .composite_properties;
            let root_t_struct_to_body = snap.root_pre_composite_props.t_parent_this;
            let root_t_inertial_struct = astrodyn::compute_t_inertial_struct(
                &root_t_struct_to_body,
                &snap.root_pre_state.rot.t_parent_this,
            );
            let combined = combine_states_at_attach(astrodyn::AttachCombineInputs {
                parent_composite: snap.root_pre_state,
                parent_mass: snap.root_pre_composite_props,
                parent_t_inertial_struct: root_t_inertial_struct,
                child_composite: snap.child_pre_state,
                child_mass: snap.child_pre_composite_props,
                combined_mass: combined_root_composite_props,
                orig_parent_cm_struct: snap.root_pre_composite_props.position,
            });

            // Write the merged translational state regardless of DOF —
            // for 3-DOF bodies this is a pure linear-momentum-conservation
            // update on velocity plus the structure-frame CoM-shift,
            // both of which are well-defined without rotational state.
            //
            // JEOD_INV: RF.10 — symmetric partner of the seed-time lift
            // above. The kernel returned `combined.composite_state` in
            // root-inertial coordinates; lower back through the root's
            // `IntegOrigin` so the writeback lands in the typed
            // `TranslationalStateTyped<IntegrationFrame>` storage. For a
            // root-integrated root the shift is `IntegOrigin::zero()`
            // and the subtraction is a bit-identical no-op; for a root
            // that integrates in `PlanetInertial<P>` the shift is the
            // only thing that prevents writing a root-inertial value
            // into integration-frame storage and silently corrupting
            // every downstream consumer of `body.trans`.
            // Symmetric partner of the seed-time lift: lower the merged
            // root-inertial composite back into integration-frame
            // coordinates via the [`CrossIntegFrameStateShift`] kernel.
            // `old = ZERO, new = root's integ origin` makes `apply`
            // subtract the integ origin from the merged state. For a
            // root-integrated root the kernel collapses to a no-op.
            let root_lower = CrossIntegFrameStateShift::between_integ_origins(
                glam::DVec3::ZERO,
                glam::DVec3::ZERO,
                snap.root_integ_origin_pos,
                snap.root_integ_origin_vel,
            );
            let (writeback_position, writeback_velocity) = root_lower.apply(
                combined.composite_state.trans.position,
                combined.composite_state.trans.velocity,
            );
            self.bodies[root_idx].trans =
                trans_raw_to_typed::<IntegrationFrame>(&TranslationalState {
                    position: writeback_position,
                    velocity: writeback_velocity,
                });
            // Write rotational state only when the root already carried
            // one — the merged body inherits the root's DOF. JEOD does
            // the same: a 3-DOF root stays 3-DOF post-merge regardless
            // of whether the child contributed `Iω` to the
            // angular-momentum solve.
            if snap.root_has_rot {
                // allowed: typed↔raw kernel-boundary lift from kinematic
                // composite-state writeback (see #397).
                let raw = RotationalState {
                    quaternion: combined.composite_state.rot.q_parent_this,
                    ang_vel_body: combined.composite_state.rot.ang_vel_this,
                };
                self.bodies[root_idx].rot = Some(rot_raw_to_self_ref(&raw));
            }
        }

        // ── Auto-flag the rerooted subject subtree as kinematic-only.
        //
        //    This block runs only on the **reroot** path — i.e. when
        //    `subject_root_id != child_id` (the subject was already a
        //    non-root body in some other tree and is now being reparented
        //    along with its ancestors). The simple root-subject attach
        //    (`subject_root_id == child_id`) deliberately does NOT
        //    auto-mark: there, the subject was a tree root with no prior
        //    integrated/derived split, and callers retain explicit
        //    control over whether the freshly-attached child should
        //    integrate or be derived kinematically (e.g.
        //    `propagate_kinematic_state_panics_on_non_kinematic_intermediate`
        //    deliberately attaches without flagging the new child
        //    kinematic to confirm the walk's fail-loud ancestor check
        //    still fires). For that case, mission code still has to
        //    call `Simulation::mark_kinematic_only(child_idx)` after
        //    `Simulation::attach(...)` — this auto-flag is **not** a
        //    drop-in replacement for that public call.
        //
        //    Per JEOD_INV: DB.17 only the integrated tree root carries
        //    integrator-written `body.trans` / `body.rot`; every interior
        //    SimBody must derive its state each tick from the root
        //    through `propagate_kinematic_state`. Pre-attach the subject
        //    root *was* the integrated root for its tree; post-attach it
        //    sits under `parent`'s tree root and must be rederived each
        //    step. We auto-flag the subject root + every descendant in
        //    its old subtree so the `propagate_kinematic_state` walk
        //    doesn't panic on its own "non-kinematic interior body"
        //    guard the next time [`Simulation::step`] runs.
        //
        //    The configuration-time path
        //    (`attach_preserving_initial_state`,
        //    `combine_writeback == false`) skips this auto-flag so
        //    builder-driven topology declarations don't bake
        //    kinematic-only flags into bodies whose initial state the
        //    user expects to be honoured verbatim.
        //
        //    Bodies without a `RotationalState` (3-DOF) cannot be made
        //    kinematic-only: `propagate_kinematic_state` derives both
        //    `trans` and `rot` from the parent's composed pose, and
        //    that walk requires every kinematic node to carry a
        //    `RotationalState`. A 3-DOF body in a rerooted subtree is
        //    therefore unreachable by the kinematic walk *and* no
        //    longer the integrated tree root, so its `body.trans` would
        //    silently go stale post-reroot. Fail loudly here — at the
        //    attach site that introduced the configuration — rather
        //    than letting the `propagate_kinematic_state` ancestor
        //    check panic on the first post-reroot step, which surfaces
        //    a confusing "non-kinematic ancestor" diagnostic far from
        //    the root cause.
        if combine_writeback && subject_root_id != child_id {
            // Use a `HashSet` rather than `Vec` so the membership check
            // inside the body loop is O(1) instead of O(n_descendants);
            // attach is a hot path on larger trees (chained docking,
            // multi-stage separation) where the subject subtree may be
            // O(10) bodies and the runner may carry O(10²) sim bodies.
            let subject_descendants: std::collections::HashSet<astrodyn::MassBodyId> = self
                .mass_tree
                .as_ref()
                .expect("attach: mass tree dropped between mutate and auto-flag")
                .subtree_ids(subject_root_id)
                .into_iter()
                .collect();
            for (body_idx, body) in self.bodies.iter_mut().enumerate() {
                if let Some(id) = body.mass_body_id {
                    if subject_descendants.contains(&id) {
                        assert!(
                            body.rot.is_some(),
                            "attach (reroot): SimBody {body_idx} (mass_body_id {id:?}) is in \
                             the rerooted subtree under subject_root {subject_root_id:?} but \
                             has no RotationalState. Kinematic propagation derives both `trans` \
                             and `rot` from the integrating root, so any attached (non-root) \
                             body must be 6-DOF — a 3-DOF body's state would go stale post-attach. \
                             Make this body 6-DOF by setting `VehicleConfig::rot = Some(...)` \
                             before the attach, or restructure the topology so this body never \
                             becomes a non-root member of the merged tree."
                        );
                        body.kinematic_only = true;
                    }
                }
            }
        }

        // ── Site B: reset the integrator history. Separate from Site A
        //    so a regression that drops this call leaves the dirty flag
        //    set (IG.37 panics on next integrate). Mirrors JEOD's
        //    `dyn_body_attach.cc::reset_integrators()` (lines 860, 871).
        Self::reset_body_integrators_by_id(&mut self.bodies, &affected_ids);
    }

    /// Read a body's composite-body inertial state into a
    /// [`RefFrameState`] suitable for [`combine_states_at_attach`] and
    /// the body-aware tree-walk used in [`detach`](Self::detach) /
    /// [`detach_subtree`](Self::detach_subtree).
    ///
    /// 3-DOF bodies have `body.rot == None`; in that case the rotational
    /// fields default to identity attitude + zero ang_vel — matching the
    /// `unwrap_or((JeodQuat::identity(), DVec3::ZERO))` pattern the Bevy
    /// adapter's `staging_system` uses for the same degenerate case so
    /// the two adapters produce bit-identical kernel inputs.
    fn body_composite_state_or_default(body: &super::types::SimBody) -> RefFrameState {
        let position = body.trans.position.raw_si();
        let velocity = body.trans.velocity.raw_si();
        let (q, w) = match body.rot {
            Some(rot) => (
                rot.q_inertial_body.to_jeod_quat(),
                rot.ang_vel_body.raw_si(),
            ),
            None => (astrodyn::JeodQuat::identity(), DVec3::ZERO),
        };
        RefFrameState {
            trans: RefFrameTrans { position, velocity },
            rot: RefFrameRot {
                q_parent_this: q,
                t_parent_this: q.left_quat_to_transformation(),
                ang_vel_this: w,
            },
        }
    }

    /// Detach a child body from its parent in the mass tree.
    ///
    /// After detachment, the former parent's *and every one of its
    /// ancestors'* composite mass properties are updated from the
    /// tree's recomputed composites (mirroring
    /// `MassTree::recompute_composites`). The child becomes a root and
    /// its own composite is also updated.
    ///
    /// The detach is the inverse of [`attach`](Self::attach)'s
    /// momentum-conservation merge:
    ///
    /// - The detached child's `body.trans` (and `body.rot` if it
    ///   carries one) is rederived from the parent's pre-detach
    ///   composite-body inertial state plus the body-aware tree-walk
    ///   offset between the two composites — the rigid-body
    ///   composition that gives the subtree's instantaneous inertial
    ///   state at the moment of separation. This mirrors JEOD's
    ///   `dyn_body_detach.cc::detach_mass_internal` which preserves
    ///   `core_body` and rederives composite from it via
    ///   `propagate_state()`.
    /// - The (former) parent's `body.trans` (and `body.rot`) shift by
    ///   the inertial-frame composite-CoM delta so the integrated
    ///   composite-body state continues to track the parent's new
    ///   (smaller) composite. The structure point doesn't move in
    ///   inertial space; only the CoM offset within the parent's
    ///   struct frame moves when the subtree leaves. This mirrors the
    ///   parent-side update done by [`detach_subtree`](Self::detach_subtree)
    ///   and by the Bevy adapter's `staging_system` detach branch.
    ///
    /// The integrators of every affected body are then reset
    /// (`dyn_body_detach.cc:271-273`'s `reset_integrators()`).
    ///
    /// # Panics
    /// Panics if the body is not in the tree or has no parent.
    pub fn detach(&mut self, child_idx: usize) {
        let child_id = self.bodies[child_idx]
            .mass_body_id
            .expect("detach: child body not in mass tree");

        // ── Site A: mark every affected body's integrators dirty
        //    BEFORE we mutate the tree. The set of bodies whose
        //    composite changes after detach is the (former) child plus
        //    the former parent's full ancestor chain, since
        //    `MassTree::detach` recomputes composites bottom-up over
        //    every tree root (the new tree containing the parent and
        //    the new tree containing the freshly detached child).
        //
        // The set is sorted + deduped so the helpers below (and the
        // mass-sync pass) can use `binary_search` for O(log n)
        // membership instead of a linear `Vec::contains` scan,
        // mirroring the Bevy path's affected-id discipline.
        let mut affected_ids: Vec<astrodyn::MassBodyId> = vec![child_id];
        let parent_id = {
            let tree_ro = self.mass_tree.as_ref().expect("detach: no mass tree");
            let pid = tree_ro
                .parent(child_id)
                .expect("detach: child body has no parent in tree");
            affected_ids.extend(tree_ro.ancestors_inclusive(pid));
            pid
        };
        affected_ids.sort_unstable();
        affected_ids.dedup();
        Self::mark_body_integrators_dirty_by_id(&mut self.bodies, &affected_ids);

        // ── Snapshot pre-detach state required by the inverse split:
        //
        //   - the (former) parent's tree root + integrated composite
        //     state, used to (a) shift the parent's body.trans by the
        //     inertial CoM-delta after the topology change, and (b)
        //     derive the detached child's instantaneous composite-body
        //     inertial state via the body-aware tree walk.
        //
        // Walk up to the integrated tree root from the (former) parent
        // — the root carries the merged composite-body state for the
        // pre-detach tree.
        let (root_id, root_idx) = {
            let tree_ro = self
                .mass_tree
                .as_ref()
                .expect("detach: no mass tree (root walk)");
            let mut walker = parent_id;
            while let Some(p) = tree_ro.parent(walker) {
                walker = p;
            }
            let idx = self
                .bodies
                .iter()
                .position(|b| b.mass_body_id == Some(walker))
                .unwrap_or_else(|| {
                    panic!(
                        "detach: tree root {walker:?} for child {child_id:?} \
                         has no Simulation body backing it — `Simulation::detach` \
                         requires the integrated tree root to be a registered body. \
                         For tree-only subtrees use `detach_subtree` instead."
                    )
                });
            (walker, idx)
        };

        // JEOD_INV: RF.10 — `body.trans` is typed
        // `TranslationalStateTyped<IntegrationFrame>`; the body-aware
        // tree walk in `derive_subtree_composite_state` composes states
        // via `propagate_forward`, which is arithmetic-valid only when
        // the seed state is in a single inertial frame. Lift the root's
        // translational state to root-inertial via its `IntegOrigin`
        // before seeding the walk. For a root-integrated root the lift
        // is a bit-identical `IntegOrigin::zero()` no-op; for a root
        // that integrates in `PlanetInertial<P>` the lift is the only
        // thing that prevents seeding the walk with a position relative
        // to the planet's origin and producing a child composite that
        // silently mixes integ-frame parent coords with inertial-frame
        // walk offsets. Mirrors `attach`'s seed-time lift.
        let root_integ_origin_pos = {
            let (p, _v) = self.frame_origin(self.bodies[root_idx].integ_frame_id);
            p
        };
        let root_integ_origin_vel = {
            let (_p, v) = self.frame_origin(self.bodies[root_idx].integ_frame_id);
            v
        };
        let child_integ_origin_pos = {
            let (p, _v) = self.frame_origin(self.bodies[child_idx].integ_frame_id);
            p
        };
        let child_integ_origin_vel = {
            let (_p, v) = self.frame_origin(self.bodies[child_idx].integ_frame_id);
            v
        };
        // Lift the (former) parent's pre-detach root state from its
        // integration frame to root-inertial via the
        // [`CrossIntegFrameStateShift`] kernel so the body-aware tree
        // walk in `derive_subtree_composite_state` runs in a single
        // inertial frame. `old = root's integ origin, new = ZERO`
        // makes `apply` add the integ origin to the stored
        // `(position, velocity)`. Same kernel as the symmetric attach
        // lift; root-integrated roots collapse to a no-op.
        let root_lift = CrossIntegFrameStateShift::between_integ_origins(
            root_integ_origin_pos,
            root_integ_origin_vel,
            glam::DVec3::ZERO,
            glam::DVec3::ZERO,
        );
        let mut root_pre_state = Self::body_composite_state_or_default(&self.bodies[root_idx]);
        (root_pre_state.trans.position, root_pre_state.trans.velocity) =
            root_lift.apply(root_pre_state.trans.position, root_pre_state.trans.velocity);
        let root_pre_composite_props = mass_typed_to_raw(
            self.bodies[root_idx]
                .mass
                .as_ref()
                .expect("detach: tree root has no mass properties"),
        );

        // The walk runs in root-inertial coordinates (the seed has been
        // lifted above), so `child_pre_state` is also root-inertial.
        // The integ-frame writeback below shifts each side back through
        // its own `IntegOrigin`.
        let child_pre_state =
            Self::derive_subtree_composite_state(self, root_id, child_id, root_pre_state);
        let child_has_rot = self.bodies[child_idx].rot.is_some();

        // ── Mutate the tree. ──
        let tree = self.mass_tree.as_mut().expect("detach: no mass tree");
        tree.detach(child_id);
        // Sync mass on every affected body from the recomputed tree.
        // `affected_ids` is sorted + deduped above; binary_search keeps
        // this O(n_bodies · log n_affected) instead of O(n²).
        for body in self.bodies.iter_mut() {
            if let Some(id) = body.mass_body_id {
                if affected_ids.binary_search(&id).is_ok() {
                    // allowed: typed↔raw kernel-boundary lift from
                    // mass-tree raw composite (see #397).
                    let raw = tree.get(id).composite_properties;
                    body.mass = Some(mass_raw_to_self_ref(&raw));
                }
            }
        }

        // Read the post-mutation composite props of the (former)
        // parent's root — this is the smaller post-detach composite
        // whose CoM has shifted within the root's struct frame.
        let root_post_composite_props = self
            .mass_tree
            .as_ref()
            .expect("detach: mass tree dropped between mutate and read")
            .get(root_id)
            .composite_properties;

        // ── Apply the parent-side composite-CoM shift. ──
        // The structure point hasn't moved in inertial space, but the
        // composite-CoM has moved within the struct frame, so the
        // composite-body inertial state must shift by the corresponding
        // kinematic offset. Same formula as `detach_subtree` and the
        // Bevy adapter; both produce bit-identical results.
        // JEOD_INV: DB.13 — composite-body propagation on topology change.
        let cm_delta_struct =
            root_post_composite_props.position - root_pre_composite_props.position;
        let t_struct_to_body = root_pre_composite_props.t_parent_this;
        let cm_delta_body = t_struct_to_body * cm_delta_struct;
        let t_inertial_to_body = root_pre_state.rot.t_parent_this;
        let cm_delta_inertial = t_inertial_to_body.transpose() * cm_delta_body;
        // Velocity offset from rigid-body rotation: ω × Δr in body
        // frame, then rotated to inertial.
        let omega_body = root_pre_state.rot.ang_vel_this;
        let dvel_inertial = t_inertial_to_body.transpose() * omega_body.cross(cm_delta_body);

        // `new_root_position` / `new_root_velocity` live in root-inertial
        // (the seed-lift above ensured the kinematic offset arithmetic
        // ran in a single inertial frame). The storage target is typed
        // `TranslationalStateTyped<IntegrationFrame>`, so lower back
        // through the root's `IntegOrigin` before the
        // `from_untyped_unchecked` cast — symmetric partner of the
        // seed-lift, mirroring the writeback pattern PR #295 established
        // for `propagate_kinematic_state`. For a root-integrated root
        // the shift is `IntegOrigin::zero()` and the subtraction is a
        // bit-identical no-op; for a root in `PlanetInertial<P>` it is
        // the only thing that keeps an inertial-coord value from being
        // labelled as integration-frame and silently corrupting every
        // downstream consumer of `body.trans`.
        let new_root_position = root_pre_state.trans.position + cm_delta_inertial;
        let new_root_velocity = root_pre_state.trans.velocity + dvel_inertial;

        // Lower the (former) parent root's post-detach state from
        // root-inertial back into its integration-frame coordinates
        // via the [`CrossIntegFrameStateShift`] kernel — symmetric
        // partner of the seed-time lift above.
        let root_lower = CrossIntegFrameStateShift::between_integ_origins(
            glam::DVec3::ZERO,
            glam::DVec3::ZERO,
            root_integ_origin_pos,
            root_integ_origin_vel,
        );
        let (root_writeback_position, root_writeback_velocity) =
            root_lower.apply(new_root_position, new_root_velocity);
        self.bodies[root_idx].trans = trans_raw_to_typed::<IntegrationFrame>(&TranslationalState {
            position: root_writeback_position,
            velocity: root_writeback_velocity,
        });
        // root.rot is unchanged — composite/core share body axes (the
        // mass tree's recompute keeps composite_properties.t_parent_this
        // == core_properties.t_parent_this throughout). A 3-DOF body
        // stays 3-DOF regardless of the detach.

        // The child becomes a free Simulation body whose `body.trans` /
        // `body.rot` should hold its instantaneous composite-body state
        // at the detach instant. Single-body detach via
        // `Simulation::detach` does NOT route through
        // `detached_subtrees` (that map is for tree-only subtrees that
        // have no Simulation body backing — the runner integrates the
        // detached child going forward).
        //
        // `child_pre_state` came out of the inertial-frame walk seeded
        // above, so it lives in root-inertial. The child's storage is
        // typed `TranslationalStateTyped<IntegrationFrame>` and the
        // integ frame here is the child's own (unchanged across detach
        // — `add_body` resolved it from `VehicleConfig::integ_source`).
        // Lower through the child's `IntegOrigin` before the cast so
        // the typed label matches the stored coordinates.
        // Lower the freshly-detached child's instantaneous
        // root-inertial composite-body state into its own
        // integration-frame coordinates via the
        // [`CrossIntegFrameStateShift`] kernel. The child's integ
        // frame is unchanged across detach (`add_body` resolved it
        // from `VehicleConfig::integ_source` and stays put), so this
        // is the symmetric writeback partner of `child_lift =
        // between_integ_origins(child_integ_origin, ZERO)` even
        // though the lift never explicitly ran (the kernel walk
        // produced a root-inertial value directly from the
        // already-lifted parent root).
        let child_lower = CrossIntegFrameStateShift::between_integ_origins(
            glam::DVec3::ZERO,
            glam::DVec3::ZERO,
            child_integ_origin_pos,
            child_integ_origin_vel,
        );
        let (child_writeback_position, child_writeback_velocity) = child_lower.apply(
            child_pre_state.trans.position,
            child_pre_state.trans.velocity,
        );
        self.bodies[child_idx].trans =
            trans_raw_to_typed::<IntegrationFrame>(&TranslationalState {
                position: child_writeback_position,
                velocity: child_writeback_velocity,
            });
        if child_has_rot {
            // allowed: typed↔raw kernel-boundary lift on detach writeback (see #397).
            let raw = RotationalState {
                quaternion: child_pre_state.rot.q_parent_this,
                ang_vel_body: child_pre_state.rot.ang_vel_this,
            };
            self.bodies[child_idx].rot = Some(rot_raw_to_self_ref(&raw));
        }
        // `parent_id` is the immediate parent in the chain; the actual
        // shift target is the integrated tree root (`root_idx`), which
        // may be the parent itself or a more remote ancestor. Bound
        // here to make the chain walk explicit even though the
        // post-detach update operates on the root.
        let _ = parent_id;

        // ── Site B: reset integrator history. Mirrors JEOD's
        //    `dyn_body_detach.cc:271-273` `reset_integrators()` call.
        //    Separated from Site A so a future regression that drops
        //    this call leaves the dirty bit set on every affected body
        //    and panics in IG.37 on the next integrate.
        Self::reset_body_integrators_by_id(&mut self.bodies, &affected_ids);

        // ── Clear `kinematic_only` on the freshly-detached child. After
        //    `tree.detach`, the child is a tree root with no parent;
        //    `propagate_kinematic_state` would panic on the next
        //    `step()` ("kinematic-only resolves to a tree root") if the
        //    flag were left set. Clearing it here keeps detach as the
        //    single fail-loud entry point: callers don't have to
        //    remember a paired `clear_kinematic_only` call, and a
        //    detached body resumes integrated dynamics on its own —
        //    matching JEOD `dyn_body_detach.cc`'s transfer of state
        //    ownership from `propagate_state_from_*` back to
        //    `integrated_frame`. No-op when the flag was already
        //    clear (the common case for non-kinematic children).
        self.bodies[child_idx].kinematic_only = false;
    }

    /// Walk root → target through the mass tree applying
    /// [`astrodyn::propagate_forward`] at each level using
    /// `composite_wrt_pstr` offsets and structure-point rotations.
    /// Returns the target node's instantaneous composite-body inertial
    /// state assuming the whole chain shares the root's rigid motion
    /// (which is the rigid-body invariant that holds at the attach /
    /// detach instant).
    ///
    /// Identical algorithm to the inline walks in
    /// [`detach_subtree`](Self::detach_subtree) and to the Bevy
    /// adapter's `staging_system` detach handler — kept as a private
    /// helper here so [`detach`](Self::detach) doesn't have to inline
    /// the same eight lines a third time.
    fn derive_subtree_composite_state(
        sim: &Simulation,
        root_id: astrodyn::MassBodyId,
        target_id: astrodyn::MassBodyId,
        root_state: RefFrameState,
    ) -> RefFrameState {
        let tree = sim
            .mass_tree
            .as_ref()
            .expect("derive_subtree_composite_state: no mass tree");
        let mut chain = Vec::<astrodyn::MassBodyId>::new();
        let mut cur = target_id;
        while cur != root_id {
            chain.push(cur);
            cur = tree.parent(cur).expect(
                "derive_subtree_composite_state: chain walk hit a parentless intermediate \
                 before reaching the tree root",
            );
        }
        chain.reverse();
        let mut current_state = root_state;
        let mut current_node_id = root_id;
        for next_id in chain {
            let next_node = tree.get(next_id);
            let current_node = tree.get(current_node_id);
            let t_current_struct_to_body = current_node.composite_properties.t_parent_this;
            let t_next_struct_to_body = next_node.composite_properties.t_parent_this;
            let offset_struct =
                next_node.composite_wrt_pstr.position - current_node.composite_properties.position;
            let offset_in_current_body = t_current_struct_to_body * offset_struct;
            let t_current_body_to_next_body = t_next_struct_to_body
                * next_node.structure_point.t_parent_this
                * t_current_struct_to_body.transpose();
            let rel = MassPointState {
                position: offset_in_current_body,
                t_parent_this: t_current_body_to_next_body,
            };
            current_state = astrodyn::propagate_forward(&current_state, &rel);
            current_node_id = next_id;
        }
        current_state
    }

    /// Mark every Simulation body whose `mass_body_id` is in
    /// `affected_ids` as having stale multi-step integrator history.
    ///
    /// Called from each topology-mutation site (attach / detach /
    /// detach_subtree / attach_subtree_aligned) **before** the
    /// matching `reset_body_integrators_by_id` call. The two-step
    /// pattern is deliberate: if a future code path adds a new
    /// topology mutation and remembers to mark dirty but forgets the
    /// reset, the dirty flag stays set and `integrate()` panics on
    /// the next step with the IG.37 diagnostic. RK4 / RKF4(5) bodies
    /// have no integrator state and are silently skipped.
    ///
    /// Mirrors JEOD's `dyn_body_attach.cc::reset_integrators()` (lines
    /// 860, 871) and `dyn_body_detach.cc:271-273`.
    ///
    /// `affected_ids` **must be sorted in ascending order and
    /// deduplicated** so the inner membership check can use
    /// `binary_search` (O(log n)) instead of `Vec::contains` (O(n)).
    /// All four call sites in this module construct it via
    /// `sort_unstable + dedup`; a `debug_assert` enforces the
    /// invariant in debug builds. Issue #274 / PR #282 review thread
    /// `PRRT_kwDORtae6c5_KoAT`.
    // JEOD_INV: IG.37 — multi-step integrator history must be reset on topology change
    pub(super) fn mark_body_integrators_dirty_by_id(
        bodies: &mut [super::types::SimBody],
        affected_ids: &[astrodyn::MassBodyId],
    ) {
        debug_assert!(
            affected_ids.windows(2).all(|w| w[0] < w[1]),
            "mark_body_integrators_dirty_by_id requires affected_ids \
             sorted ascending and deduplicated for binary_search lookup"
        );
        for body in bodies.iter_mut() {
            let Some(id) = body.mass_body_id else {
                continue;
            };
            if affected_ids.binary_search(&id).is_err() {
                continue;
            }
            if let Some(ref mut gj) = body.gj_state {
                gj.mark_topology_dirty();
            }
            if let Some(ref mut abm) = body.abm4_state {
                abm.mark_topology_dirty();
            }
        }
    }

    /// Reset multi-step integrator history on every Simulation body
    /// whose `mass_body_id` is in `affected_ids` and clear the dirty
    /// flag. Pair this with `mark_body_integrators_dirty_by_id` at the
    /// same call site — never collapse the two into one helper, since
    /// the temporal separation is what makes IG.37 fail-loud.
    ///
    /// Mirrors JEOD's `dyn_body_attach.cc::reset_integrators()` and
    /// `dyn_body_detach.cc:271-273`.
    ///
    /// `affected_ids` **must be sorted in ascending order and
    /// deduplicated** (same precondition as
    /// `mark_body_integrators_dirty_by_id`).
    // JEOD_INV: IG.37 — multi-step integrator history must be reset on topology change
    pub(super) fn reset_body_integrators_by_id(
        bodies: &mut [super::types::SimBody],
        affected_ids: &[astrodyn::MassBodyId],
    ) {
        debug_assert!(
            affected_ids.windows(2).all(|w| w[0] < w[1]),
            "reset_body_integrators_by_id requires affected_ids \
             sorted ascending and deduplicated for binary_search lookup"
        );
        for body in bodies.iter_mut() {
            let Some(id) = body.mass_body_id else {
                continue;
            };
            if affected_ids.binary_search(&id).is_err() {
                continue;
            }
            astrodyn::reset_integrators(
                body.gj_state.as_mut().map(|s| s.inner_mut()),
                body.abm4_state.as_mut().map(|s| s.inner_mut()),
            );
        }
    }

    /// Detach a tree-only subtree from its parent in the mass tree,
    /// capturing the subtree's composite-body inertial state at the
    /// moment of separation.
    ///
    /// The parent of the subtree may be either the integrated body's
    /// own mass-tree node or another *already-detached* subtree (whose
    /// state lives in [`Simulation::detached_subtrees`]). The method
    /// locates the parent automatically by walking up the tree from
    /// `subtree_root_id` to its root.
    ///
    /// `integrated_body_idx` is consulted only to identify the
    /// Simulation body whose mass-tree id matches the parent's tree
    /// root (when the parent is the integrated body).
    /// `subtree_root_id` is the [`MassBodyId`] being detached.
    ///
    /// At detach time the subtree shared the parent's rigid motion, so
    /// its composite-CoM inertial state is computed from the parent's
    /// state plus the offset between the two composites (drawn from
    /// the mass tree). The state is then stored on
    /// [`Simulation::detached_subtrees`] and propagated each step
    /// until [`attach_subtree_aligned`](Self::attach_subtree_aligned)
    /// re-attaches the subtree.
    ///
    /// If the parent is a detached subtree, its tracked composite state
    /// is updated to reflect the loss of mass (the parent's
    /// composite-CoM in inertial shifts when its mass distribution
    /// changes — even though the underlying structure point doesn't
    /// move). If the parent is the integrated body, `body.trans`
    /// (the integrated `composite_body` inertial state, post-`bd279c2`)
    /// is shifted by the inertial-frame composite-CoM delta so it
    /// continues to track the new (smaller) composite, and the body's
    /// mass is re-synced from the recomputed composite_properties.
    ///
    /// # Panics
    /// Panics if no mass tree is configured, the subtree id is not in
    /// the tree, the subtree has no parent, the parent's root has no
    /// tracked state, or a subtree with the same id is already in the
    /// detached map.
    ///
    /// # Diagnostic logging
    /// Emits a stderr trace (post-detach composite state) when the
    /// `apollo_trace` log target is enabled at the `debug` level. Enable
    /// with `RUST_LOG=apollo_trace=debug`. (Diagnostic leftover from the
    /// #248 attach-bug investigation.)
    pub fn detach_subtree(&mut self, integrated_body_idx: usize, subtree_root_id: MassBodyId) {
        let tree = self
            .mass_tree
            .as_ref()
            .expect("detach_subtree: no mass tree configured");
        // Walk up to find the root of subtree_root_id's current tree.
        tree.parent(subtree_root_id)
            .expect("detach_subtree: subtree has no parent in tree");
        let mut tree_root_id = subtree_root_id;
        while let Some(p) = tree.parent(tree_root_id) {
            tree_root_id = p;
        }

        // The parent's pre-detach composite-CoM offset in its own struct frame.
        let parent_pre_composite_props = tree.get(tree_root_id).composite_properties;

        // Determine where the parent's state lives — either an integrated
        // Simulation body or a detached subtree.
        let integrated_mass_body_id = self.bodies[integrated_body_idx].mass_body_id;
        let parent_is_integrated = integrated_mass_body_id == Some(tree_root_id);

        // Pre-detach inertial composite_body state of the parent.
        // body.trans / body.rot represent the integrated body's
        // composite_body inertial state (matching JEOD's integration
        // variable; see `attach_subtree_aligned` and the "Integration
        // target" note on `Simulation`).
        let parent_composite_state: RefFrameState = if parent_is_integrated {
            let body_trans = self.bodies[integrated_body_idx].trans;
            let body_rot = self.bodies[integrated_body_idx]
                .rot
                .expect("detach_subtree: 6-DOF integrated body required");
            RefFrameState {
                trans: RefFrameTrans {
                    position: body_trans.position.raw_si(),
                    velocity: body_trans.velocity.raw_si(),
                },
                rot: RefFrameRot {
                    q_parent_this: body_rot.q_inertial_body.to_jeod_quat(),
                    t_parent_this: body_rot
                        .q_inertial_body
                        .as_witness()
                        .left_quat_to_transformation(),
                    ang_vel_this: body_rot.ang_vel_body.raw_si(),
                },
            }
        } else {
            // Parent is itself a detached subtree.
            let detached = self
                .detached_subtrees
                .get(&tree_root_id)
                .unwrap_or_else(|| {
                    panic!(
                        "detach_subtree: parent tree-root {tree_root_id:?} of \
                         subtree {subtree_root_id:?} has no tracked state — \
                         did you forget to call detach_subtree on it first?"
                    )
                });
            detached.to_ref_frame_state()
        };

        // Walk down the tree from the root to the subtree, applying
        // propagate_forward at each level. This handles arbitrary tree
        // depth (e.g. cm → sm → s3 → lm where the subtree being
        // detached is several levels below the root). Each level uses
        // the immediate-parent-struct-frame `composite_wrt_pstr` from
        // the mass tree.
        let mut chain = Vec::<MassBodyId>::new();
        let mut current_id = subtree_root_id;
        while current_id != tree_root_id {
            chain.push(current_id);
            current_id = tree
                .parent(current_id)
                .expect("detach_subtree: chain walk hit a parentless intermediate");
        }
        chain.reverse(); // tree_root → ... → subtree
        let mut current_state = parent_composite_state;
        let mut current_node_id = tree_root_id;
        for next_id in chain {
            let next_node = tree.get(next_id);
            let current_node = tree.get(current_node_id);
            // Body-aware step:
            //   offset_in_current_body = T_current_struct_to_body
            //                          · (next.composite_wrt_pstr.position
            //                             − current.composite_properties.position)
            //   T_current_body_to_next_body = T_next_struct_to_body
            //                               · next.structure_point.t_parent_this
            //                               · T_current_body_to_struct
            // For axis-aligned bodies (struct == body) this collapses to
            // the simple struct-frame difference and the bare structure_point
            // rotation.
            let t_current_struct_to_body = current_node.composite_properties.t_parent_this;
            let t_next_struct_to_body = next_node.composite_properties.t_parent_this;
            let offset_struct =
                next_node.composite_wrt_pstr.position - current_node.composite_properties.position;
            let offset_in_current_body = t_current_struct_to_body * offset_struct;
            let t_current_body_to_next_body = t_next_struct_to_body
                * next_node.structure_point.t_parent_this
                * t_current_struct_to_body.transpose();
            let rel = MassPointState {
                position: offset_in_current_body,
                t_parent_this: t_current_body_to_next_body,
            };
            current_state = astrodyn::propagate_forward(&current_state, &rel);
            current_node_id = next_id;
        }
        let subtree_state = current_state;

        // Apply the topology change — this also recomputes parent's
        // composite_properties (now without the subtree).
        let tree = self.mass_tree.as_mut().unwrap();
        tree.detach(subtree_root_id);
        let parent_post_composite_props = tree.get(tree_root_id).composite_properties;

        if parent_is_integrated {
            // body.trans/body.rot are the integrated composite_body
            // state. JEOD's detach handler preserves core_body
            // (Pos_Vel_Att_Rate source = core_body) and rederives the
            // post-detach composite from it. The composite-CoM offset
            // in the parent's struct frame shifts when the subtree
            // leaves, so the inertial composite_body position +
            // velocity must shift by the corresponding kinematic
            // offset. Rotation/ang_vel are unchanged because
            // composite_properties.t_parent_this == core_properties
            // .t_parent_this throughout (see mass tree recompute).
            let cm_delta_struct =
                parent_post_composite_props.position - parent_pre_composite_props.position;
            // composite_properties.t_parent_this is the struct→body
            // rotation; compose with body.rot's t_parent_this to map
            // struct → inertial.
            let t_struct_to_body = parent_pre_composite_props.t_parent_this;
            let cm_delta_body = t_struct_to_body * cm_delta_struct;
            let t_inertial_to_body = parent_composite_state.rot.t_parent_this;
            let cm_delta_inertial = t_inertial_to_body.transpose() * cm_delta_body;
            // Velocity offset from rigid-body rotation: ω × Δr in body
            // frame, then rotated to inertial.
            let omega_body = parent_composite_state.rot.ang_vel_this;
            let dvel_inertial = t_inertial_to_body.transpose() * omega_body.cross(cm_delta_body);
            self.bodies[integrated_body_idx].trans =
                trans_raw_to_typed::<IntegrationFrame>(&TranslationalState {
                    position: parent_composite_state.trans.position + cm_delta_inertial,
                    velocity: parent_composite_state.trans.velocity + dvel_inertial,
                });
            // body.rot unchanged — composite/core share body axes.
            // allowed: typed↔raw kernel-boundary lift on post-detach mass writeback (see #397).
            self.bodies[integrated_body_idx].mass =
                Some(mass_raw_to_self_ref(&parent_post_composite_props));
        } else {
            // Parent is a detached subtree — update its tracked
            // composite-body state to reflect the new (smaller) composite.
            // The parent's struct origin in inertial is unchanged (rigid
            // body); only the composite-CoM has moved within the struct
            // frame. Convert the struct-frame CoM-delta to inertial via
            //
            //   T_inertial_to_struct = T_struct_to_body^T · T_inertial_to_body
            //
            // (matching `astrodyn::compute_t_inertial_struct`). The
            // earlier form `T_struct_to_body * T_inertial_to_body` was
            // only correct when `T_struct_to_body` is symmetric (identity
            // or yaw_180) and silently produced wrong CoM-shift directions
            // for non-symmetric mass-tree orientations.
            let cm_delta_struct =
                parent_post_composite_props.position - parent_pre_composite_props.position;
            let t_struct_to_body = parent_pre_composite_props.t_parent_this;
            let t_inertial_struct = astrodyn::compute_t_inertial_struct(
                &t_struct_to_body,
                &parent_composite_state.rot.t_parent_this,
            );
            let r_struct_to_inertial = t_inertial_struct.transpose();
            let cm_delta_inertial = r_struct_to_inertial * cm_delta_struct;
            // Velocity contribution from rotation: ω × Δr expressed in
            // body frame, then rotated to inertial.
            let w_body = parent_composite_state.rot.ang_vel_this;
            let cm_delta_body = t_struct_to_body * cm_delta_struct;
            let dvel_inertial =
                parent_composite_state.rot.t_parent_this.transpose() * w_body.cross(cm_delta_body);
            // Build raw f64 sums in the runtime-typed arena, then attach
            // the `RootInertial` phantom at the boundary into the typed
            // `DetachedSubtreeState` storage. Both sides of the addition
            // live in the simulation's root inertial frame (the parent's
            // composite-CoM offset shift propagates the parent's
            // pre-detach inertial pose forward by Δr in the same frame).
            use astrodyn::Vec3Ext;
            let updated_pos = parent_composite_state.trans.position + cm_delta_inertial;
            let updated_vel = parent_composite_state.trans.velocity + dvel_inertial;
            let updated = DetachedSubtreeState {
                composite_position: updated_pos.m_at::<astrodyn::RootInertial>(),
                composite_velocity: updated_vel.m_per_s_at::<astrodyn::RootInertial>(),
                composite_attitude: DetachedSubtreeState::attitude_from_raw_jeod_quat(
                    parent_composite_state.rot.q_parent_this,
                ),
                composite_ang_vel_body: parent_composite_state.rot.ang_vel_this,
            };
            self.detached_subtrees.insert(tree_root_id, updated);
        }

        if log::log_enabled!(target: "apollo_trace", log::Level::Debug) {
            eprintln!(
                "DETACH: subtree {subtree_root_id:?} state stored:\n  pos={:?}\n  vel={:?}\n  ω={:?}",
                subtree_state.trans.position,
                subtree_state.trans.velocity,
                subtree_state.rot.ang_vel_this
            );
        }

        // Insert the new subtree's state into the detached map.
        let prior = self.detached_subtrees.insert(
            subtree_root_id,
            DetachedSubtreeState::from_ref_frame_state(&subtree_state),
        );
        assert!(
            prior.is_none(),
            "detach_subtree: subtree {subtree_root_id:?} was already in detached_subtrees \
             — call attach_subtree_aligned first or use a fresh subtree id"
        );

        // JEOD_INV: IG.37 — Multi-step integrators (GJ, ABM4) carry predictor
        // history that is invalidated by the topology change. Reset the
        // integrated body's integrators (it just lost the subtree's mass,
        // so its dynamics changed and `body.trans` was shifted).
        //
        // Mark + reset are split into two distinct call sites (rather
        // than one bundled helper) so a future code path that adds a
        // new subtree-mutation method and remembers Site A but forgets
        // Site B leaves the dirty bit set, panicking on next integrate.
        if parent_is_integrated {
            let body = &mut self.bodies[integrated_body_idx];
            // Site A: mark dirty.
            if let Some(ref mut gj) = body.gj_state {
                gj.mark_topology_dirty();
            }
            if let Some(ref mut abm) = body.abm4_state {
                abm.mark_topology_dirty();
            }
            // Site B: reset history (separate observation site).
            astrodyn::reset_integrators(
                body.gj_state.as_mut().map(|s| s.inner_mut()),
                body.abm4_state.as_mut().map(|s| s.inner_mut()),
            );
        }

        // ── Clear `kinematic_only` on the freshly-detached subtree root,
        //    parallel to `Self::detach`. After `tree.detach`, the
        //    subtree's tree-root has no parent in the mass tree;
        //    `propagate_kinematic_state` would panic with the
        //    "kinematic-only resolves to a tree root" diagnostic on the
        //    next `step()` if a SimBody backing the subtree root were
        //    left flagged `kinematic_only`. Interior subtree bodies
        //    (descendants of the new tree root) keep their flag because
        //    the new root continues to drive their state via
        //    `propagate_state_via_storage`. Only one SimBody can match
        //    a given `MassBodyId` (enforced by the duplicate-id guard
        //    in `propagate_kinematic_state`), so a single linear scan
        //    is sufficient. PR #295 review thread `PRRT_kwDORtae6c5_O1JW`.
        for body in self.bodies.iter_mut() {
            if body.mass_body_id == Some(subtree_root_id) {
                body.kinematic_only = false;
                break;
            }
        }
    }

    /// Re-attach a previously-detached subtree to the integrated body's
    /// mass tree using named mass points (matching JEOD's
    /// `attach_aligned`), then update the integrated body's state via
    /// JEOD's [`combine_states_at_attach`] momentum-conservation
    /// algorithm.
    ///
    /// The integrated body's `body.trans` / `body.rot` represent the
    /// *composite_body* inertial state of the whole mass tree rooted
    /// at the integrated body — i.e. the integration variable that
    /// JEOD's `DynamicsIntegrationGroup::gravitation()` evaluates
    /// gravity at and that `DynBody::trans_integ()` integrates. The
    /// subtree state from [`Simulation::detached_subtrees`] is the
    /// subtree's composite_body frame. After the algorithm runs, the
    /// integrated body's `trans` / `rot` are set to the new combined
    /// composite_body inertial state.
    ///
    /// To compare against JEOD's logged core_body, derive core via
    /// [`Simulation::body_core_inertial`].
    ///
    /// # Panics
    /// Panics if the integrated body has no rotational state, no mass
    /// tree is configured, the parent or subtree id is not in the tree,
    /// either named mass point is missing on its body, or the subtree
    /// is not in [`Self::detached_subtrees`].
    ///
    /// # Diagnostic logging
    /// Emits a stderr trace (parent + child composite states, mass
    /// properties, combined output) when the `apollo_trace` log target is
    /// enabled at the `debug` level. Enable with `RUST_LOG=apollo_trace=debug`.
    /// (Diagnostic leftover from the #248 attach-bug investigation.)
    pub fn attach_subtree_aligned(
        &mut self,
        integrated_body_idx: usize,
        subtree_root_id: MassBodyId,
        subtree_point: &str,
        parent_id: MassBodyId,
        parent_point: &str,
    ) {
        let tree = self
            .mass_tree
            .as_ref()
            .expect("attach_subtree_aligned: no mass tree configured");
        let integrated_mass_body_id = self.bodies[integrated_body_idx]
            .mass_body_id
            .expect("attach_subtree_aligned: integrated body not registered in mass tree");

        // Read pre-attach composite mass props of the integrated body
        // (= the whole pre-attach tree without the subtree).
        let parent_pre_composite_props = tree.get(integrated_mass_body_id).composite_properties;
        let orig_parent_cm_struct = parent_pre_composite_props.position;
        let core_wrt_composite_pre = tree.get(integrated_mass_body_id).core_wrt_composite;
        // Pre-attach subtree composite mass props.
        let subtree_composite_props = tree.get(subtree_root_id).composite_properties;

        // Read the integrated body's pre-attach composite_body state
        // directly from body.trans/body.rot (post-refactor convention).
        let body_trans = self.bodies[integrated_body_idx].trans;
        let body_rot = self.bodies[integrated_body_idx]
            .rot
            .expect("attach_subtree_aligned: 6-DOF body required");
        let parent_composite_pre = RefFrameState {
            trans: RefFrameTrans {
                position: body_trans.position.raw_si(),
                velocity: body_trans.velocity.raw_si(),
            },
            rot: RefFrameRot {
                q_parent_this: body_rot.q_inertial_body.to_jeod_quat(),
                t_parent_this: body_rot
                    .q_inertial_body
                    .as_witness()
                    .left_quat_to_transformation(),
                ang_vel_this: body_rot.ang_vel_body.raw_si(),
            },
        };
        let _ = core_wrt_composite_pre; // unused under composite-body convention

        // Borrow the subtree's free-flight composite state (don't remove
        // yet — we want the entry to survive if any of the operations
        // below panic, so the caller can retry / recover instead of
        // silently losing state).
        let subtree_state = *self
            .detached_subtrees
            .get(&subtree_root_id)
            .unwrap_or_else(|| {
                panic!(
                    "attach_subtree_aligned: subtree {subtree_root_id:?} is not in \
                     detached_subtrees — call detach_subtree first or pre-populate"
                )
            });
        let child_composite = subtree_state.to_ref_frame_state();

        // Apply the topology change (also recomputes composite props).
        let tree_mut = self.mass_tree.as_mut().unwrap();
        tree_mut.attach_aligned(subtree_root_id, subtree_point, parent_id, parent_point);
        // Read post-attach composite props.
        let combined_composite_props = tree_mut.get(integrated_mass_body_id).composite_properties;

        // The combine algorithm needs `T_inertial_to_struct`, which by
        // the standard frame-chain rule is
        //
        //   T_inertial_to_struct = T_struct_to_body^T · T_inertial_to_body
        //
        // (matching `astrodyn::compute_t_inertial_struct` and JEOD's
        // `dyn_body_collect.cc` lines 219-221). composite_properties
        // .t_parent_this is the struct→body rotation. The earlier form
        // `T_struct_to_body * T_inertial_to_body` was only valid for
        // symmetric struct-to-body rotations (identity, yaw_180 — the
        // ones Apollo happens to use); non-symmetric vehicle orientations
        // would silently get a wrong torque arm in the combine algorithm.
        let t_struct_to_body = parent_pre_composite_props.t_parent_this;
        let parent_t_inertial_struct = astrodyn::compute_t_inertial_struct(
            &t_struct_to_body,
            &parent_composite_pre.rot.t_parent_this,
        );

        // APOLLO_TRACE diagnostic: dump every input to combine_states_at_attach
        // so we can diff against JEOD ground truth. Gated via the `log` crate
        // target so the regular test path is unaffected — enable with
        // `RUST_LOG=apollo_trace=debug`. (See #248 attach-bug investigation.)
        if log::log_enabled!(target: "apollo_trace", log::Level::Debug) {
            eprintln!("=== ATTACH TRACE (integrated body {integrated_body_idx} → subtree {subtree_root_id:?}) ===");
            eprintln!("  PARENT COMPOSITE (= our body.trans/body.rot):");
            eprintln!(
                "    pos    = [{:.10e} {:.10e} {:.10e}]",
                parent_composite_pre.trans.position.x,
                parent_composite_pre.trans.position.y,
                parent_composite_pre.trans.position.z
            );
            eprintln!(
                "    vel    = [{:.10e} {:.10e} {:.10e}]",
                parent_composite_pre.trans.velocity.x,
                parent_composite_pre.trans.velocity.y,
                parent_composite_pre.trans.velocity.z
            );
            eprintln!(
                "    q      = [{:.10e} {:.10e} {:.10e} {:.10e}]",
                parent_composite_pre.rot.q_parent_this.scalar(),
                parent_composite_pre.rot.q_parent_this.vector().x,
                parent_composite_pre.rot.q_parent_this.vector().y,
                parent_composite_pre.rot.q_parent_this.vector().z
            );
            eprintln!(
                "    ω_body = [{:.10e} {:.10e} {:.10e}]",
                parent_composite_pre.rot.ang_vel_this.x,
                parent_composite_pre.rot.ang_vel_this.y,
                parent_composite_pre.rot.ang_vel_this.z
            );
            eprintln!("  CHILD COMPOSITE (= detached_subtrees[{subtree_root_id:?}]):");
            eprintln!(
                "    pos    = [{:.10e} {:.10e} {:.10e}]",
                child_composite.trans.position.x,
                child_composite.trans.position.y,
                child_composite.trans.position.z
            );
            eprintln!(
                "    vel    = [{:.10e} {:.10e} {:.10e}]",
                child_composite.trans.velocity.x,
                child_composite.trans.velocity.y,
                child_composite.trans.velocity.z
            );
            eprintln!(
                "    q      = [{:.10e} {:.10e} {:.10e} {:.10e}]",
                child_composite.rot.q_parent_this.scalar(),
                child_composite.rot.q_parent_this.vector().x,
                child_composite.rot.q_parent_this.vector().y,
                child_composite.rot.q_parent_this.vector().z
            );
            eprintln!(
                "    ω_body = [{:.10e} {:.10e} {:.10e}]",
                child_composite.rot.ang_vel_this.x,
                child_composite.rot.ang_vel_this.y,
                child_composite.rot.ang_vel_this.z
            );
            eprintln!("  PARENT MASS (pre-attach):");
            eprintln!("    mass={:.10e}", parent_pre_composite_props.mass);
            eprintln!(
                "    pos_struct=[{:.10e} {:.10e} {:.10e}]",
                parent_pre_composite_props.position.x,
                parent_pre_composite_props.position.y,
                parent_pre_composite_props.position.z
            );
            eprintln!(
                "    inertia.diag=[{:.10e} {:.10e} {:.10e}]",
                parent_pre_composite_props.inertia.x_axis.x,
                parent_pre_composite_props.inertia.y_axis.y,
                parent_pre_composite_props.inertia.z_axis.z
            );
            eprintln!("  CHILD MASS (pre-attach):");
            eprintln!("    mass={:.10e}", subtree_composite_props.mass);
            eprintln!(
                "    pos_struct=[{:.10e} {:.10e} {:.10e}]",
                subtree_composite_props.position.x,
                subtree_composite_props.position.y,
                subtree_composite_props.position.z
            );
            eprintln!(
                "    inertia.diag=[{:.10e} {:.10e} {:.10e}]",
                subtree_composite_props.inertia.x_axis.x,
                subtree_composite_props.inertia.y_axis.y,
                subtree_composite_props.inertia.z_axis.z
            );
            eprintln!("  COMBINED MASS (post-attach):");
            eprintln!("    mass={:.10e}", combined_composite_props.mass);
            eprintln!(
                "    pos_struct=[{:.10e} {:.10e} {:.10e}]",
                combined_composite_props.position.x,
                combined_composite_props.position.y,
                combined_composite_props.position.z
            );
            eprintln!(
                "    inertia.diag=[{:.10e} {:.10e} {:.10e}]",
                combined_composite_props.inertia.x_axis.x,
                combined_composite_props.inertia.y_axis.y,
                combined_composite_props.inertia.z_axis.z
            );
            eprintln!(
                "  orig_parent_cm_struct=[{:.10e} {:.10e} {:.10e}]",
                orig_parent_cm_struct.x, orig_parent_cm_struct.y, orig_parent_cm_struct.z
            );
            eprintln!("=== end ATTACH TRACE ===");
        }

        // Run the JEOD combine algorithm.
        let combined = combine_states_at_attach(AttachCombineInputs {
            parent_composite: parent_composite_pre,
            parent_mass: parent_pre_composite_props,
            parent_t_inertial_struct,
            child_composite,
            child_mass: subtree_composite_props,
            combined_mass: combined_composite_props,
            orig_parent_cm_struct,
        });

        // The new whole-tree composite state is the integration target —
        // matches JEOD's `composite_body` post-attach (the source for
        // `Vel_Rate` per `set_state_source_internal` at the end of
        // `DynBody::attach_update_properties`).
        self.bodies[integrated_body_idx].trans =
            trans_raw_to_typed::<IntegrationFrame>(&TranslationalState {
                position: combined.composite_state.trans.position,
                velocity: combined.composite_state.trans.velocity,
            });
        // allowed: typed↔raw kernel-boundary lift on post-attach combine writeback (see #397).
        let raw_rot = RotationalState {
            quaternion: combined.composite_state.rot.q_parent_this,
            ang_vel_body: combined.composite_state.rot.ang_vel_this,
        };
        self.bodies[integrated_body_idx].rot = Some(rot_raw_to_self_ref(&raw_rot));
        self.bodies[integrated_body_idx].mass =
            Some(mass_raw_to_self_ref(&combined_composite_props));

        // Combine succeeded — only now remove the subtree's detached
        // entry. If any earlier step panicked (missing mass points,
        // combine preconditions, etc.) the entry survives so callers
        // can retry or inspect rather than silently losing state.
        self.detached_subtrees.remove(&subtree_root_id);

        // JEOD_INV: IG.37 — Multi-step integrators (GJ, ABM4) carry predictor
        // history that is invalidated by the topology + state combine.
        // Mirror JEOD's `dyn_body_attach.cc::reset_integrators()` for the
        // integrated body, whose `body.trans` / `body.rot` were just
        // overwritten by the combine.
        //
        // Split mark + reset across two adjacent-but-distinct call
        // sites: a future code path that adds another integrated-body
        // mutation and remembers Site A but forgets Site B leaves the
        // dirty flag set, so the next `integrate()` panics with the
        // IG.37 diagnostic instead of silently propagating stale
        // predictor history.
        let body = &mut self.bodies[integrated_body_idx];
        // Site A: mark integrators dirty.
        if let Some(ref mut gj) = body.gj_state {
            gj.mark_topology_dirty();
        }
        if let Some(ref mut abm) = body.abm4_state {
            abm.mark_topology_dirty();
        }
        // Site B: reset integrator history (separate observation site).
        astrodyn::reset_integrators(
            body.gj_state.as_mut().map(|s| s.inner_mut()),
            body.abm4_state.as_mut().map(|s| s.inner_mut()),
        );

        if log::log_enabled!(target: "apollo_trace", log::Level::Debug) {
            eprintln!(
                "  COMBINE OUTPUT: pos=[{:.4e} {:.4e} {:.4e}] ω_body=[{:.6e} {:.6e} {:.6e}]",
                combined.composite_state.trans.position.x,
                combined.composite_state.trans.position.y,
                combined.composite_state.trans.position.z,
                combined.composite_state.rot.ang_vel_this.x,
                combined.composite_state.rot.ang_vel_this.y,
                combined.composite_state.rot.ang_vel_this.z
            );
        }
    }

    /// Advance every entry in [`Simulation::detached_subtrees`] by `dt`
    /// seconds. Each subtree propagates ballistically — no gravity, no
    /// torque — matching JEOD's behavior for tree roots whose
    /// `grav_interaction.controls` is empty (which is the case for
    /// every non-LES vehicle in `SIM_Apollo`).
    pub fn step_detached_subtrees(&mut self, dt: f64) {
        for state in self.detached_subtrees.values_mut() {
            state.step_ballistic(dt);
        }
    }
}

#[cfg(test)]
mod tests {
    //! Integration-level tests for the IG.37 wiring. These tests live in
    //! the runner crate so they can poke at `SimBody::gj_state` /
    //! `abm4_state` directly — the public `body()` accessor only exposes
    //! `VehicleOutput`, which omits integrator state.

    use super::*;
    use crate::Simulation;
    use astrodyn::{
        Abm4State, GaussJacksonConfig, GaussJacksonState, GravityControl, GravityControls,
        GravityGradient, GravityModel, GravitySource, GravitySourceEntry, IntegratorType,
        MassProperties, RootInertial, SimulationTime, TranslationalState, VehicleConfig,
    };

    /// JEOD's `dyn_body_attach.cc::reset_integrators()` precedent: after an
    /// attach, both bodies' Gauss-Jackson predictor / corrector history
    /// must be reinitialized. We verify that:
    ///   1. After enough steps to leave priming, the GJ states are past
    ///      priming (`is_priming() == false`).
    ///   2. After `Simulation::attach`, both bodies' GJ states are back
    ///      in priming and the topology-dirty flag is cleared (which is
    ///      what `reset_for_topology_change` guarantees).
    ///   3. The simulation can take another step without tripping the
    ///      IG.37 assertion in `GaussJacksonState::integrate` — proving
    ///      the wiring closes the gap end-to-end.
    #[test]
    fn attach_resets_gauss_jackson_state_on_both_bodies() {
        const MU: f64 = 5.76e14;
        let dt = 1.0_f64;

        let trans_a = TranslationalState {
            position: DVec3::new(9e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 8000.0, 0.0),
        };
        // Slightly different orbit for body B so its predictor history
        // is non-trivially distinct from body A.
        let trans_b = TranslationalState {
            position: DVec3::new(9e6, 1.0, 0.0),
            velocity: DVec3::new(0.0, 7900.0, 0.0),
        };

        let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
        let mut sim = Simulation::new(time, dt);

        let earth = sim.add_source(
            "Earth",
            GravitySourceEntry {
                source: GravitySource {
                    mu: MU,
                    model: GravityModel::PointMass,
                },
                position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
                velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
                t_inertial_pfix: None,
                delta_c20: 0.0,
                rotation_model: crate::RotationModel::default(),
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
                marker_only: false,
            },
        );

        // JEOD-faithful warn-and-continue: these unit tests step GJ on synthetic
        // initial state whose non-physical configuration occasionally triggers
        // corrector non-convergence. The test asserts attach/detach mechanics,
        // not numerical correctness, so opting in to the JEOD-faithful path
        // (#485 C1) keeps the test focus on its actual subject.
        let gj_cfg = GaussJacksonConfig::with_order(8).with_allow_non_convergence(true);
        let make_cfg = |trans: TranslationalState| VehicleConfig {
            trans: trans_raw_to_typed::<RootInertial>(&trans),
            integrator: IntegratorType::GaussJackson(gj_cfg),
            mass: Some(mass_raw_to_self_ref(&(MassProperties::new(1000.0)))),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
            },
            ..Default::default()
        };
        let body_a = sim.add_body(make_cfg(trans_a));
        let body_b = sim.add_body(make_cfg(trans_b));

        // Register both in the mass tree so we can attach later.
        let id_a = sim.add_body_to_tree(body_a, "BodyA");
        let id_b = sim.add_body_to_tree(body_b, "BodyB");

        sim.validate().expect("validate failed");

        // ── Step long enough to leave GJ priming on both bodies. ──
        // GJ8 needs ~50 stages to fully bootstrap; 200 sim steps is
        // comfortably past that.
        sim.step_n(200).expect("step_n failed");

        let gj_a_pre = sim.bodies[body_a]
            .gj_state
            .as_ref()
            .expect("body A must have gj_state after validate");
        let gj_b_pre = sim.bodies[body_b]
            .gj_state
            .as_ref()
            .expect("body B must have gj_state after validate");
        assert!(
            !gj_a_pre.is_priming(),
            "test setup expected body A's GJ state past priming after 200 steps"
        );
        assert!(
            !gj_b_pre.is_priming(),
            "test setup expected body B's GJ state past priming after 200 steps"
        );
        assert!(!gj_a_pre.is_topology_dirty());
        assert!(!gj_b_pre.is_topology_dirty());
        let _ = (id_a, id_b);

        // ── Attach: triggers IG.37 reset on both bodies. ──
        sim.attach(body_a, body_b, DVec3::ZERO, DMat3::IDENTITY);

        let gj_a_post = sim.bodies[body_a]
            .gj_state
            .as_ref()
            .expect("body A must still have gj_state");
        let gj_b_post = sim.bodies[body_b]
            .gj_state
            .as_ref()
            .expect("body B must still have gj_state");
        assert!(
            gj_a_post.is_priming(),
            "body A's GJ state must be back in priming after attach (IG.37)"
        );
        assert!(
            gj_b_post.is_priming(),
            "body B's GJ state must be back in priming after attach (IG.37)"
        );
        assert!(!gj_a_post.is_topology_dirty());
        assert!(!gj_b_post.is_topology_dirty());

        // ── Step once more: the IG.37 assertion in `integrate()` must
        // not fire. With our wiring it's cleared; without it, this would
        // panic with the "stale predictor/corrector history" message.
        sim.step_n(1).expect("post-attach step failed");
    }

    /// `Simulation::detach` must reset GJ state on both the parent and
    /// the detaching child. Mirrors JEOD's `dyn_body_detach.cc:271-273`.
    #[test]
    fn detach_resets_gauss_jackson_state_on_both_bodies() {
        const MU: f64 = 5.76e14;
        let dt = 1.0_f64;
        let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
        let mut sim = Simulation::new(time, dt);

        let earth = sim.add_source(
            "Earth",
            GravitySourceEntry {
                source: GravitySource {
                    mu: MU,
                    model: GravityModel::PointMass,
                },
                position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
                velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
                t_inertial_pfix: None,
                delta_c20: 0.0,
                rotation_model: crate::RotationModel::default(),
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
                marker_only: false,
            },
        );

        // JEOD-faithful warn-and-continue: these unit tests step GJ on synthetic
        // initial state whose non-physical configuration occasionally triggers
        // corrector non-convergence. The test asserts attach/detach mechanics,
        // not numerical correctness, so opting in to the JEOD-faithful path
        // (#485 C1) keeps the test focus on its actual subject.
        let gj_cfg = GaussJacksonConfig::with_order(8).with_allow_non_convergence(true);
        let trans = TranslationalState {
            position: DVec3::new(9e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 8000.0, 0.0),
        };
        let make = |trans: TranslationalState| VehicleConfig {
            trans: trans_raw_to_typed::<RootInertial>(&trans),
            integrator: IntegratorType::GaussJackson(gj_cfg),
            mass: Some(mass_raw_to_self_ref(&(MassProperties::new(1000.0)))),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
            },
            ..Default::default()
        };
        let body_a = sim.add_body(make(trans));
        let body_b = sim.add_body(make(TranslationalState {
            position: trans.position + DVec3::new(0.0, 1.0, 0.0),
            velocity: trans.velocity,
        }));
        sim.add_body_to_tree(body_a, "BodyA");
        sim.add_body_to_tree(body_b, "BodyB");

        // Pre-attach and step into operational mode, then detach.
        sim.attach(body_b, body_a, DVec3::ZERO, DMat3::IDENTITY);
        sim.validate().expect("validate failed");
        sim.step_n(200).expect("step_n failed");
        // After 200 post-attach steps both should be operational again.
        assert!(!sim.bodies[body_a].gj_state.as_ref().unwrap().is_priming());
        assert!(!sim.bodies[body_b].gj_state.as_ref().unwrap().is_priming());

        // Detach → both GJ states must reset.
        sim.detach(body_b);
        let gj_a = sim.bodies[body_a].gj_state.as_ref().unwrap();
        let gj_b = sim.bodies[body_b].gj_state.as_ref().unwrap();
        assert!(gj_a.is_priming(), "parent's GJ must reset on detach");
        assert!(gj_b.is_priming(), "child's GJ must reset on detach");
        assert!(!gj_a.is_topology_dirty());
        assert!(!gj_b.is_topology_dirty());
    }

    /// ABM4 sibling of `attach_resets_gauss_jackson_state_on_both_bodies`.
    /// `Simulation::attach` must also reset ABM4 history on both bodies —
    /// the underlying `mark_body_integrators_dirty_by_id` /
    /// `reset_body_integrators_by_id` helpers cover both integrator
    /// kinds, but without an ABM4-specific test a regression that only
    /// breaks the ABM4 arm would slip through (PR #282 review thread
    /// `PRRT_kwDORtae6c5_J-p_`).
    #[test]
    fn attach_resets_abm4_state_on_both_bodies() {
        const MU: f64 = 5.76e14;
        let dt = 1.0_f64;

        let trans_a = TranslationalState {
            position: DVec3::new(9e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 8000.0, 0.0),
        };
        let trans_b = TranslationalState {
            position: DVec3::new(9e6, 1.0, 0.0),
            velocity: DVec3::new(0.0, 7900.0, 0.0),
        };

        let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
        let mut sim = Simulation::new(time, dt);

        let earth = sim.add_source(
            "Earth",
            GravitySourceEntry {
                source: GravitySource {
                    mu: MU,
                    model: GravityModel::PointMass,
                },
                position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
                velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
                t_inertial_pfix: None,
                delta_c20: 0.0,
                rotation_model: crate::RotationModel::default(),
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
                marker_only: false,
            },
        );

        let make_cfg = |trans: TranslationalState| VehicleConfig {
            trans: trans_raw_to_typed::<RootInertial>(&trans),
            integrator: IntegratorType::Abm4,
            mass: Some(mass_raw_to_self_ref(&(MassProperties::new(1000.0)))),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
            },
            ..Default::default()
        };
        let body_a = sim.add_body(make_cfg(trans_a));
        let body_b = sim.add_body(make_cfg(trans_b));
        sim.add_body_to_tree(body_a, "BodyA");
        sim.add_body_to_tree(body_b, "BodyB");

        sim.validate().expect("validate failed");

        // ── Step long enough to leave ABM4 priming on both bodies. ──
        // ABM4 primes after `HIST_LEN - 1 = 3` steps; 5 is comfortably past.
        sim.step_n(5).expect("step_n failed");

        let abm_a_pre = sim.bodies[body_a]
            .abm4_state
            .as_ref()
            .expect("body A must have abm4_state after validate");
        let abm_b_pre = sim.bodies[body_b]
            .abm4_state
            .as_ref()
            .expect("body B must have abm4_state after validate");
        assert!(
            !abm_a_pre.is_priming(),
            "test setup expected body A's ABM4 state past priming after 5 steps"
        );
        assert!(
            !abm_b_pre.is_priming(),
            "test setup expected body B's ABM4 state past priming after 5 steps"
        );
        assert!(!abm_a_pre.is_topology_dirty());
        assert!(!abm_b_pre.is_topology_dirty());

        // ── Attach: triggers IG.37 reset on both bodies. ──
        sim.attach(body_a, body_b, DVec3::ZERO, DMat3::IDENTITY);

        let abm_a_post = sim.bodies[body_a].abm4_state.as_ref().unwrap();
        let abm_b_post = sim.bodies[body_b].abm4_state.as_ref().unwrap();
        assert!(
            abm_a_post.is_priming(),
            "body A's ABM4 state must be back in priming after attach (IG.37)"
        );
        assert!(
            abm_b_post.is_priming(),
            "body B's ABM4 state must be back in priming after attach (IG.37)"
        );
        assert!(!abm_a_post.is_topology_dirty());
        assert!(!abm_b_post.is_topology_dirty());

        // ── Step once more: the IG.37 assertion in `abm4_translational_step`
        // must not fire. ──
        sim.step_n(1).expect("post-attach step failed");
    }

    /// ABM4 sibling of `detach_resets_gauss_jackson_state_on_both_bodies`.
    #[test]
    fn detach_resets_abm4_state_on_both_bodies() {
        const MU: f64 = 5.76e14;
        let dt = 1.0_f64;
        let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
        let mut sim = Simulation::new(time, dt);

        let earth = sim.add_source(
            "Earth",
            GravitySourceEntry {
                source: GravitySource {
                    mu: MU,
                    model: GravityModel::PointMass,
                },
                position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
                velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
                t_inertial_pfix: None,
                delta_c20: 0.0,
                rotation_model: crate::RotationModel::default(),
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
                marker_only: false,
            },
        );

        let trans = TranslationalState {
            position: DVec3::new(9e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 8000.0, 0.0),
        };
        let make = |trans: TranslationalState| VehicleConfig {
            trans: trans_raw_to_typed::<RootInertial>(&trans),
            integrator: IntegratorType::Abm4,
            mass: Some(mass_raw_to_self_ref(&(MassProperties::new(1000.0)))),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
            },
            ..Default::default()
        };
        let body_a = sim.add_body(make(trans));
        let body_b = sim.add_body(make(TranslationalState {
            position: trans.position + DVec3::new(0.0, 1.0, 0.0),
            velocity: trans.velocity,
        }));
        sim.add_body_to_tree(body_a, "BodyA");
        sim.add_body_to_tree(body_b, "BodyB");

        sim.attach(body_b, body_a, DVec3::ZERO, DMat3::IDENTITY);
        sim.validate().expect("validate failed");
        sim.step_n(5).expect("step_n failed");
        assert!(!sim.bodies[body_a].abm4_state.as_ref().unwrap().is_priming());
        assert!(!sim.bodies[body_b].abm4_state.as_ref().unwrap().is_priming());

        sim.detach(body_b);
        let abm_a = sim.bodies[body_a].abm4_state.as_ref().unwrap();
        let abm_b = sim.bodies[body_b].abm4_state.as_ref().unwrap();
        assert!(abm_a.is_priming(), "parent's ABM4 must reset on detach");
        assert!(abm_b.is_priming(), "child's ABM4 must reset on detach");
        assert!(!abm_a.is_topology_dirty());
        assert!(!abm_b.is_topology_dirty());
    }

    /// `Simulation::attach`/`detach` must reset integrators on the
    /// **full ancestor chain**, not just the directly-named bodies.
    /// `MassTree::attach`/`detach` recompute composite mass properties
    /// all the way to the root; an integrator on any ancestor is
    /// invalidated by that recompute. Builds a 3-body chain
    /// `top → middle → leaf`, attaches a fourth body underneath
    /// `middle`, and verifies that `top`'s GJ state is reset (in
    /// addition to `middle` and the new attachee). Mirrors PR #282
    /// review threads `PRRT_kwDORtae6c5_J-qF` (attach) and
    /// `PRRT_kwDORtae6c5_J-qI` (detach).
    #[test]
    fn attach_and_detach_reset_full_ancestor_chain() {
        const MU: f64 = 5.76e14;
        let dt = 1.0_f64;
        let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
        let mut sim = Simulation::new(time, dt);

        let earth = sim.add_source(
            "Earth",
            GravitySourceEntry {
                source: GravitySource {
                    mu: MU,
                    model: GravityModel::PointMass,
                },
                position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
                velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
                t_inertial_pfix: None,
                delta_c20: 0.0,
                rotation_model: crate::RotationModel::default(),
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
                marker_only: false,
            },
        );

        // Four bodies on similar orbits — only `top` is integrated; the
        // others sit on the same orbit and are attached in a chain to
        // build an ancestor relationship inside `MassTree`. Only `top`
        // needs a working integrator, but every body whose
        // `composite_properties` is recomputed by `MassTree::attach`
        // / `detach` should still see its (otherwise unused) integrator
        // reset.
        // JEOD-faithful warn-and-continue: these unit tests step GJ on synthetic
        // initial state whose non-physical configuration occasionally triggers
        // corrector non-convergence. The test asserts attach/detach mechanics,
        // not numerical correctness, so opting in to the JEOD-faithful path
        // (#485 C1) keeps the test focus on its actual subject.
        let gj_cfg = GaussJacksonConfig::with_order(8).with_allow_non_convergence(true);
        let trans = TranslationalState {
            position: DVec3::new(9e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 8000.0, 0.0),
        };
        let make_cfg = || VehicleConfig {
            trans: trans_raw_to_typed::<RootInertial>(&trans),
            integrator: IntegratorType::GaussJackson(gj_cfg),
            mass: Some(mass_raw_to_self_ref(&(MassProperties::new(1000.0)))),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
            },
            ..Default::default()
        };
        let top = sim.add_body(make_cfg());
        let middle = sim.add_body(make_cfg());
        let leaf = sim.add_body(make_cfg());
        let new_attachee = sim.add_body(make_cfg());
        sim.add_body_to_tree(top, "Top");
        sim.add_body_to_tree(middle, "Middle");
        sim.add_body_to_tree(leaf, "Leaf");
        sim.add_body_to_tree(new_attachee, "NewAttachee");

        // Build the chain: middle → top, leaf → middle. After this the
        // tree has root=top, with middle as child and leaf as grandchild.
        sim.attach(middle, top, DVec3::ZERO, DMat3::IDENTITY);
        sim.attach(leaf, middle, DVec3::ZERO, DMat3::IDENTITY);

        sim.validate().expect("validate failed");

        // ── Step long enough to leave GJ priming on `top`. ──
        sim.step_n(200).expect("step_n failed");
        assert!(
            !sim.bodies[top].gj_state.as_ref().unwrap().is_priming(),
            "test setup: top's GJ must be past priming"
        );

        // ── Attach NewAttachee underneath middle. This recomputes
        //    middle's *and* top's composite properties (top is an
        //    ancestor of middle). top's GJ state must therefore reset.
        //
        //    Pre-fix, only `middle` and `new_attachee` would be reset,
        //    leaving `top` with stale predictor history that
        //    references the pre-attach mass distribution. ──
        sim.attach(new_attachee, middle, DVec3::ZERO, DMat3::IDENTITY);
        assert!(
            sim.bodies[top].gj_state.as_ref().unwrap().is_priming(),
            "ancestor `top`'s GJ must be reset when a body is attached \
             under its descendant `middle` (IG.37 ancestor coverage)"
        );
        assert!(
            !sim.bodies[top]
                .gj_state
                .as_ref()
                .unwrap()
                .is_topology_dirty(),
            "ancestor `top`'s GJ must be reset (dirty bit cleared)"
        );

        // Step past priming again so we can test the detach branch.
        sim.step_n(200).expect("step_n failed (post-attach)");
        assert!(!sim.bodies[top].gj_state.as_ref().unwrap().is_priming());

        // ── Detach `new_attachee` — same ancestor coverage requirement. ──
        sim.detach(new_attachee);
        assert!(
            sim.bodies[top].gj_state.as_ref().unwrap().is_priming(),
            "ancestor `top`'s GJ must be reset when a descendant of \
             `middle` is detached (IG.37 ancestor coverage)"
        );
        assert!(
            !sim.bodies[top]
                .gj_state
                .as_ref()
                .unwrap()
                .is_topology_dirty(),
            "ancestor `top`'s GJ dirty bit must be cleared by reset"
        );

        // Confirm a subsequent step doesn't trip the IG.37 assertion.
        sim.step_n(1).expect("post-detach step failed");
    }

    /// Stress test the affected-id lookup in attach/detach: build a
    /// 100-body chain (one integrated GJ body + 99 tree-only nodes,
    /// linearly chained as ancestors), then attach a new body
    /// underneath the deepest ancestor and detach it again.
    /// `MassTree::attach` recomputes composites along the entire
    /// 100-deep ancestor chain, so `affected_ids` contains 100
    /// entries — and the helpers must do membership lookup against
    /// that set for every Simulation body in `self.bodies`. With the
    /// sort + binary_search pattern (PR #282 review thread
    /// `PRRT_kwDORtae6c5_KoAT`) the per-body cost is O(log 100); a
    /// regression back to `Vec::contains` would be O(100) per body,
    /// O(n²) overall.
    ///
    /// We don't time the test — what we assert is functional
    /// correctness on a deep chain: the integrated body's GJ state
    /// resets exactly once per attach / detach (never observed dirty
    /// after the call), and a follow-up `step_n(1)` doesn't trip the
    /// IG.37 panic. A scaling regression would compile and produce
    /// the same observable end-state, but on a JEOD-scale sim with
    /// thousands of bodies the cost would dominate; the named
    /// invariant — "exactly one mark + one reset per affected body
    /// per topology change" — is documented at the helper sites.
    #[test]
    fn attach_detach_scales_to_deep_ancestor_chain() {
        const MU: f64 = 5.76e14;
        const CHAIN_LEN: usize = 100;
        let dt = 1.0_f64;
        let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
        let mut sim = Simulation::new(time, dt);

        let earth = sim.add_source(
            "Earth",
            GravitySourceEntry {
                source: GravitySource {
                    mu: MU,
                    model: GravityModel::PointMass,
                },
                position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
                velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
                t_inertial_pfix: None,
                delta_c20: 0.0,
                rotation_model: crate::RotationModel::default(),
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
                marker_only: false,
            },
        );

        // One integrated GJ8 body (the only one with integrator
        // state — the rest of the chain is tree-only nodes).
        // JEOD-faithful warn-and-continue: these unit tests step GJ on synthetic
        // initial state whose non-physical configuration occasionally triggers
        // corrector non-convergence. The test asserts attach/detach mechanics,
        // not numerical correctness, so opting in to the JEOD-faithful path
        // (#485 C1) keeps the test focus on its actual subject.
        let gj_cfg = GaussJacksonConfig::with_order(8).with_allow_non_convergence(true);
        let trans = TranslationalState {
            position: DVec3::new(9e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 8000.0, 0.0),
        };
        let cm = sim.add_body(VehicleConfig {
            trans: trans_raw_to_typed::<RootInertial>(&trans),
            integrator: IntegratorType::GaussJackson(gj_cfg),
            mass: Some(mass_raw_to_self_ref(&(MassProperties::new(1000.0)))),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
            },
            ..Default::default()
        });
        let cm_id = sim.add_body_to_tree(cm, "cm");

        // Build a 100-deep ancestor chain rooted at cm. Each link is
        // a tree-only node attached underneath the previous one.
        let mut chain_ids: Vec<MassBodyId> = Vec::with_capacity(CHAIN_LEN);
        chain_ids.push(cm_id);
        {
            let tree = sim
                .mass_tree
                .as_mut()
                .expect("mass tree must exist after add_body_to_tree");
            for i in 1..CHAIN_LEN {
                let id = tree.add_body(format!("node_{i}"), MassProperties::new(10.0));
                let parent = chain_ids[i - 1];
                tree.attach(id, parent, DVec3::new(0.1, 0.0, 0.0), DMat3::IDENTITY);
                chain_ids.push(id);
            }
        }
        sim.sync_body_mass_from_tree(cm);

        sim.validate().expect("validate failed");
        sim.step_n(200).expect("step_n failed");
        assert!(
            !sim.bodies[cm].gj_state.as_ref().unwrap().is_priming(),
            "test setup: cm's GJ must be past priming"
        );

        // Attach a new body underneath the deepest tree-only node.
        // `MassTree::attach` recomputes composites for the new node
        // plus the entire 100-deep ancestor chain, so `affected_ids`
        // has 101 entries. Every Simulation body (just `cm` here)
        // does one binary_search per call — but the same code path
        // applies on a sim with thousands of bodies.
        let deepest = *chain_ids.last().expect("chain must be non-empty");
        let attachee = {
            let tree = sim.mass_tree.as_mut().unwrap();
            let id = tree.add_body("attachee".into(), MassProperties::new(5.0));
            // We need a Simulation body whose mass_body_id == this
            // attachee, so attach via `Simulation::attach` (which
            // requires both ends to be Simulation bodies). For this
            // test we instead use the tree-only attach to keep the
            // setup small — the helpers fan over Simulation bodies,
            // so what matters is the affected-id list size, not
            // whether the new node is Simulation-tracked.
            tree.attach(id, deepest, DVec3::new(0.1, 0.0, 0.0), DMat3::IDENTITY);
            id
        };
        // Manually drive the mark / reset path so we exercise the
        // helpers directly — Simulation::attach takes Simulation body
        // indices, but we want to test the affected-id discipline
        // with a long ancestor chain that includes tree-only nodes.
        let mut affected_ids: Vec<MassBodyId> = vec![attachee];
        affected_ids.extend(sim.mass_tree.as_ref().unwrap().ancestors_inclusive(deepest));
        affected_ids.sort_unstable();
        affected_ids.dedup();
        assert_eq!(
            affected_ids.len(),
            CHAIN_LEN + 1,
            "affected_ids should include the new attachee plus the full {CHAIN_LEN}-deep ancestor chain"
        );
        Simulation::mark_body_integrators_dirty_by_id(&mut sim.bodies, &affected_ids);
        Simulation::reset_body_integrators_by_id(&mut sim.bodies, &affected_ids);

        // After the helpers run, cm's GJ must be primed and clean.
        let gj_post = sim.bodies[cm].gj_state.as_ref().unwrap();
        assert!(
            gj_post.is_priming(),
            "cm's GJ must be back in priming after a topology change \
             affecting its full ancestor chain"
        );
        assert!(
            !gj_post.is_topology_dirty(),
            "cm's GJ topology-dirty flag must be cleared (binary_search lookup ran)"
        );

        // Confirm a subsequent step doesn't trip the IG.37 assertion.
        sim.step_n(1)
            .expect("post-deep-chain step failed (IG.37 must not fire)");
    }

    // ── detach_subtree / attach_subtree_aligned IG.37 wiring ────────
    //
    // The next four tests cover the inline mark+reset blocks at the
    // end of `Simulation::detach_subtree` (`parent_is_integrated`
    // branch) and `Simulation::attach_subtree_aligned`. Without them,
    // a regression that drops either reset call would still pass
    // `attach_resets_*` / `detach_resets_*` above, since those only
    // exercise the simpler `Simulation::attach` / `detach` (no
    // subtree state propagation, no momentum-conservation combine).
    // PR #282 review threads `PRRT_kwDORtae6c5_KoAQ` (detach_subtree)
    // and `PRRT_kwDORtae6c5_KoAR` (attach_subtree_aligned).
    //
    // ## Why we splice integrator state in by hand
    //
    // `Simulation::detach_subtree` and `attach_subtree_aligned` both
    // require a 6-DOF integrated body (they read `body.rot` to
    // propagate the subtree's composite-CoM state).
    // `Simulation::validate` forbids the only integrators that own
    // multi-step history (GJ / ABM4) on 6-DOF bodies — see
    // `ValidationError::GaussJacksonWith6Dof` / `Abm4With6Dof`. So
    // an end-to-end test of the inline reset block can't go through
    // `validate` + `step_n()` today.
    //
    // The reset block IS still on the production code path: it
    // guards future re-enablement of GJ/ABM4 with rotational
    // dynamics, *and* it guards external callers who construct a
    // 6-DOF SimBody and splice in `gj_state` / `abm4_state`
    // themselves (e.g. from integration-tests like these, or from a
    // downstream crate that bypasses our validator). The fail-loud
    // safety net for *those* callers is the IG.37 panic in
    // `integrate()`.
    //
    // Each test below:
    //   1. Builds a 3-body mass-tree chain `cm → middle → leaf` with
    //      a 6-DOF RK4-integrated `cm` (so it passes `validate`).
    //   2. Splices a `GaussJacksonState` / `Abm4State` directly onto
    //      `cm` after validate, then advances the state past priming
    //      using the integrator's own public API at a constant
    //      zero-accel function (we don't need realistic dynamics —
    //      only a state that is observably *past* priming).
    //   3. Asserts pre-state: `is_priming() == false` AND
    //      `is_topology_dirty() == false`.
    //   4. Triggers the topology mutation (`detach_subtree` /
    //      `attach_subtree_aligned`).
    //   5. Asserts post-state: integrator is back in priming AND not
    //      topology-dirty. Both flags moving together is the
    //      signature of `reset_for_topology_change` having run —
    //      since `mark_topology_dirty` alone never touches
    //      `is_priming`. If a regression drops the inline reset
    //      call, this assertion fails (the test fails on the
    //      `is_priming` check).
    //
    // The accompanying `dirty_*_state_panics_on_integrate` tests
    // independently verify the IG.37 fail-loud panic that backs the
    // mark site: pre-mark dirty (simulating the case where mark
    // fired but reset was forgotten), then run `integrate()` and
    // assert the panic message naming the IG.37 diagnostic.

    /// Set up a 3-body mass-tree chain `cm → middle → leaf` with a
    /// 6-DOF RK4-integrated `cm`. The chain has a non-trivial
    /// `propagate_forward` walk inside `detach_subtree(cm_idx, leaf)`
    /// (the `chain` loop iterates over both `middle` and `leaf`).
    /// Returns the simulation, the integrated-body index, and the
    /// mass-tree ids of `cm`, `middle`, `leaf`.
    fn build_three_body_chain_with_rot() -> (Simulation, usize, MassBodyId, MassBodyId, MassBodyId)
    {
        const MU: f64 = 5.76e14;
        let dt = 1.0_f64;

        let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
        let mut sim = Simulation::new(time, dt);

        let earth = sim.add_source(
            "Earth",
            GravitySourceEntry {
                source: GravitySource {
                    mu: MU,
                    model: GravityModel::PointMass,
                },
                position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
                velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
                t_inertial_pfix: None,
                delta_c20: 0.0,
                rotation_model: crate::RotationModel::default(),
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
                marker_only: false,
            },
        );

        // 6-DOF + RK4 (so it passes validate). We splice GJ/ABM4 state
        // in by hand after validate to exercise detach_subtree's IG.37
        // reset block — see the module-level comment above for why.
        let cm_idx = sim.add_body(VehicleConfig {
            trans: trans_raw_to_typed::<RootInertial>(&TranslationalState {
                position: DVec3::new(9e6, 0.0, 0.0),
                velocity: DVec3::new(0.0, 8000.0, 0.0),
            }),
            rot: Some(rot_raw_to_self_ref(&(astrodyn::RotationalState::default()))),
            integrator: IntegratorType::Rk4,
            mass: Some(mass_raw_to_self_ref(&(MassProperties::new(1000.0)))),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
            },
            ..Default::default()
        });

        let cm_id = sim.add_body_to_tree(cm_idx, "cm");

        // Tree-only subtree nodes (no Simulation body / integrator).
        let tree = sim
            .mass_tree
            .as_mut()
            .expect("mass tree must be initialised by add_body_to_tree above");
        let middle = tree.add_body("middle".into(), MassProperties::new(500.0));
        let leaf = tree.add_body("leaf".into(), MassProperties::new(250.0));

        // cm → middle → leaf at unit offsets, identity rotations.
        tree.attach(middle, cm_id, DVec3::new(1.0, 0.0, 0.0), DMat3::IDENTITY);
        tree.attach(leaf, middle, DVec3::new(1.0, 0.0, 0.0), DMat3::IDENTITY);

        sim.sync_body_mass_from_tree(cm_idx);

        (sim, cm_idx, cm_id, middle, leaf)
    }

    /// Drive a `GaussJacksonState` past priming using its public
    /// `integrate()` API at constant zero acceleration. The post-step
    /// state values aren't used — only the priming flag is.
    fn drive_gj_past_priming(gj: &mut GaussJacksonState) {
        let dt = 1.0_f64;
        let mut state = TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        };
        // GJ8 needs ~50 stages to bootstrap; 200 stages is comfortably past.
        for _ in 0..200 {
            let _ = gj.inner_mut().integrate(dt, 1.0, DVec3::ZERO, &mut state);
        }
        assert!(
            !gj.is_priming(),
            "test setup: drive_gj_past_priming did not exit priming after 200 stages"
        );
    }

    /// Drive an `Abm4State` past priming using `abm4_translational_step`
    /// at constant zero acceleration. ABM4 primes after `HIST_LEN - 1 = 3`
    /// steps; 5 is comfortably past.
    fn drive_abm4_past_priming(abm: &mut Abm4State) {
        let dt = 1.0_f64;
        let mut state = TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        };
        for _ in 0..5 {
            state = astrodyn::abm4_translational_step(
                &state,
                |_s, _t| DVec3::ZERO,
                dt,
                abm.inner_mut(),
            );
        }
        assert!(
            !abm.is_priming(),
            "test setup: drive_abm4_past_priming did not exit priming after 5 stages"
        );
    }

    /// `Simulation::detach_subtree` must reset the integrated body's
    /// Gauss-Jackson history after the topology mutation. Build a
    /// 3-body chain, splice GJ state onto the integrated body, drive
    /// past priming, detach the deepest leaf (a non-trivial chain walk
    /// inside `detach_subtree`), and verify the integrator is reset.
    /// PR #282 review thread `PRRT_kwDORtae6c5_KoAQ`.
    #[test]
    fn detach_subtree_resets_gauss_jackson_state_on_integrated_body() {
        let (mut sim, cm_idx, _cm_id, _middle, leaf) = build_three_body_chain_with_rot();
        sim.validate().expect("validate failed");

        // Splice GJ8 state onto cm post-validate (validate forbids
        // GJ+6DOF; we're exercising the inline reset block defensively).
        // JEOD-faithful warn-and-continue: test setup is synthetic and may
        // trip corrector non-convergence; opt in to JEOD's log-and-continue
        // (#485 C1) so the test stays focused on its actual subject.
        let cfg = GaussJacksonConfig::with_order(8).with_allow_non_convergence(true);
        let mut gj = GaussJacksonState::new(cfg);
        drive_gj_past_priming(&mut gj);
        assert!(!gj.is_topology_dirty());
        sim.bodies[cm_idx].gj_state = Some(gj);

        // ── detach_subtree: drops the leaf node. The chain walk
        //    `cm → middle → leaf` exercises the `propagate_forward`
        //    loop inside detach_subtree (lines 320–358). ──
        sim.detach_subtree(cm_idx, leaf);

        let gj_post = sim.bodies[cm_idx]
            .gj_state
            .as_ref()
            .expect("integrated body must still have gj_state");
        // is_priming flips back to true ONLY through
        // reset_for_topology_change — proving the inline reset call
        // inside detach_subtree fires. If a regression drops that
        // call, this assertion fails.
        assert!(
            gj_post.is_priming(),
            "cm's GJ state must be back in priming after detach_subtree (IG.37)"
        );
        assert!(
            !gj_post.is_topology_dirty(),
            "cm's GJ topology-dirty flag must be cleared by reset on detach_subtree (IG.37)"
        );
    }

    /// ABM4 sibling of `detach_subtree_resets_gauss_jackson_state_*`.
    /// PR #282 review thread `PRRT_kwDORtae6c5_KoAQ`.
    #[test]
    fn detach_subtree_resets_abm4_state_on_integrated_body() {
        let (mut sim, cm_idx, _cm_id, _middle, leaf) = build_three_body_chain_with_rot();
        sim.validate().expect("validate failed");

        let mut abm = Abm4State::new();
        drive_abm4_past_priming(&mut abm);
        assert!(!abm.is_topology_dirty());
        sim.bodies[cm_idx].abm4_state = Some(abm);

        sim.detach_subtree(cm_idx, leaf);

        let abm_post = sim.bodies[cm_idx]
            .abm4_state
            .as_ref()
            .expect("integrated body must still have abm4_state");
        assert!(
            abm_post.is_priming(),
            "cm's ABM4 state must be back in priming after detach_subtree (IG.37)"
        );
        assert!(
            !abm_post.is_topology_dirty(),
            "cm's ABM4 topology-dirty flag must be cleared by reset on detach_subtree (IG.37)"
        );
    }

    // ── attach_subtree_aligned IG.37 wiring ─────────────────────────
    //
    // The next two tests cover the inline mark+reset block at the end
    // of `Simulation::attach_subtree_aligned`. Same shape as the
    // `detach_subtree` tests above (and same constraint — `validate`
    // forbids GJ/ABM4 + 6-DOF, so we splice integrator state in by
    // hand). PR #282 review thread `PRRT_kwDORtae6c5_KoAR`.
    //
    // Setup is asymmetric: we need a *detached* subtree to attach.
    // We call `detach_subtree` first to populate
    // `Simulation::detached_subtrees`, then splice the integrator
    // state and run `attach_subtree_aligned` so its inline reset
    // block has something to clear.

    /// `Simulation::attach_subtree_aligned` must reset the integrated
    /// body's Gauss-Jackson history after the topology + state combine.
    /// PR #282 review thread `PRRT_kwDORtae6c5_KoAR`.
    #[test]
    fn attach_subtree_aligned_resets_gauss_jackson_state_on_integrated_body() {
        let (mut sim, cm_idx, _cm_id, middle, leaf) = build_three_body_chain_with_rot();

        // Add named docking points needed by attach_aligned.
        {
            let tree = sim
                .mass_tree
                .as_mut()
                .expect("mass tree must exist after build_three_body_chain_with_rot");
            tree.add_mass_point(middle, "middle.dock", DVec3::ZERO, DMat3::IDENTITY);
            tree.add_mass_point(leaf, "leaf.dock", DVec3::ZERO, DMat3::IDENTITY);
        }

        sim.validate().expect("validate failed");

        // Detach `leaf` to populate `detached_subtrees` so we have
        // something to re-attach. detach_subtree does its own IG.37
        // reset, but cm has no integrator state spliced in yet — so
        // detach is a no-op on integrators here.
        sim.detach_subtree(cm_idx, leaf);

        // Splice GJ8 state onto cm AFTER the detach so
        // attach_subtree_aligned's reset block has something to clear.
        // JEOD-faithful warn-and-continue: test setup is synthetic and may
        // trip corrector non-convergence; opt in to JEOD's log-and-continue
        // (#485 C1) so the test stays focused on its actual subject.
        let cfg = GaussJacksonConfig::with_order(8).with_allow_non_convergence(true);
        let mut gj = GaussJacksonState::new(cfg);
        drive_gj_past_priming(&mut gj);
        assert!(!gj.is_topology_dirty());
        sim.bodies[cm_idx].gj_state = Some(gj);

        // ── Re-attach. `attach_subtree_aligned` runs the
        //    combine-states algorithm, overwrites body.trans /
        //    body.rot, and must reset the integrator at the end. ──
        sim.attach_subtree_aligned(cm_idx, leaf, "leaf.dock", middle, "middle.dock");

        let gj_post = sim.bodies[cm_idx].gj_state.as_ref().unwrap();
        // `is_priming() == true` only happens through
        // `reset_for_topology_change` — so this fails if the inline
        // reset call inside `attach_subtree_aligned` is removed.
        assert!(
            gj_post.is_priming(),
            "cm's GJ state must be back in priming after attach_subtree_aligned (IG.37)"
        );
        assert!(
            !gj_post.is_topology_dirty(),
            "cm's GJ topology-dirty flag must be cleared on attach_subtree_aligned (IG.37)"
        );
    }

    /// ABM4 sibling of `attach_subtree_aligned_resets_gauss_jackson_*`.
    /// PR #282 review thread `PRRT_kwDORtae6c5_KoAR`.
    #[test]
    fn attach_subtree_aligned_resets_abm4_state_on_integrated_body() {
        let (mut sim, cm_idx, _cm_id, middle, leaf) = build_three_body_chain_with_rot();

        {
            let tree = sim
                .mass_tree
                .as_mut()
                .expect("mass tree must exist after build_three_body_chain_with_rot");
            tree.add_mass_point(middle, "middle.dock", DVec3::ZERO, DMat3::IDENTITY);
            tree.add_mass_point(leaf, "leaf.dock", DVec3::ZERO, DMat3::IDENTITY);
        }

        sim.validate().expect("validate failed");
        sim.detach_subtree(cm_idx, leaf);

        let mut abm = Abm4State::new();
        drive_abm4_past_priming(&mut abm);
        assert!(!abm.is_topology_dirty());
        sim.bodies[cm_idx].abm4_state = Some(abm);

        sim.attach_subtree_aligned(cm_idx, leaf, "leaf.dock", middle, "middle.dock");

        let abm_post = sim.bodies[cm_idx].abm4_state.as_ref().unwrap();
        assert!(
            abm_post.is_priming(),
            "cm's ABM4 state must be back in priming after attach_subtree_aligned (IG.37)"
        );
        assert!(
            !abm_post.is_topology_dirty(),
            "cm's ABM4 topology-dirty flag must be cleared on attach_subtree_aligned (IG.37)"
        );
    }

    /// Verify the IG.37 fail-loud safety net actually fires when an
    /// integrator is left dirty: simulate a regression where mark
    /// fired but reset was forgotten by manually flipping the dirty
    /// flag to true on a primed integrator, then call `integrate()`.
    /// The test passes only if the integrator panics with the IG.37
    /// diagnostic. This is the pair-half of the inline mark sites in
    /// detach_subtree / attach_subtree_aligned — together they make
    /// any regression that drops a Site B (reset) loud rather than
    /// silent.
    #[test]
    #[should_panic(expected = "topology")]
    fn dirty_gauss_jackson_state_panics_on_integrate() {
        // JEOD-faithful warn-and-continue: test setup is synthetic and may
        // trip corrector non-convergence; opt in to JEOD's log-and-continue
        // (#485 C1) so the test stays focused on its actual subject.
        let cfg = GaussJacksonConfig::with_order(8).with_allow_non_convergence(true);
        let mut gj = GaussJacksonState::new(cfg);
        drive_gj_past_priming(&mut gj);

        // Simulate a regression where Site A (mark) fired but Site B
        // (reset_for_topology_change) was forgotten.
        gj.mark_topology_dirty();
        assert!(gj.is_topology_dirty());

        // The next integrate() must fail-loud per IG.37.
        let mut state = TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        };
        let _ = gj.inner_mut().integrate(1.0, 1.0, DVec3::ZERO, &mut state);
    }

    /// ABM4 sibling of `dirty_gauss_jackson_state_panics_on_integrate`.
    #[test]
    #[should_panic(expected = "topology")]
    fn dirty_abm4_state_panics_on_integrate() {
        let mut abm = Abm4State::new();
        drive_abm4_past_priming(&mut abm);

        abm.mark_topology_dirty();
        assert!(abm.is_topology_dirty());

        let state = TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        };
        let _ = astrodyn::abm4_translational_step(&state, |_, _| DVec3::ZERO, 1.0, abm.inner_mut());
    }

    /// JEOD_INV: IG.37 — `tree.attach` + `sync_body_mass_from_tree` is the
    /// supported lower-level alternative to `Simulation::attach`, so it
    /// must reset multi-step integrator history just like the high-level
    /// path. PR #282 round-3 review thread `PRRT_kwDORtae6c5_KzS4`
    /// (Copilot): without this wiring, a Gauss-Jackson caller using the
    /// `tree.attach` + `sync_body_mass_from_tree` pattern (used by
    /// `crates/astrodyn_runner/examples/apollo.rs` and the Apollo Tier 3
    /// trajectory test for stage separation) would silently propagate
    /// stale predictor history across the topology change.
    #[test]
    fn sync_body_mass_from_tree_resets_gauss_jackson_state() {
        const MU: f64 = 5.76e14;
        let dt = 1.0_f64;

        let trans = TranslationalState {
            position: DVec3::new(9e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 8000.0, 0.0),
        };

        let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
        let mut sim = Simulation::new(time, dt);

        let earth = sim.add_source(
            "Earth",
            GravitySourceEntry {
                source: GravitySource {
                    mu: MU,
                    model: GravityModel::PointMass,
                },
                position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
                velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
                t_inertial_pfix: None,
                delta_c20: 0.0,
                rotation_model: crate::RotationModel::default(),
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
                marker_only: false,
            },
        );

        // JEOD-faithful warn-and-continue: these unit tests step GJ on synthetic
        // initial state whose non-physical configuration occasionally triggers
        // corrector non-convergence. The test asserts attach/detach mechanics,
        // not numerical correctness, so opting in to the JEOD-faithful path
        // (#485 C1) keeps the test focus on its actual subject.
        let gj_cfg = GaussJacksonConfig::with_order(8).with_allow_non_convergence(true);
        let body_idx = sim.add_body(VehicleConfig {
            trans: trans_raw_to_typed::<RootInertial>(&trans),
            integrator: IntegratorType::GaussJackson(gj_cfg),
            mass: Some(mass_raw_to_self_ref(&(MassProperties::new(1000.0)))),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
            },
            ..Default::default()
        });
        let cm_id = sim.add_body_to_tree(body_idx, "cm");

        sim.validate().expect("validate failed");
        sim.step_n(200).expect("step_n failed");
        assert!(
            !sim.bodies[body_idx].gj_state.as_ref().unwrap().is_priming(),
            "test setup: cm's GJ must be past priming"
        );

        // Mutate the mass tree directly via `as_mut()` (the lower-level
        // path that bypasses `Simulation::attach`). Add a tree-only child
        // and attach it under cm, then call the documented sync point.
        let attachee = {
            let tree = sim.mass_tree.as_mut().expect("tree must exist");
            let id = tree.add_body("attachee".into(), MassProperties::new(50.0));
            tree.attach(id, cm_id, DVec3::new(0.5, 0.0, 0.0), DMat3::IDENTITY);
            id
        };
        let _ = attachee;

        // The fix: `sync_body_mass_from_tree` mark+resets the body's
        // GJ history so the lower-level `tree.attach + sync` path is
        // IG.37-safe by construction.
        sim.sync_body_mass_from_tree(body_idx);

        let gj_post = sim.bodies[body_idx]
            .gj_state
            .as_ref()
            .expect("cm must still have gj_state");
        assert!(
            gj_post.is_priming(),
            "cm's GJ state must be back in priming after sync_body_mass_from_tree"
        );
        assert!(
            !gj_post.is_topology_dirty(),
            "cm's GJ topology-dirty flag must be cleared by sync_body_mass_from_tree"
        );

        // Confirm a subsequent step doesn't trip the IG.37 assertion.
        sim.step_n(1)
            .expect("post-sync step failed (IG.37 must not fire)");
    }

    /// ABM4 sibling of `sync_body_mass_from_tree_resets_gauss_jackson_state`.
    /// Same pattern, ABM4 instead of Gauss-Jackson — covers both
    /// multi-step integrator types whose history is invalidated by
    /// topology changes.
    #[test]
    fn sync_body_mass_from_tree_resets_abm4_state() {
        const MU: f64 = 5.76e14;
        let dt = 1.0_f64;

        let trans = TranslationalState {
            position: DVec3::new(9e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 8000.0, 0.0),
        };

        let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
        let mut sim = Simulation::new(time, dt);

        let earth = sim.add_source(
            "Earth",
            GravitySourceEntry {
                source: GravitySource {
                    mu: MU,
                    model: GravityModel::PointMass,
                },
                position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
                velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
                t_inertial_pfix: None,
                delta_c20: 0.0,
                rotation_model: crate::RotationModel::default(),
                tidal_config: None,
                planet_omega: 0.0,
                central: true,
                marker_only: false,
            },
        );

        let body_idx = sim.add_body(VehicleConfig {
            trans: trans_raw_to_typed::<RootInertial>(&trans),
            integrator: IntegratorType::Abm4,
            mass: Some(mass_raw_to_self_ref(&(MassProperties::new(1000.0)))),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
            },
            ..Default::default()
        });
        let cm_id = sim.add_body_to_tree(body_idx, "cm");

        sim.validate().expect("validate failed");
        sim.step_n(20).expect("step_n failed");
        assert!(
            !sim.bodies[body_idx]
                .abm4_state
                .as_ref()
                .unwrap()
                .is_priming(),
            "test setup: cm's ABM4 must be past priming"
        );

        let attachee = {
            let tree = sim.mass_tree.as_mut().expect("tree must exist");
            let id = tree.add_body("attachee".into(), MassProperties::new(50.0));
            tree.attach(id, cm_id, DVec3::new(0.5, 0.0, 0.0), DMat3::IDENTITY);
            id
        };
        let _ = attachee;

        sim.sync_body_mass_from_tree(body_idx);

        let abm_post = sim.bodies[body_idx]
            .abm4_state
            .as_ref()
            .expect("cm must still have abm4_state");
        assert!(
            abm_post.is_priming(),
            "cm's ABM4 state must be back in priming after sync_body_mass_from_tree"
        );
        assert!(
            !abm_post.is_topology_dirty(),
            "cm's ABM4 topology-dirty flag must be cleared by sync_body_mass_from_tree"
        );

        sim.step_n(1)
            .expect("post-sync step failed (IG.37 must not fire)");
    }

    /// Regression: detaching a kinematic child must clear its
    /// `kinematic_only` flag. Without the auto-clear, the next
    /// `step()` enters `propagate_kinematic_state` with a kinematic
    /// flag set on a body that is now a tree root (no parent), and
    /// the "kinematic-only resolves to a tree root" assert fires.
    #[test]
    fn detach_clears_kinematic_only_flag_on_child() {
        use crate::SimulationBuilderExt;
        use astrodyn::recipes::Mission;
        use astrodyn::{JeodQuat, RotationalState};

        // `iss_leo` ships a 3-DOF point-mass root; reuse it as-is. The
        // detach/auto-clear path only inspects mass-tree topology and
        // the kinematic flag on the child, so the parent never needs a
        // `rot` here.
        let mut sim = Mission::iss_leo()
            .into_builder()
            .build()
            .expect("Mission::iss_leo must validate");

        let parent_idx = 0;
        // Add a 6-DOF child SimBody, register both in the mass tree,
        // attach with a non-zero offset, and mark the child kinematic.
        let child_idx = sim.add_body(VehicleConfig {
            trans: trans_raw_to_typed::<RootInertial>(&TranslationalState::default()),
            rot: Some(rot_raw_to_self_ref(
                &(RotationalState {
                    quaternion: JeodQuat::identity(),
                    ang_vel_body: DVec3::ZERO,
                }),
            )),
            mass: Some(mass_raw_to_self_ref(&(MassProperties::new(5.0)))),
            gravity_controls: GravityControls { controls: vec![] },
            ..Default::default()
        });
        sim.add_body_to_tree(parent_idx, "parent");
        sim.add_body_to_tree(child_idx, "child");
        sim.attach(
            child_idx,
            parent_idx,
            DVec3::new(1.0, 0.0, 0.0),
            DMat3::IDENTITY,
        );
        sim.mark_kinematic_only(child_idx);
        assert!(
            sim.bodies[child_idx].kinematic_only,
            "child must be kinematic-only after mark_kinematic_only"
        );

        // One pre-detach step exercises propagation in the kinematic
        // state. The assert at the top of `propagate_kinematic_state`
        // would fire here on a regression.
        sim.step().expect("pre-detach step must succeed");

        // Detach the child. The auto-clear is what this test pins —
        // without it the next `step()` would panic with the "tree
        // root" diagnostic from `propagate_kinematic_state`.
        sim.detach(child_idx);
        assert!(
            !sim.bodies[child_idx].kinematic_only,
            "detach must clear kinematic_only on the freshly-detached child; \
             leaving it set would panic in propagate_kinematic_state on the \
             next step (the body is now a tree root with no parent)"
        );

        // Post-detach step: must run cleanly. This is the load-bearing
        // assertion — a regression that drops the auto-clear panics
        // here, not at the assertion above.
        sim.step()
            .expect("post-detach step must succeed (kinematic_only must be cleared)");
    }

    /// Sibling of `detach_clears_kinematic_only_flag_on_child` for the
    /// `detach_subtree` path. `detach_subtree` mutates the same
    /// `mass_tree` topology as `detach`, so a kinematic-flagged SimBody
    /// whose `mass_body_id` matches the subtree root must have the flag
    /// cleared at the same fail-loud entry point — otherwise
    /// `propagate_kinematic_state` panics on the next `step()` with
    /// "kinematic-only resolves to a tree root". Pins the parallel fix
    /// requested in PR #295 review thread `PRRT_kwDORtae6c5_O1JW`.
    #[test]
    fn detach_subtree_clears_kinematic_only_flag_on_subtree_root() {
        use crate::SimulationBuilderExt;
        use astrodyn::recipes::Mission;
        use astrodyn::{JeodQuat, RotationalState};

        // Same recipe-backed root as `detach_clears_kinematic_only_flag_on_child`.
        let mut sim = Mission::iss_leo()
            .into_builder()
            .build()
            .expect("Mission::iss_leo must validate");

        let parent_idx = 0;
        // `iss_leo` ships a 3-DOF root; `detach_subtree` requires the
        // integrated body to have a `rot` (it reads the parent
        // composite-body inertial state to build the chain walk).
        // Install identity attitude + zero rate so the subtree path is
        // exercised without changing the orbit.
        sim.bodies[parent_idx].rot = Some(rot_raw_to_self_ref(&RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        }));

        let child_idx = sim.add_body(VehicleConfig {
            trans: trans_raw_to_typed::<RootInertial>(&TranslationalState::default()),
            rot: Some(rot_raw_to_self_ref(
                &(RotationalState {
                    quaternion: JeodQuat::identity(),
                    ang_vel_body: DVec3::ZERO,
                }),
            )),
            mass: Some(mass_raw_to_self_ref(&(MassProperties::new(5.0)))),
            gravity_controls: GravityControls { controls: vec![] },
            ..Default::default()
        });
        sim.add_body_to_tree(parent_idx, "parent");
        let child_id = sim.add_body_to_tree(child_idx, "child");
        sim.attach(
            child_idx,
            parent_idx,
            DVec3::new(1.0, 0.0, 0.0),
            DMat3::IDENTITY,
        );
        sim.mark_kinematic_only(child_idx);
        assert!(
            sim.bodies[child_idx].kinematic_only,
            "child must be kinematic-only after mark_kinematic_only"
        );

        // Pre-detach step exercises `propagate_kinematic_state` so a
        // regression that flips the assert order shows up here.
        sim.step().expect("pre-detach step must succeed");

        // Detach via the subtree path. `child_id` is a leaf, so the
        // subtree happens to contain a single node — that's still the
        // exact case where the flag must be cleared (the freshly-rooted
        // node is the same SimBody that carried `kinematic_only`).
        sim.detach_subtree(parent_idx, child_id);
        assert!(
            !sim.bodies[child_idx].kinematic_only,
            "detach_subtree must clear kinematic_only on the SimBody backing \
             the freshly-rooted subtree; leaving it set would panic in \
             propagate_kinematic_state on the next step (tree-root with no parent)"
        );

        // Load-bearing assertion: post-detach step must run. A
        // regression that drops the auto-clear panics here.
        sim.step()
            .expect("post-detach step must succeed (kinematic_only must be cleared)");
    }

    /// Reroot path must reject a 3-DOF body in the rerooted subtree.
    ///
    /// Per JEOD_INV: DB.17, every interior SimBody derives its state
    /// each tick through `propagate_kinematic_state`, which composes
    /// both `trans` and `rot` from the parent's pose. A 3-DOF body
    /// (no `RotationalState`) cannot participate in that walk, so a
    /// chained-attach reroot that would push it from "tree root" to
    /// "interior" has no way to keep its `body.trans` live. We
    /// therefore fail loudly at the attach site rather than silently
    /// letting the body's state go stale post-reroot.
    ///
    /// Setup: parent_a (target), parent_b (subject's tree root,
    /// 6-DOF), intermediate (6-DOF child of parent_b), 3dof_child
    /// (3-DOF child of intermediate). Attaching `intermediate` to
    /// parent_a triggers the reroot path: subject_root_id is parent_b,
    /// the rerooted subtree includes 3dof_child, which fails the
    /// rot-required assertion.
    #[test]
    #[should_panic(expected = "rerooted subtree")]
    fn reroot_panics_on_3dof_body_in_subject_subtree() {
        use crate::SimulationBuilderExt;
        use astrodyn::recipes::Mission;
        use astrodyn::{JeodQuat, RotationalState};

        let mut sim = Mission::iss_leo()
            .into_builder()
            .build()
            .expect("Mission::iss_leo must validate");

        let parent_a_idx = 0;
        // iss_leo ships a 3-DOF root; promote parent_a in place to
        // 6-DOF so the post-reroot tree is well-formed (parent_a is
        // the integrated root of the merged tree).
        sim.bodies[parent_a_idx].rot = Some(rot_raw_to_self_ref(&RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        }));

        let parent_b_idx = sim.add_body(VehicleConfig {
            trans: trans_raw_to_typed::<RootInertial>(&TranslationalState {
                position: DVec3::new(7e6, 0.0, 0.0),
                velocity: DVec3::new(0.0, 7500.0, 0.0),
            }),
            rot: Some(rot_raw_to_self_ref(
                &(RotationalState {
                    quaternion: JeodQuat::identity(),
                    ang_vel_body: DVec3::ZERO,
                }),
            )),
            mass: Some(mass_raw_to_self_ref(&(MassProperties::new(100.0)))),
            gravity_controls: GravityControls { controls: vec![] },
            ..Default::default()
        });
        let intermediate_idx = sim.add_body(VehicleConfig {
            trans: trans_raw_to_typed::<RootInertial>(&TranslationalState {
                position: DVec3::new(7e6 + 1.0, 0.0, 0.0),
                velocity: DVec3::new(0.0, 7500.0, 0.0),
            }),
            rot: Some(rot_raw_to_self_ref(
                &(RotationalState {
                    quaternion: JeodQuat::identity(),
                    ang_vel_body: DVec3::ZERO,
                }),
            )),
            mass: Some(mass_raw_to_self_ref(&(MassProperties::new(50.0)))),
            gravity_controls: GravityControls { controls: vec![] },
            ..Default::default()
        });
        // 3-DOF body — no `rot` field.
        let three_dof_idx = sim.add_body(VehicleConfig {
            trans: trans_raw_to_typed::<RootInertial>(&TranslationalState {
                position: DVec3::new(7e6 + 2.0, 0.0, 0.0),
                velocity: DVec3::new(0.0, 7500.0, 0.0),
            }),
            rot: None,
            mass: Some(mass_raw_to_self_ref(&(MassProperties::new(10.0)))),
            gravity_controls: GravityControls { controls: vec![] },
            ..Default::default()
        });

        sim.add_body_to_tree(parent_a_idx, "parent_a");
        sim.add_body_to_tree(parent_b_idx, "parent_b");
        sim.add_body_to_tree(intermediate_idx, "intermediate");
        sim.add_body_to_tree(three_dof_idx, "three_dof");

        // Build subject's tree: parent_b ← intermediate ← three_dof.
        // Each attach is a simple root-subject attach (the child is
        // its own tree root before the call), so none triggers the
        // auto-flag block.
        sim.attach(
            intermediate_idx,
            parent_b_idx,
            DVec3::new(1.0, 0.0, 0.0),
            DMat3::IDENTITY,
        );
        sim.attach(
            three_dof_idx,
            intermediate_idx,
            DVec3::new(1.0, 0.0, 0.0),
            DMat3::IDENTITY,
        );

        // Reroot: attach `intermediate` to parent_a. Now
        // `child_id = intermediate_id`, `subject_root_id = parent_b_id`,
        // they differ, so the auto-flag block runs. The rerooted
        // subtree (rooted at parent_b) contains three_dof, which has
        // no RotationalState — the assertion must fire.
        sim.attach(
            intermediate_idx,
            parent_a_idx,
            DVec3::new(2.0, 0.0, 0.0),
            DMat3::IDENTITY,
        );
    }
}
