// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Bevy `Component` and `Message` newtypes for the mass tree —
//! per-body mass identifier, the parent ↔ child mass-tree edge, the
//! mass-point back-pointer, the kinematic-child gating marker, and the
//! attach / detach messages consumed by the staging system.

use astrodyn::{FrameTransform, Position, StructuralFrame, Vehicle};
use bevy::prelude::*;

#[allow(unused_imports)] // intra-doc-link resolution
use super::state::MassPropertiesC;

// ── Mass Tree (Staging) ──

/// Maps this entity to a node in the shared [`MassTreeR`](crate::MassTreeR) resource.
///
/// Entities with this component participate in the mass tree. After
/// attach/detach events are processed, the entity's [`MassPropertiesC`]
/// is synced from the tree's composite properties.
#[derive(Component, Debug, Clone, Copy)]
pub struct MassBodyIdC(pub astrodyn::MassBodyId);

/// ECS-native mass-tree relation: marks `Entity` carrying this component
/// as a sub-mass attached to the referenced parent entity in the **mass
/// tree** (deliberately distinct from Bevy's frame-tree `ChildOf`).
///
/// Mirrors JEOD's separation of `RefFrame` and `MassBody` trees (see
/// [Frame-Tree-ECS-Native § 15.2](https://github.com/simnaut/astrodyn/wiki/Frame-Tree-ECS-Native#152-mass--inertia-composition)
/// and Appendix A.3): the kinematic frame tree and the inertial mass
/// tree evolve under independent attach/detach paths, coupled only by
/// the explicit [`MassPointRef`] back-pointer. Keeping the two
/// relations as separate `Component`s makes Bevy's hierarchy +
/// observer plumbing one-to-one with JEOD's "two trees + named
/// coupling" architecture.
///
/// The component carries the **parent reference** plus the
/// attach-edge geometry (`offset` + `t_parent_child`), matching the
/// arena's per-body `MassBody::structure_point` (`offset` is the
/// child's structural origin in the *parent's* structural frame;
/// `t_parent_child` is the rotation from the parent's structural
/// frame into this body's structural frame). Edge geometry lives on
/// the child because every child has exactly one parent — which
/// matches both JEOD's `MassBody` layout and the natural ECS
/// component-per-entity grain.
///
/// The carrier entity must also have [`MassPropertiesC`]; the
/// `composite_mass_system` walks `MassChildOf` edges bottom-up via
/// the [`astrodyn::MassStorage`] trait and writes the recomputed
/// composite properties back into [`MassPropertiesC`] on every node
/// in the affected subtree.
///
/// # JEOD precedent
///
/// `MassBody` nodes form a tree via `MassBodyLinks` (see
/// `models/dynamics/mass/include/mass.hh`); `MassBody::structure_point`
/// (`MassPointState`) carries the per-attach offset + rotation that
/// `MassChildOf` mirrors here. `BodyRefFrame::mass_point`
/// (`models/dynamics/dyn_body/include/body_ref_frame.hh`) is the
/// frame-side back-pointer connecting a kinematic frame to its
/// mass-tree origin — see [`MassPointRef`] for the Bevy port.
// JEOD_INV: MA.08 — no cycle in mass tree (composite_mass_system asserts via post-order walk)
// JEOD_INV: MA.19 — no same-tree attachment (cycle prevention)
#[derive(Component, Debug, Clone, Copy)]
pub struct MassChildOf {
    /// Parent entity in the mass tree.
    pub parent: Entity,
    /// Child's structural origin expressed in the **parent's**
    /// structural frame (m). Default `[0, 0, 0]` means the child's
    /// struct origin is co-located with the parent's struct origin.
    pub offset: glam::DVec3,
    /// Rotation from the parent's structural frame into this body's
    /// structural frame. Default identity (no relative rotation).
    pub t_parent_child: glam::DMat3,
}

impl MassChildOf {
    /// Convenience constructor for an axis-aligned (identity rotation)
    /// attach at the given offset.
    pub fn new(parent: Entity, offset: glam::DVec3) -> Self {
        Self {
            parent,
            offset,
            t_parent_child: glam::DMat3::IDENTITY,
        }
    }

    /// Convenience constructor for a co-located attach (zero offset,
    /// identity rotation). The child's struct origin sits on the
    /// parent's struct origin — useful when the child's CoM offset
    /// is encoded in its own [`MassPropertiesC.center_of_mass`](MassPropertiesC).
    pub fn at_origin(parent: Entity) -> Self {
        Self {
            parent,
            offset: glam::DVec3::ZERO,
            t_parent_child: glam::DMat3::IDENTITY,
        }
    }

    /// Full constructor with explicit offset + rotation, mirroring
    /// `MassTree::attach(child, parent, offset, t_parent_child)`.
    pub fn with_rotation(parent: Entity, offset: glam::DVec3, t_parent_child: glam::DMat3) -> Self {
        Self {
            parent,
            offset,
            t_parent_child,
        }
    }
}

/// Frame-side back-pointer linking a body's frame entity to the
/// mass-tree node that supplies the body's **mass-point origin**
/// (CoM offset + struct→body rotation).
///
/// Mirrors JEOD's `BodyRefFrame::mass_point` (a `MassPoint *`) defined
/// in `models/dynamics/dyn_body/include/body_ref_frame.hh`. JEOD uses
/// this back-pointer to route kinematic state queries on a body frame
/// (which knows the mass-side point) without forcing the frame and
/// mass trees to share their hierarchy, which is the same separation
/// the Bevy adapter mirrors via [`MassChildOf`] vs Bevy's `ChildOf`.
///
/// **Optional by design.** Per [Frame-Tree-ECS-Native § 15.2](https://github.com/simnaut/astrodyn/wiki/Frame-Tree-ECS-Native#152-mass--inertia-composition)
/// the back-pointer is *absent for kinematic-only attaches* — i.e.
/// frame entities whose kinematics ride a parent without contributing
/// to that parent's mass (sensor mounts, station-keeping vehicles
/// attached only via `attach_to_frame`). Mission code attaches it
/// only when the frame entity also participates in the mass tree.
#[derive(Component, Debug, Clone, Copy)]
pub struct MassPointRef(pub Entity);

/// Marker: this entity is a kinematic non-root node in a
/// [`MassChildOf`] chain and must NOT be advanced by
/// [`integration_system`](crate::systems::integration_system).
///
/// JEOD's composite-rigid-body model integrates only the root of every
/// mass tree (`dyn_body_collect.cc:138` — every `dyn_parent != nullptr`
/// branch transmits forces upstream and computes no per-child
/// accelerations). The Bevy port mirrors this by:
///
/// 1. [`wrench_aggregation_system`](crate::wrench::wrench_aggregation_system)
///    walks every `MassChildOf` chain and folds each non-root child's
///    `(force, torque)` (with parallel-axis arm) into the root's
///    `TotalForceC`, then zeroes the children's
///    `TotalForceC` / `FrameDerivativesC`.
/// 2. The same system inserts `KinematicChildC` on every non-root
///    node and removes it from any node that becomes a root (mass tree
///    rewired or torn down).
/// 3. [`integration_system`](crate::systems::integration_system) filters
///    its body query with `Without<KinematicChildC>` so the kinematic
///    children's translational / rotational state never advances under
///    gravity (or any other contributor `integration_system` reads
///    directly). Without the marker, zeroing `TotalForceC` is not
///    enough — `integration_system` recomputes gravity at every RK
///    sub-stage from `GravityControlsC` and would still drift the
///    child's state.
///
/// This marker is purely a **gating hint** for the integrator. The
/// kinematic propagation that derives child poses *from* the root
/// each step lives at
/// [`crate::kinematic_propagation::propagate_state_from_root_system`]
/// (design-doc Section 15.3) and runs earlier in
/// `JeodSet::ForceCollection` so the wrench walk reads live
/// attitudes; non-root children's `TranslationalStateC` /
/// `RotationalStateC` are overwritten each step with the derived
/// value.
///
/// Mission code MUST NOT manage this marker manually — the
/// wrench-aggregation system owns its lifecycle. Inserting it on a
/// root-level body would freeze that body's state; removing it from a
/// non-root body would let the integrator double-count the wrench
/// (once via the aggregated root total, once via per-stage gravity on
/// the now-self-integrated child).
// JEOD_INV: DB.17 — only the root's TotalForce/FrameDerivatives drive the
// integrator (children are kinematic, gated by this marker)
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct KinematicChildC;

/// Message: attach a child body to a parent in the mass tree.
///
/// Both entities must have [`MassBodyIdC`]. Processed by `staging_system`
/// before integration each step.
///
/// # Vehicle phantoms
///
/// `AttachEvent` is parameterized by **two** vehicle phantoms:
/// `VParent` names the parent body's vehicle identity and `VChild`
/// names the child body's. The split lets the type system distinguish
/// the parent's structural frame from the child's, which is necessary
/// to type the rotation slot as
/// `FrameTransform<StructuralFrame<VParent>, StructuralFrame<VChild>>`
/// — a single-phantom shape would collapse `From == To` and lose the
/// directional guarantee at the type level.
///
/// Mission code that pins both vehicles (via
/// [`define_vehicle!`](astrodyn::define_vehicle)) gets a compile-time
/// guard against confusing one attach pair with another — e.g.
/// `AttachEvent<Iss, Soyuz>` cannot be confused with
/// `AttachEvent<Iss, Cygnus>`, and a `t_parent_child` constructed for
/// the wrong pair fails to typecheck. The compile-time guard layered
/// on top of the existing frame-kind check (structural-vs-inertial)
/// is the parent-and-child vehicle identity.
///
/// # Runtime-resolved boundary
///
/// The canonical Bevy adapter registers and consumes
/// `AttachEvent<SelfRef, SelfRef>` because per-entity storage decides
/// both parent and child vehicle identity at runtime via the entity
/// hierarchy — the message bus does not statically know which vehicle
/// pair is involved. `<SelfRef, SelfRef>` is the documented
/// runtime-resolved instantiation; mission code that mints concrete
/// pairs may register the matching `add_message::<AttachEvent<P, C>>()`
/// itself.
///
/// # Direction convention
///
/// `t_parent_child` rotates vectors expressed in the **parent's**
/// structural frame into the **child's** structural frame, matching
/// JEOD's `T_pstr_cstr` (see
/// `models/dynamics/mass/src/mass_attach.cc:151` —
/// "Transformation matrix from the new parent body's structural
/// frame to this body's structural frame"). The offset is the child's
/// structural origin expressed in the parent's structural frame
/// coordinates (JEOD `offset_pstr_cstr_pstr`).
///
/// # Cross-pair compile-time guard
///
/// Constructing an `AttachEvent<Iss, Soyuz>` whose `t_parent_child`
/// was built for a different pair (e.g. `<Iss, Iss>` — a same-vehicle
/// "self attach" rotation that happens to typecheck without the
/// split phantom) is rejected at compile time:
///
/// ```compile_fail
/// use astrodyn_bevy::AttachEvent;
/// use bevy::prelude::Entity;
/// use astrodyn::{define_vehicle, FrameTransform, StructuralFrame, Vec3Ext};
/// use glam::DVec3;
///
/// define_vehicle!(Iss);
/// define_vehicle!(Soyuz);
///
/// let _ = AttachEvent::<Iss, Soyuz> {
///     child: Entity::PLACEHOLDER,
///     parent: Entity::PLACEHOLDER,
///     offset: Vec3Ext::m_at::<StructuralFrame<Iss>>(DVec3::ZERO),
///     // Wrong pair: `<Iss, Iss>` does not match the slot's expected
///     // `<Iss, Soyuz>` — typecheck failure.
///     t_parent_child: FrameTransform::<StructuralFrame<Iss>, StructuralFrame<Iss>>::identity(),
/// };
/// ```
#[derive(Message, Debug, Clone)]
pub struct AttachEvent<VParent: Vehicle, VChild: Vehicle> {
    /// Entity of the child body.
    pub child: Entity,
    /// Entity of the parent body.
    pub parent: Entity,
    /// Child structural origin expressed in the **parent's** structural
    /// frame coordinates (m). JEOD `offset_pstr_cstr_pstr`.
    pub offset: Position<StructuralFrame<VParent>>,
    /// Rotation taking vectors expressed in the parent's structural
    /// frame into the child's structural frame. JEOD `T_pstr_cstr`.
    pub t_parent_child: FrameTransform<StructuralFrame<VParent>, StructuralFrame<VChild>>,
}

impl<VParent: Vehicle, VChild: Vehicle> AttachEvent<VParent, VChild> {
    /// Type-level witness that this attach pair carries the caller's
    /// expected `(P, C)` vehicle phantoms. Compiles only when
    /// `(VParent, VChild) == (P, C)`; on mismatch the
    /// [`astrodyn::CompatibleVehiclePair`] bound fails and surfaces a
    /// physics-language diagnostic naming both expected and found pairs
    /// instead of a `PhantomData<…>` wall.
    ///
    /// Mission code that wires a typed attach event for a specific
    /// parent/child pair calls this at the boundary to make the
    /// cross-pair guard explicit; the method itself is a no-op (returns
    /// `self`) and has zero runtime cost.
    ///
    /// # Compile-time mismatch
    ///
    /// ```compile_fail
    /// use bevy::prelude::Entity;
    /// use astrodyn_bevy::AttachEvent;
    /// use glam::{DMat3, DVec3};
    /// use astrodyn::{define_vehicle, FrameTransform, StructuralFrame, Vec3Ext};
    ///
    /// define_vehicle!(Iss);
    /// define_vehicle!(Soyuz);
    /// define_vehicle!(Cygnus);
    ///
    /// let evt: AttachEvent<Iss, Soyuz> = AttachEvent {
    ///     child: Entity::PLACEHOLDER,
    ///     parent: Entity::PLACEHOLDER,
    ///     offset: DVec3::ZERO.m_at::<StructuralFrame<Iss>>(),
    ///     t_parent_child: FrameTransform::<
    ///         StructuralFrame<Iss>,
    ///         StructuralFrame<Soyuz>,
    ///     >::from_matrix(DMat3::IDENTITY),
    /// };
    /// // Asserting the wrong child vehicle fires the
    /// // `CompatibleVehiclePair` diagnostic naming the found and
    /// // expected pairs.
    /// let _ = evt.assert_pair::<Iss, Cygnus>();
    /// ```
    #[inline]
    pub fn assert_pair<P: Vehicle, C: Vehicle>(self) -> Self
    where
        (): astrodyn::CompatibleVehiclePair<VParent, VChild, P, C>,
    {
        self
    }
}

/// Message: detach a child body from its parent in the mass tree.
///
/// The entity must have [`MassBodyIdC`] and be attached to a parent.
/// Processed by `staging_system` before integration each step.
#[derive(Message, Debug, Clone)]
pub struct DetachEvent {
    /// Entity to detach from its parent.
    pub child: Entity,
}
