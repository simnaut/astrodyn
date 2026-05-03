//! Composite-rigid-body wrench shift: parallel-axis transfer of a child's
//! `(force, torque)` pair to its parent's structural frame.
//!
//! Ports the upward-walk kernel from JEOD's
//! [`models/dynamics/dyn_body/src/dyn_body_collect.cc`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/dynamics/dyn_body/src/dyn_body_collect.cc)
//! (lines 138–202 — the `dyn_parent != nullptr` branch). JEOD's stable
//! multi-body model is *composite rigid body*: only the root integrates;
//! every attached child is kinematic. External forces / torques applied
//! to a child travel up the `MassChildOf`-equivalent chain into the
//! root's accumulators, transformed by the per-link parallel-axis arm.
//!
//! # JEOD algorithm (per-link, in parent's structural frame)
//!
//! For each child → parent edge with rotation `T_parent_this` (the
//! rotation that takes a parent-frame vector's components and produces
//! the same vector's components in the child frame, i.e.
//! `v_child = T_parent_this · v_parent`) and parent-CoM-to-child-CoM
//! offset `pcm_to_ccm` in the parent's structural frame:
//!
//! ```text
//! F_pstr   = T_parent_this^T · F_child_struct
//! tau_pstr = T_parent_this^T · tau_child_struct  +  pcm_to_ccm × F_pstr
//! ```
//!
//! `T_parent_this^T` is the inverse of `T_parent_this`, so it rotates
//! vector components from child structural to parent structural — the
//! reverse direction. JEOD encodes this as
//! `Vector3::transform_transpose(T_parent_this, v_child, v_parent)`,
//! which is bit-equivalent to `T_parent_this^T · v_child` for a
//! row-major storage convention. The arena's
//! [`crate::mass_storage::compute_node_composite`] uses the matching
//! convention for the inertia tensor rotation
//! (`t.transpose() · I_child · t` is the standard `R · A · R^T`
//! rotation with `R = t.transpose()`).
//!
//! # Frame conventions in this kernel
//!
//! The kernel is frame-agnostic in *units*: callers pass `f_child` /
//! `tau_child` in the **child's** "wrench frame" (whatever common
//! convention the orchestration layer chooses — JEOD picks structural;
//! the Bevy adapter `wrench_aggregation_system` will pick the same to
//! match), the offset `r_parent_to_child` in the **parent's** wrench
//! frame, and `t_parent_child` is the parent→child rotation matrix in
//! the same convention as
//! [`crate::mass_body::MassPointState::t_parent_this`]. Outputs are in
//! the **parent's** wrench frame.
//!
//! Because the kernel only deals in vector arithmetic + a single rotation
//! and a single cross product, the "frame phantom" boundary lives at the
//! orchestration layer ([`crate::mass_storage::MassStorage`]-driven walk
//! in `jeod_sim`) and at the Bevy adapter; this module exposes a typed
//! sibling [`shift_wrench_to_parent_typed`] for callers who want to keep
//! the phantom-tagged [`Force`]/[`Torque`] around their boundary.

use core::marker::PhantomData;

use glam::{DMat3, DVec3};
use jeod_quantities::aliases::{Force, Torque};
use jeod_quantities::frame::Frame;

/// Single-link parallel-axis wrench shift.
///
/// Given a child's wrench (`f_child`, `tau_child` — both expressed in the
/// child's wrench frame, with the torque referenced about the child's
/// composite center of mass), and the parent-frame offset
/// `r_parent_to_child` from the parent's reference point to the child's
/// reference point, return the contribution to the parent's wrench
/// expressed in the parent's wrench frame and referenced about the
/// parent's composite CoM:
///
/// ```text
/// f_parent   = t_parent_child^T · f_child
/// tau_parent = t_parent_child^T · tau_child  +  r_parent_to_child × f_parent
/// ```
///
/// `t_parent_child` is the rotation that takes parent-frame vector
/// components to child-frame components (i.e.
/// `v_child = t_parent_child · v_parent`); the transpose rotates
/// vectors back from child to parent. This matches JEOD's
/// `MassPointState::T_parent_this` and is the same convention the
/// arena's [`crate::mass_storage::compute_node_composite`] already
/// uses for child inertia rotation.
///
/// Identities:
/// - With `r_parent_to_child = 0` and `t_parent_child = I`, the output
///   is the input verbatim (no shift, no rotation).
/// - The map is linear in `(f_child, tau_child)`, so summing over
///   children and shifting commute — every consumer can sum siblings
///   first then shift, or shift each then sum, and get bit-equal
///   results modulo floating-point order. The orchestration layer in
///   `jeod_sim` shifts each child individually because that's what
///   the post-order [`crate::mass_storage::recompute_composites_via_storage`]
///   walk already does.
// JEOD_INV: DB.16 — child forces propagated to parent recursively (parallel-axis arm)
#[inline]
pub fn shift_wrench_to_parent(
    f_child: DVec3,
    tau_child: DVec3,
    r_parent_to_child: DVec3,
    t_parent_child: DMat3,
) -> (DVec3, DVec3) {
    let t_t = t_parent_child.transpose();
    let f_parent = t_t * f_child;
    let tau_parent = t_t * tau_child + r_parent_to_child.cross(f_parent);
    (f_parent, tau_parent)
}

/// Typed sibling of [`shift_wrench_to_parent`].
///
/// Operates on phantom-tagged [`Force<F>`] / [`Torque<F>`] pairs in the
/// parent's wrench frame `F`. The child's wrench is also phantom-tagged
/// `F` because the orchestration layer applies the child→parent
/// rotation *before* lifting back into the typed surface — i.e. by the
/// time `Force<F>` / `Torque<F>` exist on the child side, `F` is
/// already the parent's frame. This boundary mirrors the typed surface
/// of the existing collection kernel ([`crate::forces::TotalForceTyped`])
/// and keeps phantom propagation linear.
///
/// For consumers that want true cross-frame phantoms (`Force<ChildFrame>`
/// → `Force<ParentFrame>`), drop into [`shift_wrench_to_parent`] at the
/// boundary and re-wrap, the same pattern every other typed-sibling in
/// `jeod_dynamics::forces` uses (`raw_si()` / `from_raw_si`).
#[inline]
pub fn shift_wrench_to_parent_typed<F: Frame>(
    f_child: Force<F>,
    tau_child: Torque<F>,
    r_parent_to_child: DVec3,
    t_parent_child: DMat3,
) -> (Force<F>, Torque<F>) {
    let (f, t) = shift_wrench_to_parent(
        f_child.raw_si(),
        tau_child.raw_si(),
        r_parent_to_child,
        t_parent_child,
    );
    (Force::<F>::from_raw_si(f), Torque::<F>::from_raw_si(t))
}

/// Plain `(force, torque)` pair in some implicit frame, used by the
/// orchestration walk to stage the per-node accumulator before
/// writing back into the storage layer. The frame discipline lives at
/// the caller — by the time a [`Wrench`] crosses module boundaries,
/// the orchestration layer (`jeod_sim::wrench`) tags it with the
/// owning entity's structural frame.
///
/// Kept in `jeod_dynamics` (not `jeod_quantities`) because:
/// - it is not a typed quantity in the `jeod_quantities` sense (no
///   independent unit / phantom dimension),
/// - it is the kernel-side staging shape for an *intermediate* sum
///   in JEOD's upward walk, not a typed mission-facing surface.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Wrench {
    /// Force component (N).
    pub force: DVec3,
    /// Torque component (N·m).
    pub torque: DVec3,
    /// Phantom anchor for future migration to a phantom-tagged
    /// `Wrench<F>` once the orchestration layer is ready to expose
    /// per-node frame phantoms. Kept private so adding the phantom
    /// later is a non-breaking change.
    _frame: PhantomData<()>,
}

impl Wrench {
    /// Construct a wrench from raw components.
    #[inline]
    pub const fn new(force: DVec3, torque: DVec3) -> Self {
        Self {
            force,
            torque,
            _frame: PhantomData,
        }
    }

    /// Identity wrench (zero force, zero torque).
    #[inline]
    pub const fn zero() -> Self {
        Self::new(DVec3::ZERO, DVec3::ZERO)
    }

    /// Shift this wrench from the child's frame to the parent's frame
    /// via [`shift_wrench_to_parent`].
    #[inline]
    pub fn shift_to_parent(self, r_parent_to_child: DVec3, t_parent_child: DMat3) -> Self {
        let (f, t) =
            shift_wrench_to_parent(self.force, self.torque, r_parent_to_child, t_parent_child);
        Self::new(f, t)
    }

    /// Add another wrench's components in-place (componentwise).
    #[inline]
    pub fn accumulate(&mut self, other: Wrench) {
        self.force += other.force;
        self.torque += other.torque;
    }
}

impl core::ops::Add for Wrench {
    type Output = Wrench;
    #[inline]
    fn add(self, rhs: Wrench) -> Wrench {
        Self::new(self.force + rhs.force, self.torque + rhs.torque)
    }
}

impl core::ops::AddAssign for Wrench {
    #[inline]
    fn add_assign(&mut self, rhs: Wrench) {
        self.accumulate(rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: DVec3, b: DVec3, tol: f64, label: &str) {
        let d = (a - b).length();
        assert!(d < tol, "{label}: |Δ|={d:.3e}, a={a:?}, b={b:?}");
    }

    #[test]
    fn identity_at_zero_offset_zero_rotation() {
        // No shift, no rotation — output equals input.
        let f = DVec3::new(1.0, 2.0, 3.0);
        let t = DVec3::new(-0.5, 0.25, 7.0);
        let (f_out, t_out) = shift_wrench_to_parent(f, t, DVec3::ZERO, DMat3::IDENTITY);
        assert_close(f_out, f, 1e-15, "force");
        assert_close(t_out, t, 1e-15, "torque");
    }

    #[test]
    fn pure_offset_creates_cross_term() {
        // Identity rotation, zero child torque, non-zero force at
        // an offset — the parent torque is the parallel-axis arm
        // r × F. Sign sanity: a force +x applied at offset +y
        // produces torque -z (right-hand rule).
        let f = DVec3::new(10.0, 0.0, 0.0);
        let r = DVec3::new(0.0, 1.0, 0.0);
        let (f_out, t_out) = shift_wrench_to_parent(f, DVec3::ZERO, r, DMat3::IDENTITY);
        assert_eq!(f_out, f);
        // r × F = (0,1,0) × (10,0,0) = (1·0 - 0·0, 0·10 - 0·0, 0·0 - 1·10) = (0,0,-10)
        assert_eq!(t_out, DVec3::new(0.0, 0.0, -10.0));
    }

    #[test]
    fn pure_rotation_no_offset_matches_jeod_convention() {
        // `t_parent_child` follows JEOD's `T_parent_this`: the
        // rotation that takes parent-frame components to child-frame
        // components, so `t_parent_child · v_parent = v_child` and
        // `t_parent_child^T · v_child = v_parent`.
        //
        // 90° rotation about +Z: child basis y-axis points along
        // parent +x, child basis x-axis points along parent -y.
        //   t_parent_child columns = (0,1,0), (-1,0,0), (0,0,1).
        // (Glam stores columns; `t_pc · (1,0,0)_parent = (0,1,0)_child`,
        // i.e., parent-x → child-+y, ✓ for this rotation.)
        let t_pc = DMat3::from_cols(
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        // Force in child frame along +x_child. Geometrically,
        // child's +x axis points along parent's -y, so the force in
        // parent coordinates should be (0, -5, 0).
        let f_child = DVec3::new(5.0, 0.0, 0.0);
        let (f_parent, _) = shift_wrench_to_parent(f_child, DVec3::ZERO, DVec3::ZERO, t_pc);
        assert_close(
            f_parent,
            DVec3::new(0.0, -5.0, 0.0),
            1e-15,
            "rot child→parent",
        );
    }

    #[test]
    fn additivity_over_siblings() {
        // Two child contributions can be summed first then shifted, or
        // shifted then summed, with bit-equal results modulo float
        // ordering. Pin the linearity so an orchestration refactor that
        // changes the order of operations doesn't introduce a silent
        // numerical regression.
        let f1 = DVec3::new(1.0, 2.0, 3.0);
        let t1 = DVec3::new(0.5, 0.0, -1.0);
        let f2 = DVec3::new(-2.0, 0.5, 4.0);
        let t2 = DVec3::new(0.0, 1.0, 0.5);
        let r = DVec3::new(0.1, -0.2, 0.3);
        // Identity rotation so we can compare numerically without
        // worrying about cumulative orthonormality drift in the
        // sum-first path.
        let t_pc = DMat3::IDENTITY;

        // Path A: shift then sum.
        let (a_f1, a_t1) = shift_wrench_to_parent(f1, t1, r, t_pc);
        let (a_f2, a_t2) = shift_wrench_to_parent(f2, t2, r, t_pc);
        let a_f = a_f1 + a_f2;
        let a_t = a_t1 + a_t2;

        // Path B: sum then shift.
        let (b_f, b_t) = shift_wrench_to_parent(f1 + f2, t1 + t2, r, t_pc);

        assert_close(a_f, b_f, 1e-15, "sum-then-shift force");
        assert_close(a_t, b_t, 1e-15, "sum-then-shift torque");
    }

    #[test]
    fn sign_pure_torque_no_force() {
        // A pure torque on a child, with offset and rotation, must
        // shift to the parent without a parallel-axis cross-term
        // (cross term is `r × f_parent`, and `f_parent = 0` when
        // `f_child = 0` regardless of the rotation).
        let tau = DVec3::new(1.0, -2.0, 3.0);
        let r = DVec3::new(5.0, 5.0, 5.0);
        let (f_p, tau_p) = shift_wrench_to_parent(DVec3::ZERO, tau, r, DMat3::IDENTITY);
        assert_eq!(f_p, DVec3::ZERO);
        assert_eq!(tau_p, tau);
    }

    #[test]
    fn typed_round_trip() {
        use jeod_quantities::frame::RootInertial;
        let f = Force::<RootInertial>::from_raw_si(DVec3::new(2.0, -1.0, 4.0));
        let tau = Torque::<RootInertial>::from_raw_si(DVec3::new(0.0, 0.5, -0.25));
        let r = DVec3::new(1.0, 0.0, 0.0);
        let (f_p, tau_p) = shift_wrench_to_parent_typed(f, tau, r, DMat3::IDENTITY);
        // With identity rotation, force unchanged.
        assert_eq!(f_p.raw_si(), f.raw_si());
        // Torque gains the cross-term (1,0,0) × (2,-1,4)
        // = (0·4 - 0·(-1), 0·2 - 1·4, 1·(-1) - 0·2) = (0, -4, -1).
        let expected_tau = tau.raw_si() + DVec3::new(0.0, -4.0, -1.0);
        assert_eq!(tau_p.raw_si(), expected_tau);
    }

    #[test]
    fn wrench_helpers_compose() {
        let w = Wrench::new(DVec3::new(1.0, 0.0, 0.0), DVec3::ZERO);
        let shifted = w.shift_to_parent(DVec3::new(0.0, 1.0, 0.0), DMat3::IDENTITY);
        // r × F = (0,1,0) × (1,0,0) = (0,0,-1)
        assert_eq!(shifted.force, DVec3::new(1.0, 0.0, 0.0));
        assert_eq!(shifted.torque, DVec3::new(0.0, 0.0, -1.0));

        let mut acc = Wrench::zero();
        acc.accumulate(shifted);
        acc += Wrench::new(DVec3::new(0.0, 1.0, 0.0), DVec3::new(0.0, 0.0, 0.5));
        assert_eq!(acc.force, DVec3::new(1.0, 1.0, 0.0));
        assert_eq!(acc.torque, DVec3::new(0.0, 0.0, -0.5));
    }
}
