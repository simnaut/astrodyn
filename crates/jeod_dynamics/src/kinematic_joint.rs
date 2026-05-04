//! Declaratively driven kinematic joints.
//!
//! Pure-math kernel for advancing a single-axis kinematic joint under a
//! prescribed angular rate. The joint frame's rotation about its parent
//! is `θ(t) = θ₀ + ω · t` about a fixed axis expressed in parent-frame
//! coordinates; the angular velocity in this-frame coordinates is the
//! constant `ω · axis_in_this`. Because the joint is a pure rotation
//! about a fixed axis (the axis is parallel to itself in both parent
//! and this frames — only its direction rotates with the joint), the
//! "axis in this-frame coordinates" is the same vector as
//! "axis in parent-frame coordinates" for the kinematic-velocity
//! component.
//!
//! "Kinematic" here means: the angle (and thus the rotation and
//! angular velocity of the joint frame) is an *input*, not an
//! integrated state. There is no torque, no inertia, no momentum
//! exchange — the consumer simply specifies the angular trajectory
//! and this kernel computes the corresponding `(rotation, angular
//! velocity)` snapshot. Joint dynamics (free-swinging joints under
//! torque, inverse dynamics, constraint-derived joint forces) are
//! explicitly out of scope here and tracked separately under the
//! deferred-dynamics meta.
//!
//! The kernel is the analog of `planet_fixed_rotation_system`'s spin
//! about the planet pole, generalized to an arbitrary axis declared
//! per joint. JEOD has no native joint primitive — articulated
//! sub-trees in JEOD are wired through `attach_to_frame` (kinematic
//! attachment to a non-body reference frame) plus mission-code
//! handlers that update the attached frame's rotation each step. This
//! kernel formalises that mission-code pattern as a first-class API
//! so a Bevy mission crate can declare "this joint rotates about Z at
//! 10 deg/s" without writing a per-mission kinematics driver.

use glam::DVec3;
use jeod_frames::{RefFrameRot, RefFrameState, RefFrameTrans};
use jeod_math::JeodQuat;

/// Declarative specification for a kinematically driven single-axis
/// joint.
///
/// The joint frame's rotation about its parent is
/// `θ(t) = initial_angle_rad + rate_rad_per_s · t`, applied about
/// `axis_in_parent` (a unit vector in the parent frame).
///
/// Sign convention follows JEOD's left-transformation quaternion:
/// `q.scalar = cos(θ/2)`, `q.vector = -sin(θ/2) · axis`. A positive
/// `rate_rad_per_s` therefore rotates a vector expressed in the parent
/// frame *negatively* about `axis_in_parent` when re-expressed in the
/// child (this-) frame, matching the rest of the codebase's
/// quaternion convention.
///
/// The angular velocity expressed in this-frame coordinates is
/// `rate_rad_per_s · axis_in_parent` (the rotation axis of a single-
/// axis joint is invariant under that same rotation, so the parent
/// and this representations of the *axis direction* coincide).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointKinematicsSpec {
    /// Unit vector in the parent frame about which the joint rotates.
    /// Caller must supply a unit-norm vector — the kernel asserts this
    /// in [`evaluate`] so a mis-spec fails loudly rather than producing
    /// silently scaled rotation.
    pub axis_in_parent: DVec3,
    /// Prescribed angular rate (rad/s). May be negative.
    pub rate_rad_per_s: f64,
    /// Joint angle at `t = 0` in radians.
    pub initial_angle_rad: f64,
}

/// Tolerance on `axis_in_parent.length_squared() - 1` accepted by
/// [`evaluate`].
///
/// The kernel thresholds the squared distance from unit norm
/// (`|‖a‖² − 1| < AXIS_NORM_TOL`) rather than the linear distance
/// `|‖a‖ − 1|` because the squared form is cheaper (no `sqrt`) and is
/// the standard pattern in `glam`/`nalgebra`. Near `‖a‖ = 1` the two
/// forms differ by a factor of two to first order
/// (`‖a‖² − 1 = (‖a‖ − 1)(‖a‖ + 1) ≈ 2(‖a‖ − 1)`), so a caller that
/// pre-validates against a linear `‖a‖ − 1` bound must square the
/// comparison or scale the threshold to match.
///
/// Loose enough that float-arithmetic round-off in user-constructed
/// axes (e.g. normalising a 3-vector once at vehicle setup) does not
/// fire; tight enough that an obviously non-unit input (e.g. a raw
/// `[0, 0, 2]`) panics. Same order of magnitude as
/// `jeod_math::quaternion::NORM_LIMIT` (the JEOD fast-path bound on
/// quaternion magnitude).
///
/// Exposed as part of the public surface so callers that pre-validate
/// axes upstream of [`evaluate`] can match the kernel's exact bound
/// rather than hard-coding a parallel literal that might drift.
pub const AXIS_NORM_TOL: f64 = 1.0e-6;

/// Evaluate the joint kinematics at elapsed time `t` (seconds since
/// the joint's `initial_angle_rad` reference).
///
/// Returns:
/// - the left-transformation quaternion `q_parent_this` that maps a
///   vector expressed in the parent frame into this-frame
///   coordinates;
/// - the angular velocity of this frame relative to its parent,
///   expressed in this-frame coordinates (rad/s).
///
/// # Panics
///
/// - `axis_in_parent` is not a unit vector (within
///   [`AXIS_NORM_TOL`]). A non-unit axis would silently scale the
///   resulting rotation angle and produce an angular velocity whose
///   magnitude no longer matches `rate_rad_per_s`.
/// - `t` is non-finite. Joint angle would propagate `NaN`/`±∞` into
///   downstream consumers.
/// - `rate_rad_per_s` or `initial_angle_rad` are non-finite, for the
///   same reason.
pub fn evaluate(spec: &JointKinematicsSpec, elapsed_seconds: f64) -> (JeodQuat, DVec3) {
    let axis_norm_sq = spec.axis_in_parent.length_squared();
    assert!(
        (axis_norm_sq - 1.0).abs() < AXIS_NORM_TOL,
        "JointKinematicsSpec.axis_in_parent must be a unit vector \
         (found length² = {axis_norm_sq}, axis = {:?}). Normalize the \
         axis once at vehicle-setup time, e.g. \
         `axis_in_parent: DVec3::Z` or `axis_in_parent: raw.normalize()`.",
        spec.axis_in_parent,
    );
    assert!(
        elapsed_seconds.is_finite(),
        "joint kinematics elapsed_seconds must be finite, got {elapsed_seconds}. \
         The simulation time has gone non-finite — fix the time-update path."
    );
    assert!(
        spec.rate_rad_per_s.is_finite(),
        "JointKinematicsSpec.rate_rad_per_s must be finite, got {}. \
         Replace the spec with a finite rate or remove the joint.",
        spec.rate_rad_per_s,
    );
    assert!(
        spec.initial_angle_rad.is_finite(),
        "JointKinematicsSpec.initial_angle_rad must be finite, got {}. \
         Replace the spec with a finite angle.",
        spec.initial_angle_rad,
    );

    let angle = spec.initial_angle_rad + spec.rate_rad_per_s * elapsed_seconds;
    let q_parent_this = JeodQuat::left_quat_from_eigen_rotation(angle, spec.axis_in_parent);
    // The rotation axis of a single-axis joint is invariant under that
    // same rotation, so the axis vector is identical in parent and
    // this-frame coordinates. The angular velocity in this-frame
    // coordinates is therefore `rate · axis`.
    let ang_vel_this = spec.axis_in_parent * spec.rate_rad_per_s;
    (q_parent_this, ang_vel_this)
}

// ===========================================================================
// Sinusoidal single-axis spec
// ===========================================================================

/// Declarative specification for a single-axis joint whose angle is a
/// sinusoidal function of time:
///
/// ```text
/// θ(t) = offset_rad + amplitude_rad · sin(omega_rad_per_s · t + phase_rad)
/// ```
///
/// Useful for periodic articulation — e.g., a pointing dither, a
/// cyclic gimbal sweep, a slosh-style oscillation about a nominal
/// position. The joint frame's rotation about its parent is the
/// left-transformation quaternion for `θ(t)` about `axis_in_parent`,
/// matching [`JointKinematicsSpec`]'s convention so consumers that
/// read frame triplets (the Bevy-side `FrameRotC` / `FrameAngVelC`
/// or the `RefFrameState` storage) see a uniform sign convention
/// regardless of which kinematic style drives the joint.
///
/// The instantaneous angular rate is the analytic derivative
/// `dθ/dt = amplitude_rad · omega_rad_per_s · cos(omega_rad_per_s · t + phase_rad)`,
/// and the angular velocity in this-frame coordinates is
/// `(dθ/dt) · axis_in_parent` (the rotation axis is the eigenvector
/// of the rotation, so it is invariant between parent and this-frame
/// coordinates).
///
/// `evaluate_sinusoidal` panics on a non-unit `axis_in_parent` or any
/// non-finite scalar, matching [`evaluate`]'s fail-loud contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SinusoidalJointKinematicsSpec {
    /// Unit vector in the parent frame about which the joint rotates.
    pub axis_in_parent: DVec3,
    /// Amplitude `A` of the sinusoid (rad). May be negative — the
    /// kernel does not alias `(-A, phase)` to `(A, phase + π)` because
    /// preserving the caller's literal parametrization is more useful
    /// for diagnostics than canonicalizing.
    pub amplitude_rad: f64,
    /// Angular frequency `ω` of the sinusoid (rad/s). May be negative;
    /// the angle's time-derivative carries the sign.
    pub omega_rad_per_s: f64,
    /// Phase `φ` of the sinusoid (rad). The angle at `t = 0` is
    /// `offset + A · sin(φ)`.
    pub phase_rad: f64,
    /// DC offset `θ₀` (rad). The sinusoid oscillates about this
    /// nominal angle.
    pub offset_rad: f64,
}

/// Evaluate a [`SinusoidalJointKinematicsSpec`] at elapsed time `t`
/// (seconds since the joint's reference epoch).
///
/// Returns `(q_parent_this, ang_vel_this)` matching [`evaluate`]'s
/// semantics — quaternion is the parent→this left-transformation, and
/// angular velocity is in this-frame coordinates.
///
/// # Panics
///
/// - `axis_in_parent` is not a unit vector (within [`AXIS_NORM_TOL`]).
/// - `t` or any spec scalar is non-finite.
pub fn evaluate_sinusoidal(
    spec: &SinusoidalJointKinematicsSpec,
    elapsed_seconds: f64,
) -> (JeodQuat, DVec3) {
    let axis_norm_sq = spec.axis_in_parent.length_squared();
    assert!(
        (axis_norm_sq - 1.0).abs() < AXIS_NORM_TOL,
        "SinusoidalJointKinematicsSpec.axis_in_parent must be a unit vector \
         (found length² = {axis_norm_sq}, axis = {:?}). Normalize the axis once \
         at vehicle-setup time, e.g. `axis_in_parent: DVec3::Z` or \
         `axis_in_parent: raw.normalize()`.",
        spec.axis_in_parent,
    );
    assert!(
        elapsed_seconds.is_finite(),
        "joint kinematics elapsed_seconds must be finite, got {elapsed_seconds}. \
         The simulation time has gone non-finite — fix the time-update path."
    );
    assert!(
        spec.amplitude_rad.is_finite(),
        "SinusoidalJointKinematicsSpec.amplitude_rad must be finite, got {}. \
         Replace the spec with a finite amplitude or remove the joint.",
        spec.amplitude_rad,
    );
    assert!(
        spec.omega_rad_per_s.is_finite(),
        "SinusoidalJointKinematicsSpec.omega_rad_per_s must be finite, got {}. \
         Replace the spec with a finite frequency or remove the joint.",
        spec.omega_rad_per_s,
    );
    assert!(
        spec.phase_rad.is_finite(),
        "SinusoidalJointKinematicsSpec.phase_rad must be finite, got {}. \
         Replace the spec with a finite phase.",
        spec.phase_rad,
    );
    assert!(
        spec.offset_rad.is_finite(),
        "SinusoidalJointKinematicsSpec.offset_rad must be finite, got {}. \
         Replace the spec with a finite offset.",
        spec.offset_rad,
    );

    let phase = spec.omega_rad_per_s * elapsed_seconds + spec.phase_rad;
    let angle = spec.offset_rad + spec.amplitude_rad * phase.sin();
    let rate = spec.amplitude_rad * spec.omega_rad_per_s * phase.cos();
    let q_parent_this = JeodQuat::left_quat_from_eigen_rotation(angle, spec.axis_in_parent);
    let ang_vel_this = spec.axis_in_parent * rate;
    (q_parent_this, ang_vel_this)
}

// ===========================================================================
// Closure (fixed-transform) spec
// ===========================================================================

/// Declarative specification for a joint that pins a fixed rotation
/// about a single axis with no time dependence — the kinematic-only
/// degenerate case useful for closing kinematic loops.
///
/// The joint frame's rotation about its parent is constant
/// `θ = fixed_angle_rad` about `axis_in_parent`; the angular velocity
/// is identically zero. Mission code reaches for this when a joint in
/// an articulated chain is *constrained* to a particular pose at
/// declaration time — e.g., the second arm of a closed kinematic loop
/// where the ring-closure constraint fixes the relative attitude
/// between the two arms — but the constraint resolution itself
/// (force-controlled or geometric) is not the kinematic driver's job.
///
/// `evaluate_closure` panics on a non-unit `axis_in_parent` or
/// non-finite `fixed_angle_rad`, matching the fail-loud contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosureJointKinematicsSpec {
    /// Unit vector in the parent frame about which the joint is rotated.
    pub axis_in_parent: DVec3,
    /// Fixed joint angle (rad). The joint frame is rotated by this
    /// angle about `axis_in_parent` for all time.
    pub fixed_angle_rad: f64,
}

/// Evaluate a [`ClosureJointKinematicsSpec`].
///
/// `elapsed_seconds` is ignored (the spec is time-invariant by
/// construction) but still validated as finite so a non-finite time
/// upstream surfaces here rather than silently propagating into a
/// later consumer.
///
/// Returns `(q_parent_this, DVec3::ZERO)` — the joint is stationary,
/// so its angular velocity in this-frame coordinates is identically
/// zero.
///
/// # Panics
///
/// - `axis_in_parent` is not a unit vector (within [`AXIS_NORM_TOL`]).
/// - `t` or `fixed_angle_rad` is non-finite.
pub fn evaluate_closure(
    spec: &ClosureJointKinematicsSpec,
    elapsed_seconds: f64,
) -> (JeodQuat, DVec3) {
    let axis_norm_sq = spec.axis_in_parent.length_squared();
    assert!(
        (axis_norm_sq - 1.0).abs() < AXIS_NORM_TOL,
        "ClosureJointKinematicsSpec.axis_in_parent must be a unit vector \
         (found length² = {axis_norm_sq}, axis = {:?}). Normalize the axis once \
         at vehicle-setup time.",
        spec.axis_in_parent,
    );
    assert!(
        elapsed_seconds.is_finite(),
        "joint kinematics elapsed_seconds must be finite, got {elapsed_seconds}. \
         The simulation time has gone non-finite — fix the time-update path."
    );
    assert!(
        spec.fixed_angle_rad.is_finite(),
        "ClosureJointKinematicsSpec.fixed_angle_rad must be finite, got {}. \
         Replace the spec with a finite angle.",
        spec.fixed_angle_rad,
    );

    let q_parent_this =
        JeodQuat::left_quat_from_eigen_rotation(spec.fixed_angle_rad, spec.axis_in_parent);
    (q_parent_this, DVec3::ZERO)
}

// ===========================================================================
// Multi-DOF spec (composed single-axis kinematic chain)
// ===========================================================================

/// Maximum number of DOFs a [`MultiDofJointKinematicsSpec`] can carry.
///
/// Sized to cover the realistic kinematic-only chains mission code
/// declares — solar-array gimbals (2 DOF), antenna pointing gimbals
/// (2-3 DOF), articulated robotic arms in their kinematic-driving
/// regime (up to 6 DOF per chain). The fixed bound preserves
/// `MultiDofJointKinematicsSpec: Copy`, which keeps its parent
/// `JointKinematicsModel` trivially copyable and avoids per-tick
/// allocation in the propagation path. Chains with more than 6 DOFs
/// can be expressed as multiple chained joint frames in the frame
/// tree (one frame per DOF, each carrying its own single-DOF spec) —
/// which is also the JEOD-faithful pattern, since JEOD has no native
/// multi-DOF joint primitive and chains articulation through
/// successive `attach_to_frame` calls.
pub const MAX_MULTI_DOF_AXES: usize = 6;

/// Per-DOF kinematic style for a single axis inside a
/// [`MultiDofJointKinematicsSpec`] chain.
///
/// Each axis i carries one of these variants. The variant determines
/// how the axis's joint angle evolves with time; all variants share
/// the "single axis with a unit `axis_in_parent`-ish direction" shape
/// (the axis lives in the *intermediate* frame produced by axes
/// `0..i`, not in the chain root's parent — see
/// [`evaluate_multi_dof`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SingleDofKinematics {
    /// Constant-rate rotation about `axis_in_parent`.
    /// `θ_i(t) = initial_angle_rad + rate_rad_per_s · t`.
    ConstantRate(JointKinematicsSpec),
    /// Sinusoidal rotation about `axis_in_parent`.
    /// `θ_i(t) = offset + amplitude · sin(ω · t + phase)`.
    Sinusoidal(SinusoidalJointKinematicsSpec),
    /// Time-invariant rotation about `axis_in_parent`. Useful as a
    /// constant offset in a multi-DOF chain (e.g., a fixed mounting
    /// twist between two articulated stages).
    Closure(ClosureJointKinematicsSpec),
}

impl SingleDofKinematics {
    /// Evaluate this DOF at `elapsed_seconds`, returning
    /// `(q, ang_vel_this_axis_only)` — the rotation contribution and
    /// the angular-velocity contribution of *this axis alone* in the
    /// frame produced after this axis rotates. Composition with
    /// preceding axes is the caller's responsibility (see
    /// [`evaluate_multi_dof`]).
    fn evaluate(&self, elapsed_seconds: f64) -> (JeodQuat, DVec3) {
        match self {
            SingleDofKinematics::ConstantRate(spec) => evaluate(spec, elapsed_seconds),
            SingleDofKinematics::Sinusoidal(spec) => evaluate_sinusoidal(spec, elapsed_seconds),
            SingleDofKinematics::Closure(spec) => evaluate_closure(spec, elapsed_seconds),
        }
    }
}

/// Declarative specification for a multi-DOF kinematic joint.
///
/// A multi-DOF joint composes up to [`MAX_MULTI_DOF_AXES`] single-axis
/// kinematic stages. The semantics match a Denavit–Hartenberg–style
/// kinematic chain *with no link offsets*: each stage rotates about
/// its `axis_in_parent` declared in the *intermediate frame produced
/// by the preceding stages*, and the composed `(rotation, angular
/// velocity)` is the chain root → chain leaf state.
///
/// **Stage ordering**: stage 0 is closest to the chain's parent
/// frame; stage `n-1` is closest to the chain's leaf frame. Filled
/// stages must be a contiguous prefix — `axes[0..count]` are
/// `Some(_)`, `axes[count..]` are `None`. The kernel asserts this; a
/// "hole" in the middle is a configuration error.
///
/// **Composition** uses [`RefFrameState::incr_right`] (the same math
/// the rest of the codebase walks for hierarchy traversals). Each
/// stage's contribution is wrapped as a translation-free
/// `RefFrameState` and folded left-to-right, so the math is
/// bit-identical to a frame-tree walk through `n` joint entities each
/// carrying a single-DOF spec — a deliberate equivalence that lets
/// mission code pick whichever shape (one entity / N stages, or N
/// entities / 1 stage each) is more ergonomic without changing the
/// physics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiDofJointKinematicsSpec {
    /// Per-DOF specs, contiguous-prefix-filled. `axes[0]` is closest
    /// to the chain's parent; the last `Some(_)` is closest to the
    /// chain's leaf.
    pub axes: [Option<SingleDofKinematics>; MAX_MULTI_DOF_AXES],
}

impl MultiDofJointKinematicsSpec {
    /// Build a spec from a slice of per-DOF specs. Asserts the slice
    /// length is `<= MAX_MULTI_DOF_AXES` so a caller that tries to
    /// declare a chain longer than the kernel supports fails loudly
    /// at construction rather than at evaluation.
    ///
    /// # Panics
    ///
    /// `dofs.len() > MAX_MULTI_DOF_AXES`.
    pub fn from_slice(dofs: &[SingleDofKinematics]) -> Self {
        assert!(
            dofs.len() <= MAX_MULTI_DOF_AXES,
            "MultiDofJointKinematicsSpec supports at most {} DOFs, got {}. \
             Split the chain into multiple chained joint frame entities, \
             one per DOF, so the frame-tree walk produces the same composed \
             state.",
            MAX_MULTI_DOF_AXES,
            dofs.len(),
        );
        let mut axes = [None; MAX_MULTI_DOF_AXES];
        for (i, dof) in dofs.iter().enumerate() {
            axes[i] = Some(*dof);
        }
        Self { axes }
    }

    /// Number of populated DOFs (the contiguous-prefix count of
    /// `Some(_)` entries).
    pub fn count(&self) -> usize {
        self.axes.iter().take_while(|a| a.is_some()).count()
    }
}

/// Evaluate a [`MultiDofJointKinematicsSpec`] at elapsed time `t`.
///
/// Returns `(q_parent_this, ang_vel_this)` for the composed chain
/// root → chain leaf, matching the convention of [`evaluate`] and
/// [`evaluate_sinusoidal`]: quaternion is the parent→this left-
/// transformation, angular velocity is in this-frame coordinates.
///
/// Stages are folded with [`RefFrameState::incr_right`] so the math
/// is bit-identical to walking the equivalent N-entity chain through
/// the frame tree. A spec with zero populated DOFs returns
/// `(identity, zero)` — the no-op kinematic chain.
///
/// # Panics
///
/// - `elapsed_seconds` is non-finite (`NaN` or `±∞`). The empty-chain
///   fast path returns `(identity, zero)` without invoking any
///   per-stage evaluator, so without this top-level guard a non-finite
///   clock could silently propagate through the no-op chain and
///   violate the fail-loud contract that the other joint evaluators
///   uphold.
/// - The populated DOFs are not a contiguous prefix
///   (`axes[i]` is `None` followed by some `axes[j > i]` that is
///   `Some(_)`). A hole in the middle of the chain is a configuration
///   error.
/// - Any underlying single-DOF eval panics (non-unit axis, non-finite
///   scalars).
pub fn evaluate_multi_dof(
    spec: &MultiDofJointKinematicsSpec,
    elapsed_seconds: f64,
) -> (JeodQuat, DVec3) {
    // Top-level finite-time guard. The per-stage evaluators each
    // assert `elapsed_seconds.is_finite()`, but for a chain with zero
    // populated DOFs the loop below never enters and a non-finite
    // clock would silently fall through to the `(identity, zero)`
    // return. Asserting here makes the empty-chain fast path
    // fail-loud, matching the contract the populated chain inherits
    // from the per-stage evaluators.
    assert!(
        elapsed_seconds.is_finite(),
        "joint kinematics elapsed_seconds must be finite, got {elapsed_seconds}. \
         The simulation time has gone non-finite — fix the time-update path."
    );

    // Validate prefix-filled invariant: once we see a `None`, every
    // subsequent slot must also be `None`. A hole in the middle would
    // silently drop a stage from the composition without diagnostic.
    let mut seen_gap = false;
    for (i, slot) in spec.axes.iter().enumerate() {
        match slot {
            Some(_) => {
                assert!(
                    !seen_gap,
                    "MultiDofJointKinematicsSpec.axes must be a contiguous \
                     prefix of `Some(_)` followed by `None`, but axes[{i}] is \
                     populated after a hole. Pack the populated stages into \
                     axes[0..count].",
                );
            }
            None => {
                seen_gap = true;
            }
        }
    }

    // Fold from chain parent toward chain leaf using `incr_right`. The
    // accumulator is the chain-root → stage-i state; for each stage,
    // we wrap the per-stage `(q, ang_vel)` as a translation-free
    // RefFrameState and compose. This produces the chain-root →
    // chain-leaf state in bit-identical arithmetic to walking the
    // equivalent N-entity frame chain.
    let mut acc = RefFrameState {
        trans: RefFrameTrans {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        },
        rot: RefFrameRot {
            q_parent_this: JeodQuat::identity(),
            t_parent_this: JeodQuat::identity().left_quat_to_transformation(),
            ang_vel_this: DVec3::ZERO,
        },
    };
    for slot in spec.axes.iter() {
        let Some(dof) = slot else { break };
        let (q_stage, av_stage) = dof.evaluate(elapsed_seconds);
        let stage_state = RefFrameState {
            trans: RefFrameTrans {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
            rot: RefFrameRot {
                q_parent_this: q_stage,
                t_parent_this: q_stage.left_quat_to_transformation(),
                ang_vel_this: av_stage,
            },
        };
        acc = acc.incr_right(&stage_state);
    }
    (acc.rot.q_parent_this, acc.rot.ang_vel_this)
}

// ===========================================================================
// Unifying enum
// ===========================================================================

/// Unifying enum over the kinematic-only joint specifications shipped
/// today: constant-rate single-axis ([`JointKinematicsSpec`]),
/// sinusoidal single-axis ([`SinusoidalJointKinematicsSpec`]),
/// closure single-axis ([`ClosureJointKinematicsSpec`]), and
/// multi-DOF chains ([`MultiDofJointKinematicsSpec`]).
///
/// Mission code that needs to store a heterogeneous collection of
/// joints (or that passes the joint spec across abstractions that
/// don't know which variant they hold) reaches for this enum;
/// homogeneous collections of one variant are better expressed as
/// the variant type directly.
//
// `large_enum_variant`: `MultiDofJointKinematicsSpec` is ~384 bytes
// (six fixed slots), the other variants are <56 bytes. Boxing the
// large variant would buy the enum size back at the cost of a heap
// allocation per joint and the loss of `Copy` on the enum and on
// every component / spec collection that holds it. The kinematic-
// joint code path is intentionally `Copy`-friendly for per-tick
// fast-path evaluation, so the size asymmetry is the deliberate
// trade.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointKinematicsModel {
    /// Constant-rate single-axis joint.
    ConstantRate(JointKinematicsSpec),
    /// Sinusoidal single-axis joint.
    Sinusoidal(SinusoidalJointKinematicsSpec),
    /// Closure (fixed-angle, no time dependence) single-axis joint.
    Closure(ClosureJointKinematicsSpec),
    /// Multi-DOF kinematic chain.
    MultiDof(MultiDofJointKinematicsSpec),
}

/// Evaluate a [`JointKinematicsModel`] at elapsed time `t`, returning
/// `(q_parent_this, ang_vel_this)` for whichever variant is held.
pub fn evaluate_model(model: &JointKinematicsModel, elapsed_seconds: f64) -> (JeodQuat, DVec3) {
    match model {
        JointKinematicsModel::ConstantRate(spec) => evaluate(spec, elapsed_seconds),
        JointKinematicsModel::Sinusoidal(spec) => evaluate_sinusoidal(spec, elapsed_seconds),
        JointKinematicsModel::Closure(spec) => evaluate_closure(spec, elapsed_seconds),
        JointKinematicsModel::MultiDof(spec) => evaluate_multi_dof(spec, elapsed_seconds),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Identity case: zero rate, zero initial angle ⇒ identity rotation
    /// and zero angular velocity at any time.
    #[test]
    fn evaluate_zero_rate_zero_initial_angle_is_identity() {
        let spec = JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: 0.0,
            initial_angle_rad: 0.0,
        };
        let (q, omega) = evaluate(&spec, 12_345.678_9);
        assert_eq!(q, JeodQuat::identity());
        assert_eq!(omega, DVec3::ZERO);
    }

    /// Closed-form: angle(t) = initial + rate * t. A 10 deg/s joint
    /// starting at 0 ° hits π/2 after π/(2·rate) seconds.
    #[test]
    fn evaluate_constant_rate_quarter_turn_about_z() {
        let rate = 10.0_f64.to_radians();
        let spec = JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: rate,
            initial_angle_rad: 0.0,
        };
        let t_quarter = (PI / 2.0) / rate;
        let (q, _omega) = evaluate(&spec, t_quarter);
        let expected = JeodQuat::left_quat_from_eigen_rotation(PI / 2.0, DVec3::Z);
        // Bit-identity expected (same closed-form expression, same axis).
        assert_eq!(q, expected);
    }

    /// Angular velocity is `rate · axis` (in this-frame coordinates).
    /// Sign of `rate` flips the angular-velocity vector.
    #[test]
    fn evaluate_angular_velocity_tracks_rate_and_axis() {
        let axis = DVec3::Y;
        for rate in &[-3.5, 0.0, 1.25] {
            let spec = JointKinematicsSpec {
                axis_in_parent: axis,
                rate_rad_per_s: *rate,
                initial_angle_rad: 0.0,
            };
            let (_q, omega) = evaluate(&spec, 0.0);
            assert_eq!(omega, axis * *rate);
            // And again at a non-zero time — angular velocity is constant
            // for a constant-rate joint.
            let (_q2, omega2) = evaluate(&spec, 7.5);
            assert_eq!(omega2, axis * *rate);
        }
    }

    /// `initial_angle_rad` shifts the angle at `t = 0`.
    #[test]
    fn evaluate_respects_initial_angle() {
        let theta0 = 0.4;
        let spec = JointKinematicsSpec {
            axis_in_parent: DVec3::X,
            rate_rad_per_s: 0.0,
            initial_angle_rad: theta0,
        };
        let (q, _omega) = evaluate(&spec, 0.0);
        let expected = JeodQuat::left_quat_from_eigen_rotation(theta0, DVec3::X);
        assert_eq!(q, expected);
    }

    /// JEOD left-quat sign convention: the vector part is
    /// `-sin(θ/2)·axis`, so a positive rotation about `+Z` produces a
    /// quaternion with negative `z` component.
    #[test]
    fn evaluate_left_quat_sign_convention() {
        let spec = JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: 1.0,
            initial_angle_rad: 0.0,
        };
        let (q, _omega) = evaluate(&spec, 1.0); // angle = 1 rad
        let half = 0.5_f64;
        let scalar = half.cos();
        let zv = -half.sin();
        let dq = q.to_glam();
        assert!((dq.w - scalar).abs() < 1.0e-15);
        assert!((dq.z - zv).abs() < 1.0e-15);
    }

    /// Non-unit axis must panic — silently rescaling the rotation
    /// angle would produce a wrong physics answer with no diagnostic.
    #[test]
    #[should_panic(expected = "must be a unit vector")]
    fn evaluate_panics_on_non_unit_axis() {
        let spec = JointKinematicsSpec {
            axis_in_parent: DVec3::new(0.0, 0.0, 2.0),
            rate_rad_per_s: 1.0,
            initial_angle_rad: 0.0,
        };
        let _ = evaluate(&spec, 0.0);
    }

    #[test]
    #[should_panic(expected = "elapsed_seconds must be finite")]
    fn evaluate_panics_on_nan_elapsed() {
        let spec = JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: 1.0,
            initial_angle_rad: 0.0,
        };
        let _ = evaluate(&spec, f64::NAN);
    }

    #[test]
    #[should_panic(expected = "rate_rad_per_s must be finite")]
    fn evaluate_panics_on_nan_rate() {
        let spec = JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: f64::NAN,
            initial_angle_rad: 0.0,
        };
        let _ = evaluate(&spec, 0.0);
    }

    #[test]
    #[should_panic(expected = "initial_angle_rad must be finite")]
    fn evaluate_panics_on_nan_initial_angle() {
        let spec = JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: 0.0,
            initial_angle_rad: f64::INFINITY,
        };
        let _ = evaluate(&spec, 0.0);
    }

    /// Round-trip: the returned quaternion's transformation matrix
    /// applied to `axis_in_parent` returns the same axis (rotation
    /// about the axis fixes the axis).
    #[test]
    fn evaluate_axis_is_rotation_eigenvector() {
        let axis = DVec3::new(1.0, 1.0, 1.0).normalize();
        let spec = JointKinematicsSpec {
            axis_in_parent: axis,
            rate_rad_per_s: 0.7,
            initial_angle_rad: 0.2,
        };
        let (q, _omega) = evaluate(&spec, 3.0);
        let mat = q.left_quat_to_transformation();
        let rotated = mat * axis;
        for i in 0..3 {
            assert!(
                (rotated[i] - axis[i]).abs() < 1.0e-12,
                "axis component {i}: rotated={} axis={}",
                rotated[i],
                axis[i],
            );
        }
    }

    // ===========================================================================
    // Sinusoidal-spec tests
    // ===========================================================================

    /// At `t = 0` the angle is `offset + amplitude · sin(phase)` and the
    /// rate is `amplitude · omega · cos(phase)`. Pinning numeric values
    /// against the closed form catches sign / parameter-ordering bugs.
    #[test]
    fn sinusoidal_evaluates_at_t_zero_to_closed_form() {
        let spec = SinusoidalJointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            amplitude_rad: 0.4,
            omega_rad_per_s: 1.5,
            phase_rad: PI / 3.0,
            offset_rad: 0.1,
        };
        let (q, omega) = evaluate_sinusoidal(&spec, 0.0);
        let expected_angle = 0.1 + 0.4 * (PI / 3.0).sin();
        let expected_rate = 0.4 * 1.5 * (PI / 3.0).cos();
        let expected_q = JeodQuat::left_quat_from_eigen_rotation(expected_angle, DVec3::Z);
        for i in 0..4 {
            assert!(
                (q.data[i] - expected_q.data[i]).abs() < 1.0e-15,
                "q[{i}]: got {}, expected {}",
                q.data[i],
                expected_q.data[i]
            );
        }
        assert!((omega.x - 0.0).abs() < 1.0e-15);
        assert!((omega.y - 0.0).abs() < 1.0e-15);
        assert!((omega.z - expected_rate).abs() < 1.0e-15);
    }

    /// Sinusoidal pure-sin (offset=0, phase=0): at the quarter period
    /// `t = π/(2ω)` the angle equals the amplitude and the rate is
    /// zero (peak of the sinusoid). Different snapshot from `t = 0`,
    /// so any sign / parameter swap surfaces here even if the t=0
    /// case happens to land on a coincidence.
    #[test]
    fn sinusoidal_quarter_period_peak_angle_zero_rate() {
        let amplitude = 0.7_f64;
        let omega = 2.0_f64;
        let spec = SinusoidalJointKinematicsSpec {
            axis_in_parent: DVec3::Y,
            amplitude_rad: amplitude,
            omega_rad_per_s: omega,
            phase_rad: 0.0,
            offset_rad: 0.0,
        };
        let t_peak = PI / (2.0 * omega);
        let (q, ang_vel) = evaluate_sinusoidal(&spec, t_peak);
        let expected_q = JeodQuat::left_quat_from_eigen_rotation(amplitude, DVec3::Y);
        for i in 0..4 {
            assert!(
                (q.data[i] - expected_q.data[i]).abs() < 1.0e-12,
                "q[{i}]: got {}, expected {}",
                q.data[i],
                expected_q.data[i]
            );
        }
        // cos(π/2) ≈ 0, so the rate (and hence the angular velocity)
        // is zero at the peak — within float-trig precision.
        for i in 0..3 {
            assert!(
                ang_vel[i].abs() < 1.0e-12,
                "ang_vel component {i}: got {}, expected 0",
                ang_vel[i]
            );
        }
    }

    /// Sinusoidal with `amplitude = 0` reduces to a constant `offset`
    /// for all time, with zero angular velocity. This is the
    /// "stationary at non-zero angle" degenerate case.
    #[test]
    fn sinusoidal_zero_amplitude_is_constant_offset() {
        let spec = SinusoidalJointKinematicsSpec {
            axis_in_parent: DVec3::X,
            amplitude_rad: 0.0,
            omega_rad_per_s: 1.0,
            phase_rad: 1.5,
            offset_rad: 0.3,
        };
        let expected_q = JeodQuat::left_quat_from_eigen_rotation(0.3, DVec3::X);
        for &t in &[0.0_f64, 1.0, 7.5, 100.0] {
            let (q, ang_vel) = evaluate_sinusoidal(&spec, t);
            for i in 0..4 {
                assert!(
                    (q.data[i] - expected_q.data[i]).abs() < 1.0e-15,
                    "t={t} q[{i}]: got {}, expected {}",
                    q.data[i],
                    expected_q.data[i]
                );
            }
            assert_eq!(ang_vel, DVec3::ZERO, "t={t}: ang_vel must be zero");
        }
    }

    #[test]
    #[should_panic(expected = "must be a unit vector")]
    fn sinusoidal_panics_on_non_unit_axis() {
        let spec = SinusoidalJointKinematicsSpec {
            axis_in_parent: DVec3::new(0.0, 0.0, 2.0),
            amplitude_rad: 0.1,
            omega_rad_per_s: 1.0,
            phase_rad: 0.0,
            offset_rad: 0.0,
        };
        let _ = evaluate_sinusoidal(&spec, 0.0);
    }

    #[test]
    #[should_panic(expected = "amplitude_rad must be finite")]
    fn sinusoidal_panics_on_nan_amplitude() {
        let spec = SinusoidalJointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            amplitude_rad: f64::NAN,
            omega_rad_per_s: 1.0,
            phase_rad: 0.0,
            offset_rad: 0.0,
        };
        let _ = evaluate_sinusoidal(&spec, 0.0);
    }

    #[test]
    #[should_panic(expected = "omega_rad_per_s must be finite")]
    fn sinusoidal_panics_on_inf_omega() {
        let spec = SinusoidalJointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            amplitude_rad: 0.1,
            omega_rad_per_s: f64::INFINITY,
            phase_rad: 0.0,
            offset_rad: 0.0,
        };
        let _ = evaluate_sinusoidal(&spec, 0.0);
    }

    // ===========================================================================
    // Closure-spec tests
    // ===========================================================================

    /// The transform a closure spec produces is constant over time —
    /// asserted by sampling at multiple distinct elapsed times and
    /// requiring bit-identical output (closed-form input means the
    /// same bits come out at every t).
    #[test]
    fn closure_transform_is_time_invariant() {
        let spec = ClosureJointKinematicsSpec {
            axis_in_parent: DVec3::new(1.0, 1.0, 1.0).normalize(),
            fixed_angle_rad: 0.42,
        };
        let (q0, av0) = evaluate_closure(&spec, 0.0);
        for &t in &[1.0_f64, 5.5, 100.0, 1.0e6] {
            let (q, av) = evaluate_closure(&spec, t);
            for i in 0..4 {
                assert_eq!(
                    q.data[i].to_bits(),
                    q0.data[i].to_bits(),
                    "t={t} q[{i}]: not bit-identical"
                );
            }
            assert_eq!(av, av0);
            assert_eq!(av, DVec3::ZERO);
        }
    }

    /// Closure spec with a known angle should produce the matching
    /// left-quat about the requested axis, end-to-end sanity check.
    #[test]
    fn closure_produces_expected_quaternion() {
        let spec = ClosureJointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            fixed_angle_rad: PI / 4.0,
        };
        let (q, av) = evaluate_closure(&spec, 0.0);
        let expected = JeodQuat::left_quat_from_eigen_rotation(PI / 4.0, DVec3::Z);
        assert_eq!(q, expected);
        assert_eq!(av, DVec3::ZERO);
    }

    #[test]
    #[should_panic(expected = "must be a unit vector")]
    fn closure_panics_on_non_unit_axis() {
        let spec = ClosureJointKinematicsSpec {
            axis_in_parent: DVec3::new(0.0, 0.0, 2.0),
            fixed_angle_rad: 0.5,
        };
        let _ = evaluate_closure(&spec, 0.0);
    }

    #[test]
    #[should_panic(expected = "fixed_angle_rad must be finite")]
    fn closure_panics_on_nan_angle() {
        let spec = ClosureJointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            fixed_angle_rad: f64::NAN,
        };
        let _ = evaluate_closure(&spec, 0.0);
    }

    // ===========================================================================
    // Multi-DOF spec tests
    // ===========================================================================

    /// An empty multi-DOF chain (no populated DOFs) reduces to the
    /// no-op identity transform with zero angular velocity.
    #[test]
    fn multi_dof_empty_chain_is_identity() {
        let spec = MultiDofJointKinematicsSpec::from_slice(&[]);
        assert_eq!(spec.count(), 0);
        let (q, av) = evaluate_multi_dof(&spec, 5.0);
        assert_eq!(q, JeodQuat::identity());
        assert_eq!(av, DVec3::ZERO);
    }

    /// A 1-DOF chain matches the underlying single-axis evaluator
    /// bit-for-bit — composing one stage with an identity accumulator
    /// must not introduce any new arithmetic.
    #[test]
    fn multi_dof_single_stage_matches_constant_rate_kernel() {
        let inner = JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: 0.3,
            initial_angle_rad: 0.1,
        };
        let spec =
            MultiDofJointKinematicsSpec::from_slice(&[SingleDofKinematics::ConstantRate(inner)]);
        for &t in &[0.0_f64, 1.0, 10.0] {
            let (q_chain, av_chain) = evaluate_multi_dof(&spec, t);
            let (q_single, av_single) = evaluate(&inner, t);
            // `incr_right` re-derives `t_parent_this` from the
            // normalized quaternion, which is the same closed-form
            // expression as the single-axis kernel produces — the
            // resulting quaternion is bit-identical (a single stage
            // composed with identity is the stage itself).
            for i in 0..4 {
                assert!(
                    (q_chain.data[i] - q_single.data[i]).abs() < 1.0e-15,
                    "t={t} q[{i}]: chain {} vs single {}",
                    q_chain.data[i],
                    q_single.data[i]
                );
            }
            for i in 0..3 {
                assert!(
                    (av_chain[i] - av_single[i]).abs() < 1.0e-15,
                    "t={t} av[{i}]: chain {} vs single {}",
                    av_chain[i],
                    av_single[i]
                );
            }
        }
    }

    /// 2-DOF chain: stage 0 is constant-rate about +Z, stage 1 is
    /// sinusoidal about +Y. Asserts the composition matches manual
    /// `incr_right` of the two stages — guarding against any silent
    /// reordering or missing stage in the kernel.
    #[test]
    fn multi_dof_two_stage_matches_manual_incr_right() {
        let stage0 = JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: 0.5,
            initial_angle_rad: 0.2,
        };
        let stage1 = SinusoidalJointKinematicsSpec {
            axis_in_parent: DVec3::Y,
            amplitude_rad: 0.3,
            omega_rad_per_s: 1.2,
            phase_rad: 0.4,
            offset_rad: 0.05,
        };
        let spec = MultiDofJointKinematicsSpec::from_slice(&[
            SingleDofKinematics::ConstantRate(stage0),
            SingleDofKinematics::Sinusoidal(stage1),
        ]);

        let t = 2.5_f64;
        let (q_chain, av_chain) = evaluate_multi_dof(&spec, t);

        // Manual composition: identity -> stage0 -> stage1.
        let (q0, av0) = evaluate(&stage0, t);
        let s0 = RefFrameState {
            trans: RefFrameTrans {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
            rot: RefFrameRot {
                q_parent_this: q0,
                t_parent_this: q0.left_quat_to_transformation(),
                ang_vel_this: av0,
            },
        };
        let (q1, av1) = evaluate_sinusoidal(&stage1, t);
        let s1 = RefFrameState {
            trans: RefFrameTrans {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
            rot: RefFrameRot {
                q_parent_this: q1,
                t_parent_this: q1.left_quat_to_transformation(),
                ang_vel_this: av1,
            },
        };
        let identity = RefFrameState {
            trans: RefFrameTrans {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
            rot: RefFrameRot {
                q_parent_this: JeodQuat::identity(),
                t_parent_this: JeodQuat::identity().left_quat_to_transformation(),
                ang_vel_this: DVec3::ZERO,
            },
        };
        let expected = identity.incr_right(&s0).incr_right(&s1);

        for i in 0..4 {
            assert!(
                (q_chain.data[i] - expected.rot.q_parent_this.data[i]).abs() < 1.0e-15,
                "q[{i}]: chain {} vs manual {}",
                q_chain.data[i],
                expected.rot.q_parent_this.data[i]
            );
        }
        for i in 0..3 {
            assert!(
                (av_chain[i] - expected.rot.ang_vel_this[i]).abs() < 1.0e-15,
                "av[{i}]: chain {} vs manual {}",
                av_chain[i],
                expected.rot.ang_vel_this[i]
            );
        }
    }

    /// Two stages about the *same axis* commute — their composition
    /// is equivalent to a single stage whose angle is the sum and
    /// whose rate is the sum. This is the cleanest correctness probe
    /// for the multi-DOF evaluator: same axis, no rotation
    /// inter-leaving, closed-form predictable answer.
    #[test]
    fn multi_dof_same_axis_stages_sum_to_aggregate() {
        let axis = DVec3::Z;
        let stage_a = JointKinematicsSpec {
            axis_in_parent: axis,
            rate_rad_per_s: 0.4,
            initial_angle_rad: 0.1,
        };
        let stage_b = JointKinematicsSpec {
            axis_in_parent: axis,
            rate_rad_per_s: 0.7,
            initial_angle_rad: 0.2,
        };
        let chain = MultiDofJointKinematicsSpec::from_slice(&[
            SingleDofKinematics::ConstantRate(stage_a),
            SingleDofKinematics::ConstantRate(stage_b),
        ]);
        let aggregate = JointKinematicsSpec {
            axis_in_parent: axis,
            rate_rad_per_s: 0.4 + 0.7,
            initial_angle_rad: 0.1 + 0.2,
        };
        let t = 3.5_f64;
        let (q_chain, av_chain) = evaluate_multi_dof(&chain, t);
        let (q_agg, av_agg) = evaluate(&aggregate, t);
        // Numerical: `incr_right` normalizes the composed quaternion
        // each stage, the aggregate kernel does not — drifts by a
        // few ULPs in the worst case but stays within trig precision.
        for i in 0..4 {
            assert!(
                (q_chain.data[i] - q_agg.data[i]).abs() < 1.0e-12,
                "q[{i}]: chain {} vs aggregate {}",
                q_chain.data[i],
                q_agg.data[i]
            );
        }
        for i in 0..3 {
            assert!(
                (av_chain[i] - av_agg[i]).abs() < 1.0e-15,
                "av[{i}]: chain {} vs aggregate {}",
                av_chain[i],
                av_agg[i]
            );
        }
    }

    /// A multi-DOF chain composed entirely of `Closure` stages is
    /// time-invariant — sampling at distinct times produces the same
    /// `(rotation, ang_vel)` snapshot. Composed angular velocity
    /// must be exactly zero (each stage contributes zero, and
    /// composing zero with zero in `incr_right` stays zero
    /// bit-identically because `T * 0 + 0 = 0`).
    #[test]
    fn multi_dof_all_closure_is_time_invariant() {
        let spec = MultiDofJointKinematicsSpec::from_slice(&[
            SingleDofKinematics::Closure(ClosureJointKinematicsSpec {
                axis_in_parent: DVec3::Z,
                fixed_angle_rad: 0.5,
            }),
            SingleDofKinematics::Closure(ClosureJointKinematicsSpec {
                axis_in_parent: DVec3::Y,
                fixed_angle_rad: -0.3,
            }),
        ]);
        let (q0, av0) = evaluate_multi_dof(&spec, 0.0);
        for &t in &[1.0_f64, 100.0, 1.0e6] {
            let (q, av) = evaluate_multi_dof(&spec, t);
            for i in 0..4 {
                assert_eq!(
                    q.data[i].to_bits(),
                    q0.data[i].to_bits(),
                    "t={t} q[{i}] not bit-identical"
                );
            }
            assert_eq!(av, av0);
            assert_eq!(av, DVec3::ZERO);
        }
    }

    /// A "hole" in the middle of the axes array (populated, then
    /// `None`, then populated again) is a configuration error and
    /// must panic. This invariant prevents a silent mid-chain stage
    /// drop.
    /// An empty multi-DOF chain (`axes` all `None`) returning the
    /// no-op `(identity, zero)` must still fail loud when the clock
    /// is non-finite. Without the top-level guard, the chain's
    /// fast-path return would hide a `NaN` simulation time from
    /// every per-stage evaluator and violate the fail-loud contract
    /// the populated chain inherits.
    #[test]
    #[should_panic(expected = "elapsed_seconds must be finite")]
    fn multi_dof_empty_chain_nan_time_panics() {
        let spec = MultiDofJointKinematicsSpec {
            axes: [None; MAX_MULTI_DOF_AXES],
        };
        let _ = evaluate_multi_dof(&spec, f64::NAN);
    }

    #[test]
    #[should_panic(expected = "contiguous prefix")]
    fn multi_dof_hole_in_middle_panics() {
        let mut axes = [None; MAX_MULTI_DOF_AXES];
        axes[0] = Some(SingleDofKinematics::ConstantRate(JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: 0.1,
            initial_angle_rad: 0.0,
        }));
        // axes[1] left as None.
        axes[2] = Some(SingleDofKinematics::ConstantRate(JointKinematicsSpec {
            axis_in_parent: DVec3::Y,
            rate_rad_per_s: 0.2,
            initial_angle_rad: 0.0,
        }));
        let spec = MultiDofJointKinematicsSpec { axes };
        let _ = evaluate_multi_dof(&spec, 0.0);
    }

    /// `from_slice` panics when the caller asks for more DOFs than
    /// `MAX_MULTI_DOF_AXES` supports.
    #[test]
    #[should_panic(expected = "supports at most")]
    fn multi_dof_from_slice_too_many_panics() {
        let dof = SingleDofKinematics::ConstantRate(JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: 0.1,
            initial_angle_rad: 0.0,
        });
        let too_many = vec![dof; MAX_MULTI_DOF_AXES + 1];
        let _ = MultiDofJointKinematicsSpec::from_slice(&too_many);
    }

    // ===========================================================================
    // JointKinematicsModel dispatch
    // ===========================================================================

    /// `evaluate_model` dispatches each variant to the right kernel,
    /// returning bit-identical output to a direct call.
    #[test]
    fn evaluate_model_dispatches_to_each_variant() {
        let t = 1.7_f64;

        let const_spec = JointKinematicsSpec {
            axis_in_parent: DVec3::Z,
            rate_rad_per_s: 0.5,
            initial_angle_rad: 0.1,
        };
        let (q_direct, av_direct) = evaluate(&const_spec, t);
        let (q_model, av_model) =
            evaluate_model(&JointKinematicsModel::ConstantRate(const_spec), t);
        assert_eq!(q_model, q_direct);
        assert_eq!(av_model, av_direct);

        let sin_spec = SinusoidalJointKinematicsSpec {
            axis_in_parent: DVec3::Y,
            amplitude_rad: 0.4,
            omega_rad_per_s: 2.0,
            phase_rad: 0.3,
            offset_rad: 0.0,
        };
        let (q_direct, av_direct) = evaluate_sinusoidal(&sin_spec, t);
        let (q_model, av_model) = evaluate_model(&JointKinematicsModel::Sinusoidal(sin_spec), t);
        assert_eq!(q_model, q_direct);
        assert_eq!(av_model, av_direct);

        let close_spec = ClosureJointKinematicsSpec {
            axis_in_parent: DVec3::X,
            fixed_angle_rad: 0.7,
        };
        let (q_direct, av_direct) = evaluate_closure(&close_spec, t);
        let (q_model, av_model) = evaluate_model(&JointKinematicsModel::Closure(close_spec), t);
        assert_eq!(q_model, q_direct);
        assert_eq!(av_model, av_direct);

        let multi_spec = MultiDofJointKinematicsSpec::from_slice(&[
            SingleDofKinematics::ConstantRate(const_spec),
            SingleDofKinematics::Sinusoidal(sin_spec),
        ]);
        let (q_direct, av_direct) = evaluate_multi_dof(&multi_spec, t);
        let (q_model, av_model) = evaluate_model(&JointKinematicsModel::MultiDof(multi_spec), t);
        assert_eq!(q_model, q_direct);
        assert_eq!(av_model, av_direct);
    }
}
